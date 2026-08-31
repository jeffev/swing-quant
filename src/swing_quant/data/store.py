"""Persistência em DuckDB: preços, universo, eventos, sinais, execuções e backtests.

Schema conforme docs/05-arquitetura.md §5. Todas as escritas de preços/universo são
idempotentes (`INSERT OR REPLACE` sobre a chave primária).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import duckdb
import pandas as pd

MACRO_COLUMNS: tuple[str, ...] = ("series", "date", "value", "unit", "source")

PRICE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
  ticker      VARCHAR NOT NULL,
  date        DATE    NOT NULL,
  open        DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  adj_close   DOUBLE,
  volume      BIGINT,
  source      VARCHAR,
  PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS universe (
  index_name  VARCHAR NOT NULL,
  as_of       DATE    NOT NULL,
  ticker      VARCHAR NOT NULL,
  sector      VARCHAR,
  PRIMARY KEY (index_name, as_of, ticker)
);

CREATE TABLE IF NOT EXISTS corporate_events (
  ticker VARCHAR NOT NULL,
  date   DATE    NOT NULL,
  type   VARCHAR NOT NULL,
  value  DOUBLE,
  PRIMARY KEY (ticker, date, type)
);

CREATE SEQUENCE IF NOT EXISTS signals_id_seq;
CREATE TABLE IF NOT EXISTS signals (
  id           INTEGER PRIMARY KEY DEFAULT nextval('signals_id_seq'),
  generated_at TIMESTAMP NOT NULL,
  strategy     VARCHAR NOT NULL,
  ticker       VARCHAR NOT NULL,
  side         VARCHAR NOT NULL,
  ref_price    DOUBLE,
  stop_price   DOUBLE,
  qty          INTEGER,
  score        DOUBLE,
  max_hold     INTEGER,
  regime       VARCHAR
);

CREATE TABLE IF NOT EXISTS executions (
  signal_id   INTEGER NOT NULL,
  executed_at TIMESTAMP NOT NULL,
  price       DOUBLE NOT NULL,
  qty         INTEGER NOT NULL,
  fees        DOUBLE DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_free (
  market       VARCHAR NOT NULL,
  date         DATE    NOT NULL,
  daily_return DOUBLE  NOT NULL,
  source       VARCHAR,
  PRIMARY KEY (market, date)
);

CREATE TABLE IF NOT EXISTS macro (
  series VARCHAR NOT NULL,
  date   DATE    NOT NULL,
  value  DOUBLE  NOT NULL,
  unit   VARCHAR NOT NULL,
  source VARCHAR,
  PRIMARY KEY (series, date)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id       VARCHAR PRIMARY KEY,
  strategy     VARCHAR NOT NULL,
  params       JSON,
  period_start DATE,
  period_end   DATE,
  metrics      JSON,
  created_at   TIMESTAMP,
  git_sha      VARCHAR
);
"""


class MarketStore:
    """Wrapper fino sobre uma conexão DuckDB com o schema do projeto."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(_SCHEMA)

    # ------------------------------------------------------------------ ciclo de vida
    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ preços
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Insere/atualiza linhas OHLCV. Espera colunas `PRICE_COLUMNS`. Retorna nº de linhas."""
        if df.empty:
            return 0
        missing = set(PRICE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"colunas ausentes em prices: {sorted(missing)}")
        clean = df.loc[:, list(PRICE_COLUMNS)].copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.date
        clean = clean.dropna(subset=["close"])
        clean["volume"] = clean["volume"].fillna(0).astype("int64")
        self.con.register("_prices_in", clean)
        self.con.execute("INSERT OR REPLACE INTO prices SELECT * FROM _prices_in")
        self.con.unregister("_prices_in")
        return len(clean)

    def last_dates(self, tickers: Iterable[str] | None = None) -> dict[str, dt.date]:
        """Última data disponível por ticker (apenas os que existem no banco)."""
        sql = "SELECT ticker, max(date) AS d FROM prices"
        params: list[object] = []
        if tickers is not None:
            tl = list(tickers)
            if not tl:
                return {}
            sql += f" WHERE ticker IN ({','.join('?' * len(tl))})"
            params = list(tl)
        sql += " GROUP BY ticker"
        rows = self.con.execute(sql, params).fetchall()
        return {str(t): d for t, d in rows}

    def get_prices(
        self,
        tickers: Sequence[str] | None = None,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
    ) -> pd.DataFrame:
        """Retorna preços em formato longo ordenado por (ticker, date)."""
        where: list[str] = []
        params: list[object] = []
        if tickers:
            where.append(f"ticker IN ({','.join('?' * len(tickers))})")
            params.extend(tickers)
        if start is not None:
            where.append("date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            where.append("date <= ?")
            params.append(pd.Timestamp(end).date())
        sql = f"SELECT {', '.join(PRICE_COLUMNS)} FROM prices"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ticker, date"
        df = self.con.execute(sql, params).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def repair_high_low(self) -> int:
        """Corrige barras com `high < low` usando o envelope de open/high/low/close.

        Marca `source` com sufixo `+repair` para auditabilidade. Retorna nº de linhas corrigidas.
        """
        row = self.con.execute("SELECT count(*) FROM prices WHERE high < low").fetchone()
        n = int(row[0]) if row else 0
        if n:
            self.con.execute(
                """
                UPDATE prices SET
                    high   = greatest(open, high, low, close),
                    low    = least(open, high, low, close),
                    source = coalesce(source, '') || '+repair'
                WHERE high < low
                """
            )
        return n

    def tickers(self) -> list[str]:
        rows = self.con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
        return [str(r[0]) for r in rows]

    def price_count(self) -> int:
        row = self.con.execute("SELECT count(*) FROM prices").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ universo
    def upsert_universe(self, index_name: str, as_of: dt.date, members: pd.DataFrame) -> int:
        """Grava snapshot da composição. `members` precisa de `ticker` e opcionalmente `sector`."""
        if members.empty:
            return 0
        snap = pd.DataFrame(
            {
                "index_name": index_name,
                "as_of": as_of,
                "ticker": members["ticker"].astype(str).to_numpy(),
                "sector": members["sector"].astype("string").to_numpy()
                if "sector" in members
                else None,
            }
        )
        self.con.register("_uni_in", snap)
        self.con.execute("INSERT OR REPLACE INTO universe SELECT * FROM _uni_in")
        self.con.unregister("_uni_in")
        return len(snap)

    def universe_snapshots(self, index_name: str) -> list[dt.date]:
        rows = self.con.execute(
            "SELECT DISTINCT as_of FROM universe WHERE index_name = ? ORDER BY as_of",
            [index_name],
        ).fetchall()
        return [r[0] for r in rows]

    def universe_at(self, index_name: str, as_of: dt.date | None = None) -> pd.DataFrame:
        """Composição vigente em `as_of` (último snapshot ≤ as_of). Sem `as_of`, o mais recente."""
        if as_of is None:
            sql = """
                SELECT ticker, sector FROM universe
                WHERE index_name = ?
                  AND as_of = (SELECT max(as_of) FROM universe WHERE index_name = ?)
                ORDER BY ticker
            """
            return self.con.execute(sql, [index_name, index_name]).df()
        sql = """
            SELECT ticker, sector FROM universe
            WHERE index_name = ?
              AND as_of = (SELECT max(as_of) FROM universe WHERE index_name = ? AND as_of <= ?)
            ORDER BY ticker
        """
        return self.con.execute(sql, [index_name, index_name, as_of]).df()

    def risk_free(self, market: str) -> pd.Series:
        """Série diária de renda fixa do mercado (CDI ou T-bills), indexada por data.

        É o piso de comparação do ADR-020: entra no Sharpe como custo de oportunidade e no
        engine como rendimento do caixa. Vazia quando ainda não houve `update-riskfree`.
        """
        df = self.con.execute(
            "SELECT date, daily_return FROM risk_free WHERE market = ? ORDER BY date", [market]
        ).df()
        if df.empty:
            return pd.Series(dtype=float)
        return pd.Series(df["daily_return"].to_numpy(), index=pd.DatetimeIndex(df["date"]))

    # ------------------------------------------------------------------ eventos
    def upsert_corporate_events(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        ev = df.loc[:, ["ticker", "date", "type", "value"]].copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.date
        self.con.register("_ev_in", ev)
        self.con.execute("INSERT OR REPLACE INTO corporate_events SELECT * FROM _ev_in")
        self.con.unregister("_ev_in")
        return len(ev)

    # ------------------------------------------------------------------ macro
    def upsert_macro(self, df: pd.DataFrame) -> int:
        """Insere/atualiza séries macro. Espera `MACRO_COLUMNS`. Retorna nº de linhas."""
        if df.empty:
            return 0
        missing = set(MACRO_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"colunas ausentes em macro: {sorted(missing)}")
        clean = df.loc[:, list(MACRO_COLUMNS)].copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.date
        clean = clean.dropna(subset=["value"]).drop_duplicates(
            subset=["series", "date"], keep="last"
        )
        self.con.register("_macro_in", clean)
        self.con.execute("INSERT OR REPLACE INTO macro SELECT * FROM _macro_in")
        self.con.unregister("_macro_in")
        return len(clean)

    def macro(self, series: str) -> pd.Series:
        """Série macro gravada, indexada por data. Vazia quando ainda não houve `update-macro`.

        A unidade (`index`, `pct_month`, `daily_return`) está no catálogo de
        `swing_quant.data.macro`, não no valor: quem lê precisa saber o que está lendo.
        """
        df = self.con.execute(
            "SELECT date, value FROM macro WHERE series = ? ORDER BY date", [series]
        ).df()
        if df.empty:
            return pd.Series(dtype=float)
        return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["date"]), name=series)

    def macro_series(self) -> list[str]:
        rows = self.con.execute("SELECT DISTINCT series FROM macro ORDER BY series").fetchall()
        return [str(r[0]) for r in rows]
