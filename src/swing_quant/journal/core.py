"""Journal: persistência de sinais gerados e execuções reais (docs/04 §4, docs/05 §5).

Modelo:
- `signals`: todo sinal emitido pelo screener (side buy/sell), com preço de referência, stop,
  quantidade sugerida, score, max_hold, regime.
- `executions`: o que foi de fato executado, ligado ao sinal (`signal_id`), com `side`.
- Posição aberta = sinal de compra com execução `buy` cuja quantidade ainda não foi zerada por
  execuções `sell` ligadas ao mesmo `signal_id`.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import pandas as pd

from swing_quant.data.store import MarketStore
from swing_quant.screener.core import OpenPosition, ScreenResult

_MIGRATIONS = (
    "ALTER TABLE executions ADD COLUMN IF NOT EXISTS side VARCHAR DEFAULT 'buy'",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS as_of DATE",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS market VARCHAR",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ref_signal_id INTEGER",
)


@dataclass(frozen=True)
class ExecutionRecord:
    signal_id: int
    side: str
    price: float
    qty: int
    fees: float = 0.0
    executed_at: dt.datetime | None = None


class Journal:
    def __init__(self, store: MarketStore) -> None:
        self.store = store
        self.con = store.con
        for sql in _MIGRATIONS:
            self.con.execute(sql)

    # ------------------------------------------------------------------ sinais
    def record_screen(self, result: ScreenResult, regime: str | None = None) -> list[int]:
        """Grava entradas (buy) e saídas (sell) de um `ScreenResult`. Idempotente por
        (as_of, market, ticker, strategy, side): re-rodar o screener no mesmo dia não duplica."""
        ids: list[int] = []
        now = dt.datetime.now()
        as_of = result.as_of.date()
        regime_txt = regime or json.dumps(result.regime, default=str)
        for r in result.entries.to_dict("records"):
            ids.append(
                self._upsert_signal(
                    now,
                    as_of,
                    result.market,
                    str(r["strategy"]),
                    str(r["ticker"]),
                    "buy",
                    float(r["ref_price"]),
                    None
                    if r["stop_price"] is None or pd.isna(r["stop_price"])
                    else float(r["stop_price"]),
                    int(r["qty"]),
                    float(r["score"]),
                    int(r["max_hold"]),
                    regime_txt,
                    None,
                )
            )
        for r in result.exits.to_dict("records"):
            ref = None if pd.isna(r["ref_price"]) else float(r["ref_price"])
            ids.append(
                self._upsert_signal(
                    now,
                    as_of,
                    result.market,
                    str(r["strategy"]),
                    str(r["ticker"]),
                    "sell",
                    ref if ref is not None else 0.0,
                    None,
                    int(r["qty"]),
                    0.0,
                    0,
                    str(r["reason"]),
                    None,
                )
            )
        return ids

    def _upsert_signal(
        self,
        generated_at: dt.datetime,
        as_of: dt.date,
        market: str,
        strategy: str,
        ticker: str,
        side: str,
        ref_price: float,
        stop_price: float | None,
        qty: int,
        score: float,
        max_hold: int,
        regime: str,
        ref_signal_id: int | None,
    ) -> int:
        row = self.con.execute(
            "SELECT id FROM signals WHERE as_of = ? AND market = ? AND ticker = ? "
            "AND strategy = ? AND side = ?",
            [as_of, market, ticker, strategy, side],
        ).fetchone()
        if row:
            return int(row[0])
        out = self.con.execute(
            """
            INSERT INTO signals (generated_at, strategy, ticker, side, ref_price, stop_price,
                                 qty, score, max_hold, regime, as_of, market, ref_signal_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
            """,
            [
                generated_at,
                strategy,
                ticker,
                side,
                ref_price,
                stop_price,
                qty,
                score,
                max_hold,
                regime,
                as_of,
                market,
                ref_signal_id,
            ],
        ).fetchone()
        assert out is not None
        return int(out[0])

    def signals(self, as_of: dt.date | None = None, market: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM signals"
        where: list[str] = []
        params: list[object] = []
        if as_of is not None:
            where.append("as_of = ?")
            params.append(as_of)
        if market is not None:
            where.append("market = ?")
            params.append(market)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self.con.execute(sql + " ORDER BY id", params).df()

    # ------------------------------------------------------------------ execuções
    def record_execution(self, ex: ExecutionRecord) -> None:
        sig = self.con.execute("SELECT id FROM signals WHERE id = ?", [ex.signal_id]).fetchone()
        if sig is None:
            raise KeyError(f"sinal {ex.signal_id} não existe")
        if ex.side not in ("buy", "sell"):
            raise ValueError("side deve ser buy ou sell")
        self.con.execute(
            "INSERT INTO executions (signal_id, executed_at, price, qty, fees, side) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [ex.signal_id, ex.executed_at or dt.datetime.now(), ex.price, ex.qty, ex.fees, ex.side],
        )

    def executions(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT e.*, s.ticker, s.strategy, s.market FROM executions e "
            "JOIN signals s ON s.id = e.signal_id ORDER BY e.executed_at"
        ).df()

    # ------------------------------------------------------------------ posições
    def open_positions(self, market: str | None = None) -> list[OpenPosition]:
        """Compras executadas cuja quantidade ainda não foi zerada por vendas."""
        sql = """
            WITH buys AS (
                SELECT s.id, s.ticker, s.strategy, s.market, s.stop_price, s.max_hold,
                       min(e.executed_at) AS entry_at,
                       sum(e.qty) AS qty_buy,
                       sum(e.price * e.qty) / sum(e.qty) AS avg_price
                FROM signals s JOIN executions e ON e.signal_id = s.id AND e.side = 'buy'
                WHERE s.side = 'buy'
                GROUP BY s.id, s.ticker, s.strategy, s.market, s.stop_price, s.max_hold
            ),
            sells AS (
                SELECT signal_id, sum(qty) AS qty_sell FROM executions
                WHERE side = 'sell' GROUP BY signal_id
            )
            SELECT b.id, b.ticker, b.strategy, b.market, b.stop_price, b.max_hold, b.entry_at,
                   b.qty_buy - coalesce(s.qty_sell, 0) AS qty_open, b.avg_price
            FROM buys b LEFT JOIN sells s ON s.signal_id = b.id
            WHERE b.qty_buy - coalesce(s.qty_sell, 0) > 0
        """
        rows = self.con.execute(sql).fetchall()
        out = []
        for sid, ticker, strategy, mkt, stop, max_hold, entry_at, qty_open, avg_price in rows:
            if market is not None and mkt != market:
                continue
            out.append(
                OpenPosition(
                    ticker=str(ticker),
                    strategy=str(strategy),
                    qty=int(qty_open),
                    entry_date=pd.Timestamp(entry_at).date(),
                    entry_price=float(avg_price),
                    stop_price=None if stop is None else float(stop),
                    max_hold=int(max_hold or 0),
                    signal_id=int(sid),
                )
            )
        return out

    def realized_pnl(self, market: str | None = None) -> float:
        """P&L realizado (vendas − compras proporcionais − taxas) das posições encerradas."""
        ex = self.executions()
        if ex.empty:
            return 0.0
        if market is not None:
            ex = ex[ex["market"] == market]
        total = 0.0
        for _, g in ex.groupby("signal_id"):
            buys, sells = g[g["side"] == "buy"], g[g["side"] == "sell"]
            if sells.empty or buys.empty:
                continue
            avg_buy = float((buys["price"] * buys["qty"]).sum() / buys["qty"].sum())
            total += float(((sells["price"] - avg_buy) * sells["qty"]).sum() - g["fees"].sum())
        return total

    def equity_estimate(self, initial_capital: float, market: str | None = None) -> float:
        """Capital inicial + P&L realizado (posições abertas ao custo)."""
        return initial_capital + self.realized_pnl(market)

    def invested_at_cost(self, market: str | None = None) -> float:
        return sum(p.qty * p.entry_price for p in self.open_positions(market))
