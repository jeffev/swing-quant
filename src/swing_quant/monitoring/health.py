"""Saúde por estratégia e regra de desligamento automático (docs/04 §4).

Mensalmente: compara o realizado com o esperado do último backtest da estratégia.
Alerta se Sharpe rolling 6 meses < 0 **ou** drawdown atual pior que o p95 simulado de 1 ano
do backtest (bootstrap em blocos dos retornos diários, ADR-017).
Dois alertas consecutivos → estratégia **pausada** (o screener deixa de gerar entradas para ela;
saídas de posições abertas continuam). Reativação é decisão humana (`health --resume`).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass

import pandas as pd

from swing_quant.backtest.metrics import drawdown_series, sharpe_ratio
from swing_quant.data.store import MarketStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_status (
  strategy   VARCHAR NOT NULL,
  market     VARCHAR NOT NULL,
  as_of      DATE    NOT NULL,
  status     VARCHAR NOT NULL,      -- active | alert | paused
  reason     VARCHAR,
  sharpe_6m  DOUBLE,
  drawdown   DOUBLE,
  expected   JSON,
  consecutive_alerts INTEGER,
  PRIMARY KEY (strategy, market, as_of)
);
"""
ROLLING_WINDOW = 126  # ~6 meses
MIN_OBS = 40  # menos que isso: sem veredito (dados insuficientes)


@dataclass(frozen=True)
class Expected:
    sharpe: float = float("nan")
    cagr: float = float("nan")
    max_drawdown: float = float("nan")
    dd_p95: float = float("nan")
    """p95 do drawdown simulado do backtest — bootstrap diário (ADR-017), MC de trades se antigo."""
    run_id: str | None = None


@dataclass(frozen=True)
class HealthReport:
    strategy: str
    market: str
    as_of: dt.date
    status: str
    reason: str
    sharpe_6m: float
    drawdown: float
    consecutive_alerts: int
    expected: Expected
    n_obs: int

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["expected"] = asdict(self.expected)
        return d


def latest_expected(store: MarketStore, strategy: str, market: str) -> Expected:
    """Métricas do backtest mais recente da estratégia no mercado (tabela backtest_runs)."""
    row = store.con.execute(
        "SELECT run_id, metrics FROM backtest_runs WHERE strategy = ? "
        "ORDER BY created_at DESC LIMIT 1",
        [f"{strategy}/{market}"],
    ).fetchone()
    if row is None:
        return Expected()
    m = json.loads(row[1]) if isinstance(row[1], str) else row[1]
    full = m.get("full", {})
    # runs anteriores ao ADR-017 só têm o MC por embaralhamento de trades
    dd = m.get("dd_bootstrap") or m.get("monte_carlo", {})
    return Expected(
        sharpe=float(full.get("sharpe", float("nan"))),
        cagr=float(full.get("cagr", float("nan"))),
        max_drawdown=float(full.get("max_drawdown", float("nan"))),
        dd_p95=float(dd.get("mdd_p95", float("nan"))),
        run_id=str(row[0]),
    )


def evaluate(
    returns: pd.Series,
    expected: Expected,
    *,
    prior_alerts: int = 0,
    window: int = ROLLING_WINDOW,
    min_obs: int = MIN_OBS,
) -> tuple[str, str, float, float, int]:
    """(status, reason, sharpe_6m, drawdown, consecutive_alerts) a partir dos retornos diários
    realizados da estratégia."""
    r = returns.dropna()
    if len(r) < min_obs:
        return (
            "active",
            f"dados insuficientes ({len(r)} obs < {min_obs})",
            float("nan"),
            float("nan"),
            0,
        )
    recent = r.iloc[-window:]
    sharpe_6m = sharpe_ratio(recent)
    equity = (1 + r).cumprod()
    dd = float(drawdown_series(equity).iloc[-1])
    reasons = []
    if not math.isnan(sharpe_6m) and sharpe_6m < 0:
        reasons.append(f"Sharpe 6m {sharpe_6m:.2f} < 0")
    if not math.isnan(expected.dd_p95) and dd < expected.dd_p95:
        reasons.append(f"drawdown {dd:.1%} pior que o p95 simulado {expected.dd_p95:.1%}")
    if not reasons:
        return "active", "ok", sharpe_6m, dd, 0
    consecutive = prior_alerts + 1
    status = "paused" if consecutive >= 2 else "alert"
    return status, "; ".join(reasons), sharpe_6m, dd, consecutive


class HealthStore:
    def __init__(self, store: MarketStore) -> None:
        self.con = store.con
        self.con.execute(_SCHEMA)

    def last(self, strategy: str, market: str) -> dict[str, object] | None:
        row = self.con.execute(
            "SELECT status, consecutive_alerts, as_of FROM strategy_status "
            "WHERE strategy = ? AND market = ? ORDER BY as_of DESC LIMIT 1",
            [strategy, market],
        ).fetchone()
        if row is None:
            return None
        return {"status": row[0], "consecutive_alerts": int(row[1] or 0), "as_of": row[2]}

    def record(self, rep: HealthReport) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO strategy_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rep.strategy,
                rep.market,
                rep.as_of,
                rep.status,
                rep.reason,
                None if math.isnan(rep.sharpe_6m) else rep.sharpe_6m,
                None if math.isnan(rep.drawdown) else rep.drawdown,
                json.dumps(asdict(rep.expected)),
                rep.consecutive_alerts,
            ],
        )

    def paused(self, market: str) -> set[str]:
        rows = self.con.execute(
            """
            SELECT strategy FROM strategy_status s
            WHERE market = ? AND as_of = (
                SELECT max(as_of) FROM strategy_status
                WHERE strategy = s.strategy AND market = s.market)
              AND status = 'paused'
            """,
            [market],
        ).fetchall()
        return {str(r[0]) for r in rows}

    def resume(self, strategy: str, market: str, as_of: dt.date) -> None:
        """Reativação manual: grava um registro `active` com contador zerado."""
        self.con.execute(
            "INSERT OR REPLACE INTO strategy_status VALUES (?, ?, ?, 'active', "
            "'reativada manualmente', NULL, NULL, NULL, 0)",
            [strategy, market, as_of],
        )

    def history(self, market: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM strategy_status"
        params: list[object] = []
        if market:
            sql += " WHERE market = ?"
            params.append(market)
        return self.con.execute(sql + " ORDER BY as_of DESC, strategy", params).df()


def run_health(
    store: MarketStore,
    market: str,
    strategy_returns: dict[str, pd.Series],
    strategies: list[str],
    as_of: dt.date,
) -> list[HealthReport]:
    """Avalia todas as estratégias habilitadas e persiste o resultado."""
    hs = HealthStore(store)
    reports = []
    for name in strategies:
        expected = latest_expected(store, name, market)
        prev = hs.last(name, market)
        prior = int(str(prev["consecutive_alerts"])) if prev and prev["status"] != "paused" else 0
        if prev and prev["status"] == "paused":
            # continua pausada até reativação manual
            rep = HealthReport(
                name,
                market,
                as_of,
                "paused",
                "pausada (aguardando revisão)",
                float("nan"),
                float("nan"),
                int(str(prev["consecutive_alerts"])),
                expected,
                0,
            )
        else:
            r = strategy_returns.get(name, pd.Series(dtype=float))
            status, reason, s6, dd, cons = evaluate(r, expected, prior_alerts=prior)
            rep = HealthReport(name, market, as_of, status, reason, s6, dd, cons, expected, len(r))
        hs.record(rep)
        reports.append(rep)
    return reports
