"""Relatório mensal: realizado × esperado, aderência, slippage, saúde (docs/04 §4, roadmap F5)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from swing_quant.backtest.metrics import Metrics
from swing_quant.backtest.report import _df_table, _f, _pct
from swing_quant.monitoring.health import HealthReport
from swing_quant.monitoring.performance import (
    adherence,
    realized_metrics,
    trade_ledger,
)


@dataclass
class MonthlyReport:
    market: str
    month: str  # YYYY-MM
    equity: pd.DataFrame  # mark_to_market
    ledger: pd.DataFrame
    metrics_month: Metrics
    metrics_since_start: Metrics
    adherence: dict[str, float]
    health: list[HealthReport]
    signals_month: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthEnd(1)
    return start, end


def build_monthly(
    market: str,
    month: str,
    equity: pd.DataFrame,
    signals: pd.DataFrame,
    executions: pd.DataFrame,
    health: list[HealthReport],
) -> MonthlyReport:
    start, end = month_bounds(month)
    ledger = trade_ledger(signals, executions)
    eq_month = equity.loc[(equity.index >= start) & (equity.index <= end), "equity"]
    if eq_month.empty:
        eq_month = equity["equity"].iloc[-2:]
    ledger_month = (
        ledger[(ledger["exit_date"] >= start) & (ledger["exit_date"] <= end)]
        if not ledger.empty
        else ledger
    )
    sig_month = (
        signals[
            (pd.to_datetime(signals["as_of"]) >= start) & (pd.to_datetime(signals["as_of"]) <= end)
        ]
        if not signals.empty
        else signals
    )
    ex_month = (
        executions[executions["signal_id"].isin(sig_month["id"])]
        if not executions.empty
        else executions
    )
    notes = []
    if executions.empty:
        notes.append("Nenhuma execução registrada ainda — métricas realizadas são a curva plana.")
    return MonthlyReport(
        market=market,
        month=month,
        equity=equity,
        ledger=ledger,
        metrics_month=realized_metrics(eq_month, ledger_month),
        metrics_since_start=realized_metrics(equity["equity"], ledger),
        adherence=adherence(sig_month, ex_month),
        health=health,
        signals_month=sig_month,
        notes=notes,
    )


def render_monthly(r: MonthlyReport) -> str:
    lines = [
        f"# Relatório mensal — {r.market.upper()} — {r.month}",
        "",
        f"**Gerado em**: {dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Realizado",
        "",
        "| Métrica | Mês | Desde o início |",
        "|---|---|---|",
    ]
    m, s = r.metrics_month, r.metrics_since_start
    rows: list[tuple[str, Callable[[Metrics], str]]] = [
        ("Retorno", lambda x: _pct(x.total_return)),
        ("Sharpe", lambda x: _f(x.sharpe)),
        ("Max drawdown", lambda x: _pct(x.max_drawdown)),
        ("Trades fechados", lambda x: str(x.n_trades)),
        ("Win rate", lambda x: _pct(x.win_rate)),
        ("Profit factor", lambda x: _f(x.profit_factor)),
        ("Custos", lambda x: _f(x.fees_total, ",.2f")),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | {fn(m)} | {fn(s)} |")
    lines += [
        "",
        "## Realizado × esperado (por estratégia)",
        "",
        "| Estratégia | Status | Motivo | Sharpe 6m real | Sharpe backtest | DD atual | DD p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in r.health:
        lines.append(
            f"| {h.strategy} | {h.status} | {h.reason} | {_f(h.sharpe_6m)} | {_f(h.expected.sharpe)} | "
            f"{_pct(h.drawdown)} | {_pct(h.expected.dd_p95)} |"
        )
    lines += [
        "",
        "## Aderência sinal → execução (mês)",
        "",
        f"- Compras: {r.adherence.get('buy_signals', 0):.0f} sinais, "
        f"{_pct(r.adherence.get('buy_executed_pct'))} executados",
        f"- Vendas: {r.adherence.get('sell_signals', 0):.0f} sinais, "
        f"{_pct(r.adherence.get('sell_executed_pct'))} executados",
    ]
    if not r.ledger.empty:
        sl = r.ledger["slippage_pct"].dropna()
        if not sl.empty:
            lines += [
                "",
                "## Slippage de entrada (execução vs referência)",
                "",
                f"- Médio: {_pct(sl.mean())} · mediano: {_pct(sl.median())} · pior: {_pct(sl.max())}",
            ]
        closed = r.ledger[~r.ledger["open"]]
        if not closed.empty:
            cols = [
                "ticker",
                "strategy",
                "entry_date",
                "exit_date",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "ret",
            ]
            lines += ["", "## Trades fechados (todos)", "", _df_table(closed[cols].tail(50))]
    if not r.signals_month.empty:
        cols = ["as_of", "side", "ticker", "strategy", "ref_price", "qty", "stop_price"]
        lines += ["", "## Sinais do mês", "", _df_table(r.signals_month[cols])]
    if r.notes:
        lines += ["", "## Observações", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines) + "\n"


def save_monthly(r: MonthlyReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"monthly_{r.market}_{r.month}.md"
    path.write_text(render_monthly(r), encoding="utf-8")
    return path
