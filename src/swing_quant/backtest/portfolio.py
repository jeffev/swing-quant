"""Carteira multi-estratégia: combina painéis, aplica regime + regras de risco, relata.

Painel combinado: cada coluna é "TICKER@estrategia" (mesmos preços, sinais distintos);
`underlying` garante uma posição por ticker e `strategy_of` habilita o cap por estratégia.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from swing_quant.backtest.engine import Backtester, BacktestResult, CostModel, RiskModel
from swing_quant.backtest.metrics import Metrics, benchmark_metrics, compute_metrics
from swing_quant.backtest.panel import Panel
from swing_quant.backtest.report import _df_table, _f, _metrics_table, _pct, make_plots_basic
from swing_quant.backtest.validation import Window, time_split
from swing_quant.risk.regime import Regime


def combine_panels(panels: Mapping[str, Panel], sectors: Mapping[str, str] | None = None) -> Panel:
    """Empilha painéis de estratégias diferentes num único painel (colunas TICKER@estrategia)."""
    if not panels:
        raise ValueError("nenhum painel para combinar")
    dates = pd.DatetimeIndex(sorted(set().union(*(set(p.dates) for p in panels.values()))))
    cols: list[str] = []
    underlying: list[str] = []
    strategy_of: list[str] = []
    parts: dict[str, list[pd.DataFrame]] = {
        k: []
        for k in (
            "open",
            "high",
            "low",
            "close",
            "atr",
            "dollar_vol",
            "entry",
            "exit",
            "stop",
            "target",
            "score",
            "max_hold",
        )
    }
    for name, p in panels.items():
        rename = {t: f"{t}@{name}" for t in p.tickers}
        cols += [rename[t] for t in p.tickers]
        underlying += list(p.underlying)
        strategy_of += [name] * len(p.tickers)
        for k in parts:
            parts[k].append(getattr(p, k).reindex(dates).rename(columns=rename))

    def cat(k: str) -> pd.DataFrame:
        return pd.concat(parts[k], axis=1)

    return Panel(
        dates=dates,
        tickers=cols,
        open=cat("open"),
        high=cat("high"),
        low=cat("low"),
        close=cat("close"),
        atr=cat("atr"),
        dollar_vol=cat("dollar_vol"),
        entry=cat("entry").fillna(False).astype(bool),
        exit=cat("exit").fillna(False).astype(bool),
        stop=cat("stop"),
        target=cat("target"),
        score=cat("score"),
        max_hold=cat("max_hold").fillna(0).astype(int),
        meta={"strategies": list(panels), "n_columns": len(cols)},
        underlying=underlying,
        strategy_of=strategy_of,
        sectors=dict(sectors or {}),
    )


@dataclass
class PortfolioResult:
    market: str
    strategies: list[str]
    full: BacktestResult
    metrics_full: Metrics
    metrics_test: Metrics
    test_window: Window
    attribution: pd.DataFrame
    benchmark: dict[str, float]
    regime_summary: dict[str, float]
    risk_events: dict[str, int]
    with_vs_without: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def attribution_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["strategy", "n_trades", "pnl", "win_rate", "profit_factor"])
    g = trades.groupby("strategy")
    out = pd.DataFrame(
        {
            "n_trades": g.size(),
            "pnl": g["pnl"].sum(),
            "win_rate": g["pnl"].apply(lambda s: float((s > 0).mean())),
            "profit_factor": g["pnl"].apply(
                lambda s: (
                    float(s[s > 0].sum() / -s[s <= 0].sum()) if (s <= 0).any() else float("inf")
                )
            ),
            "avg_hold": g["bars_held"].mean(),
        }
    )
    return out.reset_index()


def run_portfolio(
    panel: Panel,
    *,
    market: str,
    costs: CostModel,
    risk: RiskModel,
    regime: Regime | None = None,
    benchmark_close: pd.Series | None = None,
    rf_daily: pd.Series | None = None,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> PortfolioResult:
    """Backtest da carteira combinada (período completo + métricas no terço de teste) e uma
    comparação com/sem regras de risco e regime para medir o efeito de cada camada."""
    split = time_split(panel.dates, split_fractions)
    bt = Backtester(
        costs,
        risk,
        allow_entries=regime.allow_entries if regime else None,
        size_factor=regime.size_factor if regime else None,
        cash_rate=rf_daily,
    )
    full = bt.run(panel)
    m_full = compute_metrics(full.equity, full.trades, full.exposure, rf=rf_daily)
    test_res = bt.run(panel.slice(split.test.start, split.test.end))
    m_test = compute_metrics(test_res.equity, test_res.trades, test_res.exposure, rf=rf_daily)

    # ablação: sem regime / sem regras de carteira / sem ambos
    plain_risk = RiskModel(
        **{
            **risk.__dict__,
            "max_sector_pct": 0.0,
            "max_strategy_pct": 0.0,
            "max_correlation": 0.0,
            "monthly_dd_reduce": 0.0,
            "circuit_breaker_dd": 0.0,
        }
    )
    variants = {
        "completo (regime + risco)": bt,
        "sem regime": Backtester(costs, risk, cash_rate=rf_daily),
        "só tendência + risco": Backtester(
            costs,
            risk,
            allow_entries=regime.allow_entries if regime else None,
            cash_rate=rf_daily,
        ),
        "só vol + risco": Backtester(
            costs,
            risk,
            size_factor=regime.size_factor if regime else None,
            cash_rate=rf_daily,
        ),
        "sem regras de carteira": Backtester(
            costs,
            plain_risk,
            allow_entries=regime.allow_entries if regime else None,
            size_factor=regime.size_factor if regime else None,
            cash_rate=rf_daily,
        ),
        "sem regime nem regras": Backtester(costs, plain_risk, cash_rate=rf_daily),
    }
    rows = []
    for label, b in variants.items():
        res = b.run(panel)
        m = compute_metrics(res.equity, res.trades, res.exposure, rf=rf_daily)
        rows.append(
            {
                "variante": label,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "n_trades": m.n_trades,
                "exposure_avg": m.exposure_avg,
            }
        )
    ablation = pd.DataFrame(rows)

    bench = (
        benchmark_metrics(
            benchmark_close.loc[full.equity.index.min() : full.equity.index.max()], rf_daily
        )
        if benchmark_close is not None
        else {}
    )
    notes = [
        "Universo = snapshot atual do índice (viés de sobrevivência).",
        "Métricas de teste usam o último terço do período; parâmetros das estratégias vêm do "
        "config.yaml (não reotimizados aqui).",
    ]
    return PortfolioResult(
        market=market,
        strategies=list(cast("list[str]", panel.meta.get("strategies", []))),
        full=full,
        metrics_full=m_full,
        metrics_test=m_test,
        test_window=split.test,
        attribution=attribution_table(full.trades),
        benchmark=bench,
        regime_summary=regime.summary() if regime else {},
        risk_events=full.risk_events,
        with_vs_without=ablation,
        notes=notes,
    )


def render_portfolio_markdown(r: PortfolioResult, plots: Mapping[str, Path] | None = None) -> str:
    lines = [
        f"# Carteira — {', '.join(r.strategies)} / {r.market.upper()}",
        "",
        f"**Gerado em**: {dt.datetime.now():%Y-%m-%d %H:%M}  ",
        f"**Teste (último terço)**: {r.test_window}",
        "",
        "## Métricas",
        "",
        _metrics_table({"Completo": r.metrics_full, "Teste": r.metrics_test}),
    ]
    if r.benchmark:
        lines += [
            "",
            f"Benchmark buy-and-hold: CAGR {_pct(r.benchmark.get('cagr'))}, "
            f"Sharpe {_f(r.benchmark.get('sharpe'))}, MDD {_pct(r.benchmark.get('max_drawdown'))}",
        ]
    if plots:
        lines += ["", "## Gráficos", ""] + [f"![{k}]({p.name})" for k, p in plots.items()]
    lines += ["", "## Atribuição por estratégia (período completo)", "", _df_table(r.attribution)]
    lines += [
        "",
        "## Efeito das camadas (ablação, período completo)",
        "",
        _df_table(r.with_vs_without),
    ]
    if r.regime_summary:
        lines += [
            "",
            "## Regime",
            "",
            f"- Dias com tendência ligada: {_pct(r.regime_summary.get('pct_days_trend_on'))}",
            f"- Dias com vol alta (sizing reduzido): {_pct(r.regime_summary.get('pct_days_high_vol'))}",
        ]
    lines += ["", "## Eventos de risco (período completo)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(r.risk_events.items())] or ["- nenhum"]
    lines += [
        "",
        "## Saídas por motivo",
        "",
        _df_table(
            r.full.trades["exit_reason"].value_counts().rename_axis("motivo").reset_index(name="n")
        ),
    ]
    lines += ["", "## Observações", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines) + "\n"


def save_portfolio_report(r: PortfolioResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{r.market}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    plots = make_plots_basic(r.full.equity, out_dir, stem, title=f"Carteira {r.market.upper()}")
    path = out_dir / f"{stem}.md"
    path.write_text(render_portfolio_markdown(r, plots), encoding="utf-8")
    r.full.trades.to_csv(out_dir / f"{stem}_trades.csv", index=False)
    return path
