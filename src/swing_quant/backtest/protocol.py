"""Orquestra o protocolo completo de validação de uma estratégia (docs/04 §2–3).

Etapas:
 1. split 60/20/20 → grid no treino, escolha por Sharpe (mín. trades)
 2. robustez (platô) no treino
 3. walk-forward em treino+validação
 4. teste OOS **uma vez** com os parâmetros escolhidos
 5. drawdown simulado — bootstrap em blocos dos retornos diários (gate, ADR-017) e Monte Carlo
    por embaralhamento de trades (informativo) —, bootstrap do Sharpe OOS
 6. sensibilidade a custos (0–3×) no período completo
 7. baseline aleatória
 8. (opcional) mercado cruzado com os mesmos parâmetros
 9. checklist de aprovação
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from swing_quant.backtest.engine import Backtester, BacktestResult, CostModel, RiskModel
from swing_quant.backtest.metrics import TRADING_DAYS, Metrics, benchmark_metrics
from swing_quant.backtest.validation import (
    PanelFactory,
    Split,
    WalkForwardResult,
    block_bootstrap_drawdown,
    bootstrap_sharpe,
    cost_sensitivity,
    evaluate,
    grid_search,
    monte_carlo_drawdown,
    plateau_ratio,
    random_baseline,
    select_best,
    time_split,
    walk_forward,
)
from swing_quant.strategies.base import Strategy


@dataclass(frozen=True)
class ApprovalThresholds:
    sharpe_oos_min: float = 0.8
    profit_factor_min: float = 1.4
    min_trades_total: int = 200
    min_trades_test: int = 30
    wf_efficiency_min: float = 0.5
    plateau_min: float = 0.7
    cost_mult_must_profit: float = 2.0
    cross_market_sharpe_min: float = 0.0
    dd_p95_max: float = 0.15
    """Gate de drawdown: p95 do bootstrap em blocos dos retornos diários em 1 ano (ADR-017)."""
    dd_horizon: int = TRADING_DAYS
    """Horizonte do gate, em pregões. MDD cresce com o horizonte — fixá-lo é o que torna
    o alvo de 15% comparável entre estratégias de giro diferente."""
    mc_mdd_p95_max: float = 0.15
    """Nível de ruína do Monte Carlo por embaralhamento de trades — informativo desde ADR-017."""


@dataclass
class ProtocolResult:
    strategy_name: str
    market: str
    params: dict[str, Any]
    split: Split
    grid: pd.DataFrame
    plateau: float
    walk_forward: WalkForwardResult
    full: BacktestResult
    metrics_full: Metrics
    metrics_train: Metrics
    metrics_test: Metrics
    benchmark: dict[str, float]
    monte_carlo: dict[str, float]
    dd_bootstrap: dict[str, float]
    dd_bootstrap_full: dict[str, float]
    bootstrap: dict[str, float]
    costs: pd.DataFrame
    baseline: dict[str, float]
    cross_market: dict[str, float] | None
    checklist: dict[str, bool]
    thresholds: ApprovalThresholds
    notes: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return all(self.checklist.values())


def approval_checklist(
    r_test: Metrics,
    r_full: Metrics,
    wf_eff: float,
    plateau: float,
    costs: pd.DataFrame,
    boot: dict[str, float],
    dd_boot: dict[str, float],
    cross: dict[str, float] | None,
    thr: ApprovalThresholds,
) -> dict[str, bool]:
    cost_row = costs[costs["cost_mult"] == thr.cost_mult_must_profit]
    profitable_2x = bool(not cost_row.empty and cost_row["total_return"].iloc[0] > 0)
    checks = {
        f"sharpe_oos >= {thr.sharpe_oos_min}": bool(r_test.sharpe >= thr.sharpe_oos_min),
        f"profit_factor_oos >= {thr.profit_factor_min}": bool(
            r_test.profit_factor >= thr.profit_factor_min
        ),
        f"n_trades_total >= {thr.min_trades_total}": r_full.n_trades >= thr.min_trades_total,
        f"n_trades_test >= {thr.min_trades_test}": r_test.n_trades >= thr.min_trades_test,
        f"wf_efficiency >= {thr.wf_efficiency_min}": bool(wf_eff >= thr.wf_efficiency_min),
        f"plateau >= {thr.plateau_min}": bool(plateau >= thr.plateau_min),
        f"profitable_at_{thr.cost_mult_must_profit:g}x_costs": profitable_2x,
        "bootstrap_ci_excludes_zero": bool(boot.get("sharpe_lo", float("nan")) > 0),
        f"dd_p95 em 1 ano <= {thr.dd_p95_max}": bool(
            dd_boot.get("mdd_p95", float("nan")) >= -thr.dd_p95_max
        ),
    }
    if cross is not None:
        checks[f"cross_market_sharpe > {thr.cross_market_sharpe_min}"] = bool(
            cross.get("sharpe", float("nan")) > thr.cross_market_sharpe_min
        )
    return checks


def run_protocol(
    strategy: Strategy,
    make_panel: PanelFactory,
    *,
    market: str,
    costs: CostModel,
    risk: RiskModel,
    grid: Mapping[str, list[Any]] | None = None,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    train_years: int = 3,
    test_years: int = 1,
    anchored: bool = False,
    cost_multipliers: tuple[float, ...] = (0, 1, 2, 3),
    mc_runs: int = 1000,
    boot_runs: int = 1000,
    baseline_runs: int = 30,
    benchmark_close: pd.Series | None = None,
    cross_panel_factory: PanelFactory | None = None,
    cross_costs: CostModel | None = None,
    thresholds: ApprovalThresholds | None = None,
    min_trades_select: int = 30,
) -> ProtocolResult:
    thr = thresholds or ApprovalThresholds()
    grid = dict(grid if grid is not None else strategy.default_grid)
    bt = Backtester(costs, risk)
    notes: list[str] = []

    base_panel = make_panel(strategy)
    split = time_split(base_panel.dates, split_fractions)

    # 1) grid no treino
    g = grid_search(strategy, grid, make_panel, bt, split.train)
    best = select_best(g, min_trades=min_trades_select)
    params = {k: best[k] for k in grid}
    chosen = strategy.with_params(**params)
    panel = make_panel(chosen)

    # 2) platô
    plat = plateau_ratio(g, params, grid)

    # 3) walk-forward em treino+validação
    wf_dates = base_panel.dates[base_panel.dates <= split.val.end]
    wf = walk_forward(
        strategy,
        grid,
        make_panel,
        bt,
        wf_dates,
        train_years=train_years,
        test_years=test_years,
        anchored=anchored,
        min_trades=min_trades_select,
    )

    # 4) teste OOS (uma vez) + treino + completo
    _, m_train = evaluate(panel, bt, split.train)
    _, m_test = evaluate(panel, bt, split.test)
    full_res, m_full = evaluate(panel, bt)

    # 5) drawdown simulado (gate = bootstrap diário; MC de trades fica como stress informativo)
    #    e bootstrap do Sharpe OOS
    daily_rets = full_res.equity.pct_change().dropna()
    dd_boot = block_bootstrap_drawdown(
        daily_rets, runs=mc_runs, ruin_level=thr.dd_p95_max, horizon=thr.dd_horizon
    )
    dd_boot_full = block_bootstrap_drawdown(
        daily_rets, runs=mc_runs, ruin_level=thr.dd_p95_max, seed=1
    )
    mc = monte_carlo_drawdown(
        full_res.trades["pnl"], risk.initial_capital, runs=mc_runs, ruin_level=thr.mc_mdd_p95_max
    )
    test_res, _ = evaluate(panel, bt, split.test)
    boot = bootstrap_sharpe(test_res.equity.pct_change().dropna(), runs=boot_runs)

    # 6) custos
    costs_df = cost_sensitivity(panel, costs, risk, cost_multipliers)

    # 7) baseline aleatória (mesmo período de teste)
    base = random_baseline(panel, bt, runs=baseline_runs, window=split.test)

    # 8) mercado cruzado
    cross: dict[str, float] | None = None
    if cross_panel_factory is not None:
        try:
            cross_panel = cross_panel_factory(chosen)
            _, m_cross = evaluate(cross_panel, Backtester(cross_costs or costs, risk))
            cross = {
                "sharpe": m_cross.sharpe,
                "cagr": m_cross.cagr,
                "profit_factor": m_cross.profit_factor,
                "n_trades": float(m_cross.n_trades),
            }
        except ValueError as exc:
            notes.append(f"mercado cruzado indisponível: {exc}")

    bench = (
        benchmark_metrics(
            benchmark_close.loc[full_res.equity.index.min() : full_res.equity.index.max()]
        )
        if benchmark_close is not None
        else {}
    )
    checklist = approval_checklist(
        m_test, m_full, wf.efficiency, plat, costs_df, boot, dd_boot, cross, thr
    )
    if base_panel.meta.get("n_tickers", 0) and "survivorship" not in " ".join(notes):
        notes.append(
            "Universo = snapshot atual do índice (sem composição histórica) → viés de "
            "sobrevivência; resultados tendem a ser otimistas até haver snapshots point-in-time."
        )
    return ProtocolResult(
        strategy_name=strategy.name,
        market=market,
        params=params,
        split=split,
        grid=g,
        plateau=plat,
        walk_forward=wf,
        full=full_res,
        metrics_full=m_full,
        metrics_train=m_train,
        metrics_test=m_test,
        benchmark=bench,
        monte_carlo=mc,
        dd_bootstrap=dd_boot,
        dd_bootstrap_full=dd_boot_full,
        bootstrap=boot,
        costs=costs_df,
        baseline=base,
        cross_market=cross,
        checklist=checklist,
        thresholds=thr,
        notes=notes,
    )
