"""Protocolo de validação (docs/04 §2): split temporal, grid/robustez, walk-forward,
Monte Carlo, bootstrap, sensibilidade a custos e baseline aleatória.

Funções puras sobre `Panel`/`BacktestResult`; a orquestração fica em `protocol.py`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from swing_quant.backtest.engine import Backtester, BacktestResult, CostModel, RiskModel
from swing_quant.backtest.metrics import Metrics, compute_metrics, max_drawdown, sharpe_ratio
from swing_quant.backtest.panel import Panel, build_panel
from swing_quant.strategies.base import Strategy

PanelFactory = Callable[[Strategy], Panel]
"""Constrói o painel para uma instância de estratégia (encapsula preços + config)."""


# --------------------------------------------------------------------------- split
@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

    def __str__(self) -> str:
        return f"{self.start.date()} → {self.end.date()}"


@dataclass(frozen=True)
class Split:
    train: Window
    val: Window
    test: Window


def time_split(dates: pd.DatetimeIndex, fractions: Sequence[float] = (0.6, 0.2, 0.2)) -> Split:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions deve ter 3 valores somando 1")
    n = len(dates)
    i1 = int(n * fractions[0])
    i2 = int(n * (fractions[0] + fractions[1]))
    return Split(
        train=Window(dates[0], dates[i1 - 1]),
        val=Window(dates[i1], dates[i2 - 1]),
        test=Window(dates[i2], dates[-1]),
    )


# --------------------------------------------------------------------------- avaliação
def evaluate(
    panel: Panel, bt: Backtester, window: Window | None = None
) -> tuple[BacktestResult, Metrics]:
    p = panel.slice(window.start, window.end) if window else panel
    res = bt.run(p)
    return res, compute_metrics(res.equity, res.trades, res.exposure)


def _row(params: Mapping[str, Any], m: Metrics) -> dict[str, Any]:
    return {
        **params,
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_drawdown": m.max_drawdown,
        "profit_factor": m.profit_factor,
        "n_trades": m.n_trades,
        "exposure_avg": m.exposure_avg,
    }


# --------------------------------------------------------------------------- grid / robustez
def grid_search(
    strategy: Strategy,
    grid: Mapping[str, list[Any]],
    make_panel: PanelFactory,
    bt: Backtester,
    window: Window | None = None,
) -> pd.DataFrame:
    """Uma linha por combinação do grid com métricas no `window`."""
    rows = []
    for inst in strategy.grid(grid):
        params = {k: inst.params.model_dump()[k] for k in grid}
        _, m = evaluate(make_panel(inst), bt, window)
        rows.append(_row(params, m))
    return pd.DataFrame(rows)


def select_best(grid_df: pd.DataFrame, metric: str = "sharpe", min_trades: int = 30) -> pd.Series:
    """Melhor linha do grid pela métrica, exigindo nº mínimo de trades (senão, relaxa)."""
    ok = grid_df[grid_df["n_trades"] >= min_trades]
    pool = ok if not ok.empty else grid_df
    return pool.sort_values(metric, ascending=False).iloc[0]


def plateau_ratio(
    grid_df: pd.DataFrame,
    best: Mapping[str, Any],
    grid: Mapping[str, list[Any]],
    metric: str = "sharpe",
) -> float:
    """Média da métrica dos vizinhos imediatos (±1 passo em um parâmetro) / métrica do ótimo.

    ≥ 0,7 indica platô (docs/04 §2.3). NaN se não houver vizinhos ou ótimo ≤ 0.
    """
    keys = list(grid)
    best_val = _lookup(grid_df, best, keys, metric)
    if best_val is None or not (best_val > 0):
        return float("nan")
    neigh: list[float] = []
    for k in keys:
        values = list(grid[k])
        i = values.index(best[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                cand = {**{kk: best[kk] for kk in keys}, k: values[j]}
                v = _lookup(grid_df, cand, keys, metric)
                if v is not None:
                    neigh.append(v)
    if not neigh:
        return float("nan")
    return float(np.nanmean(neigh) / best_val)


def _lookup(
    df: pd.DataFrame, params: Mapping[str, Any], keys: list[str], metric: str
) -> float | None:
    mask = pd.Series(True, index=df.index)
    for k in keys:
        mask &= df[k] == params[k]
    sub = df[mask]
    if sub.empty:
        return None
    v = float(sub[metric].iloc[0])
    return None if math.isnan(v) else v


# --------------------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    windows: pd.DataFrame  # train/test, params escolhidos, sharpe IS e OOS
    oos_equity: pd.Series  # equity OOS encadeada (base 1.0)
    oos_trades: pd.DataFrame
    efficiency: float  # Sharpe OOS agregado / média do Sharpe IS


def walk_forward(
    strategy: Strategy,
    grid: Mapping[str, list[Any]],
    make_panel: PanelFactory,
    bt: Backtester,
    dates: pd.DatetimeIndex,
    *,
    train_years: int = 3,
    test_years: int = 1,
    anchored: bool = False,
    min_trades: int = 30,
) -> WalkForwardResult:
    start, end = dates[0], dates[-1]
    rows: list[dict[str, Any]] = []
    oos_parts: list[pd.Series] = []
    oos_trades: list[pd.DataFrame] = []
    panels: dict[str, Panel] = {}

    def panel_for(inst: Strategy) -> Panel:
        key = repr(inst)
        if key not in panels:
            panels[key] = make_panel(inst)
        return panels[key]

    t0 = start
    while True:
        train_end = t0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(years=test_years)
        if train_end >= end:
            break
        test_end = min(test_end, end)
        train_w = Window(start if anchored else t0, train_end)
        test_w = Window(train_end + pd.Timedelta(days=1), test_end)

        g = grid_search(strategy, grid, panel_for, bt, train_w)
        best = select_best(g, min_trades=min_trades)
        params = {k: best[k] for k in grid}
        inst = strategy.with_params(**params)
        res_oos, m_oos = evaluate(panel_for(inst), bt, test_w)
        rows.append(
            {
                "train": str(train_w),
                "test": str(test_w),
                **params,
                "sharpe_is": float(best["sharpe"]),
                "sharpe_oos": m_oos.sharpe,
                "cagr_oos": m_oos.cagr,
                "n_trades_oos": m_oos.n_trades,
            }
        )
        oos_parts.append(res_oos.equity / res_oos.equity.iloc[0])
        oos_trades.append(res_oos.trades)
        t0 = t0 + pd.DateOffset(years=test_years)
        if test_end >= end:
            break

    if not oos_parts:
        raise ValueError("período curto demais para walk-forward")
    # encadeia curvas normalizadas (cada janela começa onde a anterior terminou)
    chained = [oos_parts[0]]
    for part in oos_parts[1:]:
        chained.append(part * chained[-1].iloc[-1])
    oos_equity = pd.concat(chained)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    windows = pd.DataFrame(rows)
    is_mean = float(windows["sharpe_is"].replace([np.inf, -np.inf], np.nan).mean())
    oos_sharpe = sharpe_ratio(oos_equity.pct_change().dropna())
    eff = oos_sharpe / is_mean if is_mean and is_mean > 0 else float("nan")
    return WalkForwardResult(windows, oos_equity, pd.concat(oos_trades, ignore_index=True), eff)


# --------------------------------------------------------------------------- Monte Carlo
def monte_carlo_drawdown(
    trade_pnl: pd.Series,
    initial_capital: float,
    runs: int = 1000,
    seed: int = 0,
    ruin_level: float = 0.15,
) -> dict[str, float]:
    """Embaralha a ordem dos trades; distribuição do max drawdown resultante."""
    pnl = trade_pnl.to_numpy(dtype=float)
    if len(pnl) == 0:
        return {
            "mdd_p50": float("nan"),
            "mdd_p95": float("nan"),
            "mdd_p99": float("nan"),
            "prob_dd_gt_ruin": float("nan"),
        }
    rng = np.random.default_rng(seed)
    mdds = np.empty(runs)
    for i in range(runs):
        eq = initial_capital + np.cumsum(rng.permutation(pnl))
        peak = np.maximum.accumulate(np.concatenate(([initial_capital], eq)))
        mdds[i] = float(np.min(np.concatenate(([initial_capital], eq)) / peak - 1.0))
    return {
        "mdd_p50": float(np.percentile(mdds, 50)),
        "mdd_p95": float(np.percentile(mdds, 5)),  # 5º percentil = 95% dos casos são melhores
        "mdd_p99": float(np.percentile(mdds, 1)),
        "prob_dd_gt_ruin": float(np.mean(mdds < -ruin_level)),
    }


def block_bootstrap_drawdown(
    returns: pd.Series,
    runs: int = 1000,
    block: int = 20,
    seed: int = 0,
    ruin_level: float = 0.15,
    horizon: int | None = None,
) -> dict[str, float]:
    """Distribuição do max drawdown por bootstrap circular em blocos dos retornos **diários**.

    Alternativa a `monte_carlo_drawdown` (ADR-017). Reamostrar retornos diários da carteira
    preserva três coisas que o embaralhamento de trades destrói:

    * **sobreposição** — num dia a carteira tem N posições; o retorno diário já é o efeito
      líquido delas, enquanto trades embaralhados viram uma fila sequencial;
    * **composição** — o drawdown é medido em % do patrimônio corrente, não sobre o capital
      inicial fixo, então um P&L nominal grande do fim da série não vira um DD absurdo se
      sorteado para o começo;
    * **autocorrelação** dentro do bloco (streaks de perdas sobrevivem à reamostragem).

    `horizon` é o comprimento do caminho simulado em pregões (padrão: a série inteira). O max
    drawdown **cresce com o horizonte** — um p95 sobre 16 anos não é comparável a um alvo fixo —,
    então o gate de aprovação usa um horizonte de 1 ano (`TRADING_DAYS`), que é a pergunta
    operacional: quanto se pode perder do pico nos próximos 12 meses.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block * 2:
        return {
            "mdd_p50": float("nan"),
            "mdd_p95": float("nan"),
            "mdd_p99": float("nan"),
            "prob_dd_gt_ruin": float("nan"),
        }
    h = min(horizon or n, n) if horizon else n
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(h / block)
    offsets = np.arange(block)
    mdds = np.empty(runs)
    for i in range(runs):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:h] % n
        eq = np.concatenate(([1.0], np.cumprod(1.0 + r[idx])))
        mdds[i] = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return {
        "mdd_p50": float(np.percentile(mdds, 50)),
        "mdd_p95": float(np.percentile(mdds, 5)),  # 5º percentil = 95% dos casos são melhores
        "mdd_p99": float(np.percentile(mdds, 1)),
        "prob_dd_gt_ruin": float(np.mean(mdds < -ruin_level)),
    }


def bootstrap_sharpe(
    returns: pd.Series, runs: int = 1000, block: int = 20, seed: int = 0
) -> dict[str, float]:
    """Bootstrap circular em blocos (preserva autocorrelação) do Sharpe anualizado."""
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block * 2:
        return {"sharpe_lo": float("nan"), "sharpe_hi": float("nan"), "p_sharpe_le_0": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    out = np.empty(runs)
    for i in range(runs):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = r[idx[:n]]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * math.sqrt(252) if sd > 0 else 0.0
    return {
        "sharpe_lo": float(np.percentile(out, 2.5)),
        "sharpe_hi": float(np.percentile(out, 97.5)),
        "p_sharpe_le_0": float(np.mean(out <= 0)),
    }


# --------------------------------------------------------------------------- custos
def cost_sensitivity(
    panel: Panel,
    base_costs: CostModel,
    risk: RiskModel,
    multipliers: Sequence[float] = (0, 1, 2, 3),
    window: Window | None = None,
) -> pd.DataFrame:
    rows = []
    for m in multipliers:
        _, met = evaluate(panel, Backtester(base_costs.scaled(m), risk), window)
        rows.append(
            {
                "cost_mult": m,
                "sharpe": met.sharpe,
                "cagr": met.cagr,
                "profit_factor": met.profit_factor,
                "total_return": met.total_return,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- baseline aleatória
def random_baseline(
    panel: Panel, bt: Backtester, runs: int = 50, seed: int = 0, window: Window | None = None
) -> dict[str, float]:
    """Entradas aleatórias com a mesma frequência média e mesmo `max_hold`; sem saída por sinal.

    Devolve distribuição do Sharpe do 'macaco' para comparar com a estratégia.
    """
    p = panel.slice(window.start, window.end) if window else panel
    rate = float(p.entry.to_numpy().mean())
    hold = int(p.max_hold.to_numpy().max()) or 5
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(runs):
        rnd = Panel(
            dates=p.dates,
            tickers=p.tickers,
            open=p.open,
            high=p.high,
            low=p.low,
            close=p.close,
            atr=p.atr,
            dollar_vol=p.dollar_vol,
            entry=pd.DataFrame(rng.random(p.entry.shape) < rate, index=p.dates, columns=p.tickers),
            exit=pd.DataFrame(False, index=p.dates, columns=p.tickers),
            stop=p.stop,
            score=pd.DataFrame(rng.random(p.score.shape), index=p.dates, columns=p.tickers),
            max_hold=pd.DataFrame(hold, index=p.dates, columns=p.tickers, dtype=int),
            meta={"baseline": "random"},
        )
        res = bt.run(rnd)
        sharpes.append(sharpe_ratio(res.equity.pct_change().dropna()))
    arr = np.array(sharpes, dtype=float)
    return {
        "random_sharpe_mean": float(np.nanmean(arr)),
        "random_sharpe_p95": float(np.nanpercentile(arr, 95)),
    }


# --------------------------------------------------------------------------- utilidades
def default_panel_factory(prices: pd.DataFrame, **kwargs: Any) -> PanelFactory:
    def make(inst: Strategy) -> Panel:
        return build_panel(prices, inst, **kwargs)

    return make


def all_grid_combos(grid: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, c, strict=True)) for c in product(*(grid[k] for k in keys))]


__all__ = [
    "Split",
    "WalkForwardResult",
    "Window",
    "all_grid_combos",
    "block_bootstrap_drawdown",
    "bootstrap_sharpe",
    "cost_sensitivity",
    "default_panel_factory",
    "evaluate",
    "grid_search",
    "max_drawdown",
    "monte_carlo_drawdown",
    "plateau_ratio",
    "random_baseline",
    "select_best",
    "time_split",
    "walk_forward",
]
