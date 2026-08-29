"""Métricas de desempenho (docs/04-metricas-e-validacao.md §1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    start: str
    end: str
    years: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: int
    calmar: float
    ulcer_index: float
    exposure_avg: float
    n_trades: int
    win_rate: float
    payoff: float
    profit_factor: float
    expectancy_pct: float
    expectancy_r: float
    avg_hold_bars: float
    max_consecutive_losses: int
    fees_total: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0) and not math.isnan(b) else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(profundidade máxima como fração negativa, duração em pregões do maior drawdown)."""
    if equity.empty:
        return 0.0, 0
    dd = drawdown_series(equity)
    depth = float(dd.min())
    under = dd < 0
    if not under.any():
        return depth, 0
    groups = (~under).cumsum()
    lengths = under.groupby(groups).sum()
    return depth, int(lengths.max())


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(_safe_div(excess.mean(), sd) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / TRADING_DAYS
    downside = excess.clip(upper=0.0)
    dd = math.sqrt(float((downside**2).mean()))
    return float(_safe_div(excess.mean(), dd) * math.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def max_consecutive_losses(pnl: pd.Series) -> int:
    best = cur = 0
    for v in pnl:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.Series | None = None,
    *,
    rf_annual: float = 0.0,
) -> Metrics:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    dd_depth, dd_days = max_drawdown(equity)
    dd = drawdown_series(equity)
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0.0
    c = cagr(equity)

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_win = float(wins["pnl"].sum()) if n else 0.0
    gross_loss = float(-losses["pnl"].sum()) if n else 0.0
    avg_win = float(wins["ret"].mean()) if len(wins) else 0.0
    avg_loss = float(-losses["ret"].mean()) if len(losses) else 0.0
    win_rate = len(wins) / n if n else float("nan")
    expectancy_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) if n else float("nan")

    return Metrics(
        start=str(equity.index[0].date()) if len(equity) else "",
        end=str(equity.index[-1].date()) if len(equity) else "",
        years=round(years, 2),
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        cagr=c,
        volatility=float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0,
        sharpe=sharpe_ratio(rets, rf_annual),
        sortino=sortino_ratio(rets, rf_annual),
        max_drawdown=dd_depth,
        max_drawdown_days=dd_days,
        calmar=float(_safe_div(c, abs(dd_depth))),
        ulcer_index=float(math.sqrt((dd**2).mean())) if len(dd) else 0.0,
        exposure_avg=float(exposure.mean()) if exposure is not None and len(exposure) else 0.0,
        n_trades=n,
        win_rate=win_rate,
        payoff=float(_safe_div(avg_win, avg_loss)) if n else float("nan"),
        profit_factor=float(_safe_div(gross_win, gross_loss)) if n else float("nan"),
        expectancy_pct=expectancy_pct,
        expectancy_r=float(_safe_div(expectancy_pct, avg_loss)) if n else float("nan"),
        avg_hold_bars=float(trades["bars_held"].mean()) if n else 0.0,
        max_consecutive_losses=max_consecutive_losses(trades["pnl"]) if n else 0,
        fees_total=float(trades["fees"].sum()) if n else 0.0,
    )


def benchmark_metrics(close: pd.Series) -> dict[str, float]:
    """Buy-and-hold do benchmark no mesmo período: CAGR, Sharpe, MDD."""
    close = close.dropna()
    if len(close) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    rets = close.pct_change().dropna()
    return {
        "cagr": cagr(close),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(close)[0],
    }


def rolling_drawdowns(
    equity: pd.Series, window: int = TRADING_DAYS, step: int = 21
) -> dict[str, float]:
    """Distribuição do max drawdown em janelas móveis de `window` pregões.

    Âncora empírica para o bootstrap de drawdown (ADR-017): o p95 simulado no mesmo horizonte
    deve ficar próximo do que de fato aconteceu nas janelas do histórico.
    """
    n = len(equity)
    if n <= window:
        return {"n": 0.0, "p50": float("nan"), "p95": float("nan"), "worst": float("nan")}
    dds = [max_drawdown(equity.iloc[i : i + window])[0] for i in range(0, n - window, step)]
    arr = pd.Series(dds)
    return {
        "n": float(len(arr)),
        "p50": float(arr.quantile(0.5)),
        "p95": float(arr.quantile(0.05)),  # cauda ruim = quantil inferior
        "worst": float(arr.min()),
    }


def monthly_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("ME").last().pct_change().dropna()


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    mean = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return pd.Series((mean / sd) * math.sqrt(TRADING_DAYS), index=returns.index)
