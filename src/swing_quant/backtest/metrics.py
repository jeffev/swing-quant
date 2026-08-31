"""Métricas de desempenho (docs/04-metricas-e-validacao.md §1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

TRADING_DAYS = 252
_FP_NOISE = 1e-12  # abaixo disso um desvio-padrão é erro de arredondamento, não risco

#: Custo de oportunidade: série diária da tabela `risk_free`, taxa anual escalar, ou nada.
RiskFree = pd.Series | float | None


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
    rf_cagr: float = 0.0
    """Rendimento anual da renda fixa no período (0 quando não há série)."""
    excess_over_rf: float = float("nan")
    """CAGR − `rf_cagr`. Negativo = a estratégia perdeu para deixar o dinheiro parado."""

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


def excess_returns(returns: pd.Series, rf: RiskFree = None) -> pd.Series:
    """Retornos acima do custo de oportunidade (ADR-020).

    `rf` é uma **série diária** (tabela `risk_free`) ou uma taxa anual escalar; `None` mantém o
    piso em zero. A série é o caso certo: com o CDI variando de 2% a 14% no período, uma taxa
    média achata justamente a diferença entre ganhar dinheiro e ganhar menos que o caixa.
    """
    if rf is None:
        return returns
    if isinstance(rf, pd.Series):
        return returns - rf.reindex(returns.index).fillna(0.0)
    return returns - rf / TRADING_DAYS


def sharpe_ratio(returns: pd.Series, rf: RiskFree = None) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = excess_returns(returns, rf).dropna()
    sd = excess.std(ddof=1) if len(excess) > 1 else 0.0
    if len(excess) < 2 or sd < _FP_NOISE:
        return float("nan")  # a própria renda fixa contra si mesma: 0/0, não é razão
    return float(_safe_div(excess.mean(), sd) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: RiskFree = None) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = excess_returns(returns, rf).dropna()
    downside = excess.clip(upper=0.0)
    dd = math.sqrt(float((downside**2).mean()))
    if len(excess) < 2 or dd < _FP_NOISE:
        return float("nan")
    return float(_safe_div(excess.mean(), dd) * math.sqrt(TRADING_DAYS))


def rf_cagr(rf: RiskFree, index: pd.Index) -> float:
    """Quanto a renda fixa rendeu ao ano no período — o piso que a estratégia precisa bater."""
    if rf is None or len(index) < 2:
        return 0.0
    idx = pd.DatetimeIndex(index)
    years = (idx[-1] - idx[0]).days / 365.25
    if years <= 0:
        return 0.0
    if isinstance(rf, pd.Series):
        # Mesmos períodos que o `cagr` de uma curva mede: o primeiro dia é o saldo inicial,
        # os retornos começam no segundo. Sem isso a renda fixa ganha um pregão de vantagem.
        daily = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)[1:]
        total = float((1.0 + daily).prod())
        return float(total ** (1.0 / years) - 1.0)
    return float(rf)


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
    rf: RiskFree = None,
) -> Metrics:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    rf_c = rf_cagr(rf, equity.index)
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
        sharpe=sharpe_ratio(rets, rf),
        sortino=sortino_ratio(rets, rf),
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
        rf_cagr=rf_c,
        excess_over_rf=c - rf_c,
    )


def benchmark_metrics(close: pd.Series, rf: RiskFree = None) -> dict[str, float]:
    """Buy-and-hold do benchmark no mesmo período: CAGR, Sharpe, MDD."""
    close = close.dropna()
    if len(close) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    rets = close.pct_change().dropna()
    return {
        "cagr": cagr(close),
        "sharpe": sharpe_ratio(rets, rf),
        "max_drawdown": max_drawdown(close)[0],
    }


def blended_benchmark(
    close: pd.Series, rf: RiskFree, weight: float, index: pd.Index
) -> dict[str, float]:
    """Carteira passiva com `weight` no índice e o resto na renda fixa, rebalanceada por dia.

    É o comparável honesto de uma carteira que fica a maior parte do tempo em caixa (ADR-020):
    medir uma carteira 28% investida contra um índice 100% investido mede o humor da bolsa no
    período, não as regras de entrada e saída. `weight` é a exposição média da própria
    estratégia, então as duas carregam o mesmo risco de bolsa.
    """
    px = close.reindex(index).ffill()
    eq_ret = px.pct_change().fillna(0.0)
    if rf is None:
        rf_ret = pd.Series(0.0, index=index)
    elif isinstance(rf, pd.Series):
        rf_ret = rf.reindex(index).fillna(0.0)
    else:
        rf_ret = pd.Series(rf / TRADING_DAYS, index=index)
    blend = weight * eq_ret + (1.0 - weight) * rf_ret
    blend.iloc[0] = 0.0  # dia 0 é o saldo inicial: ainda não houve pernoite para render
    curve = (1.0 + blend).cumprod()
    rets = curve.pct_change().dropna()
    return {
        "weight": weight,
        "cagr": cagr(curve),
        "sharpe": sharpe_ratio(rets, rf),
        "max_drawdown": max_drawdown(curve)[0],
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
