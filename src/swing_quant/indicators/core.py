"""Indicadores técnicos vetorizados (pandas). Todas as funções olham apenas para trás.

Convenção: recebem Series/DataFrame indexados por data e devolvem Series alinhadas, com NaN
no período de aquecimento. Nenhuma função usa `shift(-n)` ou `center=True`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def returns(s: pd.Series, n: int = 1) -> pd.Series:
    return s.pct_change(n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder (média móvel exponencial com alpha = 1/n)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Sem perdas no período -> RSI = 100; sem ganhos -> 0.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    tr.iloc[0] = high.iloc[0] - low.iloc[0] if len(tr) else np.nan
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR de Wilder."""
    return true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    """Máxima dos últimos `n` pregões **anteriores** (exclui o dia atual)."""
    return high.shift(1).rolling(n, min_periods=n).max()


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    """Mínima dos últimos `n` pregões **anteriores** (exclui o dia atual)."""
    return low.shift(1).rolling(n, min_periods=n).min()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength = (close - low) / (high - low); 0,5 quando high == low."""
    rng = high - low
    out = (close - low) / rng
    return out.where(rng > 0, 0.5)


def consecutive_down_days(close: pd.Series) -> pd.Series:
    """Nº de fechamentos consecutivos em queda até o dia (0 se subiu ou ficou igual)."""
    down = (close.diff() < 0).astype(int)
    groups = (down == 0).cumsum()
    return down.groupby(groups).cumsum()


def dollar_volume(close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    """Volume financeiro médio dos últimos `n` pregões."""
    return (close * volume).rolling(n, min_periods=n).mean()


def rolling_slope(s: pd.Series, n: int) -> pd.Series:
    """Variação relativa em `n` pregões (proxy de inclinação): s / s.shift(n) - 1."""
    return s / s.shift(n) - 1.0
