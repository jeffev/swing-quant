"""B2 — Pullback em tendência (docs/02-estrategias.md).

Filtro: SMA(fast) > SMA(mid) > SMA(slow), todas ascendentes. Entrada: mínima do dia toca a
SMA(fast) e fechamento > abertura (candle de reversão). Saída: fechamento acima da máxima dos
`target_lookback` pregões anteriores (alvo) ou `max_hold`. Stop: mínima do candle − k*ATR.
Score: inclinação da SMA(mid) (retorno em `slope_window` pregões).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, donchian_high, rolling_slope, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class PullbackParams(StrategyParams):
    fast_sma: int = Field(default=20, ge=5, le=60)
    mid_sma: int = Field(default=50, ge=20, le=150)
    slow_sma: int = Field(default=200, ge=50, le=400)
    target_lookback: int = Field(default=10, ge=3, le=60)
    max_hold: int = Field(default=10, ge=0, le=60)
    stop_atr: float = Field(default=0.5, ge=0.0, le=5.0)
    slope_window: int = Field(default=20, ge=5, le=120)
    atr_period: int = Field(default=14, ge=2, le=50)


class Pullback(Strategy):
    name: ClassVar[str] = "pullback"
    Params: ClassVar[type[StrategyParams]] = PullbackParams
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_sma": [10, 20, 30],
        "max_hold": [10, 20],
    }
    params: PullbackParams

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p.slow_sma + p.slope_window, p.target_lookback, p.atr_period) + 2

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close, high, low, opn = df["close"], df["high"], df["low"], df["open"]
        fast, mid, slow = sma(close, p.fast_sma), sma(close, p.mid_sma), sma(close, p.slow_sma)
        rising = (fast > fast.shift(1)) & (mid > mid.shift(1)) & (slow > slow.shift(1))
        aligned = (fast > mid) & (mid > slow)
        touch = (low <= fast) & (close > fast)  # tocou a média e fechou acima dela
        reversal = close > opn
        a = atr(high, low, close, p.atr_period)
        target = donchian_high(high, p.target_lookback)

        sig = self.empty_signals(df.index)
        sig["entry"] = (aligned & rising & touch & reversal).fillna(False).to_numpy()
        sig["exit"] = (close > target).fillna(False).to_numpy()
        sig["score"] = rolling_slope(mid, p.slope_window).to_numpy()
        sig["max_hold"] = p.max_hold
        sig["stop"] = (low - p.stop_atr * a).to_numpy()
        return sig
