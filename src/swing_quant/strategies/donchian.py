"""B1 — Breakout de canal Donchian (docs/02-estrategias.md).

Filtro: close > SMA(fast) > SMA(slow) e volume do dia > volume_mult * média de volume anterior.
Entrada: close > máxima dos `entry_lookback` pregões anteriores.
Saída: close < mínima dos `exit_lookback` pregões anteriores (trailing por canal).
Stop inicial: close - stop_atr * ATR. Score: (close - canal) / ATR (força do rompimento).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, donchian_high, donchian_low, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class DonchianParams(StrategyParams):
    entry_lookback: int = Field(default=20, ge=5, le=120)
    exit_lookback: int = Field(default=10, ge=3, le=60)
    volume_mult: float = Field(default=1.5, ge=0.0, le=5.0)
    volume_window: int = Field(default=20, ge=5, le=60)
    fast_sma: int = Field(default=50, ge=10, le=100)
    slow_sma: int = Field(default=200, ge=50, le=400)
    stop_atr: float | None = Field(default=2.0, gt=0)
    atr_period: int = Field(default=14, ge=2, le=50)
    max_hold: int = Field(default=0, ge=0, le=120)


class Donchian(Strategy):
    name: ClassVar[str] = "donchian"
    Params: ClassVar[type[StrategyParams]] = DonchianParams
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_lookback": [20, 40, 55],
        "exit_lookback": [10, 20],
    }
    params: DonchianParams

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p.slow_sma, p.entry_lookback, p.volume_window, p.atr_period) + 2

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        fast, slow = sma(close, p.fast_sma), sma(close, p.slow_sma)
        channel_high = donchian_high(high, p.entry_lookback)
        channel_low = donchian_low(low, p.exit_lookback)
        a = atr(high, low, close, p.atr_period)
        vol_avg = vol.shift(1).rolling(p.volume_window, min_periods=p.volume_window).mean()

        trend = (close > fast) & (fast > slow)
        vol_ok = (
            (vol > p.volume_mult * vol_avg)
            if p.volume_mult > 0
            else pd.Series(True, index=df.index)
        )
        breakout = close > channel_high

        sig = self.empty_signals(df.index)
        sig["entry"] = (trend & vol_ok & breakout).fillna(False).to_numpy()
        sig["exit"] = (close < channel_low).fillna(False).to_numpy()
        sig["score"] = ((close - channel_high) / a).to_numpy()
        sig["max_hold"] = p.max_hold
        if p.stop_atr is not None:
            sig["stop"] = (close - p.stop_atr * a).to_numpy()
        return sig
