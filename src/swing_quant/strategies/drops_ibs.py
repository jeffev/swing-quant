"""A2 — Quedas consecutivas + IBS baixo (docs/02-estrategias.md).

Filtro: close > SMA(trend_sma). Entrada: >= `min_consecutive_drops` fechamentos em queda
e IBS = (close-low)/(high-low) < `ibs_max`. Saída: close > máxima do pregão anterior, ou
`max_hold` pregões. Score: nº de quedas * (1 - IBS). Stop opcional por ATR.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, consecutive_down_days, ibs, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class DropsIBSParams(StrategyParams):
    min_consecutive_drops: int = Field(default=3, ge=1, le=8)
    ibs_max: float = Field(default=0.2, gt=0, lt=1)
    trend_sma: int = Field(default=200, ge=20, le=400)
    max_hold: int = Field(default=3, ge=0, le=30)
    stop_atr: float | None = Field(default=None, gt=0)
    atr_period: int = Field(default=14, ge=2, le=50)


class DropsIBS(Strategy):
    name: ClassVar[str] = "drops_ibs"
    Params: ClassVar[type[StrategyParams]] = DropsIBSParams
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "min_consecutive_drops": [2, 3, 4],
        "ibs_max": [0.2, 0.3],
    }
    params: DropsIBSParams

    @property
    def warmup(self) -> int:
        return max(self.params.trend_sma, self.params.atr_period) + 2

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close, high, low = df["close"], df["high"], df["low"]
        trend = sma(close, p.trend_sma)
        drops = consecutive_down_days(close)
        strength = ibs(high, low, close)
        a = atr(high, low, close, p.atr_period)

        sig = self.empty_signals(df.index)
        entry = (close > trend) & (drops >= p.min_consecutive_drops) & (strength < p.ibs_max)
        sig["entry"] = entry.fillna(False).to_numpy()
        sig["exit"] = (close > high.shift(1)).fillna(False).to_numpy()
        sig["score"] = (drops * (1.0 - strength)).to_numpy()
        sig["max_hold"] = p.max_hold
        if p.stop_atr is not None:
            sig["stop"] = (close - p.stop_atr * a).to_numpy()
        return sig
