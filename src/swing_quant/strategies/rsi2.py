"""A1 — RSI(2) de Connors (docs/02-estrategias.md).

Filtro: close > SMA(trend_sma). Entrada: RSI(rsi_period) < rsi_entry.
Saída: close > SMA(exit_sma) ou `max_hold` pregões. Stop opcional: stop_atr * ATR14.
Score: 100 - RSI (mais sobrevendido = maior prioridade).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, rsi, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class RSI2Params(StrategyParams):
    rsi_period: int = Field(default=2, ge=1, le=10)
    rsi_entry: float = Field(default=10.0, gt=0, lt=50)
    exit_sma: int = Field(default=5, ge=1, le=50)
    trend_sma: int = Field(default=200, ge=20, le=400)
    max_hold: int = Field(default=5, ge=0, le=30)
    stop_atr: float | None = Field(default=None, gt=0)
    atr_period: int = Field(default=14, ge=2, le=50)


class RSI2(Strategy):
    name: ClassVar[str] = "rsi2"
    Params: ClassVar[type[StrategyParams]] = RSI2Params
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "rsi_entry": [5.0, 10.0, 15.0],
        "exit_sma": [3, 5, 10],
    }
    params: RSI2Params

    @property
    def warmup(self) -> int:
        return max(self.params.trend_sma, self.params.exit_sma, self.params.atr_period) + 1

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        trend = sma(close, p.trend_sma)
        exit_ma = sma(close, p.exit_sma)
        r = rsi(close, p.rsi_period)
        a = atr(df["high"], df["low"], close, p.atr_period)

        sig = self.empty_signals(df.index)
        sig["entry"] = ((close > trend) & (r < p.rsi_entry)).fillna(False).to_numpy()
        sig["exit"] = (close > exit_ma).fillna(False).to_numpy()
        sig["score"] = (100.0 - r).to_numpy()
        sig["max_hold"] = p.max_hold
        if p.stop_atr is not None:
            sig["stop"] = (close - p.stop_atr * a).to_numpy()
        else:
            sig["stop"] = np.nan
        return sig
