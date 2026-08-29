"""B3 — Momentum (12-1 adaptado) com seleção cross-sectional via ranking do engine.

Score = retorno acumulado de `lookback` pregões excluindo os últimos `skip` (evita a reversão
de curto prazo). Como o engine preenche as vagas pelos maiores scores do dia, a carteira fica
com os N papéis de maior momentum entre os que passam nos filtros — a parte "cross-sectional"
da seleção. A rotação acontece por saída: momentum negativo, perda da SMA(exit_sma) ou
`max_hold` pregões (reavaliação); se ainda estiver no topo, reentra no mesmo dia.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class MomentumParams(StrategyParams):
    lookback: int = Field(default=126, ge=40, le=400)
    skip: int = Field(default=21, ge=0, le=63)
    trend_sma: int = Field(default=200, ge=20, le=400)
    exit_sma: int = Field(default=100, ge=10, le=300)
    min_momentum: float = Field(default=0.0, ge=-1.0, le=2.0)
    max_hold: int = Field(default=63, ge=0, le=252)
    stop_atr: float | None = Field(default=3.0, gt=0)
    atr_period: int = Field(default=14, ge=2, le=50)


class Momentum(Strategy):
    name: ClassVar[str] = "momentum"
    Params: ClassVar[type[StrategyParams]] = MomentumParams
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "lookback": [126, 189, 252],
        "exit_sma": [50, 100],
    }
    params: MomentumParams

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p.lookback + p.skip, p.trend_sma, p.exit_sma, p.atr_period) + 2

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        mom = close.shift(p.skip) / close.shift(p.skip + p.lookback) - 1.0
        trend = sma(close, p.trend_sma)
        exit_ma = sma(close, p.exit_sma)
        a = atr(df["high"], df["low"], close, p.atr_period)

        sig = self.empty_signals(df.index)
        sig["entry"] = ((mom > p.min_momentum) & (close > trend)).fillna(False).to_numpy()
        sig["exit"] = ((mom < 0.0) | (close < exit_ma)).fillna(False).to_numpy()
        sig["score"] = mom.to_numpy()
        sig["max_hold"] = p.max_hold
        if p.stop_atr is not None:
            sig["stop"] = (close - p.stop_atr * a).to_numpy()
        return sig
