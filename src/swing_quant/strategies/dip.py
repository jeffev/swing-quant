"""A4 — Queda desde a máxima (docs/02-estrategias.md).

Entrada: o fechamento está `drop_pct` ou mais abaixo da **máxima de fechamento dos `lookback`
pregões anteriores** — compra o papel descontado do topo recente, não o tombo de um único dia.
`trend_sma > 0` liga um filtro de tendência (só compra acima da média), que troca quantidade de
sinais por proteção contra queda estrutural.

Saída: a primeira que acontecer — alvo de `target_pct`, stop de `stop_atr`×ATR ou `max_hold`
pregões. O alvo e o stop são preços absolutos calculados sobre o fechamento do dia do sinal, e
não sobre o preço de execução (a compra sai na abertura seguinte, com slippage); é a mesma
convenção que o stop das outras estratégias já usava.

Score: profundidade da queda — entre dois candidatos no mesmo dia, o mais descontado entra
primeiro.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from swing_quant.indicators import atr, sma
from swing_quant.strategies.base import Strategy, StrategyParams


class DipParams(StrategyParams):
    drop_pct: float = Field(default=0.15, ge=0.02, le=0.60)
    lookback: int = Field(default=60, ge=10, le=252)
    target_pct: float = Field(default=0.10, ge=0.02, le=1.0)
    stop_atr: float | None = Field(default=2.0, gt=0)
    max_hold: int = Field(default=60, ge=0, le=252)
    #: 0 desliga o filtro de tendência (compra a queda pura)
    trend_sma: int = Field(default=0, ge=0, le=400)
    atr_period: int = Field(default=14, ge=2, le=50)


class Dip(Strategy):
    name: ClassVar[str] = "dip"
    Params: ClassVar[type[StrategyParams]] = DipParams
    default_grid: ClassVar[dict[str, list[Any]]] = {
        "drop_pct": [0.10, 0.15, 0.20],
        "target_pct": [0.08, 0.15],
        # o filtro de tendencia e escolhido no treino, nao por inspecao do OOS
        "trend_sma": [0, 200],
    }
    params: DipParams

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p.lookback + 1, p.trend_sma, p.atr_period) + 2

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        # máxima dos `lookback` pregões ANTERIORES: o topo de referência não inclui o dia do
        # sinal, senão a própria barra que faz o topo mexeria na queda medida nela mesma
        topo = close.shift(1).rolling(p.lookback).max()
        queda = close / topo - 1.0
        a = atr(df["high"], df["low"], close, p.atr_period)

        entrada = queda <= -p.drop_pct
        if p.trend_sma > 0:
            entrada &= close > sma(close, p.trend_sma)

        sig = self.empty_signals(df.index)
        sig["entry"] = entrada.fillna(False).to_numpy()
        # sem sinal booleano de saída: quem fecha a posição é o alvo, o stop ou o tempo
        sig["target"] = (close * (1.0 + p.target_pct)).to_numpy()
        sig["score"] = (-queda).to_numpy()
        sig["max_hold"] = p.max_hold
        if p.stop_atr is not None:
            sig["stop"] = (close - p.stop_atr * a).to_numpy()
        return sig
