"""swing_quant.strategies — registro de estratégias plugáveis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from swing_quant.strategies.base import SIGNAL_COLUMNS, Strategy, StrategyParams, validate_signals
from swing_quant.strategies.dip import Dip
from swing_quant.strategies.donchian import Donchian
from swing_quant.strategies.drops_ibs import DropsIBS
from swing_quant.strategies.momentum import Momentum
from swing_quant.strategies.pullback import Pullback
from swing_quant.strategies.rsi2 import RSI2

REGISTRY: dict[str, type[Strategy]] = {
    RSI2.name: RSI2,
    Donchian.name: Donchian,
    Dip.name: Dip,
    DropsIBS.name: DropsIBS,
    Momentum.name: Momentum,
    Pullback.name: Pullback,
}


def make_strategy(name: str, params: Mapping[str, Any] | None = None) -> Strategy:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"estratégia desconhecida: {name!r}; disponíveis: {sorted(REGISTRY)}"
        ) from exc
    return cls(params)


__all__ = [
    "REGISTRY",
    "RSI2",
    "SIGNAL_COLUMNS",
    "Dip",
    "Donchian",
    "DropsIBS",
    "Momentum",
    "Pullback",
    "Strategy",
    "StrategyParams",
    "make_strategy",
    "validate_signals",
]
