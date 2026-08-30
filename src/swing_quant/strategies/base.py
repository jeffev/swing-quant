"""Contrato de estratégia (docs/02-estrategias.md §E).

Uma estratégia recebe o OHLCV **ajustado** de um ticker (colunas open, high, low, close, volume,
indexado por data) e devolve um DataFrame alinhado com as colunas de `SIGNAL_COLUMNS`:

- entry    (bool)  sinal de compra calculado no fechamento do dia (executa em D+1)
- exit     (bool)  sinal de saída calculado no fechamento do dia (executa em D+1)
- stop     (float) preço de stop inicial para a entrada gerada neste dia (NaN = sem stop)
- target   (float) preço-alvo da entrada gerada neste dia (NaN = sem alvo); realizado
            intradiário, como o stop
- score    (float) prioridade para o screener/engine quando há mais sinais que vagas
- max_hold (int)   nº máximo de pregões em posição (stop por tempo); 0 = sem limite

Regras: nunca olhar para frente; parâmetros só via `params` (pydantic); tudo vetorizado.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from itertools import product
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

SIGNAL_COLUMNS: tuple[str, ...] = ("entry", "exit", "stop", "target", "score", "max_hold")
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class StrategyParams(BaseModel):
    """Base para parâmetros; `enabled` é lido pelo config mas ignorado pela estratégia."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    enabled: bool = True


class Strategy(ABC):
    """Classe base. Subclasses definem `name`, `Params` e `generate`."""

    name: ClassVar[str]
    Params: ClassVar[type[StrategyParams]] = StrategyParams
    #: Grid padrão para robustez/walk-forward: {param: [valores]} (≤ 3 parâmetros).
    default_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, params: StrategyParams | Mapping[str, Any] | None = None) -> None:
        if isinstance(params, StrategyParams):
            self.params = params
        else:
            self.params = self.Params.model_validate(dict(params or {}))

    # ------------------------------------------------------------------ API
    @property
    @abstractmethod
    def warmup(self) -> int:
        """Nº de pregões necessários antes do primeiro sinal válido."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe OHLCV ajustado; devolve DataFrame com SIGNAL_COLUMNS alinhado ao índice."""

    # ------------------------------------------------------------------ utilidades
    def with_params(self, **overrides: Any) -> Strategy:
        merged = {**self.params.model_dump(), **overrides}
        return type(self)(merged)

    def grid(self, grid: Mapping[str, list[Any]] | None = None) -> Iterator[Strategy]:
        """Itera instâncias para cada combinação do grid (padrão: `default_grid`)."""
        g = dict(grid if grid is not None else self.default_grid)
        if not g:
            yield self
            return
        keys = list(g)
        for combo in product(*(g[k] for k in keys)):
            yield self.with_params(**dict(zip(keys, combo, strict=True)))

    def __repr__(self) -> str:
        p = {k: v for k, v in self.params.model_dump().items() if k != "enabled"}
        return f"{self.name}({p})"

    @staticmethod
    def empty_signals(index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entry": np.zeros(len(index), dtype=bool),
                "exit": np.zeros(len(index), dtype=bool),
                "stop": np.full(len(index), np.nan),
                "target": np.full(len(index), np.nan),
                "score": np.full(len(index), np.nan),
                "max_hold": np.zeros(len(index), dtype=int),
            },
            index=index,
        )


def validate_signals(sig: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Garante colunas, tipos e alinhamento; NaN em entry/exit vira False."""
    missing = set(SIGNAL_COLUMNS) - set(sig.columns)
    if missing:
        raise ValueError(f"sinais sem colunas {sorted(missing)}")
    if not sig.index.equals(index):
        raise ValueError("índice dos sinais difere do índice do OHLCV")
    out = sig.loc[:, list(SIGNAL_COLUMNS)].copy()
    out["entry"] = out["entry"].fillna(False).astype(bool)
    out["exit"] = out["exit"].fillna(False).astype(bool)
    out["stop"] = out["stop"].astype(float)
    out["target"] = out["target"].astype(float)
    out["score"] = out["score"].astype(float)
    out["max_hold"] = out["max_hold"].fillna(0).astype(int)
    return out
