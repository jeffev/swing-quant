"""Position sizing e ranking de candidatos — **uma única implementação** usada pelo engine de
backtest e pelo screener de produção (teste de paridade em tests/test_screener.py).

qty = risco / (k*ATR), limitado por % do patrimônio, caixa, participação no volume e lote.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingParams:
    risk_per_trade: float
    atr_multiple: float
    max_position_pct: float
    max_volume_participation: float
    board_lot: int
    slippage_pct: float = 0.0
    fees_pct: float = 0.0


def round_lot(qty: float, board_lot: int) -> int:
    """Arredonda para baixo ao lote padrão quando >= lote; abaixo disso permite fracionário."""
    if math.isnan(qty) or math.isinf(qty) or qty <= 0:
        return 0
    qty_int = math.floor(qty)
    if board_lot > 1 and qty_int >= board_lot:
        qty_int -= qty_int % board_lot
    return max(qty_int, 0)


def position_size(
    price: float,
    atr_value: float,
    equity: float,
    cash: float,
    dollar_volume: float,
    factor: float,
    p: SizingParams,
) -> int:
    """Quantidade a comprar. `factor` multiplica o orçamento de risco (regime/DD mensal)."""
    if not (price > 0) or not (atr_value > 0) or math.isnan(atr_value) or math.isnan(equity):
        return 0
    risk_budget = equity * p.risk_per_trade * factor
    qty_risk = risk_budget / (p.atr_multiple * atr_value)
    qty_cap = equity * p.max_position_pct / price
    qty_cash = cash / (price * (1.0 + p.slippage_pct) * (1.0 + p.fees_pct))
    qty = min(qty_risk, qty_cap, qty_cash)
    if p.max_volume_participation > 0 and dollar_volume > 0 and not math.isnan(dollar_volume):
        qty = min(qty, dollar_volume * p.max_volume_participation / price)
    return round_lot(qty, p.board_lot)


def rank_key(score: float, dollar_volume: float) -> tuple[float, float]:
    """Chave de ordenação: maior score primeiro; empate → maior volume financeiro."""
    s = -score if not math.isnan(score) else math.inf
    v = -dollar_volume if not math.isnan(dollar_volume) else 0.0
    return (s, v)


def rank_candidates(
    candidates: Sequence[int],
    score_of: Callable[[int], float],
    dvol_of: Callable[[int], float],
) -> list[int]:
    return sorted(candidates, key=lambda j: rank_key(score_of(j), dvol_of(j)))
