"""Screener diário: aplica as estratégias ao dado mais recente e produz sinais executáveis.

Paridade com o backtest (docs/01 critério de sucesso): as entradas de `as_of` são exatamente
as que o `Backtester` compraria na abertura de D+1 partindo do mesmo estado — mesma função de
sizing (`risk/sizing.py`), mesmo ranking, mesmos filtros de liquidez/regime/subjacente.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from swing_quant.backtest.engine import CostModel, RiskModel
from swing_quant.backtest.panel import Panel, build_panel
from swing_quant.backtest.portfolio import combine_panels
from swing_quant.risk.regime import Regime
from swing_quant.risk.sizing import SizingParams, position_size, rank_key, round_lot
from swing_quant.strategies.base import Strategy

ENTRY_COLUMNS = [
    "ticker",
    "strategy",
    "ref_price",
    "qty",
    "notional",
    "stop_price",
    "max_hold",
    "score",
    "atr",
    "dollar_volume",
]
EXIT_COLUMNS = ["ticker", "strategy", "reason", "ref_price", "entry_date", "bars_held", "qty"]


@dataclass(frozen=True)
class OpenPosition:
    ticker: str  # símbolo yfinance (subjacente)
    strategy: str
    qty: int
    entry_date: dt.date
    entry_price: float
    stop_price: float | None
    max_hold: int
    signal_id: int | None = None


@dataclass
class ScreenResult:
    as_of: pd.Timestamp
    market: str
    entries: pd.DataFrame
    exits: pd.DataFrame
    equity: float
    cash: float
    open_positions: int
    slots: int
    regime: dict[str, float | bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.entries.empty and self.exits.empty


def _last_valid_date(panel: Panel, as_of: pd.Timestamp | None) -> pd.Timestamp:
    dates = panel.dates if as_of is None else panel.dates[panel.dates <= as_of]
    if len(dates) == 0:
        raise ValueError("nenhuma data disponível no painel até as_of")
    return pd.Timestamp(dates[-1])


def select_entries(
    panel: Panel,
    as_of: pd.Timestamp,
    *,
    risk: RiskModel,
    costs: CostModel,
    equity: float,
    cash: float,
    held: Sequence[str] = (),
    size_factor: float = 1.0,
    allow_entries: bool = True,
    ref_prices: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Entradas para executar na abertura do próximo pregão, ordenadas por prioridade.

    Espelha `Backtester.run` passo 5 (candidatos) + passo 2 (sizing/caps) para um dia.
    `ref_prices` (preço bruto de fechamento por subjacente) substitui o close ajustado como
    preço de referência quando fornecido.
    """
    cols = ENTRY_COLUMNS
    if not allow_entries:
        return pd.DataFrame(columns=cols)
    d = cast(int, panel.dates.get_loc(as_of))
    slots = max(risk.max_positions - len(held), 0)
    if slots == 0:
        return pd.DataFrame(columns=cols)

    held_set = set(held)
    entry = panel.entry.iloc[d].to_numpy(dtype=bool)
    score = panel.score.iloc[d].to_numpy(dtype=float)
    dvol = panel.dollar_vol.iloc[d].to_numpy(dtype=float)
    close = panel.close.iloc[d].to_numpy(dtype=float)
    atr_row = panel.atr.iloc[d].to_numpy(dtype=float)
    stop_row = panel.stop.iloc[d].to_numpy(dtype=float)
    hold_row = panel.max_hold.iloc[d].to_numpy(dtype=int)
    cands = [
        j
        for j in range(len(panel.tickers))
        if bool(entry[j])
        and panel.underlying[j] not in held_set
        and (risk.min_dollar_volume <= 0 or float(dvol[j]) >= risk.min_dollar_volume)
    ]
    cands.sort(key=lambda j: rank_key(float(score[j]), float(dvol[j])))

    params = SizingParams(
        risk_per_trade=risk.risk_per_trade,
        atr_multiple=risk.atr_multiple,
        max_position_pct=risk.max_position_pct,
        max_volume_participation=risk.max_volume_participation,
        board_lot=risk.board_lot,
        slippage_pct=costs.slippage_pct,
        fees_pct=costs.fees_pct,
    )
    rows: list[dict[str, object]] = []
    taken: set[str] = set()
    strategy_used: dict[str, float] = {}
    remaining_cash = cash
    for j in cands:
        if len(rows) >= slots:
            break
        u = panel.underlying[j]
        if u in taken:
            continue
        price = float(close[j])
        if ref_prices and u in ref_prices:
            price = float(ref_prices[u])
        qty = position_size(
            price, float(atr_row[j]), equity, remaining_cash, float(dvol[j]), size_factor, params
        )
        if risk.max_strategy_pct > 0:
            strat = panel.strategy_of[j]
            room = risk.max_strategy_pct * equity - strategy_used.get(strat, 0.0)
            qty = min(qty, round_lot(max(room, 0.0) / price, risk.board_lot))
        if qty <= 0:
            continue
        notional = qty * price
        remaining_cash -= notional * (1 + costs.slippage_pct) * (1 + costs.fees_pct)
        strategy_used[panel.strategy_of[j]] = (
            strategy_used.get(panel.strategy_of[j], 0.0) + notional
        )
        taken.add(u)
        stop = float(stop_row[j])
        rows.append(
            {
                "ticker": u,
                "strategy": panel.strategy_of[j],
                "ref_price": price,
                "qty": qty,
                "notional": notional,
                "stop_price": None if pd.isna(stop) else stop,
                "max_hold": int(hold_row[j]),
                "score": float(score[j]),
                "atr": float(atr_row[j]),
                "dollar_volume": float(dvol[j]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def check_exits(
    panel: Panel,
    as_of: pd.Timestamp,
    positions: Sequence[OpenPosition],
    ref_prices: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Para cada posição aberta: sinal de saída, stop por tempo ou stop de preço tocado hoje."""
    d = cast(int, panel.dates.get_loc(as_of))
    col_of = {
        (u, s): j for j, (u, s) in enumerate(zip(panel.underlying, panel.strategy_of, strict=True))
    }
    close = panel.close.iloc[d].to_numpy(dtype=float)
    low_row = panel.low.iloc[d].to_numpy(dtype=float)
    exit_row = panel.exit.iloc[d].to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    for pos in positions:
        j = col_of.get((pos.ticker, pos.strategy))
        if j is None:
            rows.append(_exit_row(pos, "not_in_universe", None, 0))
            continue
        entry_ts = pd.Timestamp(pos.entry_date)
        bars_held = int((panel.dates > entry_ts).sum() - (panel.dates > as_of).sum())
        price = float(close[j])
        if ref_prices and pos.ticker in ref_prices:
            price = float(ref_prices[pos.ticker])
        reason: str | None = None
        low = float(low_row[j])
        if pos.stop_price is not None and not pd.isna(low) and low <= pos.stop_price:
            reason = "stop"
        elif bool(exit_row[j]):
            reason = "signal"
        elif pos.max_hold and bars_held >= pos.max_hold:
            reason = "time"
        if reason:
            rows.append(_exit_row(pos, reason, price, bars_held))
    return pd.DataFrame(rows, columns=EXIT_COLUMNS)


def _exit_row(pos: OpenPosition, reason: str, price: float | None, bars: int) -> dict[str, object]:
    return {
        "ticker": pos.ticker,
        "strategy": pos.strategy,
        "reason": reason,
        "ref_price": price,
        "entry_date": pos.entry_date,
        "bars_held": bars,
        "qty": pos.qty,
    }


def run_screener(
    prices: pd.DataFrame,
    strategies: Mapping[str, Strategy],
    *,
    market: str,
    risk: RiskModel,
    costs: CostModel,
    equity: float,
    cash: float,
    positions: Sequence[OpenPosition] = (),
    regime: Regime | None = None,
    sectors: Mapping[str, str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> ScreenResult:
    """Pipeline completo: painéis → combinado → saídas das posições → novas entradas."""
    panels = {name: build_panel(prices, strat) for name, strat in strategies.items()}
    panel = combine_panels(panels, sectors)
    day = _last_valid_date(panel, as_of)
    raw_close: dict[str, float] = {
        str(k): float(v)
        for k, v in prices[prices["date"] == day].set_index("ticker")["close"].items()
    }
    allow, factor = True, 1.0
    regime_info: dict[str, float | bool] = {}
    if regime is not None:
        allow = bool(regime.allow_entries.reindex([day]).fillna(True).iloc[0])
        factor = float(regime.size_factor.reindex([day]).fillna(1.0).iloc[0])
        regime_info = {
            "allow_entries": allow,
            "size_factor": factor,
            "trend_on": bool(regime.trend_on.reindex([day]).fillna(False).iloc[0]),
            "high_vol": bool(regime.high_vol.reindex([day]).fillna(False).iloc[0]),
        }

    exits = check_exits(panel, day, positions, raw_close)
    exiting = set(exits["ticker"]) if not exits.empty else set()
    # tickers que saem amanhã liberam vaga só depois; hoje continuam ocupando (conservador)
    held = [p.ticker for p in positions]
    entries = select_entries(
        panel,
        day,
        risk=risk,
        costs=costs,
        equity=equity,
        cash=cash,
        held=held,
        size_factor=factor,
        allow_entries=allow,
        ref_prices=raw_close,
    )
    notes: list[str] = []
    if not allow:
        notes.append("Regime: entradas bloqueadas (tendência do benchmark desligada).")
    if factor < 1.0:
        notes.append(f"Regime: volatilidade alta — sizing × {factor:.2f}.")
    if exiting:
        notes.append(f"Saídas hoje liberam {len(exiting)} vaga(s) para o próximo screener.")
    return ScreenResult(
        as_of=day,
        market=market,
        entries=entries,
        exits=exits,
        equity=equity,
        cash=cash,
        open_positions=len(positions),
        slots=max(risk.max_positions - len(positions), 0),
        regime=regime_info,
        notes=notes,
    )
