"""Engine de backtest de carteira (long-only, diário, execução em D+1).

Regras (docs/02 §convenção, docs/03):
- Sinal no fechamento de D executa na **abertura de D+1** (entrada, saída por sinal e por tempo).
- Stop de preço: se low(D) <= stop, sai em D ao preço do stop (ou na abertura se abriu abaixo).
- Sizing por risco: qty = risco / (k*ATR) limitado por % do patrimônio, participação no volume
  e caixa disponível; arredondado ao lote quando >= lote.
- Mais candidatos que vagas → maior `score` primeiro (empate: maior volume financeiro).
- Custos por perna: corretagem fixa + taxas % + slippage % (contra o operador).

Regras de carteira (docs/03 §2–3, opcionais — 0 desliga):
- uma posição por ticker subjacente (painéis combinados);
- exposição por setor ≤ max_sector_pct; por estratégia ≤ max_strategy_pct (reduzem a qty);
- correlação 60d com posição aberta > max_correlation → não entra;
- drawdown mensal > monthly_dd_reduce → sizing × monthly_dd_size_factor até recuperar;
- drawdown do pico > circuit_breaker_dd → sem novas entradas até DD < metade do limite ou
  até vencer o cooldown (então o pico de referência é redefinido para o patrimônio atual).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swing_quant.backtest.panel import Panel
from swing_quant.risk.sizing import SizingParams, position_size, rank_key, round_lot


@dataclass(frozen=True)
class CostModel:
    commission_per_order: float = 0.0
    fees_pct: float = 0.0003
    slippage_pct: float = 0.001

    def scaled(self, mult: float) -> CostModel:
        return CostModel(
            self.commission_per_order * mult, self.fees_pct * mult, self.slippage_pct * mult
        )

    def buy_price(self, px: float) -> float:
        return px * (1.0 + self.slippage_pct)

    def sell_price(self, px: float) -> float:
        return px * (1.0 - self.slippage_pct)

    def fees(self, notional: float) -> float:
        return self.commission_per_order + notional * self.fees_pct


@dataclass(frozen=True)
class RiskModel:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    atr_multiple: float = 2.0
    max_position_pct: float = 0.20
    max_positions: int = 6
    max_volume_participation: float = 0.01
    board_lot: int = 100
    min_dollar_volume: float = 0.0
    # --- regras de carteira (0 = desligado)
    max_sector_pct: float = 0.0
    max_strategy_pct: float = 0.0
    max_correlation: float = 0.0
    corr_window: int = 60
    monthly_dd_reduce: float = 0.0
    monthly_dd_size_factor: float = 0.5
    circuit_breaker_dd: float = 0.0
    circuit_breaker_cooldown: int = 21  # pregões bloqueado; depois rearma com novo pico


@dataclass
class Position:
    col: int
    ticker: str
    underlying: str
    strategy: str
    qty: int
    entry_date: pd.Timestamp
    entry_price: float  # já com slippage
    entry_fees: float
    stop: float
    max_hold: int
    score: float
    target: float = float("nan")
    bars_held: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series  # patrimônio marcado a mercado no fechamento
    cash: pd.Series
    exposure: pd.Series  # valor investido / patrimônio
    n_positions: pd.Series
    initial_capital: float
    meta: dict[str, object] = field(default_factory=dict)
    risk_events: dict[str, int] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


TRADE_COLUMNS = [
    "ticker",
    "strategy",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "qty",
    "fees",
    "pnl",
    "ret",
    "bars_held",
    "exit_reason",
    "score",
]


class Backtester:
    def __init__(
        self,
        costs: CostModel | None = None,
        risk: RiskModel | None = None,
        *,
        allow_entries: pd.Series | None = None,
        size_factor: pd.Series | None = None,
    ) -> None:
        """`allow_entries`/`size_factor` são séries por data (filtro de regime) — opcionais."""
        self.costs = costs or CostModel()
        self.risk = risk or RiskModel()
        self.allow_entries = allow_entries
        self.size_factor = size_factor

    # ------------------------------------------------------------------ sizing
    def sizing_params(self) -> SizingParams:
        r = self.risk
        return SizingParams(
            risk_per_trade=r.risk_per_trade,
            atr_multiple=r.atr_multiple,
            max_position_pct=r.max_position_pct,
            max_volume_participation=r.max_volume_participation,
            board_lot=r.board_lot,
            slippage_pct=self.costs.slippage_pct,
            fees_pct=self.costs.fees_pct,
        )

    def _size(
        self, price: float, atr_value: float, equity: float, cash: float, dvol: float, factor: float
    ) -> int:
        return position_size(price, atr_value, equity, cash, dvol, factor, self.sizing_params())

    def _round_lot(self, qty: float) -> int:
        return round_lot(qty, self.risk.board_lot)

    # ------------------------------------------------------------------ simulação
    def run(self, panel: Panel) -> BacktestResult:
        r = self.risk
        dates = panel.dates
        tickers = panel.tickers
        underlying = panel.underlying
        strategy_of = panel.strategy_of
        sectors = panel.sectors
        n_days = len(dates)
        # arrays numpy para velocidade
        opn = panel.open.to_numpy(dtype=float)
        high = panel.high.to_numpy(dtype=float)
        low = panel.low.to_numpy(dtype=float)
        close = panel.close.to_numpy(dtype=float)
        atr_arr = panel.atr.to_numpy(dtype=float)
        dvol = panel.dollar_vol.to_numpy(dtype=float)
        entry = panel.entry.to_numpy(dtype=bool)
        exit_sig = panel.exit.to_numpy(dtype=bool)
        stop_arr = panel.stop.to_numpy(dtype=float)
        target_arr = panel.target.to_numpy(dtype=float)
        score = panel.score.to_numpy(dtype=float)
        max_hold = panel.max_hold.to_numpy(dtype=int)
        log_ret: np.ndarray | None = None
        if r.max_correlation > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                log_ret = np.diff(np.log(close), axis=0, prepend=np.nan)

        allow = (
            self.allow_entries.reindex(dates).fillna(True).to_numpy(dtype=bool)
            if self.allow_entries is not None
            else np.ones(n_days, dtype=bool)
        )
        factor = (
            self.size_factor.reindex(dates).fillna(1.0).to_numpy(dtype=float)
            if self.size_factor is not None
            else np.ones(n_days, dtype=float)
        )

        cash = r.initial_capital
        positions: dict[int, Position] = {}  # coluna -> posição
        held_underlying: set[str] = set()
        trades: list[dict[str, object]] = []
        equity_hist = np.empty(n_days)
        cash_hist = np.empty(n_days)
        expo_hist = np.empty(n_days)
        npos_hist = np.empty(n_days, dtype=int)
        events: Counter[str] = Counter()

        pending_exits: set[int] = set()
        pending_entries: list[int] = []
        # estado das regras de carteira
        peak_equity = r.initial_capital
        breaker_on = False
        breaker_days = 0
        month_start_equity = r.initial_capital
        month_key = (dates[0].year, dates[0].month) if n_days else None
        monthly_factor = 1.0

        def mark(p: Position, dd: int) -> float:
            """Valor da posição na abertura de `dd`; sem preço, usa o último fechamento válido
            e, na falta dele, o preço de entrada (painéis combinados podem ter buracos)."""
            px = opn[dd, p.col]
            if math.isnan(px) and dd > 0:
                px = close[dd - 1, p.col]
            if math.isnan(px):
                px = p.entry_price
            return float(p.qty * px)

        for d in range(n_days):
            date = dates[d]
            # ---------------------------------------------------------- 0) virada de mês
            if (date.year, date.month) != month_key:
                month_key = (date.year, date.month)
                month_start_equity = equity_hist[d - 1] if d > 0 else r.initial_capital
                monthly_factor = 1.0

            # ---------------------------------------------------------- 1) saídas na abertura
            for j in sorted(pending_exits):
                pos = positions.get(j)
                if pos is None or math.isnan(opn[d, j]):
                    continue
                reason = "time" if pos.max_hold and pos.bars_held >= pos.max_hold else "signal"
                cash += self._close_position(pos, date, opn[d, j], reason, trades)
                del positions[j]
                held_underlying.discard(pos.underlying)
            pending_exits.clear()

            # ---------------------------------------------------------- 2) entradas na abertura
            equity_open = cash + sum(mark(p, d) for p in positions.values())
            for j in pending_entries:
                if j in positions or len(positions) >= r.max_positions:
                    continue
                if underlying[j] in held_underlying:
                    events["skip_same_underlying"] += 1
                    continue
                px = opn[d, j]
                if math.isnan(px) or px <= 0:
                    continue
                if log_ret is not None and self._too_correlated(log_ret, d, j, positions):
                    events["skip_correlation"] += 1
                    continue
                qty = self._size(
                    px,
                    atr_arr[d - 1, j],
                    equity_open,
                    cash,
                    dvol[d - 1, j],
                    factor[d - 1] * monthly_factor,
                )
                if qty <= 0:
                    continue
                # caps de setor e de estratégia (reduzem a quantidade)
                if r.max_sector_pct > 0 and sectors:
                    sec = sectors.get(underlying[j])
                    if sec is not None:
                        used = sum(
                            mark(p, d)
                            for p in positions.values()
                            if sectors.get(p.underlying) == sec
                        )
                        capped = self._round_lot(max(r.max_sector_pct * equity_open - used, 0) / px)
                        if capped < qty:
                            events["cap_sector"] += 1
                            qty = capped
                if r.max_strategy_pct > 0:
                    used = sum(
                        mark(p, d) for p in positions.values() if p.strategy == strategy_of[j]
                    )
                    capped = self._round_lot(max(r.max_strategy_pct * equity_open - used, 0) / px)
                    if capped < qty:
                        events["cap_strategy"] += 1
                        qty = capped
                if qty <= 0:
                    continue
                fill = self.costs.buy_price(px)
                fees = self.costs.fees(fill * qty)
                cash -= fill * qty + fees
                positions[j] = Position(
                    col=j,
                    ticker=tickers[j],
                    underlying=underlying[j],
                    strategy=strategy_of[j],
                    qty=qty,
                    entry_date=date,
                    entry_price=fill,
                    entry_fees=fees,
                    stop=stop_arr[d - 1, j],
                    target=target_arr[d - 1, j],
                    max_hold=int(max_hold[d - 1, j]),
                    score=float(score[d - 1, j]),
                )
                held_underlying.add(underlying[j])
            pending_entries = []

            # ------------------------------------------------- 3) stops e alvos intradiários
            # Quando a barra toca os dois, assume-se o stop: sem intradiário não dá para saber
            # a ordem, e o pessimista é o único que não infla resultado.
            for j in list(positions):
                pos = positions[j]
                if not math.isnan(pos.stop) and not math.isnan(low[d, j]) and low[d, j] <= pos.stop:
                    px = min(pos.stop, opn[d, j]) if not math.isnan(opn[d, j]) else pos.stop
                    cash += self._close_position(pos, date, px, "stop", trades)
                    del positions[j]
                    held_underlying.discard(pos.underlying)
                    continue
                if (
                    not math.isnan(pos.target)
                    and not math.isnan(high[d, j])
                    and high[d, j] >= pos.target
                ):
                    # gap de abertura acima do alvo executa na abertura, não no alvo
                    px = max(pos.target, opn[d, j]) if not math.isnan(opn[d, j]) else pos.target
                    cash += self._close_position(pos, date, px, "target", trades)
                    del positions[j]
                    held_underlying.discard(pos.underlying)

            # ---------------------------------------------------------- 4) fechamento: marcação
            invested = 0.0
            for j, pos in positions.items():
                pos.bars_held += 1
                px = close[d, j] if not math.isnan(close[d, j]) else pos.entry_price
                invested += pos.qty * px
            equity = cash + invested
            equity_hist[d] = equity
            cash_hist[d] = cash
            expo_hist[d] = invested / equity if equity > 0 else 0.0
            npos_hist[d] = len(positions)

            # ---------------------------------------------------------- 4b) regras de carteira
            peak_equity = max(peak_equity, equity)
            dd = equity / peak_equity - 1.0
            if r.circuit_breaker_dd > 0:
                if not breaker_on and dd < -r.circuit_breaker_dd:
                    breaker_on = True
                    breaker_days = 0
                    events["circuit_breaker_on"] += 1
                elif breaker_on:
                    breaker_days += 1
                    if dd > -r.circuit_breaker_dd / 2:
                        breaker_on = False  # recuperou metade do drawdown
                    elif breaker_days >= r.circuit_breaker_cooldown:
                        # cooldown vencido: rearma a partir de um novo pico de referência
                        breaker_on = False
                        peak_equity = equity
                        events["circuit_breaker_reset"] += 1
            if r.monthly_dd_reduce > 0:
                m_dd = equity / month_start_equity - 1.0
                if monthly_factor == 1.0 and m_dd < -r.monthly_dd_reduce:
                    monthly_factor = r.monthly_dd_size_factor
                    events["monthly_dd_reduce"] += 1
                elif monthly_factor < 1.0 and equity >= month_start_equity:
                    monthly_factor = 1.0

            # ---------------------------------------------------------- 5) sinais para D+1
            for j, pos in positions.items():
                if exit_sig[d, j] or (pos.max_hold and pos.bars_held >= pos.max_hold):
                    pending_exits.add(j)
            if allow[d] and not breaker_on:
                idxs = np.flatnonzero(entry[d])
                cands = [
                    int(j)
                    for j in idxs
                    if j not in positions
                    and (r.min_dollar_volume <= 0 or dvol[d, j] >= r.min_dollar_volume)
                ]
                cands.sort(key=lambda j: rank_key(score[d, j], dvol[d, j]))
                pending_entries = cands
            elif breaker_on and entry[d].any():
                events["blocked_by_breaker"] += int(entry[d].sum())

        # posições abertas no fim: encerra no último fechamento (marcação, não trade real)
        last = n_days - 1
        for pos in list(positions.values()):
            px = close[last, pos.col] if not math.isnan(close[last, pos.col]) else pos.entry_price
            self._close_position(pos, dates[last], px, "end", trades)

        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        idx = pd.DatetimeIndex(dates)
        return BacktestResult(
            trades=trades_df,
            equity=pd.Series(equity_hist, index=idx, name="equity"),
            cash=pd.Series(cash_hist, index=idx, name="cash"),
            exposure=pd.Series(expo_hist, index=idx, name="exposure"),
            n_positions=pd.Series(npos_hist, index=idx, name="n_positions"),
            initial_capital=r.initial_capital,
            meta={**panel.meta, "costs": self.costs, "risk": self.risk},
            risk_events=dict(events),
        )

    # ------------------------------------------------------------------ auxiliares
    def _too_correlated(
        self, log_ret: np.ndarray, d: int, j: int, positions: dict[int, Position]
    ) -> bool:
        w = self.risk.corr_window
        lo = max(1, d - w)
        a = log_ret[lo:d, j]
        for pos in positions.values():
            b = log_ret[lo:d, pos.col]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < w // 2:
                continue
            aa, bb = a[mask], b[mask]
            if aa.std() == 0 or bb.std() == 0:
                continue
            c = float(np.corrcoef(aa, bb)[0, 1])
            if c > self.risk.max_correlation:
                return True
        return False

    def _close_position(
        self,
        pos: Position,
        date: pd.Timestamp,
        raw_px: float,
        reason: str,
        trades: list[dict[str, object]],
    ) -> float:
        fill = self.costs.sell_price(raw_px)
        fees = self.costs.fees(fill * pos.qty)
        proceeds = fill * pos.qty - fees
        cost_basis = pos.entry_price * pos.qty + pos.entry_fees
        pnl = proceeds - cost_basis
        trades.append(
            {
                "ticker": pos.ticker,
                "strategy": pos.strategy,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "exit_date": date,
                "exit_price": fill,
                "qty": pos.qty,
                "fees": pos.entry_fees + fees,
                "pnl": pnl,
                "ret": pnl / cost_basis if cost_basis else 0.0,
                "bars_held": pos.bars_held,
                "exit_reason": reason,
                "score": pos.score,
            }
        )
        return proceeds
