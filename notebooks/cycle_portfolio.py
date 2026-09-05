"""A portfolio that rebalances by where the market is in its cycle, not by the calendar.

`investor.py` prices the portfolios an investor *holds*: fixed weights, reset once a year. This
module prices the one an investor *steers* - the same asset classes, but a different target
allocation depending on which quadrant of the growth/inflation cycle the market is in, and a
rebalance triggered by the quadrant changing rather than by December arriving.

The map is the investment clock: two binary axes, four phases, one allocation each.

                       inflation falling      inflation rising
    growth up          Recovery               Overheat
    growth down        Slowdown               Stagflation

Four decisions make the numbers worth reading, and each is a place where a tactical backtest
usually cheats:

1. **The phase is a nowcast, never a hindsight label.** Growth is read from the equity market's
   own trailing excess return over cash - available the same evening - and inflation from the
   published price index, lagged by `macro_lag` months because the IPCA for a month lands in the
   middle of the next one. The phase read at the end of month *t* governs month *t+1*'s return,
   so no allocation is ever fed by the return it is about to earn.
2. **Turnover is charged.** Switching quadrants moves a third of the portfolio; at 15 bps a side
   that is a real cost, and a cycle rule that only wins gross does not win.
3. **The control has the same menu.** The comparison is not against a 60/40 but against the
   *average of the four phase allocations*, rebalanced annually: identical assets, identical
   cost model, no timing. Whatever separates the two lines is the timing and nothing else.
4. **The timing is tested against luck.** `rotation_test` slides the same phase sequence forward
   in time by every possible offset. That keeps the phase frequencies and the persistence intact
   and destroys only the alignment with the market, so the percentile it returns says how much
   of the result was the signal and how much was the shape of the era.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import asset_classes as A  # noqa: E402
import investor as INV  # noqa: E402
from asset_classes import MONTHS, ClassCurve, deflate  # noqa: E402

from swing_quant.data.assets import AssetProxy  # noqa: E402

# --------------------------------------------------------------------------- the four phases
PHASES: tuple[str, ...] = ("recovery", "overheat", "stagflation", "slowdown")

PHASE_LABEL: dict[str, str] = {
    "recovery": "Recovery (growth up, inflation falling)",
    "overheat": "Overheat (growth up, inflation rising)",
    "stagflation": "Stagflation (growth down, inflation rising)",
    "slowdown": "Slowdown (growth down, inflation falling)",
}

PHASE_PT: dict[str, str] = {
    "recovery": "Recuperação (crescimento em alta, inflação caindo)",
    "overheat": "Aquecimento (crescimento em alta, inflação subindo)",
    "stagflation": "Estagflação (crescimento em baixa, inflação subindo)",
    "slowdown": "Desaceleração (crescimento em baixa, inflação caindo)",
}

# The market's own read on growth: has the stock index beaten cash over the last year? It is a
# nowcast, not a forecast of GDP, and in Brazil the "over cash" half is not decoration - with the
# CDI at 13% a year, an index up 8% is a market saying growth is *not* happening.
GROWTH_PROXY: dict[str, str] = {"b3": "acoes_br", "us": "acoes_us"}


@dataclass(frozen=True)
class CycleConfig:
    """Everything the rule can be argued about, in one place and with a defended default.

    The windows are deliberately long and few. A quadrant that flips on a three-month reading
    trades every other month and pays for the privilege; the point of the clock is that phases
    last quarters, not weeks.
    """

    trend_months: int = 12  # window of the equity-over-cash growth read
    inflation_window: int = 12  # trailing inflation accumulation
    inflation_trend: int = 12  # its own moving average: above it = rising
    macro_lag: int = 1  # months of publication lag on the price index
    confirm_months: int = 2  # consecutive readings before the allocation actually moves
    drift_months: int = 12  # calendar reset while the phase holds
    cost_bps: float = 15.0  # one-way cost per unit of turnover


# --------------------------------------------------------------------------- allocations
# Brazil. Written as an investor would say them out loud, using only classes that exist in
# `market.duckdb` from 2011 on. Bitcoin is left out on purpose: it starts in late 2014 and would
# decide the whole table by itself. Each row is normalised before use.
BR_CYCLE: dict[str, dict[str, float]] = {
    # Falling inflation with the market already turning: the moment both duration and equity
    # risk get paid, and the one phase where small caps are worth owning.
    "recovery": {
        "acoes_br": 0.30,
        "small_caps_br": 0.10,
        "fiis": 0.15,
        "sp500_brl": 0.10,
        "tesouro_prefixado": 0.20,
        "cdi": 0.15,
    },
    # Growth holding up while prices accelerate: real assets and inflation linkers. The
    # fixed-rate bond is the thing to be out of.
    "overheat": {
        "acoes_br": 0.25,
        "fiis": 0.10,
        "sp500_brl": 0.15,
        "ouro_brl": 0.10,
        "tesouro_ipca": 0.25,
        "cdi": 0.15,
    },
    # The Brazilian nightmare quadrant. Cash is not a parking spot here, it is the position, and
    # the dollarised sleeve is what pays when the domestic story is itself the problem.
    "stagflation": {
        "acoes_br": 0.10,
        "sp500_brl": 0.15,
        "ouro_brl": 0.15,
        "tesouro_ipca": 0.20,
        "cdi": 0.40,
    },
    # Disinflation into weakness: the only phase where a pre-fixed bond is the best asset in the
    # country, because the cut is coming and the coupon is already locked.
    "slowdown": {
        "acoes_br": 0.10,
        "sp500_brl": 0.10,
        "tesouro_prefixado": 0.35,
        "tesouro_ipca": 0.20,
        "cdi": 0.25,
    },
}

US_CYCLE: dict[str, dict[str, float]] = {
    "recovery": {
        "acoes_us": 0.40,
        "small_caps_us": 0.15,
        "reits_us": 0.10,
        "bonds_us": 0.20,
        "tbills": 0.15,
    },
    "overheat": {
        "acoes_us": 0.30,
        "emergentes": 0.10,
        "commodities": 0.15,
        "ouro_usd": 0.10,
        "tips": 0.20,
        "tbills": 0.15,
    },
    "stagflation": {
        "acoes_us": 0.10,
        "commodities": 0.15,
        "ouro_usd": 0.20,
        "tips": 0.20,
        "tbills": 0.35,
    },
    "slowdown": {
        "acoes_us": 0.15,
        "treasuries_longas": 0.30,
        "bonds_us": 0.25,
        "tbills": 0.30,
    },
}

CYCLE_ALLOCATIONS: dict[str, dict[str, dict[str, float]]] = {"b3": BR_CYCLE, "us": US_CYCLE}


def normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def allocation_frame(alloc: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Phases as rows, asset classes as columns - the allocation table, ready to print."""
    df = pd.DataFrame({p: normalise(alloc[p]) for p in PHASES}).T.fillna(0.0)
    df.index = pd.Index([PHASE_LABEL[p] for p in PHASES], name="Phase")
    return df


def neutral_weights(alloc: dict[str, dict[str, float]]) -> dict[str, float]:
    """The average of the four phase allocations: the same menu with the timing removed.

    This is the control the cycle rule has to beat. Beating a 60/40 would only prove that this
    module picked a better asset menu, which is a different claim and a much cheaper one.
    """
    return normalise(dict(allocation_frame(alloc).mean(axis=0)))


def assets_of(alloc: dict[str, dict[str, float]]) -> list[str]:
    return sorted({k for phase in alloc.values() for k in phase})


# --------------------------------------------------------------------------- reading the clock
def _trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compounded return of the trailing `window` months, dated at the last month in it."""
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _confirm(raw: pd.Series, months: int) -> pd.Series:
    """Hold the current phase until a new one has been read `months` times in a row.

    Without this the allocation chases every month in which the index crosses cash by a hair.
    The confirmation costs a month of lateness at every genuine turn and saves a whole series of
    round trips at the false ones; at `months=1` it does nothing, which is how to price it.
    """
    if months <= 1:
        return raw.copy()
    values = list(raw)
    held = values[0]
    out: list[str] = []
    for i in range(len(values)):
        window = values[max(0, i - months + 1) : i + 1]
        if len(window) == months and all(v == window[-1] for v in window):
            held = window[-1]
        out.append(held)
    return pd.Series(out, index=raw.index, name="phase")


def phase_signals(
    equity: pd.Series, cash: pd.Series, inflation: pd.Series, cfg: CycleConfig | None = None
) -> pd.DataFrame:
    """The two axes and the resulting phase, as known at the **end** of each month.

    Every input is lagged to what was actually publishable: prices through the month's close,
    the price index through `macro_lag` months earlier. The caller must still shift the result by
    one month before it is allowed to earn anything - `run_cycle` does that.
    """
    cfg = cfg or CycleConfig()
    eq = _trailing(equity, cfg.trend_months)
    cs = _trailing(cash.reindex(equity.index), cfg.trend_months)
    # The gaps stay numeric until after the dropna. Comparing straight to a boolean would turn
    # every warm-up month - where the rolling window is still NaN - into a silent `False`, and
    # a False on both axes is not "no reading", it is a confident call of Slowdown.
    growth_gap = ((1.0 + eq) / (1.0 + cs) - 1.0).rename("growth_gap")

    infl = _trailing(inflation, cfg.inflation_window)
    infl_gap = (infl - infl.rolling(cfg.inflation_trend).mean()).rename("inflation_gap")

    df = pd.concat([growth_gap, infl_gap.shift(cfg.macro_lag)], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "growth_gap",
                "inflation_gap",
                "growth_up",
                "inflation_rising",
                "raw_phase",
                "phase",
            ]
        )
    df["growth_up"] = df["growth_gap"] > 0.0
    df["inflation_rising"] = df["inflation_gap"] > 0.0
    df["raw_phase"] = [
        ("overheat" if inf else "recovery") if up else ("stagflation" if inf else "slowdown")
        for up, inf in zip(df["growth_up"], df["inflation_rising"], strict=True)
    ]
    df["phase"] = _confirm(df["raw_phase"], cfg.confirm_months)
    return df


# --------------------------------------------------------------------------- the portfolio
@dataclass
class CycleRun:
    """One run of an allocation rule: what it earned, what it held, what the trading cost."""

    label: str
    returns: pd.Series  # net monthly returns, PeriodIndex
    weights: pd.DataFrame  # weights actually held during each month
    phase: pd.Series  # the phase that governed each month ("static" for the controls)
    turnover: pd.Series  # one-way turnover charged at the start of each month
    cost_bps: float
    notes: list[str] = field(default_factory=list)

    def curve(self, base: float = 100.0) -> pd.Series:
        return base * (1.0 + self.returns).cumprod()

    @property
    def annual_turnover(self) -> float:
        years = len(self.returns) / MONTHS
        return float(self.turnover.sum() / years) if years else float("nan")

    @property
    def gross_returns(self) -> pd.Series:
        """What the same rule would have earned with free trading - the cost drag, isolated."""
        return self.returns + self.turnover * self.cost_bps / 1e4

    def slice(self, start: pd.Period, end: pd.Period) -> CycleRun:
        return CycleRun(
            self.label,
            self.returns.loc[start:end],
            self.weights.loc[start:end],
            self.phase.loc[start:end],
            self.turnover.loc[start:end],
            self.cost_bps,
            list(self.notes),
        )

    def as_class(self, market: str) -> ClassCurve:
        """Wrap the run as one more asset class, so `investor`'s tables can hold it too."""
        proxy = AssetProxy(
            key=self.label,
            label=self.label,
            asset_class="portfolio",
            market=market,
            kind="ticker",
            note=f"{self.annual_turnover:.0%} de giro ao ano a {self.cost_bps:.0f} bps",
        )
        return ClassCurve(proxy, self.returns)


def _blend(
    frame: pd.DataFrame,
    targets: np.ndarray,
    governing: np.ndarray,
    *,
    drift_months: int,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The engine: drift while the phase holds, jump to target when it changes or time runs out.

    `governing[i]` is the row of `targets` that must be held during month `i`. Weights drift with
    the assets between rebalances, which is what a portfolio does when left alone, and half the
    L1 distance to the new target is charged as one-way turnover on the way in.
    """
    r = frame.to_numpy(dtype=float)
    n, k = r.shape
    held = np.empty((n, k), dtype=float)
    net = np.empty(n, dtype=float)
    turn = np.zeros(n, dtype=float)

    current = int(governing[0])
    w = targets[current].copy()
    since = 0
    for i in range(n):
        if i > 0 and (int(governing[i]) != current or since >= drift_months):
            target = targets[int(governing[i])]
            turn[i] = 0.5 * float(np.abs(target - w).sum())
            w = target.copy()
            current = int(governing[i])
            since = 0
        held[i] = w
        net[i] = float(w @ r[i]) - turn[i] * cost_rate
        grown = w * (1.0 + r[i])
        total = float(grown.sum())
        w = grown / total if total > 0 else targets[current].copy()
        since += 1
    return net, held, turn


def _prepare(
    parts: dict[str, pd.Series], alloc: dict[str, dict[str, float]]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Line the asset menu up as a dense monthly matrix plus one target vector per phase."""
    missing = [a for a in assets_of(alloc) if a not in parts]
    if missing:
        raise KeyError(f"classes de ativo ausentes nos dados: {missing}")
    assets = assets_of(alloc)
    frame = pd.DataFrame({a: parts[a] for a in assets}).dropna()
    targets = np.array(
        [[normalise(alloc[p]).get(a, 0.0) for a in assets] for p in PHASES], dtype=float
    )
    return frame, targets, assets


def run_cycle(
    parts: dict[str, pd.Series],
    phase: pd.Series,
    alloc: dict[str, dict[str, float]],
    cfg: CycleConfig | None = None,
    *,
    label: str = "ciclo",
) -> CycleRun:
    """The cycle portfolio. `phase` is dated by the month it was *read*, not the month it earns.

    The single shift below is the whole no-look-ahead guarantee: the allocation for January is
    the one the clock was showing on the last day of December.
    """
    cfg = cfg or CycleConfig()
    frame, targets, assets = _prepare(parts, alloc)
    governing_phase = phase.shift(1).reindex(frame.index).dropna()
    frame = frame.loc[governing_phase.index]
    idx = {p: i for i, p in enumerate(PHASES)}
    governing = np.array([idx[str(p)] for p in governing_phase], dtype=int)

    net, held, turn = _blend(
        frame, targets, governing, drift_months=cfg.drift_months, cost_rate=cfg.cost_bps / 1e4
    )
    return CycleRun(
        label=label,
        returns=pd.Series(net, index=frame.index),
        weights=pd.DataFrame(held, index=frame.index, columns=assets),
        phase=governing_phase.rename("phase"),
        turnover=pd.Series(turn, index=frame.index),
        cost_bps=cfg.cost_bps,
    )


def run_static(
    parts: dict[str, pd.Series],
    alloc: dict[str, dict[str, float]],
    cfg: CycleConfig | None = None,
    *,
    window: pd.PeriodIndex | None = None,
    label: str = "estática (mesmo cardápio)",
) -> CycleRun:
    """The control: the average of the four allocations, rebalanced on the calendar only."""
    cfg = cfg or CycleConfig()
    frame, _, assets = _prepare(parts, alloc)
    if window is not None:
        frame = frame.loc[frame.index.isin(window)]
    neutral = neutral_weights(alloc)
    targets = np.array([[neutral.get(a, 0.0) for a in assets]], dtype=float)
    net, held, turn = _blend(
        frame,
        targets,
        np.zeros(len(frame), dtype=int),
        drift_months=cfg.drift_months,
        cost_rate=cfg.cost_bps / 1e4,
    )
    return CycleRun(
        label=label,
        returns=pd.Series(net, index=frame.index),
        weights=pd.DataFrame(held, index=frame.index, columns=assets),
        phase=pd.Series("static", index=frame.index, name="phase"),
        turnover=pd.Series(turn, index=frame.index),
        cost_bps=cfg.cost_bps,
    )


def run_oracle(
    parts: dict[str, pd.Series],
    alloc: dict[str, dict[str, float]],
    cfg: CycleConfig | None = None,
    *,
    window: pd.PeriodIndex | None = None,
    label: str = "oráculo (com look-ahead)",
) -> CycleRun:
    """Upper bound: each month, the phase allocation that turned out to be the best one.

    Nobody can trade this. It exists to price the ceiling - if the perfect sequence of these four
    baskets is only a couple of points a year better than holding their average, then no amount
    of signal work on this menu was ever going to pay, and the honest answer is a static
    portfolio.
    """
    cfg = cfg or CycleConfig()
    frame, targets, _ = _prepare(parts, alloc)
    if window is not None:
        frame = frame.loc[frame.index.isin(window)]
    best = (frame.to_numpy(dtype=float) @ targets.T).argmax(axis=1)
    perfect = pd.Series([PHASES[i] for i in best], index=frame.index)
    return run_cycle(parts, perfect.shift(-1), alloc, cfg, label=label)


# --------------------------------------------------------------------------- is it the signal?
def rotation_test(
    parts: dict[str, pd.Series],
    phase: pd.Series,
    alloc: dict[str, dict[str, float]],
    cfg: CycleConfig | None = None,
    *,
    inflation: pd.Series,
    min_offset: int = 6,
) -> dict[str, float]:
    """Slide the phase sequence through time and see where the real alignment ranks.

    A rotation keeps the four phases in the same proportions and keeps their runs the same
    length; it breaks only the correspondence between a phase and the month it was read in. So
    this asks the one question that matters about a tactical rule - was it the timing, or was it
    that the era rewarded the average of these baskets no matter when you held which?
    """
    cfg = cfg or CycleConfig()
    actual = cagr_real(run_cycle(parts, phase, alloc, cfg).returns, inflation)
    values = phase.to_numpy()
    n = len(values)
    draws = [
        cagr_real(
            run_cycle(parts, pd.Series(np.roll(values, k), index=phase.index), alloc, cfg).returns,
            inflation,
        )
        for k in range(min_offset, n - min_offset)
    ]
    arr = np.array(draws, dtype=float)
    return {
        "actual": actual,
        "median_rotated": float(np.median(arr)),
        "p95_rotated": float(np.quantile(arr, 0.95)),
        "best_rotated": float(arr.max()),
        "percentile": float((arr < actual).mean()),
        "n_rotations": float(len(arr)),
    }


def sensitivity(
    parts: dict[str, pd.Series],
    equity: pd.Series,
    cash: pd.Series,
    inflation: pd.Series,
    alloc: dict[str, dict[str, float]],
    *,
    confirms: tuple[int, ...] = (1, 2, 3),
    trends: tuple[int, ...] = (9, 12, 18),
    cost_bps: tuple[float, ...] = (15.0, 50.0),
    with_rotation: bool = False,
) -> pd.DataFrame:
    """The same rule over a grid of its own knobs, each cell scored against its own control.

    A tactical rule that wins in one cell of a grid and loses in the neighbouring one has not
    been discovered, it has been selected. The project's backtest protocol demands a plateau
    before a strategy is believed; a portfolio rule deserves the same test, and it is cheap here
    because the whole grid is a few seconds of arithmetic.
    """
    rows = []
    for confirm in confirms:
        for trend in trends:
            for cost in cost_bps:
                cfg = CycleConfig(confirm_months=confirm, trend_months=trend, cost_bps=cost)
                signals = phase_signals(equity, cash, inflation, cfg)
                cycle = run_cycle(parts, signals["phase"], alloc, cfg)
                control = run_static(parts, alloc, cfg, window=cycle.returns.index)
                a = summary(cycle, inflation, cash)
                b = summary(control, inflation, cash)
                row = {
                    "confirm": confirm,
                    "trend": trend,
                    "cost_bps": cost,
                    "cagr_real": a["cagr_real"],
                    "control_real": b["cagr_real"],
                    "delta": a["cagr_real"] - b["cagr_real"],
                    "max_drawdown": a["max_drawdown"],
                    "control_dd": b["max_drawdown"],
                    "sharpe": a["sharpe"],
                    "control_sharpe": b["sharpe"],
                    "annual_turnover": a["annual_turnover"],
                }
                if with_rotation:
                    row["rotation_pct"] = rotation_test(
                        parts, signals["phase"], alloc, cfg, inflation=inflation
                    )["percentile"]
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics
def cagr_real(monthly: pd.Series, inflation: pd.Series) -> float:
    real = deflate(monthly, inflation)
    years = len(real) / MONTHS
    return float((1.0 + real).prod() ** (1.0 / years) - 1.0) if years else float("nan")


def summary(run: CycleRun, inflation: pd.Series, cash: pd.Series) -> dict[str, float]:
    """One row of the comparison table: return, risk, and what the trading cost to get it."""
    rets = run.returns.loc[run.returns.index.isin(inflation.index)]
    years = len(rets) / MONTHS
    excess = (rets - cash.reindex(rets.index)).dropna()
    sd = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    by_year = (1.0 + rets).groupby(rets.index.year).prod() - 1.0
    gross = run.gross_returns.loc[rets.index]
    return {
        "cagr_real": cagr_real(rets, inflation),
        "cagr_nominal": float((1.0 + rets).prod() ** (1.0 / years) - 1.0)
        if years
        else float("nan"),
        "volatility": float(rets.std(ddof=1) * math.sqrt(MONTHS)),
        "max_drawdown": A.max_drawdown_monthly(rets),
        "sharpe": float(excess.mean() / sd * math.sqrt(MONTHS)) if sd > 1e-12 else float("nan"),
        "worst_year": float(by_year.min()),
        "positive_months": float((rets > 0).mean()),
        "annual_turnover": float(run.turnover.loc[rets.index].sum() / years) if years else 0.0,
        "cost_drag": cagr_real(gross, inflation) - cagr_real(rets, inflation),
        "years": years,
    }


def compare(
    runs: list[CycleRun], inflation: pd.Series, cash: pd.Series, *, common: bool = True
) -> pd.DataFrame:
    """Every run over the same months, because a longer sample is not a better result."""
    if common and runs:
        start = max(r.returns.index[0] for r in runs)
        end = min(r.returns.index[-1] for r in runs)
        runs = [r.slice(start, end) for r in runs]
    out = pd.DataFrame({r.label: summary(r, inflation, cash) for r in runs}).T
    out.index.name = "Carteira"
    return out


def annual_returns(run: CycleRun, inflation: pd.Series, *, real: bool = True) -> pd.Series:
    """Calendar-year returns of one run, real by default.

    Nominal and real are trimmed to the same months - the deflator ends before the price data
    does, and a nominal column one month longer than the real one is not the same year.
    """
    rets = run.returns.loc[run.returns.index.isin(inflation.index)]
    return A.annual_returns(rets, inflation if real else None)


def annual_table(
    runs: list[CycleRun],
    inflation: pd.Series,
    *,
    real: bool = True,
    common: bool = True,
    phase: pd.Series | None = None,
) -> pd.DataFrame:
    """Year by year, side by side - the view a CAGR hides.

    A single annualised number says a rule won; it cannot say *when*. Two portfolios with the
    same CAGR and opposite years are not the same investment, and a tactical rule that earns its
    whole margin in one year is a bet with a backtest wrapped around it. The `meses` column keeps
    partial years honest - the first and last rows of any sample are almost never twelve months.
    """
    if common and runs:
        start = max(r.returns.index[0] for r in runs)
        end = min(r.returns.index[-1] for r in runs)
        runs = [r.slice(start, end) for r in runs]
    out = pd.DataFrame({r.label: annual_returns(r, inflation, real=real) for r in runs})
    months = runs[0].returns.loc[runs[0].returns.index.isin(inflation.index)]
    out.insert(0, "meses", months.groupby(months.index.year).size())
    if phase is not None:
        governing = phase.reindex(months.index).dropna()
        # The phase the clock spent most of the year in: enough to read the table as a story.
        out.insert(
            1,
            "fase dominante",
            governing.groupby(governing.index.year).agg(lambda s: PHASE_PT[s.mode().iloc[0]]),
        )
    out.index.name = "Ano"
    return out


def annual_edge(cycle: CycleRun, control: CycleRun, inflation: pd.Series) -> dict[str, float]:
    """How the margin over the control is spread across the years, and what one year is worth.

    A margin earned in eight years out of fifteen is a rule; a margin earned in one is that year.
    Dropping the single best year is the crudest possible robustness check and also the one that
    settles the argument fastest - if the advantage inverts without it, there was no advantage.
    """
    start = max(cycle.returns.index[0], control.returns.index[0])
    end = min(cycle.returns.index[-1], control.returns.index[-1])
    a = annual_returns(cycle.slice(start, end), inflation)
    b = annual_returns(control.slice(start, end), inflation)
    delta = (a - b).dropna()
    best = delta.idxmax()

    def _cagr(s: pd.Series) -> float:
        return float((1.0 + s).prod() ** (1.0 / len(s)) - 1.0) if len(s) else float("nan")

    return {
        "years": float(len(delta)),
        "years_ahead": float((delta > 0.0005).sum()),
        "years_behind": float((delta < -0.0005).sum()),
        "best_year": float(best),
        "best_year_delta": float(delta.max()),
        "worst_year": float(delta.idxmin()),
        "worst_year_delta": float(delta.min()),
        "edge": _cagr(a) - _cagr(b),
        "edge_without_best_year": _cagr(a.drop(best)) - _cagr(b.drop(best)),
    }


def phase_months(phase: pd.Series) -> pd.DataFrame:
    """How long the clock spent in each quadrant, and how often it moved."""
    counts = phase.value_counts().reindex(PHASES).fillna(0.0)
    changed = phase != phase.shift(1)
    runs = pd.DataFrame({"phase": phase, "run": changed.cumsum()})
    lengths = runs.groupby(["run", "phase"], observed=True).size().reset_index(name="n")
    out = pd.DataFrame(
        {
            "months": counts,
            "share": counts / counts.sum(),
            "avg_run_months": lengths.groupby("phase")["n"].mean().reindex(PHASES),
        }
    )
    out.index = pd.Index([PHASE_LABEL[p] for p in PHASES], name="Fase")
    switches = int(changed.sum() - 1)
    out.attrs["switches"] = switches
    out.attrs["switches_per_year"] = switches / (len(phase) / MONTHS)
    return out


def by_phase_returns(
    curves: dict[str, ClassCurve], phase: pd.Series, inflation: pd.Series
) -> pd.DataFrame:
    """Real annualised return of each asset class inside each phase.

    The allocations above were written against this table, so reading it as evidence *for* them
    is circular: it is in-sample by construction. It earns its place for the opposite reason - a
    phase where the intended winner did not win is a phase whose row needs rewriting, and that
    shows up here immediately.
    """
    governing = phase.shift(1).dropna()
    rows: dict[str, dict[str, float]] = {}
    for curve in curves.values():
        real = deflate(curve.returns, inflation).reindex(governing.index).dropna()
        if len(real) < MONTHS:
            continue
        grouped = real.groupby(governing.reindex(real.index), observed=True)
        rows[A.label_of(curve)] = {
            PHASE_LABEL[str(name)]: float((1.0 + g).prod() ** (MONTHS / len(g)) - 1.0)
            for name, g in grouped
            if len(g) >= 6
        }
    out = pd.DataFrame(rows).T.reindex(columns=[PHASE_LABEL[p] for p in PHASES])
    out.index.name = "Classe de ativo"
    return out


# --------------------------------------------------------------------------- what to do today
def current_stance(
    signals: pd.DataFrame, alloc: dict[str, dict[str, float]], market: str = "b3"
) -> dict[str, object]:
    """The phase the clock is showing now and the allocation it asks for - the actionable end.

    The weights are the target for *next* month, because that is the only month this reading is
    entitled to govern.
    """
    phases = list(signals["phase"])
    phase = str(phases[-1])
    i = len(phases) - 1
    while i > 0 and phases[i - 1] == phase:
        i -= 1
    return {
        "as_of": signals.index[-1],
        "governs": signals.index[-1] + 1,
        "phase": phase,
        "label": PHASE_LABEL[phase],
        "label_pt": PHASE_PT[phase],
        "in_phase_since": signals.index[i],
        "growth_up": bool(signals["growth_up"].iloc[-1]),
        "inflation_rising": bool(signals["inflation_rising"].iloc[-1]),
        "weights": dict(sorted(normalise(alloc[phase]).items(), key=lambda kv: -kv[1])),
        "market": market,
    }


# --------------------------------------------------------------------------- putting it together
@dataclass
class CycleStudy:
    """Everything the report needs, built once from the database."""

    market: str
    cfg: CycleConfig
    curves: dict[str, ClassCurve]
    inflation: pd.Series
    signals: pd.DataFrame
    runs: list[CycleRun]
    table: pd.DataFrame
    phases: pd.DataFrame
    by_phase: pd.DataFrame
    rotation: dict[str, float]
    grid: pd.DataFrame
    stance: dict[str, object]

    @property
    def cycle(self) -> CycleRun:
        return self.runs[0]


# The two static portfolios from the study that a reader will ask about anyway. They own a
# different asset menu, so they are context, not the control - the control is `run_static`.
REFERENCE_KEYS: tuple[str, ...] = ("p_6040", "p_permanent")


def build_study(
    market: str = "b3",
    cfg: CycleConfig | None = None,
    db_path: Path = A.DB_PATH,
    *,
    rotation_grid: bool = True,
) -> CycleStudy:
    """Load the classes, read the clock, run the four variants and score them on equal months."""
    cfg = cfg or CycleConfig()
    curves = A.load_classes(market, db_path)
    inflation = A.load_inflation(market, db_path)
    alloc = CYCLE_ALLOCATIONS[market]
    cash = curves[A.CASH_KEY[market]].returns
    parts = {k: c.returns for k, c in curves.items()}

    signals = phase_signals(curves[GROWTH_PROXY[market]].returns, cash, inflation, cfg)
    cycle = run_cycle(parts, signals["phase"], alloc, cfg)
    window = cycle.returns.index
    runs = [
        cycle,
        run_static(parts, alloc, cfg, window=window),
        run_oracle(parts, alloc, cfg, window=window),
    ]
    for spec in (s for s in INV.PORTFOLIOS[market] if s.key in REFERENCE_KEYS):
        built = INV.portfolio(curves, spec)
        if built is None:
            continue
        rets = built.returns.loc[window[0] : window[-1]]
        runs.append(
            CycleRun(
                label=spec.label,
                returns=rets,
                weights=pd.DataFrame(index=rets.index),
                phase=pd.Series("static", index=rets.index, name="phase"),
                turnover=pd.Series(0.0, index=rets.index),
                cost_bps=0.0,
                notes=["cardápio diferente: referência, não controle"],
            )
        )

    return CycleStudy(
        market=market,
        cfg=cfg,
        curves=curves,
        inflation=inflation,
        signals=signals,
        runs=runs,
        table=compare(runs, inflation, cash),
        phases=phase_months(cycle.phase),
        by_phase=by_phase_returns(curves, signals["phase"], inflation),
        rotation=rotation_test(parts, signals["phase"], alloc, cfg, inflation=inflation),
        grid=sensitivity(
            parts,
            curves[GROWTH_PROXY[market]].returns,
            cash,
            inflation,
            alloc,
            with_rotation=rotation_grid,
        ),
        stance=current_stance(signals, alloc, market),
    )


def verdict(study: CycleStudy) -> dict[str, float]:
    """Count how much of the grid actually beat its own control, so the reader is not left to.

    A single headline number invites the reader to keep the cell they like. These four counts
    are the whole grid at once: how often the rule beat its control on return, on risk-adjusted
    return and on drawdown, and where the timing typically ranked against a rotated calendar.
    """
    g = study.grid
    out = {
        "cells": float(len(g)),
        "share_beat_return": float((g["delta"] > 0).mean()),
        "share_beat_sharpe": float((g["sharpe"] > g["control_sharpe"]).mean()),
        "share_beat_drawdown": float((g["max_drawdown"] > g["control_dd"]).mean()),
        "median_delta": float(g["delta"].median()),
    }
    if "rotation_pct" in g:
        out["median_rotation_pct"] = float(g["rotation_pct"].median())
    return out


# --------------------------------------------------------------------------- presentation
PCT_COLS = frozenset(
    {
        "cagr_real",
        "cagr_nominal",
        "volatility",
        "max_drawdown",
        "worst_year",
        "positive_months",
        "annual_turnover",
        "cost_drag",
        "share",
        "control_real",
        "delta",
        "control_dd",
        "rotation_pct",
    }
    | set(PHASE_LABEL.values())
)

INT_COLS = frozenset({"months", "avg_run_months", "confirm", "trend", "cost_bps"})


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    """Percentages as percentages, month counts as counts, everything else left alone."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        if col in PCT_COLS:
            out[col] = series.map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        elif col in INT_COLS:
            out[col] = series.map(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        else:
            out[col] = series.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    return out


def allocation_table(
    alloc: dict[str, dict[str, float]], curves: dict[str, ClassCurve]
) -> pd.DataFrame:
    """`allocation_frame` with the catalog's names instead of the internal keys, for printing."""
    df = allocation_frame(alloc)
    df.columns = pd.Index(
        [A.label_of(curves[k]) if k in curves else k for k in df.columns], name="Classe"
    )
    return df


def _table(df: pd.DataFrame, *, percent: bool = False, digits: int = 0) -> str:
    """Markdown table. `percent` formats every float column, for frames `fmt` cannot name."""
    if percent:
        body = df.map(lambda v: f"{v:.{digits}%}" if isinstance(v, float) else v)
    else:
        body = fmt(df)
    body = body.reset_index(drop=isinstance(df.index, pd.RangeIndex))
    header = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "|" + "---|" * len(body.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in body.to_numpy()]
    return "\n".join([header, rule, *rows])


def render_report(study: CycleStudy) -> str:
    """The whole thing as one markdown file, in the shape `reports/` already uses."""
    import datetime as dt

    s = study.stance
    rot = study.rotation
    v = verdict(study)
    e = annual_edge(study.cycle, study.runs[1], study.inflation)
    cycle = study.cycle
    ph = study.phases
    lines = [
        f"# Carteira do ciclo de mercado — {study.market.upper()}",
        "",
        f"**Gerado em**: {dt.datetime.now():%Y-%m-%d %H:%M}  ",
        f"**Período**: {cycle.returns.index[0]} a {cycle.returns.index[-1]} "
        f"({len(cycle.returns) / MONTHS:.1f} anos)  ",
        f"**Regra**: relógio de crescimento × inflação, {study.cfg.confirm_months} meses de "
        f"confirmação, rebalanceamento na virada de fase (calendário a cada "
        f"{study.cfg.drift_months} meses), custo de {study.cfg.cost_bps:.0f} bps por giro",
        "",
        "## Onde o relógio está agora",
        "",
        f"- **Fase**: {s['label_pt']}",
        f"- **Lida em**: {s['as_of']} — vale para {s['governs']}",
        f"- **Nesta fase desde**: {s['in_phase_since']}",
        "- **Alvo**:",
        "",
    ]
    weights = s["weights"]
    assert isinstance(weights, dict)
    lines += [f"  - {A.label_of(study.curves[k])}: {w:.0%}" for k, w in weights.items() if w > 0]
    lines += [
        "",
        "## Comparação (retorno real, mesmos meses)",
        "",
        _table(study.table),
        "",
        "## Retorno real ano a ano",
        "",
        _table(
            annual_table(study.runs, study.inflation, phase=study.cycle.phase),
            percent=True,
            digits=1,
        ),
        "",
        "",
        f"Ciclo contra o controle estático: **{e['years_ahead']:.0f} anos a favor e "
        f"{e['years_behind']:.0f} contra** em {e['years']:.0f}. A vantagem no período é de "
        f"{e['edge']:+.1%} ao ano; sem o melhor ano ({e['best_year']:.0f}, "
        f"{e['best_year_delta']:+.1%}), fica **{e['edge_without_best_year']:+.1%}**.",
        "",
        "## Alocação por fase",
        "",
        _table(allocation_table(CYCLE_ALLOCATIONS[study.market], study.curves), percent=True),
        "",
        "## Tempo em cada fase",
        "",
        _table(ph),
        "",
        f"Trocas de fase: {ph.attrs['switches']} ({ph.attrs['switches_per_year']:.2f} por ano).",
        "",
        "## Retorno real de cada classe dentro de cada fase (in-sample)",
        "",
        _table(study.by_phase),
        "",
        "## O sinal ou a época? (teste de rotação)",
        "",
        f"- CAGR real da sequência verdadeira: **{rot['actual']:.2%}**",
        f"- Mediana das {rot['n_rotations']:.0f} rotações: {rot['median_rotated']:.2%}",
        f"- p95 das rotações: {rot['p95_rotated']:.2%} (melhor: {rot['best_rotated']:.2%})",
        f"- Percentil da sequência verdadeira: **{rot['percentile']:.0%}**",
        "",
        "## Sensibilidade aos parâmetros da regra",
        "",
        "Cada linha é a regra inteira com outro ajuste, contra o **seu próprio** controle "
        "estático. Uma vantagem que só existe numa célula da grade não foi descoberta, foi "
        "escolhida.",
        "",
        _table(study.grid),
        "",
        "### O que a grade diz",
        "",
        f"- Bateu o controle no retorno real em **{v['share_beat_return']:.0%}** das "
        f"{v['cells']:.0f} células (mediana da diferença: {v['median_delta']:+.1%} ao ano)",
        f"- Bateu o controle no Sharpe em **{v['share_beat_sharpe']:.0%}** das células",
        f"- Sofreu menos no pior drawdown em **{v['share_beat_drawdown']:.0%}** das células",
    ]
    if "median_rotation_pct" in v:
        lines.append(
            f"- Percentil mediano contra as rotações da própria sequência de fases: "
            f"**{v['median_rotation_pct']:.0%}** (50% = o calendário verdadeiro não valeu nada)"
        )
    lines += [
        "",
        "## Observações",
        "",
        "- Comparação mensal e bruta de imposto e taxa de administração; as classes são "
        "índices, que ninguém compra diretamente (ver `notebooks/etf_routes.py` para o que "
        "sobra depois do veículo).",
        "- Drawdown é de fechamento mensal: mais raso que o intradiário, mas raso pela mesma "
        "regra para todas as linhas.",
        "- As alocações por fase foram escritas olhando o mesmo histórico que as pontua. O "
        "controle `estática` usa exatamente as mesmas classes para isolar o efeito do timing, "
        "e o teste de rotação diz quanto do resultado é a época.",
        "- Um caminho histórico só, e curto para uma pergunta de ciclo: 15 anos são poucos "
        "ciclos completos.",
        "",
    ]
    return "\n".join(lines)


def main(market: str = "b3", out_dir: Path = REPO / "reports") -> Path:
    import datetime as dt

    study = build_study(market)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cycle_portfolio_{market}_{dt.datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(render_report(study), encoding="utf-8")
    return path


if __name__ == "__main__":
    for mkt in sys.argv[1:] or ["b3"]:
        print(main(mkt))
