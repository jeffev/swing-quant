"""Helpers for the *Swing Trading vs Investing* study (notebooks/swing_vs_investing.ipynb).

Research code, not production: it reuses the package's engine, panel and metrics so the
swing side of the comparison is exactly what `swing-quant portfolio` would run, and adds
the passive baselines (index B&H, equal-weight B&H, risk-free, DCA) plus two effects the
engine deliberately leaves out - interest on idle cash and taxes on realised gains.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from swing_quant.backtest.engine import Backtester, BacktestResult, CostModel, RiskModel
from swing_quant.backtest.metrics import TRADING_DAYS, cagr, drawdown_series, max_drawdown
from swing_quant.backtest.panel import Panel
from swing_quant.backtest.portfolio import combine_panels
from swing_quant.config import Config, load_config
from swing_quant.data.store import MarketStore
from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
from swing_quant.risk.regime import Regime, RegimeConfig, build_regime
from swing_quant.strategies import make_strategy

CONFIG_PATH = REPO / "config.yaml"
DB_PATH = REPO / "data" / "market.duckdb"

_FP_NOISE = 1e-12  # below this a standard deviation is rounding error, not risk

MARKET_LABEL = {"b3": "Brazil (B3)", "us": "United States"}
CURRENCY = {"b3": "R$", "us": "US$"}
RF_LABEL = {"b3": "CDI", "us": "T-bills"}


# --------------------------------------------------------------------------- data
@dataclass
class MarketData:
    """Everything the study needs for one market, already aligned on the price calendar."""

    market: str
    prices: pd.DataFrame  # long format from MarketStore
    tickers: list[str]
    sectors: dict[str, str]
    bench_close: pd.Series  # benchmark total-return index
    rf_daily: pd.Series  # risk-free daily return
    strategies: list[str]

    @property
    def currency(self) -> str:
        return CURRENCY[self.market]

    @property
    def rf_label(self) -> str:
        return RF_LABEL[self.market]


def load_market(market: str, cfg: Config | None = None) -> MarketData:
    cfg = cfg or load_config(CONFIG_PATH)
    with MarketStore(DB_PATH) as store:
        members = store.universe_at(INDEX_BY_MARKET[market])
        tickers = [to_yf_symbol(t, market) for t in members["ticker"]]
        sectors = {
            to_yf_symbol(t, market): str(s)
            for t, s in zip(members["ticker"], members["sector"], strict=True)
            if pd.notna(s)
        }
        prices = store.get_prices(tickers)
        bench = store.get_prices([cfg.market_universe(market).benchmark])
        rf = store.con.execute(
            "SELECT date, daily_return FROM risk_free WHERE market = ? ORDER BY date", [market]
        ).df()
    bench_close = bench.set_index("date")["adj_close"].sort_index()
    bench_close.index = pd.DatetimeIndex(bench_close.index)
    rf_daily = pd.Series(rf["daily_return"].to_numpy(), index=pd.DatetimeIndex(rf["date"]))
    return MarketData(
        market=market,
        prices=prices,
        tickers=tickers,
        sectors=sectors,
        bench_close=bench_close,
        rf_daily=rf_daily.sort_index(),
        strategies=cfg.enabled_strategies(market),
    )


# --------------------------------------------------------------------------- swing side
def build_models(
    market: str, cfg: Config, *, cost_mult: float = 1.0
) -> tuple[CostModel, RiskModel]:
    """The exact cost and risk models the CLI's `portfolio` command builds for this market."""
    c = cfg.market_costs(market)
    costs = CostModel(c.commission_per_order, c.fees_pct, c.slippage_pct_liquid)
    if cost_mult != 1.0:
        costs = costs.scaled(cost_mult)
    rk = cfg.risk
    risk = RiskModel(
        initial_capital=cfg.capital.for_market(market),
        risk_per_trade=rk.risk_for_market(market),
        atr_multiple=rk.atr_multiple_default,
        max_position_pct=rk.max_position_pct,
        max_positions=rk.max_positions,
        max_volume_participation=rk.max_volume_participation,
        board_lot=rk.board_lot if market == "b3" else 1,
        min_dollar_volume=cfg.market_universe(market).min_avg_dollar_volume_20d,
        max_sector_pct=rk.max_sector_pct,
        max_strategy_pct=rk.max_strategy_pct,
        max_correlation=rk.max_correlation,
        monthly_dd_reduce=rk.monthly_dd_reduce,
        circuit_breaker_dd=rk.circuit_breaker_dd,
    )
    return costs, risk


def build_panel_for(md: MarketData, cfg: Config) -> Panel:
    from swing_quant.backtest.validation import default_panel_factory

    factory = default_panel_factory(md.prices)
    panels = {n: factory(make_strategy(n, cfg.strategies.get(n, {}))) for n in md.strategies}
    return combine_panels(panels, md.sectors)


def build_regime_for(md: MarketData, cfg: Config) -> Regime:
    return build_regime(
        md.bench_close,
        RegimeConfig(
            trend_sma=cfg.regime.benchmark_sma,
            vol_percentile=cfg.regime.high_vol_percentile,
            high_vol_size_factor=cfg.regime.high_vol_size_factor,
            use_trend=cfg.regime.trend_filter,
            use_vol=cfg.regime.vol_filter,
        ),
    )


def run_swing(
    panel: Panel,
    cfg: Config,
    market: str,
    *,
    regime: Regime | None = None,
    cost_mult: float = 1.0,
) -> BacktestResult:
    costs, risk = build_models(market, cfg, cost_mult=cost_mult)
    bt = Backtester(
        costs,
        risk,
        allow_entries=regime.allow_entries if regime else None,
        size_factor=regime.size_factor if regime else None,
    )
    return bt.run(panel)


# --------------------------------------------------------------------------- idle cash
def with_cash_yield(res: BacktestResult, rf_daily: pd.Series) -> pd.Series:
    """Equity curve where the un-deployed cash earns the risk-free rate.

    The engine keeps idle cash at 0% on purpose (see `data/riskfree.py`). A swing book that
    is flat most of the time is really *equity + a money-market sleeve*, so the honest curve
    adds `cash_{t-1}/equity_{t-1} * rf_t` to each day's portfolio return and compounds.
    """
    return cash_yield_detail(res, rf_daily)[0]


def cash_yield_detail(
    res: BacktestResult, rf_daily: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(equity with cash yield, interest earned per day in currency, cash weight per day).

    The interest leg is kept separate because it is taxed under a different regime from
    trading gains - fixed income, not equities - and the study needs both bills.
    """
    eq = res.equity.dropna()
    rf = rf_daily.reindex(eq.index).fillna(0.0)
    cash_w = (res.cash.reindex(eq.index) / eq).shift(1).clip(lower=0.0, upper=1.0).fillna(0.0)
    total = eq.pct_change().fillna(0.0) + cash_w * rf
    grown = res.initial_capital * (1.0 + total).cumprod()
    interest = (cash_w * rf) * grown.shift(1).fillna(res.initial_capital)
    return grown, interest, cash_w


# --------------------------------------------------------------------------- baselines
def buy_and_hold(close: pd.Series, capital: float, index: pd.DatetimeIndex) -> pd.Series:
    """Total-return buy-and-hold, normalised to `capital` and aligned to `index`."""
    s = close.reindex(index).ffill().dropna()
    return capital * s / s.iloc[0]


def equal_weight_buy_and_hold(
    prices: pd.DataFrame, capital: float, index: pd.DatetimeIndex
) -> tuple[pd.Series, int]:
    """Equal money in every name that already traded on day 1, then never touched again.

    No rebalancing: the drifting weights are what a passive investor actually ends up with,
    and it avoids inventing turnover costs the index baseline does not pay.
    """
    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    wide = wide.reindex(index).ffill()
    alive = wide.columns[wide.iloc[0].notna()]
    wide = wide[alive].ffill()
    shares = (capital / len(alive)) / wide.iloc[0]
    return (wide * shares).sum(axis=1), len(alive)


def risk_free_curve(rf_daily: pd.Series, capital: float, index: pd.DatetimeIndex) -> pd.Series:
    rf = rf_daily.reindex(index).fillna(0.0)
    return capital * (1.0 + rf).cumprod()


def blended_curve(
    close: pd.Series, rf_daily: pd.Series, weight: float, capital: float, index: pd.DatetimeIndex
) -> pd.Series:
    """`weight` in the index and the rest at the risk-free rate, rebalanced daily.

    This is the control that makes the swing comparison fair. A swing book sits ~72% in cash,
    so crediting that cash with interest and then comparing it to a 100%-invested index is not
    a like-for-like test. The like-for-like test is a passive investor who deliberately holds
    the *same average equity exposure* and parks the remainder in the same money market.
    """
    px = close.reindex(index).ffill()
    eq_ret = px.pct_change().fillna(0.0)
    rf = rf_daily.reindex(index).fillna(0.0)
    blend = weight * eq_ret + (1.0 - weight) * rf
    return capital * (1.0 + blend).cumprod()


def vol_matched_weight(target_vol: float, close: pd.Series, index: pd.DatetimeIndex) -> float:
    """Index weight whose blended volatility matches `target_vol` (cash has ~zero vol)."""
    px = close.reindex(index).ffill()
    idx_vol = float(px.pct_change().std(ddof=1) * math.sqrt(TRADING_DAYS))
    return target_vol / idx_vol if idx_vol else float("nan")


# --------------------------------------------------------------------------- metrics
def sharpe_excess(returns: pd.Series, rf_daily: pd.Series) -> float:
    """Sharpe against the *actual* daily risk-free series, not a constant.

    With CDI averaging ~10% a year this is not a detail: a zero-rate Sharpe flatters a
    Brazilian strategy by roughly half a point.
    """
    rf = rf_daily.reindex(returns.index).fillna(0.0)
    ex = (returns - rf).dropna()
    sd = ex.std(ddof=1) if len(ex) > 1 else 0.0
    if len(ex) < 2 or sd < _FP_NOISE:
        return float("nan")  # the risk-free curve against itself: 0/0, not a real ratio
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino_excess(returns: pd.Series, rf_daily: pd.Series) -> float:
    rf = rf_daily.reindex(returns.index).fillna(0.0)
    ex = (returns - rf).dropna()
    down = ex.clip(upper=0.0)
    dd = math.sqrt(float((down**2).mean()))
    if len(ex) < 2 or dd < _FP_NOISE:
        return float("nan")
    return float(ex.mean() / dd * math.sqrt(TRADING_DAYS))


def curve_stats(
    equity: pd.Series,
    rf_daily: pd.Series,
    *,
    exposure: pd.Series | None = None,
    n_trades: int | None = None,
) -> dict[str, float]:
    """Risk/return summary of any equity curve, passive or traded."""
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    depth, days = max_drawdown(eq)
    dd = drawdown_series(eq)
    c = cagr(eq)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    rf = rf_daily.reindex(rets.index).fillna(0.0)
    rf_cagr = float((1.0 + rf).prod() ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    exp_avg = float(exposure.reindex(eq.index).mean()) if exposure is not None else 1.0
    return {
        "cagr": c,
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "final": float(eq.iloc[-1]),
        "volatility": float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_excess(rets, rf_daily),
        "sortino": sortino_excess(rets, rf_daily),
        "max_drawdown": depth,
        "max_dd_days": float(days),
        "calmar": float(c / abs(depth)) if depth else float("nan"),
        "ulcer": float(math.sqrt((dd**2).mean())),
        "excess_over_rf": c - rf_cagr,
        "exposure_avg": exp_avg,
        "return_per_exposure": c / exp_avg if exp_avg else float("nan"),
        "n_trades": float(n_trades) if n_trades is not None else 1.0,
        "years": years,
    }


def rolling_return(equity: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Trailing `window`-day total return - the experience of an investor who started then."""
    return (equity / equity.shift(window) - 1.0).dropna()


def rolling_max_drawdown(
    equity: pd.Series, window: int = TRADING_DAYS, step: int = 21
) -> pd.Series:
    idx, vals = [], []
    for i in range(0, len(equity) - window, step):
        chunk = equity.iloc[i : i + window]
        idx.append(chunk.index[-1])
        vals.append(max_drawdown(chunk)[0])
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


# --------------------------------------------------------------------------- DCA
def dca_simulate(
    returns: pd.Series, monthly: float, *, initial: float = 0.0
) -> tuple[pd.Series, pd.Series]:
    """Compound `returns` while adding `monthly` on the first trading day of each month.

    Returns (portfolio value, cumulative contributions) so the pair can be read as a
    money-weighted result rather than a time-weighted one.
    """
    r = returns.fillna(0.0)
    first_of_month = ~r.index.to_period("M").duplicated()
    value = float(initial)
    contributed = float(initial)
    vals, contribs = [], []
    for ret, is_first in zip(r.to_numpy(), first_of_month, strict=True):
        value *= 1.0 + float(ret)
        if is_first:
            value += monthly
            contributed += monthly
        vals.append(value)
        contribs.append(contributed)
    return pd.Series(vals, index=r.index), pd.Series(contribs, index=r.index)


def money_weighted_return(value: pd.Series, contributions: pd.Series) -> float:
    """Annualised IRR of the contribution schedule ending at the final portfolio value."""
    flows = contributions.diff()
    flows.iloc[0] = contributions.iloc[0]
    mask = flows != 0
    dates = list(value.index[mask]) + [value.index[-1]]
    amounts = [-float(a) for a in flows[mask]] + [float(value.iloc[-1])]
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amt = np.array(amounts)

    def npv(rate: float) -> float:
        return float((amt / (1.0 + rate) ** years).sum())

    lo, hi = -0.95, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------- taxes
@dataclass(frozen=True)
class TaxRules:
    """Realised-gain taxation. `exempt_monthly_sales` is Brazil's R$20k stock exemption.

    `interest_rate` taxes the money-market leg, which in Brazil is a different regime from
    equities: the *regressiva* table bottoms out at 15% past two years, charged either on
    redemption (a CDB or Tesouro Selic held throughout) or twice a year by `come-cotas`
    if the cash sits in a fund.
    """

    realised_rate: float
    deferred_rate: float
    interest_rate: float
    exempt_monthly_sales: float = 0.0
    label: str = ""


TAX_BR = TaxRules(
    0.15, 0.15, 0.15, 20_000.0, "15% on monthly stock gains (R$20k exemption) + 15% on interest"
)
TAX_US = TaxRules(0.25, 0.15, 0.25, 0.0, "25% short-term / 15% long-term + 25% on interest")


def interest_tax_schedule(
    interest: pd.Series, rules: TaxRules, *, mode: str = "deferred"
) -> pd.Series:
    """Tax on the cash sleeve, by month.

    `deferred` - one bill at the end, as with a CDB or Tesouro Selic held to the finish.
    `come_cotas` - every May and November, the Brazilian fund default.
    `annual` - every December, closer to how US interest income is taxed.
    """
    if interest.empty:
        return pd.Series(dtype=float)
    by_month = interest.groupby(interest.index.to_period("M")).sum()
    if mode == "deferred":
        return pd.Series({by_month.index[-1]: float(by_month.sum()) * rules.interest_rate})
    if mode == "come_cotas":
        marks = {5, 11}
    elif mode == "annual":
        marks = {12}
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    out: dict[pd.Period, float] = {}
    pending = 0.0
    for month, amount in by_month.items():
        pending += float(amount)
        if month.month in marks or month == by_month.index[-1]:
            out[month] = max(pending, 0.0) * rules.interest_rate
            pending = 0.0
    return pd.Series(out)


def monthly_tax_on_trades(trades: pd.DataFrame, rules: TaxRules) -> pd.Series:
    """Tax due per exit-month, with losses carried forward and the sales exemption applied."""
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["exit_month"] = pd.to_datetime(t["exit_date"]).dt.to_period("M")
    t["proceeds"] = t["exit_price"] * t["qty"]
    grouped = t.groupby("exit_month").agg(pnl=("pnl", "sum"), proceeds=("proceeds", "sum"))
    carry = 0.0
    out: dict[pd.Period, float] = {}
    for month, row in grouped.iterrows():
        gain = float(row["pnl"])
        if rules.exempt_monthly_sales and float(row["proceeds"]) <= rules.exempt_monthly_sales:
            out[month] = 0.0  # exempt sale: the gain is untaxed and the loss is not deductible
            continue
        taxable = gain + carry
        if taxable > 0:
            out[month] = taxable * rules.realised_rate
            carry = 0.0
        else:
            out[month] = 0.0
            carry = taxable
    return pd.Series(out)


def after_tax_curve(equity: pd.Series, tax_by_month: pd.Series) -> pd.Series:
    """Apply each month's tax bill at month end and let the smaller base compound.

    Paying tax every month is not a one-off haircut: the money removed stops compounding,
    which is exactly the drag a buy-and-hold investor avoids by never realising.
    """
    eq = equity.dropna()
    monthly = eq.resample("ME").last()
    gross_ret = monthly.pct_change()
    gross_ret.iloc[0] = monthly.iloc[0] / eq.iloc[0] - 1.0
    value = float(eq.iloc[0])
    vals = []
    for date, ret in gross_ret.items():
        value *= 1.0 + float(ret)
        gross_value = float(monthly.loc[date])
        due = float(tax_by_month.get(pd.Period(date, "M"), 0.0))
        scale = value / gross_value if gross_value else 1.0
        value = max(value - due * scale, 0.0)  # the bill scales with the shrinking book
        vals.append(value)
    return pd.Series(vals, index=monthly.index)


def deferred_tax_final(equity: pd.Series, rate: float) -> float:
    """Terminal value if the whole position is liquidated once, at the end."""
    gain = float(equity.iloc[-1] - equity.iloc[0])
    return float(equity.iloc[-1] - max(gain, 0.0) * rate)


# --------------------------------------------------------------------------- leverage
def leverage_to_match(strategy: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    """How much leverage the low-exposure book needs to reach the target's return.

    Scaling a strategy's *excess* return over the risk-free rate scales its volatility and
    drawdown roughly in proportion, so this is the honest way to ask "same return - then
    whose risk is worse?" It ignores borrowing spread, margin calls and path risk, all of
    which make the levered version worse than the arithmetic suggests.
    """
    ex_s, ex_t = strategy["excess_over_rf"], target["excess_over_rf"]
    if ex_s <= 0 or ex_t <= 0:
        return {"leverage": float("nan"), "volatility": float("nan"), "max_drawdown": float("nan")}
    k = ex_t / ex_s
    return {
        "leverage": k,
        "volatility": strategy["volatility"] * k,
        "max_drawdown": strategy["max_drawdown"] * k,
    }


# --------------------------------------------------------------------------- assembled study
@dataclass
class Study:
    """Every curve and table the notebook and the article need, for one market."""

    md: MarketData
    result: BacktestResult
    capital: float
    index: pd.DatetimeIndex
    swing_raw: pd.Series  # engine output: idle cash earns nothing
    swing: pd.Series  # idle cash earns the risk-free rate
    interest: pd.Series  # currency earned by the cash sleeve, per day
    cash_weight: pd.Series
    bench: pd.Series
    equal_weight: pd.Series
    n_alive: int
    risk_free: pd.Series
    blend_exposure: pd.Series
    blend_vol: pd.Series
    exposure_avg: float
    weight_vol_matched: float

    @property
    def market(self) -> str:
        return self.md.market

    @property
    def years(self) -> float:
        return (self.index[-1] - self.index[0]).days / 365.25

    def curves(self) -> dict[str, pd.Series]:
        rf_name = self.md.rf_label
        return {
            "Swing (cash at 0%)": self.swing_raw,
            f"Swing (cash at {rf_name})": self.swing,
            f"{self.exposure_avg:.0%} index + {rf_name}": self.blend_exposure,
            f"{self.weight_vol_matched:.0%} index + {rf_name} (vol-matched)": self.blend_vol,
            "Index buy & hold": self.bench,
            f"Equal-weight B&H ({self.n_alive})": self.equal_weight,
            rf_name: self.risk_free,
        }

    def table(self) -> pd.DataFrame:
        traded = {"Swing (cash at 0%)", f"Swing (cash at {self.md.rf_label})"}
        rows = {}
        for name, curve in self.curves().items():
            is_swing = name in traded
            rows[name] = curve_stats(
                curve,
                self.md.rf_daily,
                exposure=self.result.exposure if is_swing else None,
                n_trades=len(self.result.trades) if is_swing else 1,
            )
        return stats_table(rows)

    def attribution(self) -> dict[str, float]:
        """Split the swing CAGR into what the trades made and what the cash made."""
        rf = self.md.rf_daily.reindex(self.index).fillna(0.0)
        cash_only = self.capital * (1.0 + self.cash_weight * rf).cumprod()
        total, trading, cash = cagr(self.swing), cagr(self.swing_raw), cagr(cash_only)
        return {
            "total": total,
            "trading": trading,
            "cash": cash,
            "cross": total - trading - cash,
            "cash_share": cash / total if total else float("nan"),
            "interest_total": float(self.interest.sum()),
            "trading_pnl": float(self.result.trades["pnl"].sum()),
        }

    def rate_quartiles(self) -> pd.DataFrame:
        """Swing's edge over the index and over the cash-matched blend, by rate regime.

        The blend column is the one that matters: it holds the same cash, so the interest
        cancels and what is left is the part of the edge that is not just the rate.
        """
        rf = self.md.rf_daily.reindex(self.index).fillna(0.0)
        rate_1y = (1.0 + rf).rolling(TRADING_DAYS).apply(np.prod, raw=True) - 1.0
        j = pd.concat(
            {
                "swing": rolling_return(self.swing),
                "index": rolling_return(self.bench),
                "blend": rolling_return(self.blend_exposure),
                "rate": rate_1y,
            },
            axis=1,
            sort=True,
        ).dropna()
        j["vs_index"] = j["swing"] - j["index"]
        j["vs_blend"] = j["swing"] - j["blend"]
        q = pd.qcut(j["rate"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).rename(
            "rate_quartile"
        )
        out = j.groupby(q, observed=True).agg(
            rate=("rate", "mean"),
            swing=("swing", "mean"),
            index=("index", "mean"),
            vs_index=("vs_index", "mean"),
            vs_blend=("vs_blend", "mean"),
            n=("swing", "size"),
        )
        out.attrs["corr_vs_index"] = float(j["rate"].corr(j["vs_index"]))
        out.attrs["corr_vs_blend"] = float(j["rate"].corr(j["vs_blend"]))
        return out

    def rolling_frame(self, window: int = TRADING_DAYS) -> pd.DataFrame:
        return pd.concat(
            {
                "swing": rolling_return(self.swing, window),
                "index": rolling_return(self.bench, window),
                "blend": rolling_return(self.blend_exposure, window),
                "risk_free": rolling_return(self.risk_free, window),
            },
            axis=1,
            sort=True,
        ).dropna()


def build_study(market: str, cfg: Config | None = None) -> Study:
    """Load, backtest and assemble every baseline for one market."""
    cfg = cfg or load_config(CONFIG_PATH)
    md = load_market(market, cfg)
    panel = build_panel_for(md, cfg)
    regime = build_regime_for(md, cfg)
    res = run_swing(panel, cfg, market, regime=regime)
    cap = cfg.capital.for_market(market)
    idx = pd.DatetimeIndex(res.equity.dropna().index)
    swing, interest, cash_w = cash_yield_detail(res, md.rf_daily)
    exposure_avg = float(res.exposure.mean())
    swing_vol = float(swing.pct_change().std(ddof=1) * math.sqrt(TRADING_DAYS))
    w_vol = vol_matched_weight(swing_vol, md.bench_close, idx)
    ew, n_alive = equal_weight_buy_and_hold(md.prices, cap, idx)
    return Study(
        md=md,
        result=res,
        capital=cap,
        index=idx,
        swing_raw=res.equity,
        swing=swing,
        interest=interest,
        cash_weight=cash_w,
        bench=buy_and_hold(md.bench_close, cap, idx),
        equal_weight=ew,
        n_alive=n_alive,
        risk_free=risk_free_curve(md.rf_daily, cap, idx),
        blend_exposure=blended_curve(md.bench_close, md.rf_daily, exposure_avg, cap, idx),
        blend_vol=blended_curve(md.bench_close, md.rf_daily, w_vol, cap, idx),
        exposure_avg=exposure_avg,
        weight_vol_matched=w_vol,
    )


# --------------------------------------------------------------------------- presentation
COLORS = {
    "swing": "#2563eb",
    "swing_raw": "#93c5fd",
    "index": "#dc2626",
    "equal_weight": "#f59e0b",
    "blend": "#7c3aed",
    "risk_free": "#059669",
    "grid": "#d4d4d8",
}


def use_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


PCT_COLS = (
    "cagr",
    "total_return",
    "volatility",
    "max_drawdown",
    "ulcer",
    "excess_over_rf",
    "exposure_avg",
    "return_per_exposure",
)
RATIO_COLS = ("sharpe", "sortino", "calmar")


def stats_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in PCT_COLS:
            out[col] = out[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        elif col in RATIO_COLS:
            out[col] = out[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        elif col in ("final", "n_trades", "max_dd_days"):
            out[col] = out[col].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
        elif col == "years":
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    return out
