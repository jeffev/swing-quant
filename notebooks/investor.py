"""The four questions an investor asks that a table of 16-year CAGRs cannot answer.

`asset_classes` ranks what happened. This module asks what a person deciding where to put money
would need on top of that:

1. **Portfolios, not classes.** Nobody owns one asset. `portfolio()` blends monthly returns with
   a real rebalancing rule, so the diversification the correlation matrix implies can be priced.
2. **Distributions, not point estimates.** Every CAGR here is one draw from one path.
   `window_stats()` asks the honest version instead - if you had started in any month, what would
   five years have looked like? - and `time_to_recover()` prices the wait.
3. **Net of tax.** The headline table is gross, and Brazilian tax is not neutral across classes:
   property funds distribute tax-free, savings accounts are exempt, bonds pay the regressive
   table. It is also charged on *nominal* gains, so inflation itself is taxed.
4. **Regime dependence.** The whole Brazilian result rests on a period of high real rates.
   `by_rate_regime()` checks whether it survives the periods when they were not.
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
from asset_classes import MONTHS, ClassCurve, deflate  # noqa: E402

from swing_quant.data.assets import AssetProxy  # noqa: E402


# --------------------------------------------------------------------------- 1. portfolios
@dataclass(frozen=True)
class PortfolioSpec:
    """A portfolio as an investor would describe it: names, weights and a rebalancing habit."""

    key: str
    label: str
    weights: dict[str, float]  # asset-class key -> target weight
    note: str = ""
    rebalance_months: int = 12


def blend_returns(
    parts: dict[str, pd.Series], weights: dict[str, float], rebalance_months: int = 12
) -> pd.Series:
    """Monthly returns of a portfolio that drifts between rebalances and resets on schedule.

    Rebalancing is the whole point of holding uncorrelated things, and it is also the part that
    backtests usually cheat on: rebalancing every month is free here but not in life, so the
    default is annual - the habit an actual investor can keep.
    """
    keys = list(weights)
    frame = pd.DataFrame({k: parts[k] for k in keys}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    target = np.array([weights[k] for k in keys], dtype=float)
    target = target / target.sum()
    w = target.copy()
    out: list[float] = []
    for i, (_, row) in enumerate(frame.iterrows()):
        r = row.to_numpy(dtype=float)
        port = float(w @ r)
        out.append(port)
        grown = w * (1.0 + r)
        total = grown.sum()
        w = grown / total if total > 0 else target.copy()
        if (i + 1) % rebalance_months == 0:
            w = target.copy()
    return pd.Series(out, index=frame.index)


def portfolio(curves: dict[str, ClassCurve], spec: PortfolioSpec) -> ClassCurve | None:
    """Build one portfolio as if it were another asset class, so every table can hold both."""
    missing = [k for k in spec.weights if k not in curves]
    if missing:
        return None
    parts = {k: curves[k].returns for k in spec.weights}
    rets = blend_returns(parts, spec.weights, spec.rebalance_months)
    if len(rets) <= MONTHS:
        return None
    proxy = AssetProxy(
        key=spec.key,
        label=spec.label,
        asset_class="portfolio",
        market=next(iter(curves.values())).proxy.market,
        kind="ticker",
        note=spec.note,
    )
    return ClassCurve(proxy, rets)


# The menu is deliberately made of portfolios people actually hold or are actually sold, not of
# optimiser output: a mean-variance frontier fitted on the same 16 years it is scored against
# would be the most flattering and least honest chart in the study.
BR_PORTFOLIOS: tuple[PortfolioSpec, ...] = (
    PortfolioSpec("p_cash", "100% cash (CDI)", {"cdi": 1.0}, "the default every Brazilian has"),
    PortfolioSpec("p_stocks", "100% Brazilian stocks", {"acoes_br": 1.0}, "the other extreme"),
    PortfolioSpec(
        "p_6040",
        "60/40 Brazil",
        {"acoes_br": 0.6, "cdi": 0.4},
        "the textbook portfolio, translated to Brazil",
    ),
    PortfolioSpec(
        "p_4040_20",
        "40/40 + 20% dollarised",
        {"acoes_br": 0.4, "cdi": 0.4, "sp500_brl": 0.2},
        "the same book with a fifth of it earning in dollars",
    ),
    PortfolioSpec(
        "p_diversified",
        "Diversified Brazil",
        {"acoes_br": 0.3, "cdi": 0.3, "fiis": 0.2, "tesouro_ipca": 0.2},
        "stocks, cash, property funds and inflation-linked bonds",
    ),
    PortfolioSpec(
        "p_permanent",
        "Permanent portfolio",
        {"acoes_br": 0.25, "cdi": 0.25, "tesouro_ipca": 0.25, "ouro_brl": 0.25},
        "equal parts stocks, cash, long bonds and gold",
    ),
)

US_PORTFOLIOS: tuple[PortfolioSpec, ...] = (
    PortfolioSpec("p_cash", "100% cash (T-bills)", {"tbills": 1.0}),
    PortfolioSpec("p_stocks", "100% US stocks", {"acoes_us": 1.0}),
    PortfolioSpec("p_6040", "60/40", {"acoes_us": 0.6, "bonds_us": 0.4}, "the textbook portfolio"),
    PortfolioSpec(
        "p_global",
        "Global 60/40",
        {"acoes_us": 0.36, "intl_dev": 0.18, "emergentes": 0.06, "bonds_us": 0.4},
        "the same, with the equity sleeve spread worldwide",
    ),
    PortfolioSpec(
        "p_permanent",
        "Permanent portfolio",
        {"acoes_us": 0.25, "tbills": 0.25, "treasuries_longas": 0.25, "ouro_usd": 0.25},
    ),
)

PORTFOLIOS = {"b3": BR_PORTFOLIOS, "us": US_PORTFOLIOS}


def build_portfolios(curves: dict[str, ClassCurve], market: str) -> dict[str, ClassCurve]:
    out: dict[str, ClassCurve] = {}
    for spec in PORTFOLIOS[market]:
        built = portfolio(curves, spec)
        if built is not None:
            out[spec.key] = built
    return out


def rebalance_effect(
    curves: dict[str, ClassCurve], spec: PortfolioSpec, inflation: pd.Series
) -> dict[str, float]:
    """Real CAGR of the same weights, rebalanced annually versus left alone.

    Left alone, a 60/40 that starts in 2010 is not a 60/40 by 2026 - it is whatever the winner
    grew into. The gap between the two lines is the part of the portfolio that is a decision
    rather than a drift.
    """
    parts = {k: curves[k].returns for k in spec.weights if k in curves}
    if len(parts) != len(spec.weights):
        return {}
    annual = blend_returns(parts, spec.weights, spec.rebalance_months)
    never = blend_returns(parts, spec.weights, rebalance_months=10**6)
    return {
        "rebalanced": _cagr(deflate(annual, inflation)),
        "drifting": _cagr(deflate(never, inflation)),
    }


def _cagr(monthly: pd.Series) -> float:
    years = len(monthly) / MONTHS
    return float((1.0 + monthly).prod() ** (1.0 / years) - 1.0) if years else float("nan")


# --------------------------------------------------------------------------- 2. distributions
def rolling_windows(returns: pd.Series, years: int) -> pd.Series:
    """Annualised return of every overlapping `years`-long window, dated at its end."""
    n = years * MONTHS
    if len(returns) <= n:
        return pd.Series(dtype=float)
    growth = (1.0 + returns).cumprod()
    ratio = growth / growth.shift(n)
    return ratio.dropna() ** (1.0 / years) - 1.0


def time_to_recover(returns: pd.Series) -> dict[str, float]:
    """Longest stretch spent below a previous peak, and where the investor still is today.

    Depth is what a risk report shows; duration is what a person actually experiences. A 40%
    fall that mends in a year and a 20% fall that takes nine are not the same investment.
    """
    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    underwater = curve < peak - 1e-12
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {
        "worst_months_underwater": float(longest),
        "months_underwater_now": float(current),
        "share_underwater": float(underwater.mean()),
    }


def window_stats(
    curve: ClassCurve, inflation: pd.Series, cash: pd.Series, years: int
) -> dict[str, float]:
    """What a `years`-long holding period looked like, over every start month in the sample."""
    real = deflate(curve.returns, inflation)
    windows = rolling_windows(real, years)
    if windows.empty:
        return {}
    cash_windows = rolling_windows(deflate(cash, inflation), years).reindex(windows.index)
    both = pd.concat([windows.rename("a"), cash_windows.rename("c")], axis=1).dropna()
    return {
        "median": float(windows.median()),
        "p5": float(windows.quantile(0.05)),
        "p95": float(windows.quantile(0.95)),
        "worst": float(windows.min()),
        "best": float(windows.max()),
        "share_negative": float((windows < 0).mean()),
        "share_below_cash": float((both["a"] < both["c"]).mean()) if len(both) else float("nan"),
        "n_windows": float(len(windows)),
        **time_to_recover(real),
    }


def window_table(
    curves: dict[str, ClassCurve], inflation: pd.Series, cash_key: str, years: int
) -> pd.DataFrame:
    cash = curves[cash_key].returns
    rows = {}
    for c in curves.values():
        stats = window_stats(c, inflation, cash, years)
        if stats:
            rows[A.label_of(c)] = stats
    out = pd.DataFrame(rows).T.sort_values("median", ascending=False)
    out.index.name = "Asset class"
    return out


# --------------------------------------------------------------------------- 3. tax
@dataclass(frozen=True)
class TaxProfile:
    """How one asset class is actually taxed, for a resident individual.

    `income_yield` is the part of the annual return that arrives as cash - rent, coupons,
    dividends - and is taxed as it arrives at `income_rate`. What is left compounds untouched
    and is taxed once, at `gain_rate`, when the position is finally sold. `entry_cost` and
    `exit_cost` are the frictions that never appear in an index.
    """

    income_yield: float = 0.0
    income_rate: float = 0.0
    gain_rate: float = 0.0
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    note: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)


# Brazil. Rates are the ones a long-term individual investor actually faces: the regressive
# table bottoms out at 15% past two years, property funds distribute tax-free to individuals,
# and the savings account is exempt end to end. The yields are period averages, stated because
# they are assumptions and not measurements.
TAX_BR: dict[str, TaxProfile] = {
    "cdi": TaxProfile(gain_rate=0.15, note="regressive table, 15% past two years"),
    "tesouro_selic": TaxProfile(gain_rate=0.15, note="15% on redemption"),
    "tesouro_prefixado": TaxProfile(gain_rate=0.15, note="15% on redemption"),
    "tesouro_ipca": TaxProfile(gain_rate=0.15, note="15% on redemption"),
    "poupanca": TaxProfile(note="exempt"),
    "fiis": TaxProfile(
        income_yield=0.08,
        income_rate=0.0,
        gain_rate=0.20,
        note="distributions exempt for individuals; 20% on the price gain when sold",
    ),
    "acoes_br": TaxProfile(
        gain_rate=0.15,
        note="15% on the gain; the R$20k monthly exemption does not cover a single large sale",
    ),
    "small_caps_br": TaxProfile(gain_rate=0.15),
    "sp500_brl": TaxProfile(gain_rate=0.15, note="via a local ETF or BDR"),
    "ouro_brl": TaxProfile(gain_rate=0.15),
    "dolar": TaxProfile(gain_rate=0.15),
    "bitcoin_brl": TaxProfile(gain_rate=0.15, note="exempt below R$35k of monthly sales"),
    "imoveis": TaxProfile(
        gain_rate=0.15,
        entry_cost=0.05,
        exit_cost=0.06,
        note="ITBI and deed on the way in, agent's fee on the way out",
    ),
    "imoveis_income": TaxProfile(
        income_yield=0.04,
        income_rate=0.275,
        gain_rate=0.15,
        entry_cost=0.05,
        exit_cost=0.06,
        note="rent taxed on the progressive table; ITBI, deed and agent's fee",
    ),
}

# United States. Long-term capital gains at 15%, ordinary income at 25% for interest and REIT
# distributions, and the 28% collectibles rate that catches physical-gold ETFs.
TAX_US: dict[str, TaxProfile] = {
    "tbills": TaxProfile(income_yield=0.013, income_rate=0.25, note="interest taxed yearly"),
    "bonds_us": TaxProfile(income_yield=0.028, income_rate=0.25),
    "treasuries_longas": TaxProfile(income_yield=0.028, income_rate=0.25),
    "tips": TaxProfile(income_yield=0.025, income_rate=0.25, note="phantom income on the accrual"),
    "acoes_us": TaxProfile(income_yield=0.018, income_rate=0.15, gain_rate=0.15),
    "nasdaq": TaxProfile(income_yield=0.008, income_rate=0.15, gain_rate=0.15),
    "small_caps_us": TaxProfile(income_yield=0.014, income_rate=0.15, gain_rate=0.15),
    "intl_dev": TaxProfile(income_yield=0.030, income_rate=0.15, gain_rate=0.15),
    "emergentes": TaxProfile(income_yield=0.025, income_rate=0.15, gain_rate=0.15),
    "reits_us": TaxProfile(
        income_yield=0.038,
        income_rate=0.25,
        gain_rate=0.15,
        note="distributions are ordinary income, not qualified dividends",
    ),
    "ouro_usd": TaxProfile(gain_rate=0.28, note="collectibles rate, not the 15% equity rate"),
    "commodities": TaxProfile(gain_rate=0.28),
    "bitcoin_usd": TaxProfile(gain_rate=0.15),
}

TAX_BY_MARKET = {"b3": TAX_BR, "us": TAX_US}


def _after_tax_growth(returns: pd.Series, tax: TaxProfile) -> tuple[float, float]:
    """Walk the position month by month, tracking value and cost basis. Returns (final, basis).

    The split matters. The income leg - rent, coupons, dividends - is taxed when it arrives and
    what is left is reinvested, which *raises the cost basis*: money already taxed as income is
    not taxed again as a capital gain when the position is finally sold. Ignoring that would
    double-tax an exempt property fund into looking like a taxed one.
    """
    monthly_yield = (1.0 + tax.income_yield) ** (1 / MONTHS) - 1.0
    value = 1.0 - tax.entry_cost
    basis = 1.0
    for r in returns.to_numpy(dtype=float):
        value *= 1.0 + (r - monthly_yield)  # the price leg
        income = value * monthly_yield
        kept = income * (1.0 - tax.income_rate)
        value += kept
        basis += kept
    return value * (1.0 - tax.exit_cost), basis


def after_tax_cagr(curve: ClassCurve, tax: TaxProfile, inflation: pd.Series) -> dict[str, float]:
    """Real CAGR before and after tax, plus the part of the bill that is charged on inflation.

    Brazilian capital-gains tax is levied on the *nominal* gain. Over sixteen years of 5.8%
    inflation that is not a rounding error: a position that merely kept its purchasing power
    still owes tax, and `inflation_tax` prices exactly that - the difference between the rule as
    it is and the same rate applied to the real gain.
    """
    # Gross and net have to be measured over the same months or the comparison is meaningless.
    rets = curve.returns.reindex(inflation.index).dropna()
    infl = inflation.reindex(rets.index)
    years = len(rets) / MONTHS
    if years <= 0:
        return {}
    price_level = float((1.0 + infl).prod())

    final, basis = _after_tax_growth(rets, tax)
    gain = max(final - basis, 0.0)
    net_nominal = final - gain * tax.gain_rate

    # counterfactual: the same rate applied to the gain net of inflation, as an indexed system
    real_gain = max(final / price_level - basis, 0.0)
    indexed_nominal = final - real_gain * tax.gain_rate

    return {
        "gross_real": _cagr(deflate(rets, infl)),
        "net_real": (net_nominal / price_level) ** (1.0 / years) - 1.0,
        "tax_drag": _cagr(deflate(rets, infl))
        - ((net_nominal / price_level) ** (1.0 / years) - 1.0),
        "inflation_tax": ((indexed_nominal / price_level) ** (1.0 / years) - 1.0)
        - ((net_nominal / price_level) ** (1.0 / years) - 1.0),
    }


def tax_table(curves: dict[str, ClassCurve], inflation: pd.Series, market: str) -> pd.DataFrame:
    profiles = TAX_BY_MARKET[market]
    rows = {}
    for key, c in curves.items():
        if key not in profiles:
            continue
        rows[A.label_of(c)] = {
            **after_tax_cagr(c, profiles[key], inflation),
            "note": profiles[key].note,
        }
    out = pd.DataFrame(rows).T
    for col in ("gross_real", "net_real", "tax_drag", "inflation_tax"):
        out[col] = out[col].astype(float)
    out = out.sort_values("net_real", ascending=False)
    out.index.name = "Asset class"
    return out


# --------------------------------------------------------------------------- 4. rate regime
def real_rate(cash: pd.Series, inflation: pd.Series, window: int = MONTHS) -> pd.Series:
    """Trailing 12-month real return of the money market - the rate that sets the hurdle."""
    nom = (1.0 + cash).rolling(window).apply(np.prod, raw=True) - 1.0
    inf = (1.0 + inflation.reindex(cash.index)).rolling(window).apply(np.prod, raw=True) - 1.0
    return ((1.0 + nom) / (1.0 + inf) - 1.0).dropna()


def by_rate_regime(
    curves: dict[str, ClassCurve], inflation: pd.Series, cash_key: str, n_buckets: int = 3
) -> pd.DataFrame:
    """Each class's real return, grouped by how well the money market was paying at the time.

    If the answer to "what should I own" is really "whatever the Selic is doing", it shows up
    here: the classes that only win in the high-rate bucket are not investments, they are a bet
    on the rate staying put.
    """
    rr = real_rate(curves[cash_key].returns, inflation)
    labels = ["low real rate", "middle", "high real rate"][:n_buckets]
    bucket = pd.qcut(rr, n_buckets, labels=labels)
    rows = {}
    for c in curves.values():
        real = deflate(c.returns, inflation).reindex(rr.index).dropna()
        if len(real) < MONTHS:
            continue
        grouped = real.groupby(bucket.reindex(real.index), observed=True)
        rows[A.label_of(c)] = {
            str(name): float((1.0 + g).prod() ** (MONTHS / len(g)) - 1.0)
            for name, g in grouped
            if len(g) >= 6
        }
    out = pd.DataFrame(rows).T
    out.attrs["bounds"] = {
        str(name): (float(g.min()), float(g.max())) for name, g in rr.groupby(bucket, observed=True)
    }
    out.attrs["months"] = {str(name): int(len(g)) for name, g in rr.groupby(bucket, observed=True)}
    out.index.name = "Asset class"
    return out


# --------------------------------------------------------------------------- presentation
def money(value: float, currency: str = "R$") -> str:
    return f"{currency} {value:,.0f}".replace(",", " ")


def purchasing_power(curve: ClassCurve, inflation: pd.Series, capital: float) -> float:
    """What `capital` invested at the start is worth at the end, in start-date money."""
    return float(capital * (1.0 + deflate(curve.returns, inflation)).prod())


PCT_COLS = (
    "median",
    "p5",
    "p95",
    "worst",
    "best",
    "share_negative",
    "share_below_cash",
    "gross_real",
    "net_real",
    "tax_drag",
    "inflation_tax",
    "rebalanced",
    "drifting",
    "low real rate",
    "middle",
    "high real rate",
    "share_underwater",
)


def fmt(df: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Percentages as percentages, month counts as years, everything else left alone."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        if col in PCT_COLS:
            out[col] = series.map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        elif str(col).endswith("months_underwater") or str(col).startswith("worst_months"):
            out[col] = series.map(lambda v: f"{v / 12:.1f}y" if pd.notna(v) else "—")
        elif col == "n_windows":
            out[col] = series.map(lambda v: f"{v:.0f}")
        else:
            out[col] = series
    if labels:
        out = out.rename(columns=labels)
    return out


def sharpe_like(returns: pd.Series, cash: pd.Series) -> float:
    excess = (returns - cash.reindex(returns.index)).dropna()
    sd = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    return float(excess.mean() / sd * math.sqrt(MONTHS)) if sd > 1e-12 else float("nan")
