"""Buying US equities from Brazil: what actually lands in reais, after the dollar, fees and tax.

Sections 13-17 rank asset *classes* using indices. Nobody can buy an index. This module prices
the two vehicles a Brazilian household actually uses to own the same US equity risk, and asks
what is left in reais once every leg is charged:

1. **The B3 route** (`IVVB11`, `WRLD11`, ...) - a Brazilian index fund, quoted in reais, that
   holds the US ETF abroad. It distributes nothing, so the 15% capital-gains tax is deferred to
   the day of the sale and compounds untouched until then.
2. **The US route** (`VTI`, `VOO`, ...) - the American share bought through a broker abroad. It
   pays dividends, the IRS withholds 30% of them, converting reais costs IOF plus a spread on
   the way out and on the way back, and since Lei 14.754/2023 the gain is taxed at a flat 15% in
   the annual return, with the currency move inside the taxable gain.

Three things shape every number here:

- **The fund fee is already inside the price.** An ETF's quote is net of its own expense ratio,
  so charging the published fee on top would double-count it. What this module measures instead
  is the *observed* gap between the Brazilian fund and its own underlying ETF converted at the
  spot rate - the fee plus tracking error plus cash drag plus the FX the fund actually got.
- **Return in reais is a product, not a sum.** "The dollar rose 6% and the ETF rose 10%" is not
  16%: it is 16.6%. The decomposition here keeps the cross term visible instead of hiding it.
- **The dollar leg is not free money.** Under interest parity the expected currency move is
  roughly the CDI-minus-T-bill gap, which is exactly what the investor gives up by not leaving
  the money in Brazilian cash. `forward_table` prices the two sides against each other rather
  than adding the dollar to the equity return and calling it an edge.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import asset_classes as A  # noqa: E402
from asset_classes import MONTHS, deflate  # noqa: E402

from swing_quant.data.etfs import (  # noqa: E402
    US_DIVIDEND_WITHHOLDING,
    USDBRL,
    EtfVehicle,
    etfs_for,
    peers,
    repair_unadjusted_splits,
)
from swing_quant.data.store import MarketStore  # noqa: E402

DB_PATH = REPO / "data" / "market.duckdb"

# The rest of the study is a snapshot ending in August 2026; the ETF series are cut at the same
# month so a reader can put this section's numbers next to section 13's without a footnote.
STUDY_END = "2026-08-31"

# A fund's first weeks are quoted, not traded: SPXB11 opened at R$10 on volumes of a few hundred
# reais and spent six months converging to its own net asset value. Requiring a median day of
# R$100 thousand before the series starts drops that noise without hand-picking dates.
MIN_DAILY_VALUE = 100_000.0


# --------------------------------------------------------------------------- the two routes
@dataclass(frozen=True)
class RouteRules:
    """What one route charges a resident individual, entry to exit.

    Costs are one-off fractions of the amount invested; `gain_tax` is charged once, on the
    nominal gain in reais, when the position is sold. `dividend_withholding` is the slice of
    every distribution that never arrives - on the B3 route it is withheld inside the fund and
    is already in the quote, on the US route it is withheld from the investor's own cash.
    """

    key: str
    label: str
    entry_cost: float
    exit_cost: float
    dividend_withholding: float
    gain_tax: float
    note: str
    sources: tuple[str, ...] = field(default_factory=tuple)


# Brokerage on Brazilian ETFs is zero at every large retail broker; what is left is the exchange
# fee (emolumentos plus settlement, about 0.0325% a side on a small order). The 15% rate has no
# R$20k monthly exemption: that exemption is for individual shares, not for index funds.
B3_ROUTE = RouteRules(
    key="b3",
    label="B3-listed ETF, in reais",
    entry_cost=0.00035,
    exit_cost=0.00035,
    dividend_withholding=US_DIVIDEND_WITHHOLDING,
    gain_tax=0.15,
    note=(
        "zero brokerage plus the exchange fee on both sides; 15% on the gain when sold, with "
        "no R$20k monthly exemption and no periodic taxation; the fund accumulates the dividend"
    ),
    sources=(
        "Lei 11.033/2004 art. 3 - the R$20k exemption covers shares, not index funds",
        "B3 cash-market fee schedule",
    ),
)

# Converting reais into dollars for an investment abroad costs IOF at 1.1% since the June 2025
# decree, plus the broker's exchange spread. Bringing the money home pays IOF again at 0.38%
# plus another spread. Both are one-off, both are stated so a reader with a cheaper broker can
# rerun the numbers.
FX_SPREAD = 0.005
IOF_OUT = 0.011
IOF_IN = 0.0038

US_ROUTE = RouteRules(
    key="us",
    label="US-listed ETF, broker abroad",
    entry_cost=IOF_OUT + FX_SPREAD,
    exit_cost=IOF_IN + FX_SPREAD,
    dividend_withholding=US_DIVIDEND_WITHHOLDING,
    gain_tax=0.15,
    note=(
        "1.1% IOF on the investment remittance plus a 0.5% spread on the way out, 0.38% plus "
        "spread on the way back; 30% withheld by the IRS on dividends, creditable against the "
        "15% owed in Brazil; 15% on the gain in reais, currency move included, no exemption"
    ),
    sources=(
        "Lei 14.754/2023 - flat 15% from 2024, and the end of the R$35k monthly exemption",
        "Decreto 6.306/2007 as amended in 2025 - 1.1% IOF on an investment remittance",
        "IRS - 30% withholding on dividends paid to a non-resident with no treaty",
    ),
)

ROUTES: dict[str, RouteRules] = {"b3": B3_ROUTE, "us": US_ROUTE}


# --------------------------------------------------------------------------- loading
def _monthly_last(daily: pd.Series) -> pd.Series:
    s = daily.dropna().sort_index()
    return s.groupby(pd.PeriodIndex(s.index, freq="M")).last()


@dataclass
class VehicleData:
    """One ETF as the monthly legs a Brazilian investor actually experiences.

    `fx_return` is always the real currency move, including for a fund quoted in reais. A B3
    fund's return already contains it, so the walk must not apply it twice - but the *reader*
    still wants it split out, because "how much of this was the dollar?" is the question the
    section exists to answer. `quoted_brl` is what keeps those two uses apart.
    """

    vehicle: EtfVehicle
    price_return: pd.Series  # price-only return, in the ETF's own currency
    dividend_yield: pd.Series  # cash distributions over the previous month's price
    fx_return: pd.Series  # BRL per USD, always the real one
    splits: list = field(default_factory=list)  # unadjusted splits repaired on the way in

    @property
    def quoted_brl(self) -> bool:
        return self.vehicle.venue == "b3"

    @property
    def key(self) -> str:
        return self.vehicle.ticker

    @property
    def months(self) -> pd.PeriodIndex:
        return self.price_return.index

    def gross_return_brl(self, withholding: float = 0.0) -> pd.Series:
        """Total return in reais, with `withholding` of every distribution left behind.

        On the B3 route the distribution leg is empty: the fund keeps the cash, and whatever the
        IRS took from it is already inside the quote.
        """
        local = (1.0 + self.price_return) * (1.0 + self.dividend_yield * (1.0 - withholding)) - 1.0
        if self.quoted_brl:
            return local.dropna()
        return ((1.0 + local) * (1.0 + self.fx_return) - 1.0).dropna()

    def dollar_leg(self, withholding: float = 0.0) -> pd.Series:
        """The ETF's own return in dollars - measured for a US fund, implied for a B3 one."""
        brl = self.gross_return_brl(withholding)
        fx = self.fx_return.reindex(brl.index)
        return ((1.0 + brl) / (1.0 + fx) - 1.0).dropna()


def load_vehicles(
    venue: str | None = None,
    db_path: Path = DB_PATH,
    end: str = STUDY_END,
    min_daily_value: float = MIN_DAILY_VALUE,
) -> dict[str, VehicleData]:
    """Rebuild every ETF in the catalog as price, dividend and currency legs.

    Two pieces of hygiene happen here, both of them visible afterwards. Unadjusted splits are
    repaired - Yahoo does not receive B3 corporate events, and SPXI11's 8-for-1 in January 2026
    would otherwise read as an 88% crash. And the series starts at the first month the fund
    actually traded, because a quote nobody transacted at is not a price anyone could have got.
    """
    with MarketStore(db_path) as store:
        fx_daily = store.get_prices([USDBRL], end=end).set_index("date")["adj_close"]
        fx = _monthly_last(fx_daily).pct_change()
        out: dict[str, VehicleData] = {}
        for vehicle in etfs_for(venue):
            prices = store.get_prices([vehicle.ticker], end=end)
            if prices.empty:
                continue
            daily = prices.set_index("date")
            repaired, splits = repair_unadjusted_splits(daily["close"])
            traded = _first_liquid_month(repaired, daily["volume"], min_daily_value)
            close = _monthly_last(repaired).loc[traded:]
            px = close.pct_change().dropna()
            if px.empty:
                continue
            div = _dividend_yield(store, vehicle.ticker, close, end)
            out[vehicle.ticker] = VehicleData(
                vehicle=vehicle,
                price_return=px,
                dividend_yield=div.reindex(px.index).fillna(0.0),
                fx_return=fx.reindex(px.index).fillna(0.0),
                splits=[(str(pd.Timestamp(d).date()), f) for d, f in splits],
            )
    return out


def _first_liquid_month(close: pd.Series, volume: pd.Series, min_daily_value: float) -> pd.Period:
    """First month whose median day traded more than `min_daily_value` in local currency."""
    value = (close * volume.reindex(close.index).fillna(0.0)).dropna()
    by_month = value.groupby(pd.PeriodIndex(value.index, freq="M")).median()
    liquid = by_month[by_month >= min_daily_value]
    return liquid.index[0] if len(liquid) else by_month.index[0]


def _dividend_yield(store: MarketStore, ticker: str, close: pd.Series, end: str) -> pd.Series:
    """Cash paid in each month over the price that started it - the leg the IRS taxes.

    Prices and dividends come out of `fetch_etf_history` on the same split basis, so this is a
    measured yield rather than an assumed one. That matters most for a fund like SCHD, where
    the distribution is a third of the total return.
    """
    events = store.con.execute(
        "SELECT date, value FROM corporate_events "
        "WHERE ticker = ? AND type = 'dividend' AND date <= ? ORDER BY date",
        [ticker, pd.Timestamp(end).date()],
    ).df()
    if events.empty:
        return pd.Series(0.0, index=close.index[1:])
    paid = pd.Series(events["value"].to_numpy(), index=pd.DatetimeIndex(events["date"]))
    monthly = paid.groupby(pd.PeriodIndex(paid.index, freq="M")).sum()
    base = close.shift(1)
    return (monthly.reindex(base.index).fillna(0.0) / base).dropna()


def load_reference(db_path: Path = DB_PATH, end: str = STUDY_END) -> dict[str, pd.Series]:
    """The rulers this section is scored against: inflation, cash and the currency itself."""
    with MarketStore(db_path) as store:
        cdi = store.risk_free("b3")
        tbill = store.risk_free("us")
        fx_daily = store.get_prices([USDBRL], end=end).set_index("date")["adj_close"]
    cut = pd.Timestamp(end)
    return {
        "ipca": A.load_inflation("b3", db_path),
        "cdi": _compound_monthly(cdi[cdi.index <= cut]),
        "tbill": _compound_monthly(tbill[tbill.index <= cut]),
        "fx": _monthly_last(fx_daily).pct_change().dropna(),
    }


def _compound_monthly(daily: pd.Series) -> pd.Series:
    r = daily.dropna().sort_index()
    return r.groupby(pd.PeriodIndex(r.index, freq="M")).apply(lambda x: float(np.prod(1 + x) - 1))


# --------------------------------------------------------------------------- the wealth walk
@dataclass
class RouteOutcome:
    """One ETF held through one route over one window, from gross return to money in hand."""

    ticker: str
    label: str
    route: str
    months: int
    gross_cagr: float  # in reais, before any cost or tax
    gross_real_cagr: float  # the same, deflated by IPCA on the net leg's own deflator
    net_cagr: float  # after entry, exit, withholding and the 15% on the sale
    net_real_cagr: float  # the same, deflated by IPCA
    final: float  # what R$1 became, net
    tax_paid: float
    cost_paid: float
    withheld: float
    fx_cagr: float
    asset_cagr: float  # the ETF's own return in its own currency
    cross: float  # the part of the return that is neither leg alone
    max_drawdown: float
    start: str
    end: str

    def per_10k(self) -> float:
        return 10_000.0 * self.final


def _cagr(monthly: pd.Series) -> float:
    years = len(monthly) / MONTHS
    return float((1.0 + monthly).prod() ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def walk(data: VehicleData, rules: RouteRules, window: tuple[str, str] | None = None) -> dict:
    """Walk R$1 through the route month by month, tracking value, cost basis and every leak.

    The basis matters as much as the value. A distribution that arrives, loses 30% to the IRS
    and is reinvested has already been taxed: it raises the acquisition cost, so it is not taxed
    a second time as a capital gain on the way out. Skipping that would make the US route look
    worse than it is, in exactly the way a naive spreadsheet does.
    """
    px, div, fx = data.price_return, data.dividend_yield, data.fx_return
    if window is not None:
        sl = slice(pd.Period(window[0], freq="M"), pd.Period(window[1], freq="M"))
        px, div, fx = px.loc[sl], div.loc[sl], fx.loc[sl]
    if px.empty:
        return {}

    withholding = rules.dividend_withholding if data.vehicle.distributes else 0.0
    if data.quoted_brl:  # the quote already carries the currency; applying it again doubles it
        fx = pd.Series(0.0, index=fx.index)
    value = 1.0 - rules.entry_cost
    basis = 1.0
    cost_paid = rules.entry_cost
    withheld = 0.0
    path = [value]
    for r_px, y, r_fx in zip(px.to_numpy(), div.to_numpy(), fx.to_numpy(), strict=True):
        opening = value
        value *= (1.0 + r_px) * (1.0 + r_fx)  # the price leg, in reais
        income = opening * y * (1.0 + r_fx)  # the cash leg, converted at the same rate
        withheld += income * withholding
        kept = income * (1.0 - withholding)
        value += kept
        basis += kept  # already taxed abroad: it is cost, not gain
        path.append(value)

    gross = data.gross_return_brl(withholding).loc[px.index]
    before_exit = value
    value *= 1.0 - rules.exit_cost
    cost_paid += before_exit * rules.exit_cost
    tax = max(value - basis, 0.0) * rules.gain_tax
    curve = pd.Series(path[1:], index=px.index)
    return {
        "final": value - tax,
        "gross_final": float((1.0 + gross).prod()),
        "tax_paid": tax,
        "cost_paid": cost_paid,
        "withheld": withheld,
        "curve": curve,
        "returns": gross,
        "months": len(px),
    }


def outcome(
    data: VehicleData,
    rules: RouteRules,
    inflation: pd.Series,
    window: tuple[str, str] | None = None,
) -> RouteOutcome | None:
    res = walk(data, rules, window)
    if not res or res["months"] <= MONTHS:
        return None
    rets = res["returns"]
    years = res["months"] / MONTHS
    price_level = float((1.0 + inflation.reindex(rets.index).fillna(0.0)).prod())
    fx_leg = data.fx_return.loc[rets.index]
    local_leg = data.dollar_leg(
        rules.dividend_withholding if data.vehicle.distributes else 0.0
    ).loc[rets.index]
    curve = (1.0 + rets).cumprod()
    return RouteOutcome(
        ticker=data.key,
        label=data.vehicle.label,
        route=rules.key,
        months=res["months"],
        gross_cagr=_cagr(rets),
        # deflated with the same price level the net figure uses, so "before tax" and "after
        # tax" sit on one ruler: mixing a nominal gross with a real net turns inflation into
        # tax and makes a short series look like it gained from being taxed.
        gross_real_cagr=float((float((1.0 + rets).prod()) / price_level) ** (1.0 / years) - 1.0),
        net_cagr=float(res["final"] ** (1.0 / years) - 1.0),
        net_real_cagr=float((res["final"] / price_level) ** (1.0 / years) - 1.0),
        final=float(res["final"]),
        tax_paid=float(res["tax_paid"]),
        cost_paid=float(res["cost_paid"]),
        withheld=float(res["withheld"]),
        fx_cagr=_cagr(fx_leg),
        asset_cagr=_cagr(local_leg),
        cross=_cagr(rets) - _cagr(fx_leg) - _cagr(local_leg),
        max_drawdown=float((curve / curve.cummax() - 1.0).min()),
        start=str(rets.index[0]),
        end=str(rets.index[-1]),
    )


def route_of(vehicle: EtfVehicle) -> RouteRules:
    return ROUTES[vehicle.venue]


def outcomes(
    vehicles: dict[str, VehicleData],
    inflation: pd.Series,
    window: tuple[str, str] | None = None,
) -> list[RouteOutcome]:
    out = []
    for data in vehicles.values():
        got = outcome(data, route_of(data.vehicle), inflation, window)
        if got is not None:
            out.append(got)
    return sorted(out, key=lambda o: o.net_real_cagr, reverse=True)


def outcome_frame(items: list[RouteOutcome]) -> pd.DataFrame:
    rows = {
        o.label: {
            "route": ROUTES[o.route].label,
            "gross_cagr": o.gross_cagr,
            "gross_real_cagr": o.gross_real_cagr,
            "net_cagr": o.net_cagr,
            "net_real_cagr": o.net_real_cagr,
            "asset_cagr": o.asset_cagr,
            "fx_cagr": o.fx_cagr,
            "max_drawdown": o.max_drawdown,
            "per_10k": o.per_10k(),
            "years": o.months / MONTHS,
        }
        for o in items
    }
    return pd.DataFrame(rows).T


def common_window(vehicles: dict[str, VehicleData], min_years: float = 4.0) -> tuple[str, str]:
    """First and last month in which every *established* ETF already traded.

    A fund that only became liquid last year would otherwise shrink the head-to-head window for
    all the others, and a two-year window says nothing about a 15% tax charged once at the end.
    The latecomers still appear in the own-history table, with their own start date on the row.
    """
    eligible = [v for v in vehicles.values() if len(v.months) >= min_years * MONTHS]
    pool = eligible or list(vehicles.values())
    return str(max(v.months[0] for v in pool)), str(min(v.months[-1] for v in pool))


def established(vehicles: dict[str, VehicleData], min_years: float = 4.0) -> dict[str, VehicleData]:
    """The subset with enough history to survive the common window."""
    return {k: v for k, v in vehicles.items() if len(v.months) >= min_years * MONTHS}


# --------------------------------------------------------------------------- route vs route
def route_gap(
    vehicles: dict[str, VehicleData], inflation: pd.Series, window: tuple[str, str] | None = None
) -> pd.DataFrame:
    """Same exposure, two vehicles: what the wrapper costs and what the currency desk costs.

    `wrapper_drag` is the Brazilian fund's gross return in reais minus its own underlying ETF's
    return in reais, both net of the same 30% dividend withholding. It is the honest version of
    "the fee", because it also catches tracking error, cash drag and the rate the fund got.
    """
    rows = []
    for br, us in peers():
        if br.ticker not in vehicles or us.ticker not in vehicles:
            continue
        a, b = vehicles[br.ticker], vehicles[us.ticker]
        win = window or (str(max(a.months[0], b.months[0])), str(min(a.months[-1], b.months[-1])))
        oa = outcome(a, B3_ROUTE, inflation, win)
        ob = outcome(b, US_ROUTE, inflation, win)
        if oa is None or ob is None:
            continue
        rows.append(
            {
                "exposure": exposure_label(br),
                "b3": br.label,
                "us": us.label,
                "b3_gross": oa.gross_cagr,
                "us_gross": ob.gross_cagr,
                "wrapper_drag": oa.gross_cagr - ob.gross_cagr,
                "b3_net": oa.net_cagr,
                "us_net": ob.net_cagr,
                "net_gap": oa.net_cagr - ob.net_cagr,
                "b3_per_10k": oa.per_10k(),
                "us_per_10k": ob.per_10k(),
                "years": oa.months / MONTHS,
                "start": win[0],
                "end": win[1],
            }
        )
    return pd.DataFrame(rows)


def breakeven_years(
    annual_gap: float, entry_gap: float, rate: float = 0.10, max_years: int = 60
) -> float:
    """How long the cheaper-but-costlier-to-enter route needs to overtake the other.

    The US route pays about 1.6% at the door and gives back a few tenths of a point a year in
    lower fees. That trade has a break-even, and for a short holding period it is simply a bad
    deal - which is the practical answer to "should I open an account abroad for R$5.000?".
    """
    if annual_gap <= 0:
        return float("nan")
    for y in range(1, max_years + 1):
        if (1.0 - entry_gap) * (1.0 + rate + annual_gap) ** y >= (1.0 + rate) ** y:
            return float(y)
    return float("nan")


# --------------------------------------------------------------------------- the currency leg
def parity_check(ref: dict[str, pd.Series]) -> dict[str, float]:
    """Did the dollar pay more or less than the interest gap said it would?

    Uncovered interest parity says a currency whose rates are higher should depreciate by the
    difference. Over this window CDI ran far above T-bills, so parity predicted a steadily
    weaker real. Whether it delivered exactly that is the difference between a currency bet and
    a carry trade that happened to work.
    """
    cdi, tbill, fx = ref["cdi"], ref["tbill"], ref["fx"]
    idx = cdi.index.intersection(tbill.index).intersection(fx.index)
    cdi, tbill, fx = cdi.loc[idx], tbill.loc[idx], fx.loc[idx]
    implied = ((1.0 + cdi) / (1.0 + tbill) - 1.0).dropna()
    years = len(idx) / MONTHS
    return {
        "months": float(len(idx)),
        "years": years,
        "fx_cagr": _cagr(fx),
        "parity_cagr": _cagr(implied),
        "excess": _cagr(fx) - _cagr(implied),
        "cdi_cagr": _cagr(cdi),
        "tbill_cagr": _cagr(tbill),
        "start": str(idx[0]),
        "end": str(idx[-1]),
    }


def fx_windows(ref: dict[str, pd.Series], years: int = 5) -> dict[str, float]:
    """Every overlapping window of the currency leg - the spread the point estimate hides."""
    fx = ref["fx"]
    n = years * MONTHS
    growth = (1.0 + fx).cumprod()
    roll = (growth / growth.shift(n)).dropna() ** (1.0 / years) - 1.0
    return {
        "years": float(years),
        "n_windows": float(len(roll)),
        "median": float(roll.median()),
        "p5": float(roll.quantile(0.05)),
        "p95": float(roll.quantile(0.95)),
        "share_negative": float((roll < 0).mean()),
        "worst": float(roll.min()),
        "best": float(roll.max()),
    }


# --------------------------------------------------------------------------- forward view
@dataclass(frozen=True)
class Outlook:
    """The forward assumptions, all of them stated, none of them fitted to this sample."""

    label: str
    usd_equity_real: float  # expected real return of US equities, in dollars
    us_inflation: float
    br_inflation: float
    cdi: float
    tbill: float
    fx_adjustment: float = 0.0  # deviation from parity, if the reader wants to bet on one

    @property
    def fx_drift(self) -> float:
        """Expected BRL move per year: the rate gap, plus whatever view the reader adds."""
        return (1.0 + self.cdi) / (1.0 + self.tbill) - 1.0 + self.fx_adjustment

    @property
    def usd_nominal(self) -> float:
        return (1.0 + self.usd_equity_real) * (1.0 + self.us_inflation) - 1.0

    @property
    def brl_nominal(self) -> float:
        return (1.0 + self.usd_nominal) * (1.0 + self.fx_drift) - 1.0


def current_rates(ref: dict[str, pd.Series], months: int = 12) -> dict[str, float]:
    """Trailing one-year CDI and T-bill - the starting point of the forward assumptions."""
    out = {}
    for key in ("cdi", "tbill"):
        tail = ref[key].dropna().iloc[-months:]
        out[key] = float((1.0 + tail).prod() ** (MONTHS / len(tail)) - 1.0)
    tail = ref["ipca"].dropna().iloc[-months:]
    out["ipca"] = float((1.0 + tail).prod() ** (MONTHS / len(tail)) - 1.0)
    return out


def project(
    outlook: Outlook,
    rules: RouteRules,
    years: int,
    dividend_yield: float,
    distributes: bool,
    wrapper_drag: float = 0.0,
    amount: float = 10_000.0,
) -> dict[str, float]:
    """Expected money in hand, in reais, after costs, withholding and the 15% on the sale.

    Both routes lose 30% of every dividend to the IRS - the Brazilian fund suffers it inside its
    own portfolio, the investor abroad suffers it in his own account. What differs is what
    happens next, and it is the opposite of the usual story about accumulating funds. On the US
    route the surviving cash arrives and is reinvested, which *raises the cost basis*: it was
    already taxed abroad, so it is never taxed again. On the B3 route it stays inside the fund
    and is eventually taxed at 15% along with everything else.

    Deferral buys nothing here, because on this route nothing was going to be taxed early
    anyway - the 30% credit already covers the 15% owed on the dividend. What is left is the
    basis step-up, and it belongs to the distributing fund. The accumulating wrapper is simpler
    to hold and cheaper to enter; it is not, on these rules, more tax-efficient.
    """
    withholding = rules.dividend_withholding
    gross = outlook.brl_nominal + wrapper_drag
    price_growth = (1.0 + gross) / (1.0 + dividend_yield) - 1.0
    value = amount * (1.0 - rules.entry_cost)
    basis = amount
    for _ in range(years):
        opening = value
        value *= 1.0 + price_growth
        kept = opening * dividend_yield * (1.0 - withholding)
        value += kept
        if distributes:  # taxed abroad already: it is acquisition cost, not gain
            basis += kept
    value *= 1.0 - rules.exit_cost
    tax = max(value - basis, 0.0) * rules.gain_tax
    net = value - tax
    real = net / (1.0 + outlook.br_inflation) ** years
    cash_gross = amount * (1.0 + outlook.cdi) ** years
    cash = amount + (cash_gross - amount) * (1.0 - 0.15)  # CDI, 15% once at redemption
    return {
        "gross": value,
        "tax": tax,
        "net": net,
        "real": real,
        "net_cagr": (net / amount) ** (1.0 / years) - 1.0,
        "real_cagr": (real / amount) ** (1.0 / years) - 1.0,
        "vs_cdi": net - cash,
        "cdi_net": cash,
        "cdi_real": cash / (1.0 + outlook.br_inflation) ** years,
    }


def forward_table(
    outlooks: list[Outlook],
    horizons: tuple[int, ...] = (5, 10, 20),
    dividend_yield: float = 0.014,
    wrapper_drag: float = -0.0043,
    amount: float = 10_000.0,
) -> pd.DataFrame:
    """Both routes, every scenario, every horizon - the table the question actually wants.

    The defaults describe the S&P 500 case: a 1.4% distribution yield and the 0.43 point a year
    the Brazilian wrapper has actually lagged its own underlying ETF since 2014.
    """
    rows = []
    for outlook in outlooks:
        for years in horizons:
            b3 = project(outlook, B3_ROUTE, years, dividend_yield, False, wrapper_drag, amount)
            us = project(outlook, US_ROUTE, years, dividend_yield, True, 0.0, amount)
            rows.append(
                {
                    "scenario": outlook.label,
                    "years": years,
                    "usd_real": outlook.usd_equity_real,
                    "fx_drift": outlook.fx_drift,
                    "brl_nominal": outlook.brl_nominal,
                    "b3_final": b3["net"],
                    "b3_real_final": b3["real"],
                    "b3_cagr": b3["net_cagr"],
                    "b3_real_cagr": b3["real_cagr"],
                    "us_final": us["net"],
                    "us_real_final": us["real"],
                    "us_cagr": us["net_cagr"],
                    "us_real_cagr": us["real_cagr"],
                    "cdi_final": b3["cdi_net"],
                    "cdi_real_final": b3["cdi_real"],
                    "b3_vs_cdi_final": b3["vs_cdi"],
                    "us_vs_cdi_final": us["vs_cdi"],
                }
            )
    return pd.DataFrame(rows)


def vehicle_forward(
    vehicles: dict[str, VehicleData],
    outlook: Outlook,
    years: int,
    drags: dict[str, float],
    yields: dict[str, float],
    default_drag: float = 0.0,
    default_yield: float = 0.014,
    amount: float = 10_000.0,
) -> pd.DataFrame:
    """Expected money in hand per ETF, holding the market view fixed across all of them.

    Every row assumes the *same* equity return, on purpose. Forecasting that the Nasdaq will
    beat the world index is a market call this study has no evidence for; what the data does
    support is how much each vehicle takes out of whatever the market delivers - its measured
    lag against its own underlying, its distribution yield, and the toll of its route. This
    table isolates exactly that, so the reader compares tickers instead of prophecies.
    """
    rows = []
    for key, data in vehicles.items():
        etf = data.vehicle
        rules = route_of(etf)
        drag = drags.get(key, default_drag if etf.venue == "b3" else 0.0)
        div = yields.get(key, default_yield)
        res = project(outlook, rules, years, div, etf.distributes, drag, amount)
        rows.append(
            {
                "label": etf.label,
                "route": rules.label,
                "exposure": exposure_label(etf),
                "expense_ratio": etf.expense_ratio,
                "measured_drag": drag,
                "dividend_yield": div,
                "withholding_cost": withholding_cost(div),
                "final": res["net"],
                "real_final": res["real"],
                "net_cagr": res["net_cagr"],
                "real_cagr": res["real_cagr"],
                "vs_cdi_final": res["vs_cdi"],
            }
        )
    out = pd.DataFrame(rows).set_index("label").sort_values("final", ascending=False)
    out.index.name = "ETF"
    return out


def default_outlooks(rates: dict[str, float], parity_gap: float = 0.0) -> list[Outlook]:
    """Three futures for US equities plus one for the currency, none of them fitted here.

    The real-return band is deliberately wide and deliberately below what the last sixteen years
    delivered: 11% real a year in dollars is a bull market, not a planning assumption. The
    currency needs no band of its own - parity ties it to the rate gap - but it does deserve one
    counter-scenario, because over this sample the real depreciated `parity_gap` points a year
    *less* than the gap said it would. Feeding that deviation forward is the closest thing to a
    bear case for the dollar leg that the data itself supports.
    """
    common = {
        "us_inflation": 0.023,
        "br_inflation": rates["ipca"],
        "cdi": rates["cdi"],
        "tbill": rates["tbill"],
    }
    outlooks = [
        Outlook("Bear - 3% real in dollars", 0.03, **common),
        Outlook("Base - 5% real in dollars", 0.05, **common),
        Outlook("Bull - 7% real in dollars", 0.07, **common),
    ]
    if parity_gap:
        outlooks.append(
            Outlook(
                f"Base, currency off parity by its historical gap ({parity_gap:+.1%} a year)",
                0.05,
                fx_adjustment=parity_gap,
                **common,
            )
        )
    return outlooks


def dividend_yields(vehicles: dict[str, VehicleData], months: int = 60) -> dict[str, float]:
    """Trailing distribution yield of each US ETF - what the 30% withholding actually bites."""
    out = {}
    for key, data in vehicles.items():
        if not data.vehicle.distributes:
            continue
        tail = data.dividend_yield.iloc[-months:]
        out[key] = float((1.0 + tail).prod() ** (MONTHS / len(tail)) - 1.0)
    return out


def withholding_cost(yield_pct: float, withholding: float = US_DIVIDEND_WITHHOLDING) -> float:
    """Points of annual return the IRS keeps - the fee nobody puts in the fee comparison."""
    return yield_pct * withholding


# --------------------------------------------------------------------------- presentation
# One formatter for every table in this section. A column of money is anything ending in
# `_final` or `per_10k`; the rest of this set is a rate. Keeping the two apart in one place is
# what stops a table from printing R$ 44.228 as 4,422,800%.
PCT = (
    "gross_cagr",
    "gross_real_cagr",
    "net_cagr",
    "net_real_cagr",
    "real_cagr",
    "asset_cagr",
    "fx_cagr",
    "max_drawdown",
    "wrapper_drag",
    "net_gap",
    "b3_gross",
    "us_gross",
    "b3_net",
    "us_net",
    "b3_cagr",
    "us_cagr",
    "b3_real_cagr",
    "us_real_cagr",
    "expense_ratio",
    "measured_drag",
    "dividend_yield",
    "withholding_cost",
    "usd_real",
    "fx_drift",
    "brl_nominal",
)


def _is_money(col: str) -> bool:
    return col.endswith("_final") or col.endswith("per_10k")


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if _is_money(col):
            out[col] = out[col].map(lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "-")
        elif col in PCT:
            out[col] = out[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
        elif col == "years":
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    return out


# The package speaks Portuguese and the study speaks English, the same split `asset_classes`
# uses for its class labels. One dict keeps the two in sync.
EN_EXPOSURE: dict[str, str] = {
    "sp500": "S&P 500",
    "world": "World equities",
    "nasdaq": "Nasdaq 100",
    "us_total": "Total US market",
    "intl": "Equities outside the US",
    "dividend": "US dividend equities",
    "bonds": "US bonds",
}


def exposure_label(vehicle: EtfVehicle) -> str:
    return EN_EXPOSURE.get(vehicle.exposure, vehicle.exposure_label)


ROUTE_COLOR = {"b3": "#2563eb", "us": "#dc2626"}
EXPOSURE_COLOR = {
    "sp500": "#dc2626",
    "world": "#2563eb",
    "nasdaq": "#7c3aed",
    "us_total": "#f59e0b",
    "intl": "#059669",
    "dividend": "#0891b2",
    "bonds": "#64748b",
}


def annualised(monthly: pd.Series) -> float:
    return _cagr(monthly)


def volatility(monthly: pd.Series) -> float:
    return float(monthly.std(ddof=1) * math.sqrt(MONTHS))
