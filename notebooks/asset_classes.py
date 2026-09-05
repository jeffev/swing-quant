"""Asset-class baselines for the study: everything a household can actually buy, on one ruler.

`study_lib` answers "did the swing book beat the index and the cash it sat on?". This module
answers the question underneath it - **which asset class should hold the money at all** - by
rebuilding a total-return curve for each one from `market.duckdb` and putting them side by side.

Three choices shape every number here:

1. **Monthly is the canonical frequency.** Physical property (IVG-R) and the savings account
   only exist monthly. Forcing them into a daily grid would invent prices they never had, so
   everything is compared on month-end closes instead. Drawdowns are therefore month-end
   drawdowns: shallower than the intraday truth, but shallower by the same rule for all.
2. **Real returns are the headline.** Over 16 years Brazilian inflation compounds to more than
   the nominal return of several of these assets. A nominal table would rank them wrong.
3. **Two windows.** Each class over its own longest history, plus a strict head-to-head over the
   window where every class exists - the honest comparison, and the shorter one.
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

from swing_quant.data.assets import AssetProxy, proxies_for  # noqa: E402
from swing_quant.data.macro import MACRO_CATALOG  # noqa: E402
from swing_quant.data.store import MarketStore  # noqa: E402

DB_PATH = REPO / "data" / "market.duckdb"
MONTHS = 12
DEFLATOR = {"b3": "ipca", "us": "cpi_us"}
CURRENCY = {"b3": "R$", "us": "US$"}
CASH_KEY = {"b3": "cdi", "us": "tbills"}


# --------------------------------------------------------------------------- building blocks
def _to_monthly(daily: pd.Series) -> pd.Series:
    """Month-end close of a daily level series, indexed by period."""
    s = daily.dropna().sort_index()
    return s.groupby(pd.PeriodIndex(s.index, freq="M")).last()


def _monthly_from_daily_returns(daily: pd.Series) -> pd.Series:
    """Compound daily returns inside each month."""
    r = daily.dropna().sort_index()
    return r.groupby(pd.PeriodIndex(r.index, freq="M")).apply(
        lambda x: float(np.prod(1.0 + x) - 1.0)
    )


def _as_period(s: pd.Series) -> pd.Series:
    """Reindex a series stamped on the first day of its reference month by period."""
    out = s.dropna().sort_index()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out[~out.index.duplicated(keep="last")]


@dataclass
class ClassCurve:
    """One asset class as a monthly return series, plus what the reader must know to trust it."""

    proxy: AssetProxy
    returns: pd.Series  # nominal monthly returns, indexed by PeriodIndex

    @property
    def key(self) -> str:
        return self.proxy.key

    @property
    def label(self) -> str:
        return self.proxy.label

    @property
    def start(self) -> pd.Period:
        return self.returns.index[0]

    def curve(self, base: float = 100.0) -> pd.Series:
        return base * (1.0 + self.returns).cumprod()


# --------------------------------------------------------------------------- loading
def _price_series(store: MarketStore, symbol: str) -> pd.Series:
    df = store.get_prices([symbol])
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adj_close"].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s.dropna()


def _macro_returns(store: MarketStore, series: str) -> pd.Series:
    """Monthly returns from a macro series, reading its unit from the catalog."""
    raw = store.macro(series)
    if raw.empty:
        return pd.Series(dtype=float)
    unit = MACRO_CATALOG[series].unit
    if unit == "pct_month":
        return _as_period(raw) / 100.0
    if unit == "index":
        return _as_period(raw).pct_change().dropna()
    if unit == "daily_return":
        return _monthly_from_daily_returns(raw)
    raise ValueError(f"unidade desconhecida em {series}: {unit}")


def build_class_curve(store: MarketStore, proxy: AssetProxy) -> ClassCurve | None:
    """Rebuild one asset class as monthly total returns, or None when the data is missing."""
    if proxy.kind in {"ticker", "b3_index"}:
        rets = _to_monthly(_price_series(store, proxy.symbols[0])).pct_change().dropna()
    elif proxy.kind == "fx":
        asset, fx = (_price_series(store, s) for s in proxy.symbols)
        local = (asset * fx.reindex(asset.index).ffill()).dropna()
        rets = _to_monthly(local).pct_change().dropna()
    elif proxy.kind == "macro":
        rets = _macro_returns(store, proxy.series)
    elif proxy.kind == "riskfree":
        rets = _monthly_from_daily_returns(store.risk_free(proxy.market))
    else:
        raise ValueError(f"kind desconhecido: {proxy.kind}")
    rets = rets.dropna()
    return ClassCurve(proxy, rets) if len(rets) > MONTHS else None


def load_classes(market: str, db_path: Path = DB_PATH) -> dict[str, ClassCurve]:
    with MarketStore(db_path) as store:
        out = {}
        for proxy in proxies_for(market):
            curve = build_class_curve(store, proxy)
            if curve is not None:
                out[proxy.key] = curve
    return out


def load_inflation(market: str, db_path: Path = DB_PATH) -> pd.Series:
    """Monthly inflation of the market's currency, as a return series."""
    with MarketStore(db_path) as store:
        return _macro_returns(store, DEFLATOR[market])


# --------------------------------------------------------------------------- statistics
def _annualise(monthly: pd.Series) -> float:
    years = len(monthly) / MONTHS
    return float((1.0 + monthly).prod() ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def deflate(nominal: pd.Series, inflation: pd.Series) -> pd.Series:
    """Nominal monthly returns to real ones - the only way to rank across 16 years of IPCA."""
    infl = inflation.reindex(nominal.index)
    return ((1.0 + nominal) / (1.0 + infl) - 1.0).dropna()


def annual_returns(monthly: pd.Series, inflation: pd.Series | None = None) -> pd.Series:
    """Calendar-year returns of a monthly series, real when `inflation` is given.

    The canonical yearly view for the whole study: a CAGR says a thing won, only the years say
    when, and two curves with the same CAGR and opposite years are not the same investment.
    """
    rets = monthly if inflation is None else deflate(monthly, inflation)
    return (1.0 + rets).groupby(rets.index.year).prod() - 1.0


def cagr(monthly: pd.Series, inflation: pd.Series | None = None) -> float:
    rets = monthly if inflation is None else deflate(monthly, inflation)
    years = len(rets) / MONTHS
    return float((1.0 + rets).prod() ** (1.0 / years) - 1.0) if years else float("nan")


def max_drawdown_monthly(returns: pd.Series) -> float:
    curve = (1.0 + returns).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def class_stats(
    returns: pd.Series, inflation: pd.Series, cash: pd.Series | None = None
) -> dict[str, float]:
    """Risk and return of one class over whatever window `returns` covers."""
    real = deflate(returns, inflation)
    vol = float(returns.std(ddof=1) * math.sqrt(MONTHS))
    sharpe = float("nan")
    if cash is not None:
        excess = (returns - cash.reindex(returns.index)).dropna()
        sd = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        if sd > 1e-12:
            sharpe = float(excess.mean() / sd * math.sqrt(MONTHS))
    by_year = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    return {
        "cagr_nominal": _annualise(returns),
        "cagr_real": _annualise(real),
        "volatility": vol,
        "max_drawdown": max_drawdown_monthly(returns),
        "sharpe": sharpe,
        "worst_year": float(by_year.min()),
        "best_year": float(by_year.max()),
        "positive_months": float((returns > 0).mean()),
        "total_real": float((1.0 + real).prod() - 1.0),
        "months": float(len(returns)),
        "years": len(returns) / MONTHS,
    }


def common_window(curves: dict[str, ClassCurve]) -> tuple[pd.Period, pd.Period]:
    """First and last month in which *every* class has a return."""
    starts = [c.returns.index[0] for c in curves.values()]
    ends = [c.returns.index[-1] for c in curves.values()]
    return max(starts), min(ends)


def returns_frame(
    curves: dict[str, ClassCurve], window: tuple[pd.Period, pd.Period] | None = None
) -> pd.DataFrame:
    df = pd.DataFrame({k: c.returns for k, c in curves.items()})
    if window is not None:
        df = df.loc[window[0] : window[1]]
    return df


def stats_table(
    curves: dict[str, ClassCurve],
    inflation: pd.Series,
    cash_key: str,
    window: tuple[pd.Period, pd.Period] | None = None,
) -> pd.DataFrame:
    """One row per asset class, sorted by real CAGR - the ranking the reader came for."""
    cash = curves[cash_key].returns if cash_key in curves else None
    rows = {}
    for key, c in curves.items():
        rets = c.returns if window is None else c.returns.loc[window[0] : window[1]]
        if len(rets) <= MONTHS:
            continue
        stats = class_stats(rets, inflation, cash)
        stats["start"] = float(rets.index[0].year + (rets.index[0].month - 1) / 12)
        rows[label_of(c)] = stats
    out = pd.DataFrame(rows).T.sort_values("cagr_real", ascending=False)
    out.index.name = "Asset class"
    return out


def correlation_matrix(
    curves: dict[str, ClassCurve], window: tuple[pd.Period, pd.Period] | None = None
) -> pd.DataFrame:
    df = returns_frame(curves, window).dropna()
    df.columns = [label_of(curves[k]) for k in df.columns]
    return df.corr()


# The catalog's group names are Portuguese; the study needs the language-neutral key plus an
# English name, so a chart legend and the exported JSON agree with the rest of the write-up.
CLASS_GROUP_EN = {
    "equity": "Equities",
    "real_estate": "Property",
    "fixed_income": "Bonds",
    "cash": "Cash",
    "commodity": "Commodities",
    "crypto": "Crypto",
    "fx": "Currency",
}


def asset_class_of(curves: dict[str, ClassCurve]) -> dict[str, str]:
    """Label -> asset-class key (`equity`, `real_estate`, ...), not its display name."""
    return {label_of(c): c.proxy.asset_class for c in curves.values()}


# --------------------------------------------------------------------------- derived classes
def swing_as_class(equity: pd.Series, market: str, label: str) -> ClassCurve:
    """Wrap the study's swing equity curve as one more asset class, on the same monthly ruler."""
    monthly = _to_monthly(equity).pct_change().dropna()
    proxy = AssetProxy(
        key="swing",
        label=label,
        asset_class="equity",
        market=market,
        kind="ticker",
        note="a carteira do projeto, com o caixa remunerado",
    )
    return ClassCurve(proxy, monthly)


# Aluguel líquido de um imóvel residencial: yield bruto de ~5,5% ao ano no período (FipeZap),
# menos IR sobre o aluguel, condomínio e IPTU dos meses vagos, manutenção e vacância. 4% é uma
# hipótese deliberadamente generosa - o ponto do gráfico é que só o aluguel salva a classe.
NET_RENTAL_YIELD = 0.04


def with_income(curve: ClassCurve, annual_yield: float, label: str) -> ClassCurve:
    """Add a constant income stream to a price-only class - rent on a property, in practice.

    The IVG-R tracks what apartments are worth, not what they pay. Comparing it against total
    return series without adding rent would flatter every other class; adding rent as a constant
    is crude, but it is crude in the direction of honesty and the rate is stated on the chart.
    """
    monthly_yield = (1.0 + annual_yield) ** (1 / MONTHS) - 1.0
    proxy = AssetProxy(
        key=f"{curve.key}_income",
        label=label,
        asset_class=curve.proxy.asset_class,
        market=curve.proxy.market,
        kind=curve.proxy.kind,
        note=f"IVG-R mais {annual_yield:.1%} ao ano de aluguel líquido",
    )
    return ClassCurve(proxy, curve.returns + monthly_yield)


# The package speaks Portuguese, the study speaks English. One dict keeps the two in sync
# instead of scattering translations through the notebook.
EN_LABEL: dict[str, str] = {
    "acoes_br": "Brazilian stocks (Ibovespa)",
    "small_caps_br": "Brazilian small caps (SMLL)",
    "fiis": "Listed property funds (IFIX)",
    "imoveis": "Physical property (IVG-R)",
    "imoveis_income": f"Physical property + net rent ({NET_RENTAL_YIELD:.0%})",
    "sp500_brl": "S&P 500 in reais",
    "ouro_brl": "Gold in reais",
    "dolar": "US dollars under the mattress",
    "bitcoin_brl": "Bitcoin in reais",
    "cdi": "CDI (cash)",
    "poupanca": "Savings account",
    "tesouro_selic": "Floating-rate government bond",
    "tesouro_prefixado": "Fixed-rate government bond (~4y)",
    "tesouro_ipca": "Inflation-linked government bond (~10y)",
    "acoes_us": "US stocks (S&P 500)",
    "nasdaq": "Nasdaq 100",
    "small_caps_us": "US small caps",
    "intl_dev": "Developed ex-US",
    "emergentes": "Emerging markets",
    "reits_us": "REITs",
    "bonds_us": "US aggregate bonds",
    "treasuries_longas": "Long treasuries (20y+)",
    "tips": "TIPS",
    "ouro_usd": "Gold",
    "commodities": "Commodities",
    "bitcoin_usd": "Bitcoin",
    "tbills": "T-bills (cash)",
}


def label_of(curve: ClassCurve) -> str:
    """English label for the study, falling back to the catalog's Portuguese one."""
    return EN_LABEL.get(curve.key, curve.label)


# --------------------------------------------------------------------------- presentation
PCT_COLS = (
    "cagr_nominal",
    "cagr_real",
    "volatility",
    "max_drawdown",
    "worst_year",
    "best_year",
    "positive_months",
    "total_real",
)
COL_LABEL = {
    "cagr_real": "CAGR real",
    "cagr_nominal": "CAGR nominal",
    "volatility": "Vol.",
    "max_drawdown": "Max drawdown",
    "sharpe": "Sharpe vs cash",
    "worst_year": "Worst year",
    "best_year": "Best year",
    "positive_months": "Positive months",
    "total_real": "Total real",
    "years": "Years",
}


def fmt_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, label in COL_LABEL.items():
        if col not in df.columns:
            continue
        if col in PCT_COLS:
            out[label] = df[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        elif col == "years":
            out[label] = df[col].map(lambda v: f"{v:.1f}")
        else:
            out[label] = df[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    return out


# One colour per asset class, so a reader can see "everything red is equity" without a legend
# lookup. The swing sleeve keeps the blue it has everywhere else in the study.
CLASS_COLORS = {
    "equity": "#dc2626",
    "real_estate": "#f59e0b",
    "fixed_income": "#7c3aed",
    "cash": "#059669",
    "commodity": "#b45309",
    "crypto": "#0891b2",
    "fx": "#64748b",
}
SWING_COLOR = "#2563eb"


def color_of(curve: ClassCurve) -> str:
    if curve.key == "swing":
        return SWING_COLOR
    return CLASS_COLORS[curve.proxy.asset_class]


def color_map(curves: dict[str, ClassCurve]) -> dict[str, str]:
    """Label -> colour, ready to index by a stats table's row names."""
    return {label_of(c): color_of(c) for c in curves.values()}


def annotate_scatter(ax, points: list[tuple[float, float, str, str]], min_gap: float) -> None:
    """Label every point without letting the labels overlap.

    A risk/return chart puts half the classes in one corner - cash, savings, the short bonds all
    sit at ~1% volatility - so plain annotations pile on top of each other. Sort by height and
    push each label up until it clears the previous one: the leader lines stay short and every
    name stays readable.
    """
    ordered = sorted(points, key=lambda p: p[1])
    last = float("-inf")
    for x, y, text, color in ordered:
        y_label = max(y, last + min_gap)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + min_gap * 0.35, y_label),
            fontsize=7.2,
            color=color,
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color, "alpha": 0.55}
            if y_label - y > min_gap * 0.1
            else None,
        )
        last = y_label
