"""Everything the money could have been in, year by year, on one ruler.

The study answers its questions one comparison at a time: `study_lib` puts the swing book next
to the index and the cash it sat on, `asset_classes` ranks the asset classes, `cycle_portfolio`
asks whether steering between them pays. This module puts all of them in a single table with one
row per candidate and one column per calendar year - the view that a CAGR cannot give, because a
CAGR says a thing won without ever saying when.

Three things make the rows comparable, and each is a decision:

1. **The swing sleeves earn interest on idle cash.** The engine pays 0% on un-deployed cash on
   purpose, and the book sits in cash roughly three quarters of the time. Comparing that curve
   against the CDI would be comparing a portfolio that is mostly cash against cash, with the
   cash leg deleted from one side. `study_lib.with_cash_yield` puts it back.
2. **Every strategy is run on both markets.** The config knows which sleeve is in production
   where; the table shows all six against both, because a strategy that only works in the market
   it was fitted on is the single most common way a backtest lies, and it is visible here at a
   glance.
3. **Each row keeps its own history, and the CAGR does not.** A row starts when its data starts
   - bitcoin in 2014, the cycle portfolio in 2012 - so the year cells are honest about what
   existed. The CAGR column is computed over one shared window instead, which is the only way
   the numbers can be ranked against each other. Crypto is the exception that proves the rule:
   it is four and a half years shorter than everything else, so it is kept out of the window
   calculation rather than allowed to charge every other row a quarter of its sample.
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import asset_classes as A  # noqa: E402
import cycle_portfolio as CY  # noqa: E402
import investor as INV  # noqa: E402
import study_lib as S  # noqa: E402

from swing_quant.config import Config, load_config  # noqa: E402
from swing_quant.strategies import REGISTRY, make_strategy  # noqa: E402

SWING = "Swing"
PORTFOLIOS = "Carteiras"
CLASSES = "Classes de ativo"
GROUP_ORDER = (SWING, PORTFOLIOS, CLASSES)


@dataclass
class Line:
    """One candidate for the money: a label, a group, and monthly returns. Nothing else."""

    label: str
    group: str
    returns: pd.Series  # nominal monthly returns, PeriodIndex
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- swing
def _monthly_from_equity(equity: pd.Series) -> pd.Series:
    s = equity.dropna().sort_index()
    return s.groupby(pd.PeriodIndex(s.index, freq="M")).last().pct_change().dropna()


def swing_lines(market: str, cfg: Config | None = None, *, with_cash: bool = True) -> list[Line]:
    """Every strategy in the registry, run alone on this market, plus the production sleeve.

    Running the disabled ones is the point, not an oversight: `rsi2` and `drops_ibs` were failed
    by the protocol in August 2026 and `dip`/`pullback` never passed it. Their year columns show
    what that verdict looks like from the outside, which is more useful than their absence.
    """
    cfg = cfg or load_config(S.CONFIG_PATH)
    md = S.load_market(market, cfg)
    regime = S.build_regime_for(md, cfg)
    live = set(cfg.enabled_strategies(market))

    from swing_quant.backtest.validation import default_panel_factory

    factory = default_panel_factory(md.prices)
    out: list[Line] = []
    for name in sorted(REGISTRY):
        panel = factory(make_strategy(name, cfg.strategies.get(name, {})))
        res = S.run_swing(panel, cfg, market, regime=regime)
        equity = S.with_cash_yield(res, md.rf_daily) if with_cash else res.equity
        status = "sleeve de produção" if name in live else "reprovada/não habilitada"
        out.append(
            Line(
                label=f"swing {name}" + (" ★" if name in live else ""),
                group=SWING,
                returns=_monthly_from_equity(equity),
                note=f"{status}; {len(res.trades)} trades",
                tags=("live",) if name in live else (),
            )
        )

    if live:
        combined = S.build_panel_for(md, cfg)
        res = S.run_swing(combined, cfg, market, regime=regime)
        equity = S.with_cash_yield(res, md.rf_daily) if with_cash else res.equity
        out.append(
            Line(
                label=f"swing carteira ({', '.join(sorted(live))}) ★",
                group=SWING,
                returns=_monthly_from_equity(equity),
                note="o que o `swing-quant portfolio` roda, com juros sobre o caixa parado",
                tags=("live",),
            )
        )
    return out


# --------------------------------------------------------------------------- portfolios
def portfolio_lines(
    market: str, curves: dict[str, A.ClassCurve], inflation: pd.Series
) -> list[Line]:
    """The cycle portfolio, its static control, and the two textbook portfolios."""
    cfg = CY.CycleConfig()
    alloc = CY.CYCLE_ALLOCATIONS[market]
    cash = curves[A.CASH_KEY[market]].returns
    parts = {k: c.returns for k, c in curves.items()}
    signals = CY.phase_signals(curves[CY.GROWTH_PROXY[market]].returns, cash, inflation, cfg)
    cycle = CY.run_cycle(parts, signals["phase"], alloc, cfg)
    static = CY.run_static(parts, alloc, cfg, window=cycle.returns.index)

    out = [
        Line("carteira do ciclo", PORTFOLIOS, cycle.returns, "rebalanceada pela virada de fase"),
        Line("ciclo — controle estático", PORTFOLIOS, static.returns, "média das quatro fases"),
    ]
    for spec in INV.PORTFOLIOS[market]:
        # "100% ações" e "100% caixa" são a classe de ativo com outro nome: a linha já existe
        # embaixo, e duas linhas idênticas na mesma tabela só fazem o leitor procurar a diferença.
        if len(spec.weights) == 1:
            continue
        built = INV.portfolio(curves, spec)
        if built is not None:
            out.append(Line(spec.label.lower(), PORTFOLIOS, built.returns, spec.note))
    return out


# --------------------------------------------------------------------------- assembly
def build_lines(market: str) -> tuple[list[Line], pd.Series]:
    curves = A.load_classes(market)
    inflation = A.load_inflation(market)
    lines = swing_lines(market)
    lines += portfolio_lines(market, curves, inflation)
    lines += [
        Line(
            A.label_of(c),
            CLASSES,
            c.returns,
            A.CLASS_GROUP_EN[c.proxy.asset_class],
            # Crypto starts in late 2014, four and a half years after everything else. It stays
            # in the table - it is a real thing a person could have bought - but it does not get
            # to set the comparison window: letting it would cost every other row a quarter of
            # its sample to accommodate one.
            tags=("late_start",) if c.proxy.asset_class == "crypto" else (),
        )
        for c in curves.values()
    ]
    return lines, inflation


def common_window(lines: list[Line], inflation: pd.Series) -> tuple[pd.Period, pd.Period]:
    """The months in which every row exists - the only window a ranking may use.

    Rows tagged `late_start` are excluded from the calculation and keep their own shorter
    history; the `desde` column is what tells the reader their CAGR is not on the same ruler.
    """
    considered = [ln for ln in lines if "late_start" not in ln.tags] or lines
    start = max([ln.returns.index[0] for ln in considered] + [inflation.index[0]])
    end = min([ln.returns.index[-1] for ln in considered] + [inflation.index[-1]])
    return start, end


def annual_frame(lines: list[Line], inflation: pd.Series, *, real: bool = True) -> pd.DataFrame:
    """Rows are candidates, columns are years. Real returns unless `real=False`.

    Transposed on purpose: fifteen years fit across a page, twenty-five candidates do not.
    """
    deflator = inflation if real else None
    start, end = common_window(lines, inflation)
    rows: dict[str, dict[str, object]] = {}
    for ln in lines:
        rets = ln.returns.loc[ln.returns.index.isin(inflation.index)]
        annual = A.annual_returns(rets, deflator)
        rows[ln.label] = {
            "grupo": ln.group,
            "desde": rets.index[0].year,
            **{str(year): value for year, value in annual.items()},
            "CAGR": A.cagr(rets.loc[start:end], deflator),
        }
    out = pd.DataFrame(rows).T
    years = sorted(c for c in out.columns if str(c).isdigit())
    out = out[["grupo", "desde", *years, "CAGR"]]
    out = out.sort_values(
        ["grupo", "CAGR"],
        ascending=[True, False],
        key=lambda s: s.map({g: i for i, g in enumerate(GROUP_ORDER)}) if s.name == "grupo" else s,
    )
    out.index.name = "Candidato"
    out.attrs["window"] = (start, end)
    out.attrs["real"] = real
    return out


# --------------------------------------------------------------------------- presentation
def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in {"grupo", "desde"}:
            out[col] = df[col]
        else:
            out[col] = df[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    return out


def _table(df: pd.DataFrame) -> str:
    body = fmt(df).reset_index()
    header = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "|" + "---|" * len(body.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in body.to_numpy()]
    return "\n".join([header, rule, *rows])


def render_report(market: str, frame: pd.DataFrame, lines: list[Line]) -> str:
    start, end = frame.attrs["window"]
    notes = {ln.label: ln.note for ln in lines}
    body = [
        f"# Retorno real ano a ano — tudo junto ({market.upper()})",
        "",
        f"**Gerado em**: {dt.datetime.now():%Y-%m-%d %H:%M}  ",
        f"**Janela comum (usada só na coluna CAGR)**: {start} a {end}  ",
        "**Retorno real**: deflacionado por "
        + ("IPCA" if market == "b3" else "CPI")
        + ", bruto de imposto e de taxa de administração",
        "",
        "As linhas de swing incluem os juros sobre o caixa parado (o motor paga 0% de propósito, "
        "e a carteira fica a maior parte do tempo em caixa). ★ marca o sleeve de produção do "
        "mercado. Cada linha começa quando seu dado começa — a coluna `desde` diz quando.",
        "",
        _table(frame),
        "",
        "## O que é cada linha de swing",
        "",
    ]
    body += [
        f"- `{label}`: {note}"
        for label, note in notes.items()
        if label in frame.index and frame.loc[label, "grupo"] == SWING
    ]
    body += [
        "",
        "## Observações",
        "",
        "- Universo do swing é o snapshot atual do índice (viés de sobrevivência); as classes "
        "são índices, que ninguém compra diretamente.",
        "- Retorno mensal de fechamento de mês, porque imóvel e poupança só existem nessa "
        "frequência — o drawdown intradiário do swing não aparece aqui.",
        "- Toda estratégia é rodada nos dois mercados, inclusive onde não é sleeve de produção: "
        "é o teste cruzado, e o resultado fora do mercado de origem é para ser lido com "
        "desconfiança, não com entusiasmo.",
        "",
    ]
    return "\n".join(body)


def main(market: str = "b3", out_dir: Path = REPO / "reports") -> Path:
    lines, inflation = build_lines(market)
    frame = annual_frame(lines, inflation)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"annual_returns_{market}_{dt.datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(render_report(market, frame, lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    for mkt in sys.argv[1:] or ["b3"]:
        print(main(mkt))
