"""The build step the article never had: study numbers straight into the published HTML.

The article carries its own data in a `<script id="study-data" type="application/json">` block,
so the page is a single self-contained file. That is a good property and it came with a bad one:
until now the block was updated by hand, which means every number in the article was one
copy-paste away from disagreeing with the notebook that produced it.

This module closes that loop for the sections it owns. It rebuilds the cycle-portfolio and
year-by-year payloads from `market.duckdb`, merges them into the block of every article file,
and leaves the rest of the block untouched - so re-running it can add or refresh those keys but
can never silently drop what it does not know about.

Keys are camelCase because that is what the article's JavaScript reads; the Python side of the
study is snake_case. The translation lives here, in one place, instead of in a person's head.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import annual_comparison as AC  # noqa: E402
import asset_classes as A  # noqa: E402
import cycle_portfolio as CY  # noqa: E402

ARTICLES = (
    REPO / "notebooks" / "article" / "swing_vs_investing.html",
    REPO / "notebooks" / "article" / "swing_vs_investing_pt.html",
)
DATA_RE = re.compile(r'(<script id="study-data" type="application/json">)(.*?)(</script>)', re.S)
MARKETS = ("b3", "us")

# The study speaks English and the Portuguese edition of the article does not. Both labels ride
# in the payload so each edition reads the key it wants and the two can never drift apart - which
# is exactly what happened the last time the translation lived in the HTML by hand.
PT_LABEL: dict[str, str] = {
    # classes, Brasil
    "Brazilian stocks (Ibovespa)": "Ações Brasil (Ibovespa)",
    "Brazilian small caps (SMLL)": "Small caps Brasil (SMLL)",
    "Listed property funds (IFIX)": "FIIs (IFIX)",
    "Physical property (IVG-R)": "Imóvel físico (IVG-R)",
    "Physical property + net rent (4%)": "Imóvel físico + aluguel líquido (4%)",
    "S&P 500 in reais": "S&P 500 em reais",
    "Gold in reais": "Ouro em reais",
    "US dollars under the mattress": "Dólar guardado (sem render)",
    "Bitcoin in reais": "Bitcoin em reais",
    "CDI (cash)": "CDI (caixa)",
    "Savings account": "Poupança",
    "Floating-rate government bond": "Tesouro Selic",
    "Fixed-rate government bond (~4y)": "Tesouro Prefixado (~4a)",
    "Inflation-linked government bond (~10y)": "Tesouro IPCA+ (~10a)",
    # classes, EUA
    "US stocks (S&P 500)": "Ações EUA (S&P 500)",
    "Nasdaq 100": "Nasdaq 100",
    "US small caps": "Small caps EUA",
    "Developed ex-US": "Desenvolvidos ex-EUA",
    "Emerging markets": "Emergentes",
    "REITs": "REITs",
    "US aggregate bonds": "Renda fixa agregada (AGG)",
    "Long treasuries (20y+)": "Treasuries longas (20a+)",
    "TIPS": "TIPS (indexadas à inflação)",
    "Gold": "Ouro",
    "Commodities": "Commodities",
    "Bitcoin": "Bitcoin",
    "T-bills (cash)": "T-bills (caixa)",
    # carteiras
    "60/40 Brazil": "60/40 Brasil",
    "40/40 + 20% dollarised": "40/40 + 20% dolarizada",
    "Diversified Brazil": "Diversificada Brasil",
    "Permanent portfolio": "Carteira permanente",
    "60/40": "60/40",
    "Global 60/40": "60/40 global",
    "100% cash (CDI)": "100% caixa (CDI)",
    "100% Brazilian stocks": "100% ações Brasil",
    "100% cash (T-bills)": "100% caixa (T-bills)",
    "100% US stocks": "100% ações EUA",
}


def _pt(label: str) -> str:
    """Portuguese label, falling back to the original - a missing entry must not blank a row."""
    lowered = {k.lower(): v for k, v in PT_LABEL.items()}
    return PT_LABEL.get(label) or lowered.get(label.lower(), label)


def _n(value: Any, digits: int = 4) -> Any:
    """Round for transport. NaN becomes null: JSON has no NaN, and `JSON.parse` would throw."""
    if value is None:
        return None
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(f) or math.isinf(f) else round(f, digits)


# --------------------------------------------------------------------------- cycle portfolio
def cycle_payload(market: str) -> dict[str, Any]:
    """Everything the cycle section of the article shows, for one market."""
    study = CY.build_study(market, rotation_grid=True)
    cycle, control = study.runs[0], study.runs[1]
    curves = study.curves
    alloc = CY.CYCLE_ALLOCATIONS[market]
    annual = CY.annual_table(study.runs, study.inflation, phase=cycle.phase)
    labels = [r.label for r in study.runs]

    phases = []
    for key in CY.PHASES:
        row = study.phases.loc[CY.PHASE_LABEL[key]]
        phases.append(
            {
                "key": key,
                "labelEn": CY.PHASE_LABEL[key],
                "labelPt": CY.PHASE_PT[key],
                "months": int(row["months"]),
                "share": _n(row["share"]),
                "avgRun": _n(row["avg_run_months"], 1),
            }
        )

    return {
        "window": [str(cycle.returns.index[0]), str(cycle.returns.index[-1])],
        "config": {
            "trendMonths": study.cfg.trend_months,
            "confirmMonths": study.cfg.confirm_months,
            "driftMonths": study.cfg.drift_months,
            "costBps": study.cfg.cost_bps,
            "macroLag": study.cfg.macro_lag,
        },
        "labels": labels,
        "labelsPt": [_pt(x) for x in labels],
        "table": [
            {
                "label": label,
                "labelPt": _pt(label),
                **{k: _n(v) for k, v in study.table.loc[label].items()},
            }
            for label in labels
        ],
        "alloc": [
            {
                "phase": key,
                "weights": [
                    {
                        "key": asset,
                        "label": A.label_of(curves[asset]),
                        "labelPt": _pt(A.label_of(curves[asset])),
                        "w": _n(weight),
                    }
                    for asset, weight in sorted(
                        CY.normalise(alloc[key]).items(), key=lambda kv: -kv[1]
                    )
                ],
            }
            for key in CY.PHASES
        ],
        "phases": phases,
        "switches": int(study.phases.attrs["switches"]),
        "switchesPerYear": _n(study.phases.attrs["switches_per_year"], 2),
        "annual": [
            {
                "year": int(year),
                "months": int(row["meses"]),
                "phase": str(row["fase dominante"]),
                "values": [_n(row[label]) for label in labels],
            }
            for year, row in annual.iterrows()
        ],
        "byPhase": [
            {"label": str(idx), "labelPt": _pt(str(idx)), "values": [_n(v) for v in row]}
            for idx, row in study.by_phase.iterrows()
        ],
        "rotation": {k: _n(v) for k, v in study.rotation.items()},
        "edge": {k: _n(v) for k, v in CY.annual_edge(cycle, control, study.inflation).items()},
        "verdict": {k: _n(v) for k, v in CY.verdict(study).items()},
        "grid": [{k: _n(v) for k, v in row.items()} for _, row in study.grid.iterrows()],
        "stance": {
            "asOf": str(study.stance["as_of"]),
            "governs": str(study.stance["governs"]),
            "phase": study.stance["phase"],
            "since": str(study.stance["in_phase_since"]),
        },
    }


# --------------------------------------------------------------------------- year by year
def annual_payload(market: str) -> dict[str, Any]:
    """The one-row-per-candidate, one-column-per-year table, for one market."""
    lines, inflation = AC.build_lines(market)
    frame = AC.annual_frame(lines, inflation)
    start, end = frame.attrs["window"]
    years = [c for c in frame.columns if str(c).isdigit()]
    notes = {ln.label: ln.note for ln in lines}
    live = {ln.label for ln in lines if "live" in ln.tags}
    late = {ln.label for ln in lines if "late_start" in ln.tags}
    return {
        "window": [str(start), str(end)],
        "years": [int(y) for y in years],
        "rows": [
            {
                "label": str(label),
                "labelPt": _pt(str(label)),
                "group": str(row["grupo"]),
                "since": int(row["desde"]),
                "cagr": _n(row["CAGR"]),
                "live": label in live,
                "shortHistory": label in late,
                "note": notes.get(str(label), ""),
                "values": [_n(row[y]) for y in years],
            }
            for label, row in frame.iterrows()
        ],
    }


# --------------------------------------------------------------------------- merge
def build_payload() -> dict[str, Any]:
    return {
        "cycle": {m: cycle_payload(m) for m in MARKETS},
        "annual": {m: annual_payload(m) for m in MARKETS},
    }


def merge_into(path: Path, payload: dict[str, Any]) -> int:
    """Merge `payload` into the file's study-data block, keeping every key it does not own.

    Returns the new block's size in bytes. The merge is top-level and by key: `cycle` and
    `annual` are replaced wholesale, everything else in the block survives untouched.
    """
    text = path.read_text(encoding="utf-8")
    match = DATA_RE.search(text)
    if match is None:
        raise ValueError(f"{path.name}: bloco study-data não encontrado")
    data = json.loads(match.group(2))
    data.update(payload)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text[: match.start(2)] + blob + text[match.end(2) :], encoding="utf-8")
    return len(blob)


def main(paths: tuple[Path, ...] = ARTICLES) -> None:
    payload = build_payload()
    for path in paths:
        if path.exists():
            size = merge_into(path, payload)
            print(f"{path.name}: study-data agora com {size / 1024:.0f} KB")
        else:
            print(f"{path.name}: ausente, pulado")


if __name__ == "__main__":
    pd.set_option("mode.chained_assignment", None)
    main()
