"""A carteira do ciclo (notebooks/cycle_portfolio.py): o relógio, o giro e o look-ahead.

Como no `test_investor`, o resto do módulo é pesquisa e não tem teste. Estas contas têm porque
erram em silêncio: uma alocação alimentada pelo retorno que ela está prestes a ganhar produz um
gráfico lindo e mentiroso, e um giro cobrado no mês errado transforma custo em lucro. Os dois
casos passariam despercebidos numa leitura de tabela.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import cycle_portfolio as cyc  # noqa: E402


def _months(n: int, start: str = "2010-01") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="M")


def _flat(value: float, n: int = 60, start: str = "2010-01") -> pd.Series:
    return pd.Series([value] * n, index=_months(n, start))


ALLOC = {
    "recovery": {"risky": 1.0},
    "overheat": {"risky": 1.0},
    "stagflation": {"safe": 1.0},
    "slowdown": {"safe": 1.0},
}


# --------------------------------------------------------------------------- alocações
def test_every_phase_has_an_allocation_that_sums_to_one() -> None:
    for market, alloc in cyc.CYCLE_ALLOCATIONS.items():
        assert set(alloc) == set(cyc.PHASES), market
        for phase, weights in alloc.items():
            assert all(w > 0 for w in weights.values()), (market, phase)
            assert sum(weights.values()) == pytest.approx(1.0), (market, phase)


def test_neutral_weights_are_the_average_of_the_four_phases() -> None:
    neutral = cyc.neutral_weights(cyc.BR_CYCLE)
    assert sum(neutral.values()) == pytest.approx(1.0)
    # o CDI aparece nas quatro fases; o alvo neutro é a média simples dos quatro pesos
    expected = sum(cyc.BR_CYCLE[p]["cdi"] for p in cyc.PHASES) / 4
    assert neutral["cdi"] == pytest.approx(expected)


# --------------------------------------------------------------------------- o relógio
def test_phase_maps_the_two_axes_onto_the_right_quadrant() -> None:
    """Crescimento acima do caixa e inflação subindo é aquecimento, e assim por diante."""
    n = 40
    # ações rendendo 2% ao mês contra caixa a 0,5%: crescimento ligado o tempo todo
    equity, cash = _flat(0.02, n), _flat(0.005, n)
    # Inflação acelerando na segunda metade. O eixo é a *aceleração*, não o nível: uma inflação
    # alta e estável volta a ficar em cima da própria média móvel e deixa de contar como subindo.
    infl = pd.Series([0.002] * 20 + [0.002 * i for i in range(1, 21)], index=_months(n))
    cfg = cyc.CycleConfig(trend_months=3, inflation_window=3, inflation_trend=3, confirm_months=1)
    out = cyc.phase_signals(equity, cash, infl, cfg)
    assert out["phase"].iloc[0] == "recovery"
    assert out["phase"].iloc[-1] == "overheat"

    weak = cyc.phase_signals(_flat(0.0, n), cash, infl, cfg)
    assert weak["phase"].iloc[0] == "slowdown"
    assert weak["phase"].iloc[-1] == "stagflation"


def test_inflation_is_lagged_by_the_publication_delay() -> None:
    """O índice de preços de um mês só é publicado no mês seguinte — a leitura anda junto."""
    n = 40
    equity, cash = _flat(0.02, n), _flat(0.005, n)
    infl = pd.Series([0.002] * 20 + [0.02] * 20, index=_months(n))
    args = {"trend_months": 3, "inflation_window": 3, "inflation_trend": 3, "confirm_months": 1}
    now = cyc.phase_signals(equity, cash, infl, cyc.CycleConfig(macro_lag=0, **args))
    late = cyc.phase_signals(equity, cash, infl, cyc.CycleConfig(macro_lag=2, **args))
    first_now = now.index[now["inflation_rising"]][0]
    first_late = late.index[late["inflation_rising"]][0]
    assert (first_late - first_now).n == 2


def test_confirmation_delays_the_switch_and_ignores_a_single_month() -> None:
    raw = pd.Series(
        ["slowdown"] * 5 + ["recovery"] + ["slowdown"] * 4 + ["recovery"] * 5, index=_months(15)
    )
    held = cyc._confirm(raw, months=2)
    assert list(held[:6]) == ["slowdown"] * 6  # o mês solto não troca a carteira
    assert list(held[10:]) == ["slowdown"] + ["recovery"] * 4  # a virada real custa um mês


# --------------------------------------------------------------------------- o motor
def test_allocation_is_governed_by_the_previous_month_reading() -> None:
    """O teste que impede o look-ahead: a fase lida no fim de dezembro rende em janeiro.

    O ativo `risky` só sobe no mês em que a fase lida é `stagflation` — que manda ficar em
    `safe`. Uma carteira com look-ahead pegaria essa alta; a correta fica de fora dela e pega o
    mês seguinte, quando o `risky` está parado.
    """
    n = 6
    parts = {
        "risky": pd.Series([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], index=_months(n)),
        "safe": _flat(0.0, n),
    }
    phase = pd.Series(
        ["recovery", "stagflation", "recovery", "recovery", "recovery", "recovery"],
        index=_months(n),
    )
    run = cyc.run_cycle(parts, phase, ALLOC, cyc.CycleConfig(cost_bps=0.0))
    assert run.phase.loc[pd.Period("2010-03", "M")] == "stagflation"
    assert run.returns.loc[pd.Period("2010-03", "M")] == pytest.approx(0.0)
    assert run.returns.sum() == pytest.approx(0.0)


def test_turnover_is_charged_when_the_phase_flips_and_not_otherwise() -> None:
    n = 6
    parts = {"risky": _flat(0.0, n), "safe": _flat(0.0, n)}
    phase = pd.Series(["recovery"] * 3 + ["stagflation"] * 3, index=_months(n))
    run = cyc.run_cycle(parts, phase, ALLOC, cyc.CycleConfig(cost_bps=100.0, drift_months=10**6))
    charged = run.turnover[run.turnover > 0]
    assert len(charged) == 1  # 100% em `risky` -> 100% em `safe`: um giro, uma cobrança
    assert float(charged.iloc[0]) == pytest.approx(1.0)
    assert run.returns.loc[charged.index[0]] == pytest.approx(-0.01)  # 100 bps
    assert run.gross_returns.sum() == pytest.approx(0.0)


def test_weights_drift_between_rebalances_and_reset_on_the_calendar() -> None:
    """Sem troca de fase a carteira anda sozinha; o calendário é que a traz de volta ao alvo."""
    n = 26
    parts = {
        "a": pd.Series([0.10] * n, index=_months(n)),
        "b": pd.Series([0.0] * n, index=_months(n)),
    }
    alloc = {p: {"a": 0.5, "b": 0.5} for p in cyc.PHASES}
    phase = pd.Series(["recovery"] * n, index=_months(n))
    cfg = cyc.CycleConfig(cost_bps=0.0, drift_months=12)
    run = cyc.run_cycle(parts, phase, alloc, cfg)
    drifted = run.weights["a"].iloc[11]
    assert drifted > 0.7  # onze meses de 10% ao mês deixam `a` dominando
    assert run.weights["a"].iloc[12] == pytest.approx(0.5)  # o reset anual desfaz o drift


def test_static_control_holds_the_average_allocation() -> None:
    n = 24
    parts = {"risky": _flat(0.01, n), "safe": _flat(0.0, n)}
    run = cyc.run_static(parts, ALLOC, cyc.CycleConfig(cost_bps=0.0))
    assert run.weights["risky"].iloc[0] == pytest.approx(0.5)
    assert run.returns.iloc[0] == pytest.approx(0.005)


def test_oracle_beats_the_rule_it_bounds() -> None:
    """O oráculo é o teto: se ele não ganha da regra, o motor está errado, não a regra."""
    rng = np.random.default_rng(7)
    n = 60
    parts = {
        "risky": pd.Series(rng.normal(0.01, 0.05, n), index=_months(n)),
        "safe": pd.Series(rng.normal(0.004, 0.005, n), index=_months(n)),
    }
    phase = pd.Series(rng.choice(cyc.PHASES, n), index=_months(n))
    cfg = cyc.CycleConfig(cost_bps=0.0)
    rule = cyc.run_cycle(parts, phase, ALLOC, cfg).returns
    oracle = cyc.run_oracle(parts, ALLOC, cfg).returns
    common = rule.index.intersection(oracle.index)
    assert (1 + oracle.loc[common]).prod() > (1 + rule.loc[common]).prod()


def test_missing_asset_class_is_a_loud_error() -> None:
    with pytest.raises(KeyError, match="safe"):
        cyc.run_cycle({"risky": _flat(0.0)}, pd.Series(["recovery"] * 60, index=_months(60)), ALLOC)


# --------------------------------------------------------------------------- leitura de hoje
def test_current_stance_reports_the_last_phase_and_when_it_started() -> None:
    signals = pd.DataFrame(
        {
            "growth_up": [False] * 3 + [True] * 3,
            "inflation_rising": [True] * 6,
            "raw_phase": ["stagflation"] * 3 + ["overheat"] * 3,
            "phase": ["stagflation"] * 3 + ["overheat"] * 3,
        },
        index=_months(6),
    )
    stance = cyc.current_stance(signals, cyc.BR_CYCLE)
    assert stance["phase"] == "overheat"
    assert stance["in_phase_since"] == pd.Period("2010-04", "M")
    assert stance["governs"] == pd.Period("2010-07", "M")  # a leitura vale para o mês seguinte
    assert sum(dict(stance["weights"]).values()) == pytest.approx(1.0)  # type: ignore[arg-type]
