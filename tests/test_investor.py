"""A camada de análise do investidor (notebooks/investor.py): carteiras, janelas e imposto.

São contas que entram no artigo, e duas delas erram silenciosamente se ninguém olhar: uma
carteira que rebalanceia sozinha e um imposto que cobra duas vezes o mesmo dinheiro produzem
números plausíveis e errados. O resto do notebook não tem teste por ser pesquisa; estas têm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import investor as inv  # noqa: E402
from asset_classes import ClassCurve  # noqa: E402

from swing_quant.data.assets import AssetProxy  # noqa: E402


def _months(n: int, start: str = "2010-01") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="M")


def _curve(values: list[float] | float, n: int = 24, key: str = "x") -> ClassCurve:
    data = [values] * n if isinstance(values, float) else values
    proxy = AssetProxy(key, key, "equity", "b3", "ticker", ("X",))
    return ClassCurve(proxy, pd.Series(data, index=_months(len(data))))


# --------------------------------------------------------------------------- carteiras
def test_blend_of_identical_returns_is_that_return() -> None:
    parts = {
        "a": pd.Series([0.01] * 12, index=_months(12)),
        "b": pd.Series([0.01] * 12, index=_months(12)),
    }
    out = inv.blend_returns(parts, {"a": 0.5, "b": 0.5})
    assert out.round(10).tolist() == [0.01] * 12


def test_weights_are_normalised() -> None:
    parts = {
        "a": pd.Series([0.10] * 12, index=_months(12)),
        "b": pd.Series([0.0] * 12, index=_months(12)),
    }
    out = inv.blend_returns(parts, {"a": 3.0, "b": 1.0})  # 75/25 escrito sem normalizar
    assert out.iloc[0] == pytest.approx(0.075)


def test_drift_between_rebalances_moves_the_weights() -> None:
    """Sem rebalanceamento o vencedor domina; com rebalanceamento anual a carteira volta ao alvo.

    O ativo `a` sobe 10% ao mês e `b` fica parado. Depois de doze meses `a` multiplicou por
    1,1^12 = 3,14, então vale 76% da carteira que nunca rebalanceou — e o retorno do 13º mês,
    7,6%, denuncia esse peso. A carteira anual voltou a 50/50 e rende 5%.
    """
    idx = _months(13)
    parts = {"a": pd.Series([0.10] * 13, index=idx), "b": pd.Series([0.0] * 13, index=idx)}
    drifting = inv.blend_returns(parts, {"a": 0.5, "b": 0.5}, rebalance_months=10**6)
    annual = inv.blend_returns(parts, {"a": 0.5, "b": 0.5}, rebalance_months=12)

    assert drifting.iloc[0] == pytest.approx(0.05)  # 50/50 no primeiro mês, nos dois casos
    assert annual.iloc[0] == pytest.approx(0.05)
    weight_a = 1.10**12 / (1.10**12 + 1)
    assert drifting.iloc[12] == pytest.approx(0.10 * weight_a, abs=1e-6)
    assert annual.iloc[12] == pytest.approx(0.05)  # reset para 50/50 no aniversário


def test_portfolio_needs_every_leg() -> None:
    curves = {"cdi": _curve(0.01, key="cdi")}
    spec = inv.PortfolioSpec("p", "p", {"cdi": 0.5, "nao_existe": 0.5})
    assert inv.portfolio(curves, spec) is None


# --------------------------------------------------------------------------- janelas
def test_rolling_window_of_constant_returns_is_the_annualised_rate() -> None:
    monthly = pd.Series([0.01] * 36, index=_months(36))
    out = inv.rolling_windows(monthly, years=1)
    assert out.iloc[0] == pytest.approx(1.01**12 - 1)
    assert len(out) == 36 - 12


def test_rolling_window_needs_a_full_window() -> None:
    assert inv.rolling_windows(pd.Series([0.01] * 10, index=_months(10)), years=1).empty


def test_time_to_recover_counts_the_wait_not_the_depth() -> None:
    """Cai um mês, fica no fundo três, volta: quatro meses debaixo d'água."""
    rets = [0.0, -0.20, 0.0, 0.0, 0.0, 0.30, 0.0]
    out = inv.time_to_recover(pd.Series(rets, index=_months(len(rets))))
    assert out["worst_months_underwater"] == 4
    assert out["months_underwater_now"] == 0


def test_time_to_recover_reports_an_open_drawdown() -> None:
    rets = [0.10, -0.30, 0.0, 0.0]
    out = inv.time_to_recover(pd.Series(rets, index=_months(len(rets))))
    assert out["months_underwater_now"] == 3
    assert out["share_underwater"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- imposto
_FLAT_INFLATION = pd.Series([0.0] * 24, index=_months(24))


def test_exempt_profile_costs_nothing() -> None:
    out = inv.after_tax_cagr(_curve(0.01), inv.TaxProfile(), _FLAT_INFLATION)
    assert out["tax_drag"] == pytest.approx(0.0, abs=1e-12)


def test_reinvested_exempt_income_is_not_taxed_again_as_a_gain() -> None:
    """Regressão: o dividendo isento de um FII entra na base de custo, não no ganho de capital.

    Um fundo cujo retorno é *todo* distribuição isenta não deve nada — nem no caminho, porque a
    distribuição é isenta, nem na venda, porque o que foi reinvestido já é custo de aquisição.
    Antes desta correção o modelo cobrava 20% sobre dezesseis anos de dividendos reinvestidos.
    """
    yield_a_year = 0.08
    monthly = (1 + yield_a_year) ** (1 / 12) - 1
    fii = inv.TaxProfile(income_yield=yield_a_year, income_rate=0.0, gain_rate=0.20)
    out = inv.after_tax_cagr(_curve(monthly), fii, _FLAT_INFLATION)
    assert out["tax_drag"] == pytest.approx(0.0, abs=1e-9)


def test_taxable_income_is_charged_as_it_arrives() -> None:
    taxed = inv.TaxProfile(income_yield=0.08, income_rate=0.275, gain_rate=0.0)
    out = inv.after_tax_cagr(_curve((1.08) ** (1 / 12) - 1), taxed, _FLAT_INFLATION)
    assert out["net_real"] == pytest.approx(0.08 * (1 - 0.275), abs=2e-3)


def test_entry_and_exit_costs_show_up_in_the_drag() -> None:
    plain = inv.after_tax_cagr(_curve(0.01), inv.TaxProfile(), _FLAT_INFLATION)
    costly = inv.after_tax_cagr(
        _curve(0.01), inv.TaxProfile(entry_cost=0.05, exit_cost=0.06), _FLAT_INFLATION
    )
    assert costly["net_real"] < plain["net_real"]
    assert costly["tax_drag"] > 0.05  # 11% de atrito diluído em dois anos


def test_inflation_tax_is_zero_without_inflation_and_positive_with_it() -> None:
    """O IR brasileiro incide sobre o ganho nominal: com inflação, tributa-se poder de compra."""
    flat = inv.after_tax_cagr(_curve(0.01), inv.TaxProfile(gain_rate=0.15), _FLAT_INFLATION)
    assert flat["inflation_tax"] == pytest.approx(0.0, abs=1e-12)

    inflation = pd.Series([0.005] * 24, index=_months(24))
    with_inflation = inv.after_tax_cagr(_curve(0.01), inv.TaxProfile(gain_rate=0.15), inflation)
    assert with_inflation["inflation_tax"] > 0.001


def test_gross_and_net_are_measured_over_the_same_months() -> None:
    """A curva vai além da série de inflação; comparar períodos diferentes daria drag negativo."""
    curve = _curve(0.01, n=36)
    out = inv.after_tax_cagr(curve, inv.TaxProfile(gain_rate=0.15), _FLAT_INFLATION)
    assert out["tax_drag"] > 0


# --------------------------------------------------------------------------- regime de juros
def test_real_rate_discounts_inflation_from_the_cash_leg() -> None:
    cash = pd.Series([(1.10) ** (1 / 12) - 1] * 24, index=_months(24))
    inflation = pd.Series([(1.05) ** (1 / 12) - 1] * 24, index=_months(24))
    out = inv.real_rate(cash, inflation)
    assert out.iloc[-1] == pytest.approx(1.10 / 1.05 - 1, abs=1e-4)
