"""As duas rotas de ETF (notebooks/etf_routes.py) e a higiene de série que elas dependem.

Três contas aqui erram em silêncio e produzem números plausíveis: aplicar o câmbio duas vezes
num fundo já cotado em reais, tributar de novo um dividendo que já pagou 30% lá fora, e deixar
passar um desdobramento não ajustado — o SPXI11 desdobrou 8 para 1 em jan/2026 e, sem correção,
16 anos de S&P 500 viram prejuízo no gráfico. O resto do módulo é pesquisa; estas têm teste.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import etf_routes as R  # noqa: E402, N812

from swing_quant.data.etfs import (  # noqa: E402
    ETF_BY_TICKER,
    ETF_VEHICLES,
    EtfVehicle,
    detect_unadjusted_splits,
    etf_tickers,
    peers,
    repair_unadjusted_splits,
)


# --------------------------------------------------------------------------- catálogo
def test_tickers_are_unique() -> None:
    assert len(ETF_BY_TICKER) == len(ETF_VEHICLES)


def test_every_peer_exists_and_shares_the_exposure() -> None:
    """Um par que compara S&P com Nasdaq mede a bolsa, não a rota — o pareamento tem que casar."""
    for br, us in peers():
        assert br.venue == "b3"
        assert us.venue == "us"
        assert br.exposure == us.exposure


def test_b3_funds_accumulate_and_us_funds_distribute() -> None:
    for etf in ETF_VEHICLES:
        assert etf.distributes == (etf.venue == "us")


def test_ticker_list_carries_the_currency() -> None:
    assert "USDBRL=X" in etf_tickers()


# --------------------------------------------------------------------------- splits
def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="D"))


def test_split_is_detected_and_repaired() -> None:
    """8 para 1: o histórico anterior é dividido por 8 e o retorno diário volta a fazer sentido."""
    raw = _series([400.0, 402.0, 50.0, 50.5])
    fixed, events = repair_unadjusted_splits(raw)
    assert len(events) == 1
    assert events[0][1] == pytest.approx(8.0, rel=0.05)
    assert fixed.iloc[0] == pytest.approx(50.0)
    assert fixed.pct_change().abs().max() < 0.05


def test_reverse_split_is_repaired() -> None:
    fixed, events = repair_unadjusted_splits(_series([5.0, 5.1, 51.0, 50.0]))
    assert events and events[0][1] == pytest.approx(0.1, rel=0.05)
    assert fixed.iloc[0] == pytest.approx(50.0, rel=0.05)


def test_a_crash_is_not_a_split() -> None:
    """Uma queda de 45% que não bate com nenhum fator redondo é mercado, e fica como está."""
    raw = _series([100.0, 100.0, 55.0, 56.0])
    assert detect_unadjusted_splits(raw) == []
    fixed, events = repair_unadjusted_splits(raw)
    assert not events
    assert fixed.equals(raw)


# --------------------------------------------------------------------------- a caminhada
def _vehicle(venue: str, distributes: bool) -> EtfVehicle:
    return EtfVehicle(
        ticker="TEST",
        label="teste",
        venue=venue,
        exposure="sp500",
        exposure_label="S&P 500",
        manager="-",
        expense_ratio=0.0,
        distributes=distributes,
    )


def _data(venue: str, distributes: bool, px: float, div: float, fx: float, n: int = 24):
    idx = pd.period_range("2020-01", periods=n, freq="M")
    return R.VehicleData(
        vehicle=_vehicle(venue, distributes),
        price_return=pd.Series(px, index=idx),
        dividend_yield=pd.Series(div, index=idx),
        fx_return=pd.Series(fx, index=idx),
    )


def test_a_brl_quoted_fund_does_not_get_the_currency_twice() -> None:
    """O erro mais caro do módulo: o IVVB11 já é em reais; aplicar o dólar de novo dobra tudo."""
    data = _data("b3", False, px=0.01, div=0.0, fx=0.02)
    monthly = data.gross_return_brl()
    assert monthly.iloc[0] == pytest.approx(0.01)
    res = R.walk(data, R.B3_ROUTE)
    expected = (1.0 - R.B3_ROUTE.entry_cost) * 1.01**24
    assert res["curve"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_a_us_fund_multiplies_the_two_legs() -> None:
    data = _data("us", True, px=0.01, div=0.0, fx=0.02)
    assert data.gross_return_brl().iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)


def test_the_dollar_leg_is_recovered_from_a_brl_quote() -> None:
    """Decomposição: dado o retorno em reais e o câmbio, sobra o retorno do ativo em dólar."""
    data = _data("b3", False, px=0.0302, div=0.0, fx=0.02)
    assert data.dollar_leg().iloc[0] == pytest.approx(1.0302 / 1.02 - 1.0, rel=1e-9)


def test_reinvested_dividends_are_not_taxed_twice() -> None:
    """O dividendo já perdeu 30% para o IRS: ele sobe o custo de aquisição, não o ganho.

    Sem isso, a rota americana paga 15% sobre dinheiro que já foi tributado e aparece pior do
    que é — que é exatamente o erro que uma planilha ingênua comete.
    """
    yield_only = _data("us", True, px=0.0, div=0.01, fx=0.0)
    res = R.walk(yield_only, R.US_ROUTE)
    assert res["tax_paid"] == pytest.approx(0.0, abs=1e-9)
    assert res["withheld"] > 0


def test_the_accumulating_route_defers_the_same_gain() -> None:
    """Mesma exposição bruta: quem acumula paga o imposto uma vez, no fim, sobre tudo."""
    fund = _data("b3", False, px=0.01, div=0.0, fx=0.0)
    res = R.walk(fund, R.B3_ROUTE)
    gain = res["curve"].iloc[-1] - 1.0
    assert res["tax_paid"] == pytest.approx(max(gain, 0.0) * 0.15, rel=0.02)


def test_costs_are_charged_on_both_ends() -> None:
    flat = _data("us", True, px=0.0, div=0.0, fx=0.0)
    res = R.walk(flat, R.US_ROUTE)
    assert res["final"] < 1.0  # nada rendeu e o câmbio cobrou nas duas pontas
    assert res["tax_paid"] == 0.0
    assert res["cost_paid"] == pytest.approx(
        R.US_ROUTE.entry_cost + (1 - R.US_ROUTE.entry_cost) * R.US_ROUTE.exit_cost, rel=1e-6
    )


# --------------------------------------------------------------------------- o modelo forward
def _outlook(**kw) -> R.Outlook:
    base = {
        "label": "teste",
        "usd_equity_real": 0.05,
        "us_inflation": 0.02,
        "br_inflation": 0.04,
        "cdi": 0.10,
        "tbill": 0.03,
    }
    return R.Outlook(**{**base, **kw})


def test_parity_sets_the_currency_drift() -> None:
    o = _outlook()
    assert o.fx_drift == pytest.approx(1.10 / 1.03 - 1.0)
    assert o.brl_nominal == pytest.approx((1 + o.usd_nominal) * (1 + o.fx_drift) - 1.0)


def test_a_view_on_the_currency_moves_the_drift() -> None:
    assert _outlook(fx_adjustment=-0.02).fx_drift == pytest.approx(_outlook().fx_drift - 0.02)


def test_the_basis_step_up_beats_deferral() -> None:
    """Contraintuitivo e testado: com o mesmo custo, o fundo que distribui termina com mais.

    O dividendo já pagou 30% lá fora, e os 30% cobrem os 15% brasileiros — não sobra imposto
    para adiar. O que sobra é o degrau de custo de aquisição do dinheiro reinvestido, e ele é do
    fundo que distribui. Se este teste inverter, o modelo voltou a cobrar imposto duas vezes.
    """
    o = _outlook()
    free = R.RouteRules("free", "sem custo", 0.0, 0.0, 0.30, 0.15, "")
    keeps = R.project(o, free, 20, dividend_yield=0.02, distributes=False)
    pays = R.project(o, free, 20, dividend_yield=0.02, distributes=True)
    assert pays["net"] > keeps["net"]
    assert pays["gross"] == pytest.approx(keeps["gross"], rel=1e-9)  # o caminho é o mesmo
    assert pays["tax"] < keeps["tax"]


def test_without_dividends_the_two_treatments_coincide() -> None:
    o = _outlook()
    free = R.RouteRules("free", "sem custo", 0.0, 0.0, 0.30, 0.15, "")
    a = R.project(o, free, 10, dividend_yield=0.0, distributes=False)
    b = R.project(o, free, 10, dividend_yield=0.0, distributes=True)
    assert a["net"] == pytest.approx(b["net"], rel=1e-12)


def test_the_entry_cost_needs_years_to_pay_for_itself() -> None:
    assert R.breakeven_years(0.0043, 0.0153, rate=0.18) > 1
    assert pd.isna(R.breakeven_years(-0.001, 0.0153))
