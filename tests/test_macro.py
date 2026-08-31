"""Séries macro: gravação idempotente e a curva rolada de título público.

O download (BCB, BLS e Tesouro Direto) fica fora daqui — só a parte pura, sem rede.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swing_quant.data.macro import (
    MACRO_CATALOG,
    TD_SERIES,
    rolled_bond_returns,
    tesouro_curves,
)
from swing_quant.data.store import MACRO_COLUMNS, MarketStore


def _serie(key: str, dias: list[str], valor: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series": key,
            "date": [dt.date.fromisoformat(d) for d in dias],
            "value": valor,
            "unit": MACRO_CATALOG[key].unit,
            "source": "teste",
        },
        columns=list(MACRO_COLUMNS),
    )


def test_save_is_idempotent_by_series_and_date() -> None:
    store = MarketStore(":memory:")
    dias = ["2024-01-31", "2024-02-29", "2024-03-31"]
    assert store.upsert_macro(_serie("ipca", dias, 0.5)) == 3
    store.upsert_macro(_serie("ipca", dias, 0.6))  # mesma chave, valor novo
    store.upsert_macro(_serie("poupanca", dias, 0.5))

    ipca = store.macro("ipca")
    assert len(ipca) == 3
    assert ipca.iloc[0] == pytest.approx(0.6)
    assert store.macro_series() == ["ipca", "poupanca"]


def test_unknown_series_is_empty_not_an_error() -> None:
    store = MarketStore(":memory:")
    assert store.macro("nao_existe").empty


def test_upsert_macro_rejects_missing_columns() -> None:
    store = MarketStore(":memory:")
    with pytest.raises(ValueError, match="colunas ausentes"):
        store.upsert_macro(pd.DataFrame({"series": ["ipca"], "date": ["2024-01-31"]}))


# --------------------------------------------------------------------------- Tesouro Direto
def _td_fixture() -> pd.DataFrame:
    """Dois papéis: um vencendo em ~10 anos (o alvo) e um curto que não deve ser escolhido.

    O papel longo dobra de preço no segundo dia; o curto cai pela metade. Se a curva pegar o
    papel errado — ou trocar de papel e comparar PUs diferentes — o retorno denuncia.
    """
    datas = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    longo, curto = pd.Timestamp("2034-01-01"), pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "kind": "Tesouro IPCA+",
            "date": list(datas) * 2,
            "maturity": [longo] * 3 + [curto] * 3,
            "pu": [100.0, 200.0, 200.0, 100.0, 50.0, 50.0],
        }
    )


def test_rolled_bond_picks_the_maturity_closest_to_target() -> None:
    out = rolled_bond_returns(_td_fixture(), "Tesouro IPCA+", target_years=10.0)
    assert list(out["value"].round(6)) == [1.0, 0.0]  # o longo dobrou, não o curto caiu


def test_rolling_to_a_new_bond_does_not_create_return() -> None:
    """A troca de papel não pode virar ganho: o retorno do dia é sempre do papel da véspera."""
    datas = pd.to_datetime(["2024-01-02", "2024-01-03"])
    velho, novo = pd.Timestamp("2034-01-01"), pd.Timestamp("2034-06-01")
    td = pd.DataFrame(
        {
            "kind": "Tesouro IPCA+",
            "date": list(datas) * 2,
            "maturity": [velho] * 2 + [novo] * 2,
            # O papel novo é dez vezes mais caro; só a variação do velho é retorno de verdade.
            "pu": [100.0, 110.0, 1000.0, 1100.0],
        }
    )
    out = rolled_bond_returns(td, "Tesouro IPCA+", target_years=10.0)
    assert out["value"].iloc[0] == pytest.approx(0.10)


def test_rolled_bond_ignores_other_kinds() -> None:
    td = _td_fixture().assign(kind="Tesouro Prefixado")
    assert rolled_bond_returns(td, "Tesouro IPCA+", 10.0).empty


def test_tesouro_curves_labels_every_series_in_the_catalog() -> None:
    td = pd.concat(
        [_td_fixture().assign(kind=kind) for kind, _ in TD_SERIES.values()], ignore_index=True
    )
    out = tesouro_curves(td)
    assert set(out["series"]) == set(TD_SERIES)
    assert set(out["unit"]) == {"daily_return"}
    assert set(out.columns) == set(MACRO_COLUMNS)


def test_tesouro_curves_respects_start_date() -> None:
    td = _td_fixture()
    out = tesouro_curves(td, start=dt.date(2024, 1, 4))
    ipca = out[out["series"] == "tesouro_ipca"]
    assert list(pd.to_datetime(ipca["date"]).dt.date) == [dt.date(2024, 1, 4)]
