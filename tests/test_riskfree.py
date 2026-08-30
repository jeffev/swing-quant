"""Renda fixa de referência: gravação idempotente e acumulação por ano.

O download (BCB e yfinance) fica fora daqui — só a parte pura, sem rede.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swing_quant.data.riskfree import (
    RISK_FREE_COLUMNS,
    annual_returns,
    risk_free_daily,
    save_risk_free,
)
from swing_quant.data.store import MarketStore


def _serie(market: str, dias: list[str], taxa: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": market,
            "date": [dt.date.fromisoformat(d) for d in dias],
            "daily_return": taxa,
            "source": "teste",
        },
        columns=list(RISK_FREE_COLUMNS),
    )


def test_save_is_idempotent_by_market_and_date() -> None:
    store = MarketStore(":memory:")
    dias = ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert save_risk_free(store, _serie("b3", dias, 0.0004)) == 3
    save_risk_free(store, _serie("b3", dias, 0.0005))  # mesma chave, valor novo
    save_risk_free(store, _serie("us", dias, 0.0001))

    b3 = risk_free_daily(store, "b3")
    assert len(b3) == 3
    assert b3["daily_return"].unique().tolist() == [0.0005]  # sobrescreveu, não duplicou
    assert len(risk_free_daily(store, "us")) == 3
    assert save_risk_free(store, pd.DataFrame()) == 0
    store.close()


def test_annual_returns_compounds_within_the_year() -> None:
    daily = pd.concat(
        [
            _serie("b3", ["2023-12-28", "2023-12-29"], 0.01),
            _serie("b3", ["2024-01-02", "2024-01-03", "2024-01-04"], 0.01),
        ]
    )
    anual = annual_returns(daily[["date", "daily_return"]])
    assert anual[2023] == pytest.approx(1.01**2 - 1)
    assert anual[2024] == pytest.approx(1.01**3 - 1)
    assert annual_returns(pd.DataFrame()).empty
