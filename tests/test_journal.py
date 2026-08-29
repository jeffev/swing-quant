import datetime as dt

import pandas as pd
import pytest

from swing_quant.data.store import MarketStore
from swing_quant.journal.core import ExecutionRecord, Journal
from swing_quant.screener.core import ENTRY_COLUMNS, EXIT_COLUMNS, ScreenResult


def _result(as_of: str = "2026-08-27") -> ScreenResult:
    entries = pd.DataFrame(
        [
            {
                "ticker": "PETR4.SA",
                "strategy": "donchian",
                "ref_price": 37.5,
                "qty": 300,
                "notional": 11250.0,
                "stop_price": 35.2,
                "max_hold": 0,
                "score": 1.8,
                "atr": 1.1,
                "dollar_volume": 5e8,
            },
            {
                "ticker": "VALE3.SA",
                "strategy": "donchian",
                "ref_price": 60.0,
                "qty": 100,
                "notional": 6000.0,
                "stop_price": None,
                "max_hold": 5,
                "score": 1.2,
                "atr": 1.5,
                "dollar_volume": 8e8,
            },
        ],
        columns=ENTRY_COLUMNS,
    )
    exits = pd.DataFrame(
        [
            {
                "ticker": "ITUB4.SA",
                "strategy": "donchian",
                "reason": "signal",
                "ref_price": 30.0,
                "entry_date": dt.date(2026, 8, 1),
                "bars_held": 18,
                "qty": 200,
            }
        ],
        columns=EXIT_COLUMNS,
    )
    return ScreenResult(
        as_of=pd.Timestamp(as_of),
        market="b3",
        entries=entries,
        exits=exits,
        equity=100_000,
        cash=80_000,
        open_positions=1,
        slots=5,
        regime={"allow_entries": True},
    )


@pytest.fixture
def journal() -> Journal:
    return Journal(MarketStore(":memory:"))


def test_record_screen_is_idempotent(journal: Journal) -> None:
    ids1 = journal.record_screen(_result())
    ids2 = journal.record_screen(_result())
    assert len(ids1) == 3 and ids1 == ids2
    sig = journal.signals(as_of=dt.date(2026, 8, 27), market="b3")
    assert len(sig) == 3
    assert set(sig["side"]) == {"buy", "sell"}
    assert sig.loc[sig["ticker"] == "PETR4.SA", "stop_price"].iloc[0] == pytest.approx(35.2)
    assert pd.isna(sig.loc[sig["ticker"] == "VALE3.SA", "stop_price"].iloc[0])


def test_execution_lifecycle_and_pnl(journal: Journal) -> None:
    ids = journal.record_screen(_result())
    petr = int(journal.signals()[lambda d: d["ticker"] == "PETR4.SA"]["id"].iloc[0])
    assert petr in ids
    journal.record_execution(
        ExecutionRecord(
            petr, "buy", 37.6, 300, fees=1.0, executed_at=dt.datetime(2026, 8, 28, 10, 5)
        )
    )
    pos = journal.open_positions("b3")
    assert len(pos) == 1
    p = pos[0]
    assert p.ticker == "PETR4.SA" and p.qty == 300 and p.entry_date == dt.date(2026, 8, 28)
    assert p.stop_price == pytest.approx(35.2) and p.signal_id == petr
    assert journal.invested_at_cost("b3") == pytest.approx(300 * 37.6)
    assert journal.realized_pnl() == 0.0

    # venda parcial -> ainda aberta com 100; venda total -> fechada com P&L
    journal.record_execution(ExecutionRecord(petr, "sell", 39.0, 200, fees=1.0))
    assert journal.open_positions()[0].qty == 100
    journal.record_execution(ExecutionRecord(petr, "sell", 38.0, 100, fees=1.0))
    assert journal.open_positions() == []
    expected = (39.0 - 37.6) * 200 + (38.0 - 37.6) * 100 - 3.0
    assert journal.realized_pnl() == pytest.approx(expected)
    assert journal.equity_estimate(100_000) == pytest.approx(100_000 + expected)


def test_execution_validation(journal: Journal) -> None:
    with pytest.raises(KeyError):
        journal.record_execution(ExecutionRecord(999, "buy", 1.0, 1))
    ids = journal.record_screen(_result())
    with pytest.raises(ValueError):
        journal.record_execution(ExecutionRecord(ids[0], "short", 1.0, 1))


def test_signals_filter_by_market(journal: Journal) -> None:
    journal.record_screen(_result())
    assert journal.signals(market="us").empty
    assert len(journal.signals(market="b3")) == 3
