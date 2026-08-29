import datetime as dt

import pandas as pd
import pytest

from swing_quant.data.store import MarketStore


def test_upsert_and_get_prices(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    n = store.upsert_prices(sample_prices)
    assert n == len(sample_prices)
    out = store.get_prices(["AAA3.SA"])
    assert set(out["ticker"]) == {"AAA3.SA"}
    assert out["date"].is_monotonic_increasing
    assert store.price_count() == len(sample_prices)


def test_upsert_is_idempotent(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    store.upsert_prices(sample_prices)
    store.upsert_prices(sample_prices)
    assert store.price_count() == len(sample_prices)


def test_upsert_replaces_on_conflict(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    store.upsert_prices(sample_prices)
    row = sample_prices.iloc[[0]].copy()
    row["close"] = 999.0
    store.upsert_prices(row)
    out = store.get_prices(
        [row["ticker"].iloc[0]], start=row["date"].iloc[0], end=row["date"].iloc[0]
    )
    assert out["close"].iloc[0] == 999.0


def test_last_dates(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    store.upsert_prices(sample_prices)
    last = store.last_dates()
    assert last["AAA3.SA"] == sample_prices["date"].max().date()
    assert store.last_dates(["ZZZ"]) == {}
    assert store.last_dates([]) == {}


def test_missing_columns_raise(store: MarketStore) -> None:
    with pytest.raises(ValueError, match="colunas ausentes"):
        store.upsert_prices(pd.DataFrame({"ticker": ["X"], "date": ["2024-01-02"]}))


def test_get_prices_date_filter(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    store.upsert_prices(sample_prices)
    out = store.get_prices(start="2024-06-01", end="2024-06-30")
    assert out["date"].min() >= pd.Timestamp("2024-06-01")
    assert out["date"].max() <= pd.Timestamp("2024-06-30")


def test_universe_snapshots(store: MarketStore) -> None:
    m1 = pd.DataFrame({"ticker": ["PETR4", "VALE3"], "sector": ["Energia", "Mineração"]})
    m2 = pd.DataFrame({"ticker": ["PETR4", "ITUB4"], "sector": ["Energia", "Financeiro"]})
    store.upsert_universe("IBRX100", dt.date(2024, 1, 31), m1)
    store.upsert_universe("IBRX100", dt.date(2024, 6, 30), m2)

    assert store.universe_snapshots("IBRX100") == [dt.date(2024, 1, 31), dt.date(2024, 6, 30)]
    latest = store.universe_at("IBRX100")
    assert set(latest["ticker"]) == {"PETR4", "ITUB4"}
    # point-in-time: em março vale o snapshot de janeiro
    march = store.universe_at("IBRX100", dt.date(2024, 3, 15))
    assert set(march["ticker"]) == {"PETR4", "VALE3"}
    # antes do primeiro snapshot: vazio
    assert store.universe_at("IBRX100", dt.date(2023, 1, 1)).empty


def test_universe_without_sector(store: MarketStore) -> None:
    store.upsert_universe("SP500", dt.date(2024, 1, 31), pd.DataFrame({"ticker": ["AAPL"]}))
    out = store.universe_at("SP500")
    assert out["ticker"].tolist() == ["AAPL"]
    assert out["sector"].isna().all()


def test_corporate_events(store: MarketStore) -> None:
    ev = pd.DataFrame(
        {"ticker": ["PETR4.SA"], "date": ["2024-05-10"], "type": ["dividend"], "value": [0.5]}
    )
    assert store.upsert_corporate_events(ev) == 1
    assert store.upsert_corporate_events(ev) == 1  # idempotente
    n = store.con.execute("SELECT count(*) FROM corporate_events").fetchone()
    assert n is not None and n[0] == 1


def test_context_manager(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "sub" / "m.duckdb"
    with MarketStore(path) as s:
        assert s.price_count() == 0
    assert path.exists()


def test_repair_high_low(store: MarketStore, sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    df.loc[0, ["open", "high", "low", "close"]] = [36.0, 18.9, 36.0, 36.15]
    store.upsert_prices(df)
    assert store.repair_high_low() == 1
    assert store.repair_high_low() == 0  # idempotente
    fixed = store.get_prices([df.loc[0, "ticker"]], start=df.loc[0, "date"], end=df.loc[0, "date"])
    assert fixed["high"].iloc[0] == 36.15
    assert fixed["low"].iloc[0] == 18.9
    assert fixed["source"].iloc[0].endswith("+repair")
