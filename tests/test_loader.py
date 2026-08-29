import datetime as dt

import pandas as pd

from swing_quant.data.loader import INCREMENTAL_LOOKBACK_DAYS, to_long, update_prices
from swing_quant.data.store import PRICE_COLUMNS, MarketStore
from tests.conftest import make_prices


def _yf_wide(tickers: list[str], n: int = 5) -> pd.DataFrame:
    """Imita a saída de yf.download(group_by='ticker', auto_adjust=False)."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", name="Date")
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    df = pd.DataFrame(1.0, index=idx, columns=cols)
    for t in tickers:
        df[(t, "Volume")] = 1000
        df[(t, "High")] = 1.1
        df[(t, "Low")] = 0.9
    return df


def test_to_long_multiindex() -> None:
    raw = _yf_wide(["AAA.SA", "BBB.SA"])
    out = to_long(raw, ["AAA.SA", "BBB.SA"])
    assert list(out.columns) == list(PRICE_COLUMNS)
    assert len(out) == 10
    assert set(out["ticker"]) == {"AAA.SA", "BBB.SA"}
    assert out["source"].eq("yfinance").all()
    assert out["date"].dt.tz is None


def test_to_long_single_ticker_flat_columns() -> None:
    raw = _yf_wide(["AAA.SA"])["AAA.SA"]
    out = to_long(raw, ["AAA.SA"])
    assert len(out) == 5
    assert out["ticker"].eq("AAA.SA").all()


def test_to_long_missing_ticker_skipped() -> None:
    raw = _yf_wide(["AAA.SA"])
    out = to_long(raw, ["AAA.SA", "NAOEXISTE.SA"])
    assert set(out["ticker"]) == {"AAA.SA"}


def test_to_long_empty() -> None:
    assert to_long(pd.DataFrame(), ["X"]).empty


def test_update_prices_incremental(store: MarketStore) -> None:
    calls: list[tuple[list[str], dt.date]] = []

    def fake_dl(tickers, start, end):  # type: ignore[no-untyped-def]
        calls.append((list(tickers), pd.Timestamp(start).date()))
        return make_prices(list(tickers), pd.Timestamp(start).date(), pd.Timestamp(end).date())

    hist_start = dt.date(2024, 1, 2)
    # 1ª carga: tudo desde history_start
    r1 = update_prices(
        store, ["AAA3.SA", "BBB4.SA"], hist_start, as_of=dt.date(2024, 3, 29), downloader=fake_dl
    )
    assert r1.tickers_updated == 2 and r1.tickers_failed == []
    assert calls[0][1] == hist_start
    n1 = store.price_count()

    # 2ª carga: incremental, começa lookback dias antes da última data
    last = store.last_dates()["AAA3.SA"]
    r2 = update_prices(
        store, ["AAA3.SA", "BBB4.SA"], hist_start, as_of=dt.date(2024, 4, 30), downloader=fake_dl
    )
    assert calls[1][1] == last - dt.timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    assert r2.up_to_date == 0
    assert store.price_count() > n1

    # 3ª carga: já em dia -> nenhuma requisição
    before = len(calls)
    r3 = update_prices(store, ["AAA3.SA", "BBB4.SA"], hist_start, as_of=last, downloader=fake_dl)
    assert r3.up_to_date == 2 and len(calls) == before


def test_update_prices_full_ignores_existing(store: MarketStore) -> None:
    calls: list[dt.date] = []

    def fake_dl(tickers, start, end):  # type: ignore[no-untyped-def]
        calls.append(pd.Timestamp(start).date())
        return make_prices(list(tickers), pd.Timestamp(start).date(), pd.Timestamp(end).date())

    hist = dt.date(2024, 1, 2)
    update_prices(store, ["AAA3.SA"], hist, as_of=dt.date(2024, 2, 29), downloader=fake_dl)
    update_prices(
        store, ["AAA3.SA"], hist, as_of=dt.date(2024, 3, 29), full=True, downloader=fake_dl
    )
    assert calls == [hist, hist]


def test_update_prices_records_failures(store: MarketStore) -> None:
    def fake_dl(tickers, start, end):  # type: ignore[no-untyped-def]
        # devolve só o primeiro ticker; o segundo "falhou"
        return make_prices([tickers[0]], pd.Timestamp(start).date(), pd.Timestamp(end).date())

    r = update_prices(
        store,
        ["OK.SA", "FAIL.SA"],
        dt.date(2024, 1, 2),
        as_of=dt.date(2024, 1, 31),
        downloader=fake_dl,
    )
    assert r.tickers_failed == ["FAIL.SA"]
    assert r.tickers_updated == 1


def test_update_prices_batch_exception(store: MarketStore) -> None:
    def boom(tickers, start, end):  # type: ignore[no-untyped-def]
        raise ConnectionError("rede caiu")

    r = update_prices(store, ["A.SA", "B.SA"], dt.date(2024, 1, 2), downloader=boom)
    assert set(r.tickers_failed) == {"A.SA", "B.SA"}
    assert store.price_count() == 0


def test_update_prices_repairs_high_lt_low(store: MarketStore) -> None:
    def fake_dl(tickers, start, end):  # type: ignore[no-untyped-def]
        df = make_prices(list(tickers), pd.Timestamp(start).date(), pd.Timestamp(end).date())
        df.loc[3, "high"] = df.loc[3, "low"] / 2  # barra corrompida
        return df

    r = update_prices(
        store, ["AAA3.SA"], dt.date(2024, 1, 2), as_of=dt.date(2024, 1, 31), downloader=fake_dl
    )
    assert r.repaired_rows == 1
    out = store.get_prices(["AAA3.SA"])
    assert (out["high"] >= out["low"]).all()
