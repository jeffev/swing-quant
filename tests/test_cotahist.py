import datetime as dt

import pandas as pd
import pytest

from swing_quant.data.cotahist import compare_with_store, download_cotahist, parse_cotahist
from swing_quant.data.store import MarketStore
from tests.conftest import make_prices


def _rec(date: str, ticker: str, close: float, codbdi: str = "02", tpmerc: str = "010") -> str:
    """Monta um registro tipo 01 com largura fixa de 245 caracteres."""
    price = f"{round(close * 100):013d}"
    line = (
        "01"
        + date
        + codbdi
        + f"{ticker:<12}"
        + tpmerc
        + " " * 29  # NOMRES(12) ESPECI(10) PRAZOT(3) MODREF(4)
        + price  # PREABE 57-69
        + price  # PREMAX
        + price  # PREMIN
        + price  # PREMED
        + price  # PREULT 109-121
        + price  # PREOFC
        + price  # PREOFV
        + "00001"  # TOTNEG 148-152
        + f"{1000:018d}"  # QUATOT 153-170
        + f"{int(close * 1000 * 100):018d}"  # VOLTOT 171-188
    )
    return line.ljust(245)


def test_parse_cotahist_filters_and_scales() -> None:
    text = "\n".join(
        [
            "00COTAHIST.2024BOVESPA 20241230".ljust(245),  # header
            _rec("20240102", "PETR4", 37.78),
            _rec("20240102", "PETR4F", 37.80, codbdi="96"),  # fracionário → fora
            _rec("20240102", "PETRA100", 1.23, tpmerc="070"),  # opção → fora
            _rec("20240103", "PETR4", 38.96),
            "99COTAHIST.2024BOVESPA".ljust(245),  # trailer
        ]
    )
    df = parse_cotahist(text)
    assert df["ticker"].tolist() == ["PETR4", "PETR4"]
    assert df["close"].tolist() == [37.78, 38.96]
    assert df["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["quantity"].tolist() == [1000, 1000]


def test_parse_cotahist_empty() -> None:
    assert parse_cotahist("").empty


def test_compare_with_store_statuses(store: MarketStore) -> None:
    prices = make_prices(["AAA3.SA", "BBB4.SA"], dt.date(2024, 1, 2), dt.date(2024, 3, 28))
    store.upsert_prices(prices)

    # COTAHIST sintético: AAA3 igual ao store; BBB4 com preços 2x (split) exceto uma data errada
    aaa = prices[prices["ticker"] == "AAA3.SA"]
    bbb = prices[prices["ticker"] == "BBB4.SA"]
    cot = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA3", "date": aaa["date"], "close": aaa["close"]}),
            pd.DataFrame({"ticker": "BBB4", "date": bbb["date"], "close": bbb["close"] * 2}),
        ]
    )
    cot.loc[cot.index[-1], "close"] = cot["close"].iloc[-1] * 1.37  # divergência genuína
    cot = cot.reset_index(drop=True)

    out = compare_with_store(store, cot, ["AAA3.SA", "BBB4.SA", "ZZZ9.SA"], n_dates=100)
    by = out.groupby("ticker")["status"].value_counts()
    assert by.loc[("AAA3.SA", "ok")] == len(aaa)
    assert by.loc[("BBB4.SA", "adjusted_split_or_bonus")] >= len(bbb) - 1
    assert by.loc[("ZZZ9.SA", "missing_in_cotahist")] == 1
    assert out["status"].isin(["mismatch"]).sum() <= 1


@pytest.mark.network
@pytest.mark.slow
def test_download_cotahist_live() -> None:
    df = download_cotahist(2024)
    assert len(df) > 50_000
    petr = df[(df["ticker"] == "PETR4") & (df["date"] == "2024-01-02")]
    assert len(petr) == 1
    assert abs(petr["close"].iloc[0] - 37.78) < 0.05


def test_compare_detects_constant_bonus_ratio(store: MarketStore) -> None:
    prices = make_prices(["CCC3.SA"], dt.date(2024, 1, 2), dt.date(2024, 2, 29))
    store.upsert_prices(prices)
    # bonificação de 10%: yfinance ajusta (÷1.1), COTAHIST não -> razão constante 1/1.1
    cot = pd.DataFrame({"ticker": "CCC3", "date": prices["date"], "close": prices["close"] * 1.1})
    out = compare_with_store(store, cot, ["CCC3.SA"], n_dates=10)
    assert set(out["status"]) == {"adjusted_split_or_bonus"}


def test_compare_detects_two_adjustment_regimes(store: MarketStore) -> None:
    prices = make_prices(["DDD3.SA"], dt.date(2024, 1, 2), dt.date(2024, 6, 28))
    store.upsert_prices(prices)
    # duas bonificações no ano: fator 1.133 até março, 1.03 depois
    cot = pd.DataFrame({"ticker": "DDD3", "date": prices["date"], "close": prices["close"]})
    before = cot["date"] < "2024-03-15"
    cot.loc[before, "close"] *= 1.133
    cot.loc[~before, "close"] *= 1.03
    out = compare_with_store(store, cot, ["DDD3.SA"], n_dates=40)
    assert set(out["status"]) == {"adjusted_split_or_bonus"}
