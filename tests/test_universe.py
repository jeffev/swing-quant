import pytest

from swing_quant.data.universe import (
    fetch_b3_index,
    fetch_sp500,
    from_yf_symbol,
    parse_b3_portfolio,
    parse_sp500_html,
    to_yf_symbol,
)


def test_symbol_roundtrip() -> None:
    assert to_yf_symbol("PETR4", "b3") == "PETR4.SA"
    assert to_yf_symbol("PETR4.SA", "b3") == "PETR4.SA"
    assert to_yf_symbol("BRK.B", "us") == "BRK-B"
    assert from_yf_symbol("PETR4.SA", "b3") == "PETR4"
    assert from_yf_symbol("BRK-B", "us") == "BRK.B"


def test_parse_b3_portfolio_filters_subtotals() -> None:
    payload = {
        "results": [
            {"cod": "PETR4", "asset": "PETROBRAS", "segment": "Petróleo", "part": "7,123"},
            {"cod": "VALE3", "asset": "VALE", "segment": "Mineração", "part": "10,5"},
            {"cod": "", "asset": "Subtotal Petróleo", "segment": "Petróleo", "part": "7,123"},
            {"cod": "PETR4", "asset": "dup", "segment": "Petróleo", "part": "7,123"},
        ]
    }
    df = parse_b3_portfolio(payload)
    assert df["ticker"].tolist() == ["PETR4", "VALE3"]
    assert df["sector"].tolist() == ["Petróleo", "Mineração"]
    assert df["weight"].round(3).tolist() == [7.123, 10.5]


def test_parse_b3_portfolio_empty() -> None:
    assert parse_b3_portfolio({}).empty
    assert parse_b3_portfolio({"results": []}).empty


def test_parse_sp500_html() -> None:
    html = """
    <table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
    <tr><td>AAPL</td><td>Apple</td><td>Information Technology</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td></tr>
    <tr><td>AAPL</td><td>Apple dup</td><td>Information Technology</td></tr>
    </table>
    """
    df = parse_sp500_html(html)
    assert df["ticker"].tolist() == ["AAPL", "BRK.B"]
    assert df["sector"].tolist() == ["Information Technology", "Financials"]


@pytest.mark.network
def test_fetch_b3_index_live() -> None:
    df = fetch_b3_index("IBRX100")
    assert 90 <= len(df) <= 110
    assert "PETR4" in set(df["ticker"])


@pytest.mark.network
def test_fetch_sp500_live() -> None:
    df = fetch_sp500()
    assert len(df) >= 490
    assert "AAPL" in set(df["ticker"])
