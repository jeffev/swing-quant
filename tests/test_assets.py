"""Catálogo de proxies de classe de ativo: coerência interna e isolamento do universo de trade."""

from __future__ import annotations

from swing_quant.data.assets import (
    ASSET_CLASS_LABEL,
    ASSET_PROXIES,
    PROXY_BY_KEY,
    b3_index_symbols,
    index_name_of,
    macro_series_needed,
    proxies_for,
    proxy_tickers,
)
from swing_quant.data.macro import MACRO_CATALOG

_KINDS = {"ticker", "fx", "b3_index", "macro", "riskfree"}


def test_keys_are_unique() -> None:
    assert len(PROXY_BY_KEY) == len(ASSET_PROXIES)


def test_every_proxy_is_buildable() -> None:
    """Cada proxy precisa dizer de onde vem sua curva — sem isso o estudo não consegue montá-la."""
    for p in ASSET_PROXIES:
        assert p.kind in _KINDS, p.key
        assert p.market in {"b3", "us"}, p.key
        assert p.asset_class in ASSET_CLASS_LABEL, p.key
        if p.kind in {"ticker", "b3_index"}:
            assert p.symbols, p.key
        if p.kind == "fx":
            assert len(p.symbols) == 2, p.key  # ativo em moeda estrangeira + câmbio
        if p.kind == "macro":
            assert p.series in MACRO_CATALOG, p.key
        if p.kind == "riskfree":
            assert not p.symbols and not p.series, p.key


def test_b3_indices_are_kept_out_of_the_yfinance_download() -> None:
    """Um índice da B3 não existe no yfinance; pedi-lo lá só geraria falha de download."""
    assert "^IFIX" in b3_index_symbols("b3")
    assert not set(proxy_tickers()) & set(b3_index_symbols())
    assert index_name_of("^IFIX") == "IFIX"


def test_proxy_tickers_are_deduplicated_across_proxies() -> None:
    """USDBRL aparece em quatro proxies brasileiros; baixar quatro vezes seria desperdício."""
    tickers = proxy_tickers("b3")
    assert len(tickers) == len(set(tickers))
    assert "USDBRL=X" in tickers


def test_market_filter_splits_the_catalog() -> None:
    br, us = proxies_for("b3"), proxies_for("us")
    assert len(br) + len(us) == len(ASSET_PROXIES)
    assert not set(proxy_tickers("us")) & {"USDBRL=X"}


def test_macro_series_needed_matches_the_macro_catalog() -> None:
    assert set(macro_series_needed()) <= set(MACRO_CATALOG)
    assert "ivgr" in macro_series_needed("b3")


def test_proxies_never_reach_the_trading_universe() -> None:
    """Os proxies vivem na tabela `prices`, mas quem define candidatos a trade é `universe`.

    O teste é do contrato: nenhum símbolo de proxy pode ser um ticker do IBrX/S&P — se um FII
    ou o dólar entrasse no universo, o screener passaria a gerar sinal para ele.
    """
    from swing_quant.data.store import MarketStore

    store = MarketStore(":memory:")
    assert store.universe_at("IBRX100").empty
    for symbol in proxy_tickers():
        assert not symbol.endswith("3.SA"), symbol  # ações ordinárias da B3
        assert not symbol.endswith("4.SA"), symbol  # preferenciais
