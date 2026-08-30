import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swing_quant.backtest.panel import adjust_ohlc, build_panel
from swing_quant.strategies import REGISTRY, RSI2, make_strategy, validate_signals
from swing_quant.strategies.base import SIGNAL_COLUMNS
from tests.conftest import make_prices


def _ohlcv(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 50 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )


def test_registry_and_factory() -> None:
    assert "rsi2" in REGISTRY
    s = make_strategy("rsi2", {"rsi_entry": 5, "enabled": True})
    assert isinstance(s, RSI2)
    assert s.params.rsi_entry == 5
    with pytest.raises(KeyError, match="desconhecida"):
        make_strategy("nope")


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        RSI2({"rsi_entry": 80})  # < 50 obrigatório


def test_generate_shape_and_types() -> None:
    df = _ohlcv()
    sig = validate_signals(RSI2().generate(df), df.index)
    assert list(sig.columns) == list(SIGNAL_COLUMNS)
    assert sig["entry"].dtype == bool and sig["exit"].dtype == bool
    assert (sig["max_hold"] == 5).all()
    assert sig["stop"].isna().all()  # Connors original sem stop


def test_entry_requires_trend_and_oversold() -> None:
    df = _ohlcv(400)
    strat = RSI2({"trend_sma": 50, "rsi_entry": 10})
    sig = strat.generate(df)
    from swing_quant.indicators import rsi, sma

    trend_ok = df["close"] > sma(df["close"], 50)
    oversold = rsi(df["close"], 2) < 10
    expected = (trend_ok & oversold).fillna(False)
    assert (sig["entry"] == expected).all()
    assert sig["entry"].iloc[:50].sum() == 0  # warmup


def test_stop_atr_option() -> None:
    df = _ohlcv()
    sig = RSI2({"stop_atr": 2.0}).generate(df)
    valid = sig["stop"].dropna()
    assert len(valid) > 0
    assert (valid < df.loc[valid.index, "close"]).all()


def test_grid_iterates_all_combos() -> None:
    combos = list(RSI2().grid())
    assert len(combos) == 9
    assert {c.params.rsi_entry for c in combos} == {5.0, 10.0, 15.0}


def test_no_lookahead_in_strategy() -> None:
    df = _ohlcv(400)
    df2 = df.copy()
    df2.iloc[300:, :4] *= 2.0
    a = RSI2().generate(df).iloc[:300]
    b = RSI2().generate(df2).iloc[:300]
    pd.testing.assert_frame_equal(a, b)


def test_adjust_ohlc_applies_factor() -> None:
    long = make_prices(["AAA3.SA"], dt.date(2024, 1, 2), dt.date(2024, 1, 31))
    long["adj_close"] = long["close"] * 0.5  # provento de 50% "ajustado"
    adj = adjust_ohlc(long)
    assert np.allclose(adj["close"], long["close"].to_numpy() * 0.5)
    assert np.allclose(adj["open"], long["open"].to_numpy() * 0.5)
    assert np.allclose(adj["raw_close"], long["close"].to_numpy())


def test_build_panel_alignment_and_min_rows() -> None:
    long = make_prices(["AAA3.SA", "BBB4.SA"], dt.date(2022, 1, 3), dt.date(2024, 12, 30))
    short = make_prices(["CCC3.SA"], dt.date(2024, 11, 1), dt.date(2024, 12, 30))
    panel = build_panel(pd.concat([long, short]), RSI2())
    assert panel.tickers == ["AAA3.SA", "BBB4.SA"]  # CCC3 descartado (pouco histórico)
    assert panel.close.shape == (len(panel.dates), 2)
    assert panel.entry.dtypes.eq(bool).all()
    sl = panel.slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))
    assert sl.dates.min() >= pd.Timestamp("2024-01-01")
    assert sl.dates.max() <= pd.Timestamp("2024-06-30")


# ---------------------------------------------------------------- Donchian / DropsIBS
def test_registry_has_every_strategy() -> None:
    assert set(REGISTRY) == {"rsi2", "donchian", "dip", "drops_ibs", "momentum", "pullback"}


def test_donchian_breakout_logic() -> None:
    from swing_quant.indicators import donchian_high
    from swing_quant.strategies import Donchian

    df = _ohlcv(400)
    strat = Donchian(
        {
            "entry_lookback": 20,
            "exit_lookback": 10,
            "volume_mult": 0.0,
            "fast_sma": 20,
            "slow_sma": 50,
        }
    )
    sig = validate_signals(strat.generate(df), df.index)
    ch = donchian_high(df["high"], 20)
    # toda entrada exige close acima do canal dos 20 dias anteriores
    assert (df.loc[sig["entry"], "close"] > ch[sig["entry"]]).all()
    assert sig["stop"].notna().sum() > 0
    assert (sig["max_hold"] == 0).all()
    assert sig["entry"].iloc[:50].sum() == 0


def test_donchian_volume_filter_reduces_entries() -> None:
    from swing_quant.strategies import Donchian

    df = _ohlcv(400)
    rng = np.random.default_rng(9)
    df["volume"] = rng.integers(500_000, 2_000_000, len(df)).astype(float)
    base = {"fast_sma": 20, "slow_sma": 50}
    no_filter = Donchian({**base, "volume_mult": 0.0}).generate(df)["entry"].sum()
    with_filter = Donchian({**base, "volume_mult": 1.5}).generate(df)["entry"].sum()
    assert with_filter < no_filter


def test_drops_ibs_entry_and_exit() -> None:
    from swing_quant.strategies import DropsIBS

    idx = pd.date_range("2023-01-02", periods=260, freq="B")
    close = pd.Series(np.linspace(100, 130, 260), index=idx)  # tendência de alta (close > SMA)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=idx,
    )
    # 3 quedas consecutivas no fim com fechamento na mínima (IBS = 0)
    for k, v in zip(range(-3, 0), [125.0, 124.0, 123.0], strict=True):
        df.iloc[k, df.columns.get_loc("close")] = v
        df.iloc[k, df.columns.get_loc("low")] = v
        df.iloc[k, df.columns.get_loc("high")] = v + 2
    sig = DropsIBS({"min_consecutive_drops": 3, "ibs_max": 0.2, "trend_sma": 200}).generate(df)
    assert bool(sig["entry"].iloc[-1]) is True
    assert bool(sig["entry"].iloc[-2]) is False  # só 2 quedas até ali
    assert sig["score"].iloc[-1] == pytest.approx(3.0)
    assert (sig["max_hold"] == 3).all()
    # saída: close > máxima do dia anterior
    assert bool(sig["exit"].iloc[100]) == bool(df["close"].iloc[100] > df["high"].iloc[99])


# ---------------------------------------------------------------- Momentum / Pullback
def test_momentum_score_and_signals() -> None:
    from swing_quant.strategies import Momentum

    df = _ohlcv(500)
    strat = Momentum({"lookback": 126, "skip": 21, "trend_sma": 50, "exit_sma": 50})
    sig = validate_signals(strat.generate(df), df.index)
    close = df["close"]
    mom = close.shift(21) / close.shift(147) - 1.0
    valid = sig["score"].notna()
    assert np.allclose(sig.loc[valid, "score"], mom[valid])
    # entrada exige momentum positivo; saída inclui momentum negativo
    assert (mom[sig["entry"]] > 0).all()
    assert (sig["exit"] | ~(mom < 0).fillna(False)).all()
    assert (sig["max_hold"] == 63).all() and sig["stop"].notna().sum() > 0
    assert sig["entry"].iloc[: strat.warmup - 2].sum() == 0


def test_pullback_entry_requires_alignment_and_touch() -> None:
    from swing_quant.indicators import sma
    from swing_quant.strategies import Pullback

    df = _ohlcv(600, seed=8)
    strat = Pullback({"fast_sma": 10, "mid_sma": 30, "slow_sma": 100})
    sig = validate_signals(strat.generate(df), df.index)
    fast, mid, slow = sma(df["close"], 10), sma(df["close"], 30), sma(df["close"], 100)
    e = sig["entry"]
    if e.any():
        assert (fast[e] > mid[e]).all() and (mid[e] > slow[e]).all()
        assert (df.loc[e, "low"] <= fast[e]).all() and (
            df.loc[e, "close"] > df.loc[e, "open"]
        ).all()
    # stop sempre abaixo da mínima do candle
    valid = sig["stop"].notna()
    assert (sig.loc[valid, "stop"] <= df.loc[valid, "low"]).all()
    assert (sig["max_hold"] == 10).all()


def test_new_strategies_no_lookahead() -> None:
    from swing_quant.strategies import Dip, Momentum, Pullback

    df = _ohlcv(600, seed=4)
    df2 = df.copy()
    df2.iloc[450:, :4] *= 1.5
    for strat in (Momentum({"trend_sma": 50}), Pullback({"slow_sma": 100}), Dip({"lookback": 40})):
        a = strat.generate(df).iloc[:450]
        b = strat.generate(df2).iloc[:450]
        pd.testing.assert_frame_equal(a, b)


def test_dip_entry_after_drop_from_the_previous_high() -> None:
    """Sobe até 14 e cai: só entra quando o fechamento está `drop_pct` abaixo do topo anterior."""
    from swing_quant.strategies import Dip

    subida = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]
    close = [*subida, 12.0, 10.0, 10.5]
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [1_000_000.0] * len(close),
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="B"),
    )
    params = {"drop_pct": 0.20, "lookback": 10, "target_pct": 0.10, "stop_atr": None}
    sig = Dip(params).generate(df)

    # topo dos 10 pregões anteriores = 14,5; quedas: -17,2% (i=10), -31,0% (i=11), -27,6% (i=12)
    assert sig["entry"].tolist() == [False] * 11 + [True, True]
    assert sig["target"].iloc[11] == pytest.approx(11.0)  # 10 * 1,10
    assert sig["score"].iloc[11] == pytest.approx(1 - 10.0 / 14.5)
    assert sig["score"].iloc[11] > sig["score"].iloc[12]  # queda mais funda entra primeiro
    assert sig["stop"].isna().all()  # stop_atr=None desliga o stop


def test_dip_trend_filter_only_buys_above_the_average() -> None:
    from swing_quant.strategies import Dip

    df = _ohlcv(400, seed=11)
    sem = Dip({"drop_pct": 0.05, "lookback": 20, "trend_sma": 0}).generate(df)
    com = Dip({"drop_pct": 0.05, "lookback": 20, "trend_sma": 200}).generate(df)
    assert com["entry"].sum() <= sem["entry"].sum()
    # o filtro só remove sinais, nunca cria
    assert not (com["entry"] & ~sem["entry"]).any()
