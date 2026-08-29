import numpy as np
import pandas as pd
import pytest

from swing_quant.indicators import (
    atr,
    bollinger,
    consecutive_down_days,
    dollar_volume,
    donchian_high,
    donchian_low,
    ibs,
    rsi,
    sma,
    true_range,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_sma_basic() -> None:
    s = pd.Series([1, 2, 3, 4, 5], index=_idx(5), dtype=float)
    out = sma(s, 3)
    assert out.isna().sum() == 2
    assert out.tolist()[2:] == [2.0, 3.0, 4.0]


def test_rsi_all_up_is_100_and_all_down_is_0() -> None:
    up = pd.Series(np.arange(1, 31, dtype=float), index=_idx(30))
    down = pd.Series(np.arange(30, 0, -1, dtype=float), index=_idx(30))
    assert rsi(up, 2).dropna().eq(100.0).all()
    assert rsi(down, 2).dropna().eq(0.0).all()


def test_rsi_range_and_warmup() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=_idx(200))
    r = rsi(s, 14)
    assert r.iloc[:14].isna().all()
    valid = r.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_known_value() -> None:
    # Sequência clássica: 2 subidas de 1 e 1 queda de 1 com RSI(2) de Wilder.
    s = pd.Series([10, 11, 12, 11, 12, 13], index=_idx(6), dtype=float)
    r = rsi(s, 2)
    # dia 3 (índice 2): ganhos médios = 1, perdas = 0 -> 100
    assert r.iloc[2] == 100.0
    # dia 4: avg_gain = 0.5*1 + 0.5*0 = 0.5; avg_loss = 0.5 -> RS=1 -> 50
    assert r.iloc[3] == pytest.approx(50.0)


def test_true_range_and_atr_constant_range() -> None:
    n = 40
    close = pd.Series(np.full(n, 100.0), index=_idx(n))
    high = close + 1.0
    low = close - 1.0
    tr = true_range(high, low, close)
    assert (tr == 2.0).all()
    a = atr(high, low, close, 14)
    assert a.iloc[:13].isna().all()
    assert a.dropna().round(9).eq(2.0).all()


def test_atr_uses_gaps() -> None:
    idx = _idx(3)
    close = pd.Series([100.0, 110.0, 110.0], index=idx)
    high = pd.Series([101.0, 111.0, 111.0], index=idx)
    low = pd.Series([99.0, 109.0, 109.0], index=idx)
    tr = true_range(high, low, close)
    # dia 2: max(111-109, |111-100|, |109-100|) = 11
    assert tr.iloc[1] == 11.0


def test_donchian_excludes_current_bar() -> None:
    idx = _idx(6)
    high = pd.Series([1, 2, 3, 10, 4, 5], index=idx, dtype=float)
    low = pd.Series([1, 2, 3, 0.5, 4, 5], index=idx, dtype=float)
    dh = donchian_high(high, 3)
    dl = donchian_low(low, 3)
    assert dh.iloc[3] == 3.0  # máxima de [1,2,3], não inclui o 10 do próprio dia
    assert dh.iloc[4] == 10.0
    assert dl.iloc[4] == 0.5


def test_bollinger_shape() -> None:
    s = pd.Series(np.linspace(10, 20, 50), index=_idx(50))
    bb = bollinger(s, 20, 2.0)
    assert list(bb.columns) == ["mid", "upper", "lower"]
    valid = bb.dropna()
    assert (valid["upper"] >= valid["mid"]).all() and (valid["mid"] >= valid["lower"]).all()


def test_ibs() -> None:
    idx = _idx(3)
    high = pd.Series([10, 10, 10], index=idx, dtype=float)
    low = pd.Series([8, 8, 10], index=idx, dtype=float)
    close = pd.Series([8, 10, 10], index=idx, dtype=float)
    out = ibs(high, low, close)
    assert out.tolist() == [0.0, 1.0, 0.5]


def test_consecutive_down_days() -> None:
    s = pd.Series([5, 4, 3, 3, 2, 6, 5], index=_idx(7), dtype=float)
    assert consecutive_down_days(s).tolist() == [0, 1, 2, 0, 1, 0, 1]


def test_dollar_volume() -> None:
    idx = _idx(3)
    close = pd.Series([10, 10, 10], index=idx, dtype=float)
    vol = pd.Series([100, 200, 300], index=idx, dtype=float)
    dv = dollar_volume(close, vol, 2)
    assert dv.tolist()[1:] == [1500.0, 2500.0]


def test_no_lookahead() -> None:
    """Alterar valores futuros não pode mudar indicadores passados."""
    rng = np.random.default_rng(1)
    s = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=_idx(300))
    s2 = s.copy()
    s2.iloc[200:] *= 3.0
    for f in (lambda x: sma(x, 20), lambda x: rsi(x, 2), lambda x: bollinger(x, 20)["lower"]):
        pd.testing.assert_series_equal(f(s).iloc[:200], f(s2).iloc[:200])
