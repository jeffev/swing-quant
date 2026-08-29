import numpy as np
import pandas as pd

from swing_quant.risk.regime import (
    RegimeConfig,
    build_regime,
    high_volatility,
    realized_volatility,
    trend_filter,
)


def _close(n: int = 1500, seed: int = 0, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.012, n)), index=idx)


def test_trend_filter_uptrend_vs_downtrend() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    up = pd.Series(np.linspace(100, 200, 300), index=idx)
    down = pd.Series(np.linspace(200, 100, 300), index=idx)
    assert trend_filter(up, 50).iloc[-1]
    assert not trend_filter(down, 50).iloc[-1]
    assert not trend_filter(up, 50).iloc[:49].any()  # aquecimento -> False


def test_realized_volatility_scale() -> None:
    c = _close()
    vol = realized_volatility(c, 20)
    assert vol.iloc[:20].isna().all()
    # ruído diário 1,2% -> ~19% a.a.; média deve ficar perto
    assert 0.15 < vol.dropna().mean() < 0.24


def test_high_volatility_flags_shock_without_lookahead() -> None:
    c = _close(1500)
    shocked = c.copy()
    rng = np.random.default_rng(1)
    # choque de vol nos últimos 60 dias
    shocked.iloc[-60:] = shocked.iloc[-61] * np.cumprod(1 + rng.normal(0, 0.05, 60))
    hv_base = high_volatility(c)
    hv_shock = high_volatility(shocked)
    assert hv_shock.iloc[-30:].mean() > 0.8
    # antes do choque, séries idênticas -> mesmas flags (sem look-ahead)
    pd.testing.assert_series_equal(hv_base.iloc[:-61], hv_shock.iloc[:-61])
    assert hv_base.mean() < 0.2  # ~10% dos dias acima do percentil 90


def test_build_regime_shapes_and_factor() -> None:
    c = _close()
    r = build_regime(c, RegimeConfig(high_vol_size_factor=0.5))
    assert r.allow_entries.index.equals(c.index)
    assert set(r.size_factor.unique()) <= {0.5, 1.0}
    assert (r.size_factor[r.high_vol] == 0.5).all()
    s = r.summary()
    assert 0 <= s["pct_days_high_vol"] <= 1 and 0 <= s["pct_days_trend_on"] <= 1


def test_build_regime_flags_disable_layers() -> None:
    c = _close()
    r = build_regime(c, RegimeConfig(use_trend=False, use_vol=True))
    assert r.allow_entries.all()
    assert (r.size_factor < 1).any()
    r2 = build_regime(c, RegimeConfig(use_trend=True, use_vol=False))
    assert (r2.size_factor == 1.0).all()
    assert not r2.allow_entries.all()
