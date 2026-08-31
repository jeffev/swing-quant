import math

import numpy as np
import pandas as pd
import pytest

from swing_quant.backtest.metrics import (
    blended_benchmark,
    cagr,
    compute_metrics,
    max_consecutive_losses,
    max_drawdown,
    rf_cagr,
    sharpe_ratio,
)


def _equity(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_max_drawdown_depth_and_duration() -> None:
    eq = _equity([100, 110, 99, 90, 95, 120, 118])
    depth, days = max_drawdown(eq)
    assert depth == pytest.approx(90 / 110 - 1)
    assert days == 3  # 99, 90, 95 abaixo do pico 110


def test_max_drawdown_monotonic() -> None:
    assert max_drawdown(_equity([1, 2, 3, 4])) == (0.0, 0)


def test_cagr_doubling_in_two_years() -> None:
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])
    eq = pd.Series([100.0, 200.0], index=idx)
    assert cagr(eq) == pytest.approx(math.sqrt(2) - 1, rel=1e-3)


def test_sharpe_constant_positive_returns_is_inf_or_nan_safe() -> None:
    r = pd.Series([0.001] * 50)
    s = sharpe_ratio(r)
    assert math.isnan(s) or s > 0  # desvio zero -> nan por convenção


def test_sharpe_sign() -> None:
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 1000))
    bad = pd.Series(rng.normal(-0.001, 0.01, 1000))
    assert sharpe_ratio(good) > 0 > sharpe_ratio(bad)


def test_max_consecutive_losses() -> None:
    assert max_consecutive_losses(pd.Series([1, -1, -1, 2, -1, -1, -1, 3])) == 3
    assert max_consecutive_losses(pd.Series([1, 2])) == 0


def test_compute_metrics_trade_stats() -> None:
    eq = _equity([100, 101, 103, 102, 105])
    trades = pd.DataFrame(
        {
            "pnl": [10.0, -5.0, 20.0, -5.0],
            "ret": [0.10, -0.05, 0.20, -0.05],
            "bars_held": [3, 2, 5, 1],
            "fees": [1.0, 1.0, 1.0, 1.0],
        }
    )
    m = compute_metrics(eq, trades, exposure=pd.Series([0.5] * 5, index=eq.index))
    assert m.n_trades == 4
    assert m.win_rate == 0.5
    assert m.profit_factor == pytest.approx(30 / 10)
    assert m.payoff == pytest.approx(0.15 / 0.05)
    assert m.expectancy_pct == pytest.approx(0.5 * 0.15 - 0.5 * 0.05)
    assert m.avg_hold_bars == pytest.approx(2.75)
    assert m.exposure_avg == 0.5
    assert m.fees_total == 4.0
    assert m.total_return == pytest.approx(0.05)


def test_compute_metrics_no_trades() -> None:
    eq = _equity([100, 100, 100])
    m = compute_metrics(eq, pd.DataFrame(columns=["pnl", "ret", "bars_held", "fees"]))
    assert m.n_trades == 0
    assert math.isnan(m.win_rate)
    assert m.max_drawdown == 0.0


def test_sharpe_against_a_daily_rate_series_not_zero() -> None:
    """ADR-020: uma carteira que rende menos que o CDI tem Sharpe negativo, não positivo."""
    idx = pd.date_range("2024-01-01", periods=504, freq="B")
    noise = np.random.default_rng(0).normal(0, 0.004, len(idx))
    rets = pd.Series(0.0002 + noise, index=idx)  # ~5%/ano
    cdi = pd.Series(0.0004, index=idx)  # ~10,5%/ano: o dobro

    assert sharpe_ratio(rets) > 0  # contra zero, parece boa
    assert sharpe_ratio(rets, cdi) < 0  # contra o CDI, destrói valor
    assert sharpe_ratio(rets, 0.105) < 0  # taxa anual escalar leva à mesma conclusão


def test_risk_free_against_itself_is_not_a_ratio() -> None:
    """Excesso identicamente zero é 0/0 — NaN, não um Sharpe qualquer vindo de ruído numérico."""
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    cdi = pd.Series(0.0004, index=idx)
    assert math.isnan(sharpe_ratio(cdi, cdi))


def test_blended_benchmark_matches_exposure() -> None:
    """Peso 0 = pura renda fixa; peso 1 = índice puro; no meio, uma mistura das duas."""
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    close = pd.Series(100.0 * (1.004 ** np.arange(len(idx))), index=idx)
    rf = pd.Series(0.0002, index=idx)

    only_rf = blended_benchmark(close, rf, 0.0, idx)
    only_eq = blended_benchmark(close, rf, 1.0, idx)
    half = blended_benchmark(close, rf, 0.5, idx)

    assert only_rf["cagr"] == pytest.approx(rf_cagr(rf, idx), rel=1e-6)
    assert only_eq["cagr"] == pytest.approx(cagr(close), rel=1e-6)
    assert only_rf["cagr"] < half["cagr"] < only_eq["cagr"]
    assert half["max_drawdown"] >= only_eq["max_drawdown"]
