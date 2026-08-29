import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swing_quant.backtest.engine import Backtester, CostModel, RiskModel
from swing_quant.backtest.protocol import ApprovalThresholds, run_protocol
from swing_quant.backtest.validation import (
    Window,
    block_bootstrap_drawdown,
    bootstrap_sharpe,
    cost_sensitivity,
    default_panel_factory,
    grid_search,
    monte_carlo_drawdown,
    plateau_ratio,
    select_best,
    time_split,
    walk_forward,
)
from swing_quant.strategies import RSI2
from tests.conftest import make_prices

RISK = RiskModel(initial_capital=100_000, board_lot=1, max_positions=4)
COSTS = CostModel(0.0, 0.0003, 0.001)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(
        ["AAA3.SA", "BBB4.SA", "CCC3.SA", "DDD4.SA"],
        dt.date(2014, 1, 2),
        dt.date(2020, 12, 30),
        seed=11,
    )


def test_time_split_fractions() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    s = time_split(dates, (0.6, 0.2, 0.2))
    assert s.train.start == dates[0] and s.train.end == dates[59]
    assert s.val.start == dates[60] and s.val.end == dates[79]
    assert s.test.start == dates[80] and s.test.end == dates[99]
    with pytest.raises(ValueError):
        time_split(dates, (0.5, 0.5, 0.5))


def test_grid_search_and_select(prices: pd.DataFrame) -> None:
    grid = {"rsi_entry": [10.0, 15.0], "exit_sma": [3, 5]}
    g = grid_search(RSI2(), grid, default_panel_factory(prices), Backtester(COSTS, RISK))
    assert len(g) == 4
    assert {"rsi_entry", "exit_sma", "sharpe", "n_trades"} <= set(g.columns)
    best = select_best(g, min_trades=1)
    assert best["sharpe"] == g["sharpe"].max()


def test_plateau_ratio_flat_vs_spike() -> None:
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    rows = [{"a": a, "b": b, "sharpe": 1.0} for a in grid["a"] for b in grid["b"]]
    flat = pd.DataFrame(rows)
    assert plateau_ratio(flat, {"a": 2, "b": 10}, grid) == pytest.approx(1.0)
    spike = flat.copy()
    spike.loc[(spike["a"] == 2) & (spike["b"] == 10), "sharpe"] = 5.0
    assert plateau_ratio(spike, {"a": 2, "b": 10}, grid) == pytest.approx(0.2)
    # ótimo na borda: só vizinhos existentes contam
    assert plateau_ratio(flat, {"a": 1, "b": 10}, grid) == pytest.approx(1.0)


def test_walk_forward_windows_and_chaining(prices: pd.DataFrame) -> None:
    grid = {"rsi_entry": [10.0, 15.0]}
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    wf = walk_forward(
        RSI2(),
        grid,
        default_panel_factory(prices),
        Backtester(COSTS, RISK),
        dates,
        train_years=2,
        test_years=1,
        min_trades=1,
    )
    # 2014-2020 com treino 2a / teste 1a -> janelas de teste 2016..2020 (5)
    assert len(wf.windows) == 5
    assert wf.oos_equity.index.is_monotonic_increasing
    assert wf.oos_equity.iloc[0] == pytest.approx(1.0)
    assert not wf.oos_equity.index.duplicated().any()
    assert {"train", "test", "rsi_entry", "sharpe_is", "sharpe_oos"} <= set(wf.windows.columns)


def test_monte_carlo_drawdown_properties() -> None:
    pnl = pd.Series([100.0, -50.0] * 100)
    mc = monte_carlo_drawdown(pnl, 10_000.0, runs=200, seed=1, ruin_level=0.15)
    assert -1 <= mc["mdd_p99"] <= mc["mdd_p95"] <= mc["mdd_p50"] <= 0
    assert 0 <= mc["prob_dd_gt_ruin"] <= 1
    empty = monte_carlo_drawdown(pd.Series(dtype=float), 10_000.0)
    assert np.isnan(empty["mdd_p95"])


def test_block_bootstrap_drawdown_properties() -> None:
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.0004, 0.008, 2000))
    dd = block_bootstrap_drawdown(rets, runs=300, block=20, seed=1, ruin_level=0.15)
    assert -1 < dd["mdd_p99"] <= dd["mdd_p95"] <= dd["mdd_p50"] < 0
    assert 0 <= dd["prob_dd_gt_ruin"] <= 1
    # série curta demais para os blocos -> sem veredito
    assert np.isnan(block_bootstrap_drawdown(pd.Series([0.01] * 10), block=20)["mdd_p95"])


def test_block_bootstrap_drawdown_grows_with_horizon() -> None:
    """Motivo de o gate fixar o horizonte (ADR-017): MDD cresce com o comprimento do caminho."""
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0.0004, 0.008, 3000))
    p95 = [
        block_bootstrap_drawdown(rets, runs=300, seed=2, horizon=h)["mdd_p95"]
        for h in (126, 252, 1000, None)
    ]
    assert p95 == sorted(p95, reverse=True)  # cada horizonte maior é mais negativo
    # horizonte maior que a série é truncado no tamanho dela
    assert block_bootstrap_drawdown(rets, runs=100, seed=2, horizon=99_999)[
        "mdd_p95"
    ] == pytest.approx(block_bootstrap_drawdown(rets, runs=100, seed=2)["mdd_p95"])


def test_block_bootstrap_matches_realized_1y_drawdowns() -> None:
    """Calibração: no mesmo horizonte, o simulado deve ficar perto do realizado."""
    from swing_quant.backtest.metrics import rolling_drawdowns

    rng = np.random.default_rng(9)
    rets = pd.Series(
        rng.normal(0.0005, 0.009, 2600),
        index=pd.date_range("2012-01-02", periods=2600, freq="B"),
    )
    equity = (1 + rets).cumprod()
    realized = rolling_drawdowns(equity, window=252)
    sim = block_bootstrap_drawdown(rets, runs=500, seed=3, horizon=252)
    assert realized["n"] > 50
    assert sim["mdd_p50"] == pytest.approx(realized["p50"], abs=0.04)
    assert sim["mdd_p95"] <= realized["p50"]  # a cauda simulada é pior que a mediana realizada


def test_block_bootstrap_is_scale_free() -> None:
    """Ao contrário do MC sobre P&L nominal, o resultado não depende do tamanho do capital."""
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0005, 0.01, 1500))
    a = block_bootstrap_drawdown(rets, runs=200, seed=5)
    b = block_bootstrap_drawdown(rets, runs=200, seed=5)
    assert a["mdd_p95"] == pytest.approx(b["mdd_p95"])  # determinístico pela seed
    # o MC por trades escala com o capital inicial; o bootstrap diário, não
    pnl = pd.Series([500.0, -300.0] * 150)
    mc_small = monte_carlo_drawdown(pnl, 10_000.0, runs=200, seed=5)
    mc_big = monte_carlo_drawdown(pnl, 1_000_000.0, runs=200, seed=5)
    assert mc_small["mdd_p95"] < mc_big["mdd_p95"]


def test_bootstrap_sharpe_ci() -> None:
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 1500))
    b = bootstrap_sharpe(good, runs=300, seed=0)
    assert b["sharpe_lo"] > 0 and b["sharpe_hi"] > b["sharpe_lo"]
    assert b["p_sharpe_le_0"] < 0.05
    short = bootstrap_sharpe(pd.Series([0.01] * 10))
    assert np.isnan(short["sharpe_lo"])


def test_cost_sensitivity_monotonic(prices: pd.DataFrame) -> None:
    panel = default_panel_factory(prices)(RSI2())
    df = cost_sensitivity(panel, COSTS, RISK, (0, 1, 2, 3))
    assert df["cost_mult"].tolist() == [0, 1, 2, 3]
    # mais custo nunca melhora o retorno total
    assert df["total_return"].is_monotonic_decreasing


def test_run_protocol_end_to_end(prices: pd.DataFrame) -> None:
    grid = {"rsi_entry": [10.0, 15.0]}
    r = run_protocol(
        RSI2(),
        default_panel_factory(prices),
        market="b3",
        costs=COSTS,
        risk=RISK,
        grid=grid,
        train_years=2,
        test_years=1,
        mc_runs=50,
        boot_runs=50,
        baseline_runs=3,
        thresholds=ApprovalThresholds(),
        min_trades_select=1,
        cross_panel_factory=default_panel_factory(prices),
        cross_costs=COSTS,
    )
    assert r.strategy_name == "rsi2" and r.market == "b3"
    assert set(r.params) == {"rsi_entry"}
    assert len(r.checklist) == 10  # 9 + mercado cruzado
    assert any("dd_p95 em 1 ano" in k for k in r.checklist)  # gate do ADR-017
    assert set(r.dd_bootstrap) == set(r.dd_bootstrap_full) == set(r.monte_carlo)
    # horizonte completo nunca é mais brando que 1 ano
    assert r.dd_bootstrap_full["mdd_p95"] <= r.dd_bootstrap["mdd_p95"]
    assert isinstance(r.approved, bool)
    assert r.cross_market is not None and "sharpe" in r.cross_market
    assert any("sobrevivência" in n for n in r.notes)
    assert r.metrics_test.start >= str(r.split.test.start.date())


def test_evaluate_window_slices(prices: pd.DataFrame) -> None:
    from swing_quant.backtest.validation import evaluate

    panel = default_panel_factory(prices)(RSI2())
    w = Window(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31"))
    res, m = evaluate(panel, Backtester(COSTS, RISK), w)
    assert res.equity.index.min() >= w.start and res.equity.index.max() <= w.end
    assert m.start.startswith("2018")
