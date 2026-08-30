"""Exploração dos runs registrados: achatamento das métricas e agregação por ação."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from swing_quant.backtest.explore import (
    benchmark_yearly,
    by_exit_reason,
    by_ticker,
    load_runs,
    load_trades,
    trades_path,
    yearly_returns,
)
from swing_quant.data.store import MarketStore


def _insert_run(store: MarketStore, run_id: str, strategy: str, sharpe: float) -> None:
    metrics = {
        "test": {
            "sharpe": sharpe,
            "cagr": 0.24,
            "max_drawdown": -0.09,
            "n_trades": 362,
            "win_rate": 0.47,
            "profit_factor": 2.73,
            "avg_hold_bars": 13.8,
        },
        "dd_bootstrap": {"mdd_p95": -0.138},
        "approved": True,
    }
    store.con.execute(
        "INSERT INTO backtest_runs VALUES (?, ?, ?, '2010-01-04', '2026-08-27', ?, now(), 'abc')",
        [run_id, strategy, json.dumps({"lookback": 126.0, "exit_sma": 100.0}), json.dumps(metrics)],
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA3.SA", "AAA3.SA", "BBB4.SA", "CCC5.SA"],
            "entry_date": pd.to_datetime(["2024-01-02", "2024-03-01", "2024-01-10", "2024-02-01"]),
            "exit_date": pd.to_datetime(["2024-01-20", "2024-03-05", "2024-02-20", "2024-02-04"]),
            "pnl": [100.0, -40.0, 60.0, -20.0],
            "ret": [0.10, -0.04, 0.06, -0.02],
            "bars_held": [12, 3, 28, 2],
            "exit_reason": ["signal", "stop", "signal", "stop"],
        }
    )


def test_load_runs_flattens_metrics() -> None:
    store = MarketStore(":memory:")
    assert load_runs(store).empty
    _insert_run(store, "momentum_us_20260829_214721_9aceb1", "momentum/us", 2.11)
    _insert_run(store, "donchian_b3_20260828_224441_5259c5", "donchian/b3", 0.79)

    runs = load_runs(store)
    assert len(runs) == 2
    linha = runs[runs["estrategia"] == "momentum"].iloc[0]
    assert linha["mercado"] == "us"
    assert linha["sharpe_oos"] == pytest.approx(2.11)
    assert linha["dd_p95_1a"] == pytest.approx(-0.138)
    assert linha["aprovada"]
    # params viram texto legível, sem o ruído do float do numpy
    assert linha["params"] == "lookback=126, exit_sma=100"

    assert list(load_runs(store, market="b3")["estrategia"]) == ["donchian"]
    store.close()


def test_trades_path_drops_the_run_hash(tmp_path: Path) -> None:
    assert (
        trades_path("momentum_us_20260829_214721_9aceb1", tmp_path).name
        == "momentum_us_20260829_214721_trades.csv"
    )
    # run sem CSV correspondente não explode: devolve DataFrame vazio
    assert load_trades("sumiu_2026_abc", tmp_path).empty


def test_load_trades_reads_csv(tmp_path: Path) -> None:
    _trades().to_csv(tmp_path / "run_20260101_120000_trades.csv", index=False)
    df = load_trades("run_20260101_120000_abc123", tmp_path)
    assert len(df) == 4
    assert df["run_id"].unique().tolist() == ["run_20260101_120000_abc123"]
    assert pd.api.types.is_datetime64_any_dtype(df["entry_date"])


def test_by_ticker_aggregates_and_shares_pnl() -> None:
    out = by_ticker(_trades())
    assert list(out["ticker"]) == ["AAA3.SA", "BBB4.SA", "CCC5.SA"]  # ordenado por P&L
    aaa = out[out["ticker"] == "AAA3.SA"].iloc[0]
    assert aaa["trades"] == 2
    assert aaa["pnl"] == pytest.approx(60.0)
    assert aaa["win_rate"] == pytest.approx(0.5)
    assert aaa["permanencia_mediana"] == pytest.approx(7.5)
    # a contribuição soma 100% do P&L do run
    assert out["contribuicao"].sum() == pytest.approx(1.0)
    assert by_ticker(pd.DataFrame()).empty


def test_by_exit_reason_shares_sum_to_one() -> None:
    out = by_exit_reason(_trades())
    assert out["share"].sum() == pytest.approx(1.0)
    stop = out[out["exit_reason"] == "stop"].iloc[0]
    assert stop["trades"] == 2
    assert stop["bars_mediano"] == pytest.approx(2.5)
    assert by_exit_reason(pd.DataFrame()).empty


def test_yearly_returns_compound_to_the_period_total() -> None:
    """O acumulado dos anos tem de bater com o P&L total sobre o capital inicial."""
    trades = pd.DataFrame(
        {
            "exit_date": pd.to_datetime(["2024-03-01", "2024-11-20", "2025-02-10"]),
            "pnl": [10_000.0, -2_000.0, 5_400.0],
        }
    )
    anual = yearly_returns(trades, capital=100_000.0)
    assert anual[2024] == pytest.approx(0.08)  # 8.000 sobre 100.000
    assert anual[2025] == pytest.approx(0.05)  # 5.400 sobre 108.000
    acumulado = float((1 + anual).prod() - 1)
    assert acumulado == pytest.approx(13_400.0 / 100_000.0)
    assert yearly_returns(pd.DataFrame()).empty


def test_benchmark_yearly_uses_first_and_last_close_of_each_year() -> None:
    store = MarketStore(":memory:")
    linhas = [
        ("^BVSP", "2024-01-02", 100.0),
        ("^BVSP", "2024-12-27", 120.0),
        ("^BVSP", "2025-01-02", 120.0),
        ("^BVSP", "2025-12-29", 90.0),
    ]
    for ticker, data, preco in linhas:
        store.con.execute(
            "INSERT INTO prices VALUES (?, ?, NULL, NULL, NULL, ?, ?, NULL, 'teste')",
            [ticker, data, preco, preco],
        )
    anual = benchmark_yearly(store, "^BVSP")
    assert anual[2024] == pytest.approx(0.20)
    assert anual[2025] == pytest.approx(-0.25)
    assert benchmark_yearly(store, "NAO_EXISTE").empty
    store.close()
