"""Monitoramento: marcação a mercado, ledger, aderência, saúde/pausa e relatório mensal."""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from swing_quant.data.store import MarketStore
from swing_quant.journal.core import ExecutionRecord, Journal
from swing_quant.monitoring.health import (
    Expected,
    HealthStore,
    evaluate,
    latest_expected,
    run_health,
)
from swing_quant.monitoring.monthly import build_monthly, render_monthly, save_monthly
from swing_quant.monitoring.performance import (
    adherence,
    mark_to_market,
    strategy_daily_returns,
    trade_ledger,
)
from swing_quant.screener.core import ENTRY_COLUMNS, EXIT_COLUMNS, ScreenResult
from tests.conftest import make_prices


def _screen(as_of: str, ticker: str, price: float, qty: int) -> ScreenResult:
    entries = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "strategy": "donchian",
                "ref_price": price,
                "qty": qty,
                "notional": price * qty,
                "stop_price": price * 0.9,
                "max_hold": 0,
                "score": 1.0,
                "atr": 1.0,
                "dollar_volume": 1e8,
            }
        ],
        columns=ENTRY_COLUMNS,
    )
    return ScreenResult(
        pd.Timestamp(as_of),
        "b3",
        entries,
        pd.DataFrame(columns=EXIT_COLUMNS),
        100_000,
        100_000,
        0,
        6,
    )


@pytest.fixture
def journal_with_trades() -> tuple[Journal, pd.DataFrame]:
    store = MarketStore(":memory:")
    j = Journal(store)
    prices = make_prices(["AAA3.SA", "BBB4.SA"], dt.date(2024, 1, 2), dt.date(2024, 3, 28))
    store.upsert_prices(prices)
    px = prices.set_index(["ticker", "date"])["close"]
    d1, d2, d3 = pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-25"), pd.Timestamp("2024-02-15")
    # AAA: sinal d1-1, compra d1, venda d2 (fechado). BBB: sinal d3-1, compra d3 (aberto)
    sid_a = j.record_screen(
        _screen("2024-01-09", "AAA3.SA", float(px[("AAA3.SA", d1)]) * 0.99, 100)
    )[0]
    j.record_execution(
        ExecutionRecord(
            sid_a, "buy", float(px[("AAA3.SA", d1)]), 100, 2.0, executed_at=d1.to_pydatetime()
        )
    )
    j.record_execution(
        ExecutionRecord(
            sid_a, "sell", float(px[("AAA3.SA", d2)]), 100, 2.0, executed_at=d2.to_pydatetime()
        )
    )
    sid_b = j.record_screen(_screen("2024-02-14", "BBB4.SA", float(px[("BBB4.SA", d3)]), 50))[0]
    j.record_execution(
        ExecutionRecord(
            sid_b, "buy", float(px[("BBB4.SA", d3)]), 50, 1.0, executed_at=d3.to_pydatetime()
        )
    )
    return j, prices


def test_mark_to_market_consistency(journal_with_trades: tuple[Journal, pd.DataFrame]) -> None:
    j, prices = journal_with_trades
    ex = j.executions()
    mtm = mark_to_market(ex, prices, 100_000)
    px = prices.set_index(["ticker", "date"])["close"]
    d1, d2, d3 = pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-25"), pd.Timestamp("2024-02-15")
    # antes da primeira execução: tudo caixa
    assert mtm.loc[pd.Timestamp("2024-01-05"), "equity"] == 100_000
    # com AAA aberta: caixa = 100k - custo; investido = 100 × close do dia
    day = pd.Timestamp("2024-01-15")
    assert mtm.loc[day, "invested"] == pytest.approx(100 * px[("AAA3.SA", day)])
    assert mtm.loc[day, "cash"] == pytest.approx(100_000 - 100 * px[("AAA3.SA", d1)] - 2.0)
    assert mtm.loc[day, "n_positions"] == 1
    # após venda: caixa realizado, nada investido até BBB
    after = pd.Timestamp("2024-02-01")
    expected_cash = 100_000 - 100 * px[("AAA3.SA", d1)] - 2.0 + 100 * px[("AAA3.SA", d2)] - 2.0
    assert mtm.loc[after, "cash"] == pytest.approx(expected_cash)
    assert mtm.loc[after, "invested"] == 0.0
    # BBB aberta no fim
    last = mtm.index[-1]
    assert mtm.loc[last, "invested"] == pytest.approx(50 * px[("BBB4.SA", last)])
    assert (mtm["equity"] > 0).all()
    _ = d3


def test_mark_to_market_without_executions() -> None:
    prices = make_prices(["AAA3.SA"], dt.date(2024, 1, 2), dt.date(2024, 1, 31))
    mtm = mark_to_market(pd.DataFrame(), prices, 50_000)
    assert (mtm["equity"] == 50_000).all() and (mtm["n_positions"] == 0).all()


def test_trade_ledger_and_adherence(journal_with_trades: tuple[Journal, pd.DataFrame]) -> None:
    j, _ = journal_with_trades
    ledger = trade_ledger(j.signals(), j.executions())
    assert len(ledger) == 2
    a = ledger[ledger["ticker"] == "AAA3.SA"].iloc[0]
    b = ledger[ledger["ticker"] == "BBB4.SA"].iloc[0]
    assert not a["open"] and b["open"]
    assert a["pnl"] == pytest.approx((a["exit_price"] - a["entry_price"]) * 100 - 4.0)
    assert a["slippage_pct"] == pytest.approx(a["entry_price"] / a["ref_price"] - 1, rel=1e-6)
    assert a["slippage_pct"] > 0  # referência 1% abaixo do executado
    assert a["bars_held"] > 0
    adh = adherence(j.signals(), j.executions())
    assert adh["buy_signals"] == 2 and adh["buy_executed_pct"] == 1.0


def test_strategy_daily_returns_sum_to_pnl(
    journal_with_trades: tuple[Journal, pd.DataFrame],
) -> None:
    j, prices = journal_with_trades
    ledger = trade_ledger(j.signals(), j.executions())
    mtm = mark_to_market(j.executions(), prices, 100_000)
    rets = strategy_daily_returns(ledger, mtm["equity"])
    assert set(rets) == {"donchian"}
    closed_pnl = ledger[~ledger["open"]]["pnl"].sum()
    approx_pnl = (rets["donchian"] * mtm["equity"].shift(1).fillna(mtm["equity"].iloc[0])).sum()
    assert approx_pnl == pytest.approx(closed_pnl, rel=1e-6)


# ---------------------------------------------------------------- saúde
def test_evaluate_rules() -> None:
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.003, 0.005, 300))  # Sharpe claramente positivo, DD pequeno
    bad = pd.Series(rng.normal(-0.003, 0.005, 300))
    exp = Expected(sharpe=1.0, dd_p95=-0.15)
    assert evaluate(good, exp)[0] == "active"
    status, reason, _, dd, cons = evaluate(bad, exp)
    assert status == "alert" and cons == 1 and "Sharpe 6m" in reason
    assert evaluate(bad, exp, prior_alerts=1)[0] == "paused"
    # dados insuficientes -> sem veredito
    st, reason, *_ = evaluate(good.iloc[:10], exp)
    assert st == "active" and "insuficientes" in reason
    # drawdown pior que o p95 simulado dispara mesmo com Sharpe positivo recente
    crash = pd.concat([pd.Series(rng.normal(0.002, 0.005, 200)), pd.Series([-0.03] * 10)])
    st, reason, _, dd, _ = evaluate(crash.reset_index(drop=True), Expected(dd_p95=-0.10))
    assert dd < -0.10 and st == "alert" and "p95 simulado" in reason


def test_health_store_pause_and_resume() -> None:
    store = MarketStore(":memory:")
    hs = HealthStore(store)
    rng = np.random.default_rng(1)
    bad = pd.Series(rng.normal(-0.003, 0.01, 200))
    as_of = dt.date(2026, 9, 1)
    r1 = run_health(store, "b3", {"donchian": bad}, ["donchian"], as_of)
    assert r1[0].status == "alert" and hs.paused("b3") == set()
    r2 = run_health(store, "b3", {"donchian": bad}, ["donchian"], dt.date(2026, 10, 1))
    assert r2[0].status == "paused" and hs.paused("b3") == {"donchian"}
    # continua pausada mesmo se melhorar, até reativação manual
    good = pd.Series(rng.normal(0.003, 0.01, 200))
    r3 = run_health(store, "b3", {"donchian": good}, ["donchian"], dt.date(2026, 11, 1))
    assert r3[0].status == "paused"
    hs.resume("donchian", "b3", dt.date(2026, 11, 2))
    assert hs.paused("b3") == set()
    hist = hs.history("b3")
    assert len(hist) == 4 and hist["status"].iloc[0] == "active"


def test_latest_expected_from_backtest_runs() -> None:
    store = MarketStore(":memory:")
    assert np.isnan(latest_expected(store, "donchian", "b3").sharpe)
    metrics = {
        "full": {"sharpe": 0.79, "cagr": 0.051, "max_drawdown": -0.108},
        "dd_bootstrap": {"mdd_p95": -0.155},
        "monte_carlo": {"mdd_p95": -0.42},
    }
    store.con.execute(
        "INSERT INTO backtest_runs VALUES ('r1', 'donchian/b3', '{}', '2010-01-04', '2026-08-27', ?, "  # noqa: E501
        "now(), 'abc')",
        [json.dumps(metrics)],
    )
    e = latest_expected(store, "donchian", "b3")
    # prefere o bootstrap diário (ADR-017) ao MC por embaralhamento
    assert e.sharpe == pytest.approx(0.79) and e.dd_p95 == pytest.approx(-0.155)
    assert e.run_id == "r1"


def test_latest_expected_falls_back_to_monte_carlo() -> None:
    """Runs gravados antes do ADR-017 só têm `monte_carlo`."""
    store = MarketStore(":memory:")
    metrics = {"full": {"sharpe": 0.79}, "monte_carlo": {"mdd_p95": -0.42}}
    store.con.execute(
        "INSERT INTO backtest_runs VALUES ('r0', 'donchian/b3', '{}', '2010-01-04', '2026-08-27', ?, "  # noqa: E501
        "now(), 'abc')",
        [json.dumps(metrics)],
    )
    assert latest_expected(store, "donchian", "b3").dd_p95 == pytest.approx(-0.42)


# ---------------------------------------------------------------- mensal
def test_monthly_report(  # type: ignore[no-untyped-def]
    journal_with_trades: tuple[Journal, pd.DataFrame], tmp_path
) -> None:
    j, prices = journal_with_trades
    mtm = mark_to_market(j.executions(), prices, 100_000)
    health = run_health(j.store, "b3", {}, ["donchian"], dt.date(2024, 2, 1))
    rep = build_monthly("b3", "2024-01", mtm, j.signals(), j.executions(), health)
    assert rep.metrics_month.n_trades == 1  # AAA fechou em janeiro
    assert rep.adherence["buy_signals"] == 1
    md = render_monthly(rep)
    for s in (
        "# Relatório mensal",
        "## Realizado",
        "## Realizado × esperado",
        "## Aderência",
        "## Slippage",
        "## Trades fechados",
        "## Sinais do mês",
    ):
        assert s in md
    assert "insuficientes" in md
    path = save_monthly(rep, tmp_path)
    assert path.name == "monthly_b3_2024-01.md" and path.exists()
