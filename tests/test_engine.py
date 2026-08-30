"""Testes determinísticos do engine com painéis construídos à mão."""

import numpy as np
import pandas as pd
import pytest

from swing_quant.backtest.engine import Backtester, CostModel, RiskModel
from swing_quant.backtest.panel import Panel


def _panel(
    tickers: list[str],
    opens: dict[str, list[float]],
    entry: dict[str, list[bool]],
    exit_: dict[str, list[bool]] | None = None,
    stop: dict[str, list[float]] | None = None,
    max_hold: int = 0,
    lows: dict[str, list[float]] | None = None,
    highs: dict[str, list[float]] | None = None,
    target: dict[str, list[float]] | None = None,
    scores: dict[str, list[float]] | None = None,
) -> Panel:
    n = len(next(iter(opens.values())))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")

    def wide(d: dict[str, list], fill, dtype=float) -> pd.DataFrame:  # type: ignore[no-untyped-def, type-arg]
        return pd.DataFrame({t: d.get(t, [fill] * n) for t in tickers}, index=idx, dtype=dtype)

    opn = wide(opens, np.nan)
    close = opn.copy()  # fechamento = abertura (sem variação intradiária, simplifica contas)
    low = wide(lows, np.nan) if lows else close.copy()
    high = wide(highs, np.nan) if highs else close.copy()
    return Panel(
        dates=idx,
        tickers=tickers,
        open=opn,
        high=high,
        low=low,
        close=close,
        atr=pd.DataFrame(1.0, index=idx, columns=tickers),
        dollar_vol=pd.DataFrame(1e9, index=idx, columns=tickers),
        entry=wide(entry, False, bool),
        exit=wide(exit_ or {}, False, bool),
        stop=wide(stop or {}, np.nan),
        target=wide(target or {}, np.nan),
        score=wide(scores or {}, 1.0),
        max_hold=pd.DataFrame(max_hold, index=idx, columns=tickers, dtype=int),
    )


NO_COST = CostModel(0.0, 0.0, 0.0)
RISK = RiskModel(initial_capital=100_000, risk_per_trade=0.01, atr_multiple=2.0, board_lot=1)


def test_entry_next_open_and_exit_by_signal() -> None:
    # sinal de entrada em D0 -> compra na abertura de D1 (10.0); sinal de saída em D2 -> vende D3
    p = _panel(
        ["A"],
        opens={"A": [9.0, 10.0, 11.0, 12.0, 13.0]},
        entry={"A": [True, False, False, False, False]},
        exit_={"A": [False, False, True, False, False]},
    )
    res = Backtester(NO_COST, RISK).run(p)
    t = res.trades
    assert len(t) == 1
    tr = t.iloc[0]
    assert tr["entry_date"] == p.dates[1] and tr["entry_price"] == 10.0
    assert tr["exit_date"] == p.dates[3] and tr["exit_price"] == 12.0
    assert tr["exit_reason"] == "signal"
    # sizing: risco 1% de 100k = 1000 / (2*ATR=2) = 500 ações
    assert tr["qty"] == 500
    assert tr["pnl"] == pytest.approx(500 * 2.0)
    assert res.equity.iloc[-1] == pytest.approx(101_000.0)


def test_time_stop() -> None:
    p = _panel(
        ["A"],
        opens={"A": [10.0] * 8},
        entry={"A": [True] + [False] * 7},
        max_hold=3,
    )
    res = Backtester(NO_COST, RISK).run(p)
    tr = res.trades.iloc[0]
    # entra D1; bars_held = 3 no fechamento de D3 -> sai na abertura de D4
    assert tr["entry_date"] == p.dates[1]
    assert tr["exit_date"] == p.dates[4]
    assert tr["exit_reason"] == "time"
    assert tr["bars_held"] == 3


def test_price_stop_intraday_and_gap() -> None:
    # stop em 9.5: D2 low toca 9.4 -> sai a 9.5; segundo caso gap abaixo -> sai na abertura
    p = _panel(
        ["A", "B"],
        opens={"A": [10.0, 10.0, 10.0, 10.0], "B": [10.0, 10.0, 9.0, 10.0]},
        lows={"A": [10.0, 10.0, 9.4, 10.0], "B": [10.0, 10.0, 8.9, 10.0]},
        entry={"A": [True, False, False, False], "B": [True, False, False, False]},
        stop={"A": [9.5] * 4, "B": [9.5] * 4},
    )
    res = Backtester(NO_COST, RISK).run(p)
    t = res.trades.set_index("ticker")
    assert t.loc["A", "exit_price"] == 9.5 and t.loc["A", "exit_reason"] == "stop"
    assert t.loc["B", "exit_price"] == 9.0  # gap: executa na abertura, pior que o stop
    assert (t["exit_date"] == p.dates[2]).all()


def test_costs_applied_both_legs() -> None:
    costs = CostModel(commission_per_order=5.0, fees_pct=0.001, slippage_pct=0.01)
    p = _panel(
        ["A"],
        opens={"A": [10.0, 10.0, 10.0, 10.0]},
        entry={"A": [True, False, False, False]},
        exit_={"A": [False, True, False, False]},
    )
    res = Backtester(costs, RISK).run(p)
    tr = res.trades.iloc[0]
    assert tr["entry_price"] == pytest.approx(10.1)
    assert tr["exit_price"] == pytest.approx(9.9)
    qty = tr["qty"]
    expected_fees = 2 * 5.0 + qty * 10.1 * 0.001 + qty * 9.9 * 0.001
    assert tr["fees"] == pytest.approx(expected_fees)
    assert tr["pnl"] == pytest.approx(qty * (9.9 - 10.1) - expected_fees)


def test_max_positions_and_score_ranking() -> None:
    p = _panel(
        ["A", "B", "C"],
        opens={t: [10.0] * 3 for t in "ABC"},
        entry={t: [True, False, False] for t in "ABC"},
        scores={"A": [1.0] * 3, "B": [3.0] * 3, "C": [2.0] * 3},
    )
    res = Backtester(NO_COST, RiskModel(**{**RISK.__dict__, "max_positions": 2})).run(p)
    assert set(res.trades["ticker"]) == {"B", "C"}
    assert res.n_positions.max() == 2


def test_position_size_caps() -> None:
    # max_position_pct=5% -> 5000/10 = 500 ações mesmo com risco permitindo 1000/(2*0.5)=1000
    risk = RiskModel(
        initial_capital=100_000,
        risk_per_trade=0.01,
        atr_multiple=2.0,
        max_position_pct=0.05,
        board_lot=100,
    )
    p = _panel(["A"], opens={"A": [10.0, 10.0, 10.0]}, entry={"A": [True, False, False]})
    p.atr.iloc[:, :] = 0.5
    res = Backtester(NO_COST, risk).run(p)
    assert res.trades.iloc[0]["qty"] == 500


def test_board_lot_rounding_and_fractional() -> None:
    p = _panel(["A"], opens={"A": [10.0, 10.0, 10.0]}, entry={"A": [True, False, False]})
    # 1000 / (2*1) = 500 -> lote 100 mantém 500; ATR 1.3 -> 384 -> arredonda para 300
    p.atr.iloc[:, :] = 1.3
    res = Backtester(NO_COST, RiskModel(**{**RISK.__dict__, "board_lot": 100})).run(p)
    assert res.trades.iloc[0]["qty"] == 300
    # abaixo de um lote: 50 ações -> fracionário permitido
    p.atr.iloc[:, :] = 10.0
    res = Backtester(NO_COST, RiskModel(**{**RISK.__dict__, "board_lot": 100})).run(p)
    assert res.trades.iloc[0]["qty"] == 50


def test_regime_filter_blocks_entries() -> None:
    p = _panel(["A"], opens={"A": [10.0] * 4}, entry={"A": [True, True, False, False]})
    allow = pd.Series([False, True, True, True], index=p.dates)
    res = Backtester(NO_COST, RISK, allow_entries=allow).run(p)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_date"] == p.dates[2]  # sinal de D1 executado em D2


def test_liquidity_filter() -> None:
    p = _panel(["A"], opens={"A": [10.0] * 3}, entry={"A": [True, False, False]})
    p.dollar_vol.iloc[:, :] = 1_000.0
    risk = RiskModel(**{**RISK.__dict__, "min_dollar_volume": 10_000.0})
    assert Backtester(NO_COST, risk).run(p).trades.empty


def test_open_positions_closed_at_end() -> None:
    p = _panel(["A"], opens={"A": [10.0, 10.0, 12.0]}, entry={"A": [True, False, False]})
    res = Backtester(NO_COST, RISK).run(p)
    assert res.trades.iloc[0]["exit_reason"] == "end"
    assert res.equity.iloc[-1] == pytest.approx(100_000 + 500 * 2.0)


def test_equity_and_cash_consistency() -> None:
    p = _panel(
        ["A"],
        opens={"A": [10.0, 10.0, 11.0, 11.0, 11.0]},
        entry={"A": [True, False, False, False, False]},
        exit_={"A": [False, False, True, False, False]},
    )
    res = Backtester(NO_COST, RISK).run(p)
    assert res.cash.iloc[0] == 100_000
    assert res.cash.iloc[1] == pytest.approx(100_000 - 500 * 10.0)
    assert res.exposure.iloc[1] == pytest.approx(5000 / 100_000)
    assert res.exposure.iloc[-1] == 0.0
    assert res.equity.iloc[-1] == pytest.approx(res.cash.iloc[-1])


def test_price_target_intraday_and_gap() -> None:
    # alvo em 11: D2 a máxima de A toca 11.2 -> sai a 11; B abre em 12 -> gap paga melhor
    p = _panel(
        ["A", "B"],
        opens={"A": [10.0, 10.0, 10.0, 10.0], "B": [10.0, 10.0, 12.0, 10.0]},
        highs={"A": [10.0, 10.0, 11.2, 10.0], "B": [10.0, 10.0, 12.0, 10.0]},
        entry={"A": [True, False, False, False], "B": [True, False, False, False]},
        target={"A": [11.0] * 4, "B": [11.0] * 4},
    )
    res = Backtester(NO_COST, RISK).run(p)
    t = res.trades.set_index("ticker")
    assert t.loc["A", "exit_price"] == 11.0 and t.loc["A", "exit_reason"] == "target"
    assert t.loc["B", "exit_price"] == 12.0  # gap: executa na abertura, melhor que o alvo
    assert (t["exit_date"] == p.dates[2]).all()


def test_stop_wins_when_the_same_bar_touches_stop_and_target() -> None:
    """Sem intradiário não dá para saber a ordem; o engine assume o pior caso."""
    p = _panel(
        ["A"],
        opens={"A": [10.0, 10.0, 10.0, 10.0]},
        highs={"A": [10.0, 10.0, 11.5, 10.0]},
        lows={"A": [10.0, 10.0, 9.0, 10.0]},
        entry={"A": [True, False, False, False]},
        stop={"A": [9.5] * 4},
        target={"A": [11.0] * 4},
    )
    res = Backtester(NO_COST, RISK).run(p)
    assert res.trades["exit_reason"].tolist() == ["stop"]
    assert res.trades["exit_price"].tolist() == [9.5]


def test_no_target_keeps_the_old_behaviour() -> None:
    """Painel sem alvo (o caso de todas as estratégias anteriores) não fecha por alvo."""
    p = _panel(
        ["A"],
        opens={"A": [10.0, 10.0, 20.0, 10.0]},
        highs={"A": [10.0, 10.0, 50.0, 10.0]},
        entry={"A": [True, False, False, False]},
        exit_={"A": [False, False, True, False]},
    )
    res = Backtester(NO_COST, RISK).run(p)
    assert res.trades["exit_reason"].tolist() == ["signal"]
