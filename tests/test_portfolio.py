"""Regras de carteira do engine e painel combinado multi-estratégia."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swing_quant.backtest.engine import Backtester, CostModel, RiskModel
from swing_quant.backtest.panel import Panel
from swing_quant.backtest.portfolio import (
    attribution_table,
    combine_panels,
    render_portfolio_markdown,
    run_portfolio,
)
from swing_quant.backtest.validation import default_panel_factory
from swing_quant.risk.regime import build_regime
from swing_quant.strategies import RSI2, Donchian, DropsIBS
from tests.conftest import make_prices
from tests.test_engine import NO_COST, RISK, _panel


def _risk(**kw: object) -> RiskModel:
    return RiskModel(**{**RISK.__dict__, **kw})  # type: ignore[arg-type]


def test_one_position_per_underlying() -> None:
    p = _panel(
        ["A@s1", "A@s2"],
        opens={"A@s1": [10.0] * 4, "A@s2": [10.0] * 4},
        entry={"A@s1": [True, False, False, False], "A@s2": [True, False, False, False]},
        scores={"A@s1": [2.0] * 4, "A@s2": [1.0] * 4},
    )
    p.underlying = ["A", "A"]
    p.strategy_of = ["s1", "s2"]
    res = Backtester(NO_COST, RISK).run(p)
    assert len(res.trades) == 1 and res.trades.iloc[0]["strategy"] == "s1"
    assert res.risk_events.get("skip_same_underlying") == 1


def test_sector_cap_reduces_quantity() -> None:
    p = _panel(
        ["A", "B"],
        opens={"A": [10.0] * 3, "B": [10.0] * 3},
        entry={"A": [True, False, False], "B": [True, False, False]},
        scores={"A": [2.0] * 3, "B": [1.0] * 3},
    )
    p.sectors = {"A": "Energia", "B": "Energia"}
    # sem cap: 500 ações cada (5.000 = 5% do capital). Cap setorial 7% -> B fica com 200
    res = Backtester(NO_COST, _risk(max_sector_pct=0.07, board_lot=100)).run(p)
    t = res.trades.set_index("ticker")
    assert t.loc["A", "qty"] == 500
    assert t.loc["B", "qty"] == 200
    assert res.risk_events.get("cap_sector") == 1


def test_strategy_cap() -> None:
    p = _panel(
        ["A@s1", "B@s1", "C@s2"],
        opens={t: [10.0] * 3 for t in ["A@s1", "B@s1", "C@s2"]},
        entry={t: [True, False, False] for t in ["A@s1", "B@s1", "C@s2"]},
        scores={"A@s1": [3.0] * 3, "B@s1": [2.0] * 3, "C@s2": [1.0] * 3},
    )
    p.underlying = ["A", "B", "C"]
    p.strategy_of = ["s1", "s1", "s2"]
    res = Backtester(NO_COST, _risk(max_strategy_pct=0.05, board_lot=1)).run(p)
    t = res.trades.set_index("ticker")
    assert t.loc["A@s1", "qty"] == 500  # 5% cheio
    assert t.loc["B@s1", "qty"] == 0 if "B@s1" in t.index else True  # sem espaço em s1
    assert t.loc["C@s2", "qty"] == 500  # estratégia s2 não afetada
    assert "B@s1" not in t.index


def test_correlation_filter_skips_correlated_candidate() -> None:
    n = 90
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.normal(0, 0.01, n)) * 10
    a = base
    b = base * (1 + rng.normal(0, 0.0005, n))  # quase idêntico -> corr ~1
    c = np.cumprod(1 + rng.normal(0, 0.01, n)) * 10  # independente
    entry = [False] * n
    entry[70] = True
    p = _panel(
        ["A", "B", "C"],
        opens={"A": list(a), "B": list(b), "C": list(c)},
        entry={"A": entry, "B": entry, "C": entry},
        scores={"A": [3.0] * n, "B": [2.0] * n, "C": [1.0] * n},
    )
    res = Backtester(NO_COST, _risk(max_correlation=0.8, corr_window=60)).run(p)
    assert set(res.trades["ticker"]) == {"A", "C"}
    assert res.risk_events.get("skip_correlation") == 1


def test_circuit_breaker_blocks_new_entries() -> None:
    # A cai 30% após a entrada -> DD da carteira > 15% ? posição = 5% do capital, então não.
    # Usar max_position_pct alto e ATR pequeno para que a posição seja ~100% do capital.
    opens = [10.0] * 3 + [7.0] * 3 + [7.0] * 4
    entry = [True] + [False] * 9
    entry[6] = True  # novo sinal durante o drawdown
    p = _panel(
        ["A", "B"],
        opens={"A": opens, "B": [10.0] * 10},
        entry={"A": entry, "B": [False] * 6 + [True] + [False] * 3},
        exit_={"A": [False, False, False, False, True] + [False] * 5},
    )
    p.atr.iloc[:, :] = 0.02
    risk = _risk(max_position_pct=1.0, risk_per_trade=0.05, circuit_breaker_dd=0.15, board_lot=1)
    res = Backtester(NO_COST, risk).run(p)
    assert res.risk_events.get("circuit_breaker_on", 0) >= 1
    assert res.risk_events.get("blocked_by_breaker", 0) >= 1
    assert set(res.trades["ticker"]) == {"A"}  # B nunca entrou


def test_monthly_dd_reduces_sizing() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    opens = [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    p = _panel(
        ["A", "B"],
        opens={"A": opens, "B": [10.0] * 12},
        entry={"A": [True] + [False] * 11, "B": [False] * 5 + [True] + [False] * 6},
        exit_={"A": [False, False, True] + [False] * 9},
    )
    p.dates = idx
    for name in (
        "open",
        "high",
        "low",
        "close",
        "atr",
        "dollar_vol",
        "entry",
        "exit",
        "stop",
        "score",
        "max_hold",
    ):
        getattr(p, name).index = idx
    p.atr.iloc[:, :] = 0.02
    risk = _risk(max_position_pct=1.0, risk_per_trade=0.05, monthly_dd_reduce=0.05, board_lot=1)
    res = Backtester(NO_COST, risk).run(p)
    assert res.risk_events.get("monthly_dd_reduce") == 1
    b = res.trades.set_index("ticker").loc["B"]
    # sem redução: 0.05*eq/(2*0.02) limitado por max_position 100% -> ~equity/10 ações;
    # com fator 0,5 o risco cai pela metade
    full_qty = int(res.equity.iloc[4] * 0.05 / (2 * 0.02))
    assert b["qty"] < full_qty


def test_combine_panels_and_run_portfolio() -> None:
    prices = make_prices(
        ["AAA3.SA", "BBB4.SA", "CCC3.SA"], dt.date(2015, 1, 2), dt.date(2020, 12, 30), seed=21
    )
    factory = default_panel_factory(prices)
    panels = {
        "rsi2": factory(RSI2()),
        "donchian": factory(Donchian({"volume_mult": 0.0})),
        "drops_ibs": factory(DropsIBS()),
    }
    sectors = {"AAA3.SA": "X", "BBB4.SA": "X", "CCC3.SA": "Y"}
    combined = combine_panels(panels, sectors)
    assert len(combined.tickers) == 9
    assert combined.tickers[0].endswith("@rsi2")
    assert combined.underlying[:3] == ["AAA3.SA", "BBB4.SA", "CCC3.SA"]
    assert set(combined.strategy_of) == {"rsi2", "donchian", "drops_ibs"}
    assert combined.entry.dtypes.eq(bool).all()

    bench = prices[prices["ticker"] == "AAA3.SA"].set_index("date")["adj_close"]
    regime = build_regime(bench)
    risk = RiskModel(
        initial_capital=100_000,
        board_lot=1,
        max_positions=5,
        max_sector_pct=0.5,
        max_strategy_pct=0.6,
        max_correlation=0.95,
        monthly_dd_reduce=0.06,
        circuit_breaker_dd=0.15,
    )
    r = run_portfolio(
        combined,
        market="b3",
        costs=CostModel(0, 0.0003, 0.001),
        risk=risk,
        regime=regime,
        benchmark_close=bench,
    )
    assert set(r.attribution["strategy"]) <= {"rsi2", "donchian", "drops_ibs"}
    assert len(r.with_vs_without) == 6
    assert r.strategies == ["rsi2", "donchian", "drops_ibs"]
    md = render_portfolio_markdown(r)
    for s in ("## Atribuição", "## Efeito das camadas", "## Regime", "## Eventos de risco"):
        assert s in md


def test_attribution_table_empty() -> None:
    assert attribution_table(pd.DataFrame(columns=["strategy", "pnl", "bars_held"])).empty


def test_panel_defaults_underlying_and_strategy() -> None:
    p = _panel(["A"], opens={"A": [1.0, 1.0]}, entry={"A": [False, False]})
    assert p.underlying == ["A"] and p.strategy_of == ["default"]
    assert isinstance(p, Panel)
    with pytest.raises(ValueError):
        combine_panels({})


def test_held_position_with_missing_prices_does_not_nan_equity() -> None:
    """Regressão: ticker sem open nem close em alguns dias (união de datas) enquanto há
    posição aberta -> equity_open não pode virar NaN nem quebrar o sizing."""
    nan = float("nan")
    p = _panel(
        ["A", "B"],
        opens={"A": [10.0, 10.0, nan, nan, 10.0, 10.0], "B": [10.0] * 6},
        entry={"A": [True] + [False] * 5, "B": [False, False, True, False, False, False]},
    )
    p.close.loc[p.dates[2:4], "A"] = nan
    p.low.loc[p.dates[2:4], "A"] = nan
    res = Backtester(NO_COST, RISK).run(p)
    assert not res.equity.isna().any()
    assert set(res.trades["ticker"]) == {"A", "B"}  # B entrou em D3 apesar do buraco em A


def test_circuit_breaker_cooldown_rearms() -> None:
    """Após o cooldown, o breaker desarma e redefine o pico: novas entradas voltam a ocorrer
    mesmo sem recuperar o drawdown."""
    n = 30
    opens = [10.0] * 3 + [7.0] * (n - 3)  # queda e estagnação: nunca recupera
    entry_a = [True] + [False] * (n - 1)
    entry_b = [False] * n
    entry_b[5] = True  # durante o bloqueio (breaker liga em D3, cooldown até D8)
    entry_b[20] = True  # após o cooldown de 5 pregões
    p = _panel(
        ["A", "B"],
        opens={"A": opens, "B": [10.0] * n},
        entry={"A": entry_a, "B": entry_b},
        exit_={"A": [False] * 4 + [True] + [False] * (n - 5)},
    )
    p.atr.iloc[:, :] = 0.02
    risk = _risk(
        max_position_pct=1.0,
        risk_per_trade=0.05,
        circuit_breaker_dd=0.15,
        board_lot=1,
        circuit_breaker_cooldown=5,
    )
    res = Backtester(NO_COST, risk).run(p)
    assert res.risk_events.get("circuit_breaker_on") == 1
    assert res.risk_events.get("circuit_breaker_reset") == 1
    assert res.risk_events.get("blocked_by_breaker") == 1  # só o sinal do dia 5
    assert "B" in set(res.trades["ticker"])  # entrou no dia 21
