"""Screener: paridade com o engine, seleção de entradas e detecção de saídas."""

import datetime as dt

import pandas as pd

from swing_quant.backtest.engine import Backtester, CostModel, RiskModel
from swing_quant.backtest.portfolio import combine_panels
from swing_quant.backtest.validation import default_panel_factory
from swing_quant.risk.regime import RegimeConfig, build_regime
from swing_quant.screener.core import OpenPosition, check_exits, run_screener, select_entries
from swing_quant.strategies import RSI2, Donchian
from tests.conftest import make_prices

COSTS = CostModel(0.0, 0.0003, 0.001)
RISK = RiskModel(initial_capital=100_000, risk_per_trade=0.01, board_lot=100, max_positions=4)


def _prices() -> pd.DataFrame:
    return make_prices(
        [f"T{i:02d}.SA" for i in range(12)], dt.date(2018, 1, 2), dt.date(2020, 12, 30), seed=42
    )


def test_parity_with_engine_on_signal_days() -> None:
    """Para cada dia com sinais, as entradas do screener == compras do engine em D+1 partindo
    do mesmo estado (sem posições, mesmo caixa). O engine executa ao open de D+1; para comparar
    quantidades, o screener recebe esse mesmo preço como referência (em produção usa o close de
    D, então a quantidade sugerida é indicativa — o ranking e o conjunto são idênticos)."""
    prices = _prices()
    panel = default_panel_factory(prices)(RSI2({"rsi_entry": 15.0, "trend_sma": 50}))
    signal_days = panel.dates[panel.entry.any(axis=1).to_numpy()]
    assert len(signal_days) > 20
    checked = 0
    for day in signal_days[:40]:
        pos = panel.dates.get_loc(day)
        if pos + 1 >= len(panel.dates):
            continue
        nxt = panel.dates[pos + 1]
        open_next = {t: float(panel.open.loc[nxt, t]) for t in panel.tickers}
        scr = select_entries(
            panel, day, risk=RISK, costs=COSTS, equity=100_000, cash=100_000, ref_prices=open_next
        )
        eng = Backtester(COSTS, RISK).run(panel.slice(day, nxt))
        bought = eng.trades[eng.trades["entry_date"] == nxt]
        assert list(scr["ticker"]) == list(bought["ticker"]), f"tickers divergem em {day.date()}"
        assert list(scr["qty"]) == list(bought["qty"]), f"quantidades divergem em {day.date()}"
        checked += 1
    assert checked >= 20


def test_select_entries_respects_slots_and_held() -> None:
    prices = _prices()
    panel = default_panel_factory(prices)(RSI2({"rsi_entry": 30.0, "trend_sma": 50}))
    day = panel.dates[panel.entry.sum(axis=1).to_numpy() >= 3][0]
    full = select_entries(panel, day, risk=RISK, costs=COSTS, equity=100_000, cash=100_000)
    assert 1 <= len(full) <= RISK.max_positions
    # com 3 posições abertas sobra 1 vaga
    held = [t for t in panel.tickers if t not in set(full["ticker"])][:3]
    one = select_entries(
        panel, day, risk=RISK, costs=COSTS, equity=100_000, cash=100_000, held=held
    )
    assert len(one) == 1 and one["ticker"].iloc[0] == full["ticker"].iloc[0]
    # ticker já em carteira nunca é sugerido
    again = select_entries(
        panel,
        day,
        risk=RISK,
        costs=COSTS,
        equity=100_000,
        cash=100_000,
        held=[full["ticker"].iloc[0]],
    )
    assert full["ticker"].iloc[0] not in set(again["ticker"])
    # regime bloqueado -> vazio; sizing reduzido -> menos quantidade
    assert select_entries(
        panel, day, risk=RISK, costs=COSTS, equity=1e5, cash=1e5, allow_entries=False
    ).empty
    half = select_entries(panel, day, risk=RISK, costs=COSTS, equity=1e5, cash=1e5, size_factor=0.5)
    assert (half["qty"].to_numpy() <= full["qty"].to_numpy()[: len(half)]).all()


def test_check_exits_reasons() -> None:
    prices = _prices()
    panel = default_panel_factory(prices)(RSI2({"max_hold": 3, "trend_sma": 50}))
    day = panel.dates[-1]
    t = panel.tickers[0]
    entry_date = panel.dates[-2].date()  # 1 pregão em carteira
    positions = [
        OpenPosition(t, "default", 100, entry_date, 10.0, stop_price=None, max_hold=3),
        OpenPosition(panel.tickers[1], "default", 100, panel.dates[-6].date(), 10.0, None, 3),
        OpenPosition(
            panel.tickers[2],
            "default",
            100,
            entry_date,
            10.0,
            stop_price=float(panel.high.iloc[-1, 2]) * 2,
            max_hold=0,
        ),  # stop acima -> toca
        OpenPosition("ZZZ9.SA", "default", 100, entry_date, 10.0, None, 3),
    ]
    ex = check_exits(panel, day, positions).set_index("ticker")
    assert ex.loc[panel.tickers[1], "reason"] == "time"
    assert ex.loc[panel.tickers[1], "bars_held"] >= 3
    assert ex.loc[panel.tickers[2], "reason"] == "stop"
    assert ex.loc["ZZZ9.SA", "reason"] == "not_in_universe"
    # a primeira só sai se houver sinal de saída hoje
    if t in ex.index:
        assert ex.loc[t, "reason"] == "signal"


def test_run_screener_end_to_end_with_regime_and_multi_strategy() -> None:
    prices = _prices()
    bench = prices[prices["ticker"] == "T00.SA"].set_index("date")["adj_close"]
    regime = build_regime(bench, RegimeConfig(use_trend=False))
    strategies = {
        "rsi2": RSI2({"trend_sma": 50}),
        "donchian": Donchian({"volume_mult": 0.0, "slow_sma": 100}),
    }
    res = run_screener(
        prices,
        strategies,
        market="b3",
        risk=RISK,
        costs=COSTS,
        equity=100_000,
        cash=100_000,
        regime=regime,
        sectors={"T00.SA": "X"},
    )
    assert res.as_of == pd.Timestamp("2020-12-30")
    assert set(res.entries.columns) >= {"ticker", "strategy", "qty", "ref_price", "notional"}
    assert res.regime["allow_entries"] is True
    assert res.slots == RISK.max_positions
    # preço de referência é o close bruto (não ajustado) do último dia
    if not res.entries.empty:
        t = res.entries["ticker"].iloc[0]
        raw = prices[(prices["ticker"] == t) & (prices["date"] == res.as_of)]["close"].iloc[0]
        assert res.entries["ref_price"].iloc[0] == float(raw)
    # as_of anterior funciona (reprocessar um dia passado)
    past = run_screener(
        prices,
        strategies,
        market="b3",
        risk=RISK,
        costs=COSTS,
        equity=1e5,
        cash=1e5,
        as_of=pd.Timestamp("2020-06-15"),
    )
    assert past.as_of <= pd.Timestamp("2020-06-15")


def test_combined_panel_one_position_per_underlying_in_screener() -> None:
    prices = _prices()
    f = default_panel_factory(prices)
    panel = combine_panels(
        {
            "a": f(RSI2({"rsi_entry": 30.0, "trend_sma": 50})),
            "b": f(RSI2({"rsi_entry": 30.0, "trend_sma": 50})),
        }
    )
    day = panel.dates[panel.entry.any(axis=1).to_numpy()][-1]
    out = select_entries(panel, day, risk=RISK, costs=COSTS, equity=1e5, cash=1e5)
    assert out["ticker"].is_unique
