"""Dashboard Streamlit — sinais do dia, posições, equity real × esperado, saúde.

Rodar: `uv run --extra dashboard streamlit run dashboard/app.py`
(ou `uv run swing-quant dashboard`). Lê o DuckDB em modo leitura.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_quant.backtest.metrics import drawdown_series, rolling_sharpe  # noqa: E402
from swing_quant.config import load_config  # noqa: E402
from swing_quant.data.store import MarketStore  # noqa: E402
from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol  # noqa: E402
from swing_quant.journal.core import Journal  # noqa: E402
from swing_quant.monitoring.health import HealthStore, latest_expected  # noqa: E402
from swing_quant.monitoring.performance import (  # noqa: E402
    mark_to_market,
    realized_metrics,
    trade_ledger,
)

st.set_page_config(page_title="Swing Quant", page_icon="📈", layout="wide")
cfg = load_config(ROOT / "config.yaml")
market = st.sidebar.selectbox("Mercado", ["b3", "us"], index=0)
lookback_days = st.sidebar.slider("Janela (dias)", 60, 720, 365, step=30)
st.sidebar.caption(f"Banco: `{cfg.data.db_path}`")


@st.cache_data(ttl=300)
def load(market: str, lookback_days: int) -> dict[str, pd.DataFrame]:
    with MarketStore(ROOT / cfg.data.db_path) as store:
        j = Journal(store)
        hs = HealthStore(store)
        members = store.universe_at(INDEX_BY_MARKET[market])
        tickers = [to_yf_symbol(t, market) for t in members["ticker"]]
        start = dt.date.today() - dt.timedelta(days=lookback_days)
        prices = store.get_prices(tickers, start=start)
        bench = store.get_prices([cfg.market_universe(market).benchmark], start=start)
        signals = j.signals(market=market)
        executions = j.executions()
        executions = (
            executions[executions["market"] == market] if not executions.empty else executions
        )
        positions = pd.DataFrame([p.__dict__ for p in j.open_positions(market)])
        health = hs.history(market)
        runs = store.con.execute(
            "SELECT run_id, strategy, params, period_start, period_end, created_at, metrics "
            "FROM backtest_runs WHERE strategy LIKE ? ORDER BY created_at DESC LIMIT 20",
            [f"%/{market}"],
        ).df()
        expected = {
            name: latest_expected(store, name, market).__dict__
            for name, v in cfg.strategies.items()
            if v.get("enabled")
        }
    return {
        "prices": prices,
        "bench": bench,
        "signals": signals,
        "executions": executions,
        "positions": positions,
        "health": health,
        "runs": runs,
        "expected": pd.DataFrame(expected).T if expected else pd.DataFrame(),
    }


data = load(market, lookback_days)
prices, signals, executions = data["prices"], data["signals"], data["executions"]

st.title(f"📈 Swing Quant — {market.upper()}")
st.caption(
    "Esta página é o **acompanhamento do que está valendo**: sinais do screener, posições "
    "abertas e desempenho realizado — fica vazia enquanto não houver execuções no journal. "
    "Para explorar backtests e o resultado ação por ação, abra a página **Backtests** na "
    "barra lateral."
)

# ---------------------------------------------------------------- sinais do dia
st.header("Sinais mais recentes")
if signals.empty:
    st.info("Nenhum sinal no journal ainda. Rode `swing-quant screen`.")
else:
    last_day = signals["as_of"].max()
    today = signals[signals["as_of"] == last_day]
    st.caption(f"Screener de {pd.Timestamp(last_day):%d/%m/%Y} — {len(today)} sinal(is)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Entradas")
        st.dataframe(
            today[today["side"] == "buy"][
                ["id", "ticker", "strategy", "ref_price", "qty", "stop_price", "score"]
            ],
            hide_index=True,
            width="stretch",
        )
    with c2:
        st.subheader("Saídas")
        st.dataframe(
            today[today["side"] == "sell"][
                ["id", "ticker", "strategy", "ref_price", "qty", "regime"]
            ],
            hide_index=True,
            width="stretch",
        )

# ---------------------------------------------------------------- posições e equity
st.header("Carteira real")
positions = data["positions"]
if positions.empty:
    st.info("Sem posições abertas.")
else:
    st.dataframe(
        positions[
            [
                "signal_id",
                "ticker",
                "strategy",
                "qty",
                "entry_date",
                "entry_price",
                "stop_price",
                "max_hold",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

if not prices.empty:
    mtm = mark_to_market(executions, prices, cfg.capital.initial)
    ledger = trade_ledger(signals, executions)
    m = realized_metrics(mtm["equity"], ledger)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Patrimônio", f"{mtm['equity'].iloc[-1]:,.0f}")
    k2.metric("Retorno", f"{m.total_return:.1%}")
    k3.metric("Sharpe", f"{m.sharpe:.2f}" if m.sharpe == m.sharpe else "-")
    k4.metric("Max DD", f"{m.max_drawdown:.1%}")
    k5.metric("Trades fechados", str(m.n_trades))

    bench = data["bench"]
    chart = pd.DataFrame({"carteira": mtm["equity"] / mtm["equity"].iloc[0]})
    if not bench.empty:
        b = bench.set_index("date")["adj_close"].reindex(mtm.index).ffill()
        chart["benchmark"] = b / b.iloc[0]
    st.subheader("Equity (base 1) × benchmark")
    st.line_chart(chart)
    st.subheader("Drawdown")
    st.area_chart(drawdown_series(mtm["equity"]))
    rs = rolling_sharpe(mtm["equity"].pct_change().fillna(0.0), 63).dropna()
    if not rs.empty:
        st.subheader("Sharpe rolling (3 meses)")
        st.line_chart(rs)
    if not ledger.empty:
        st.subheader("Ledger de trades")
        st.dataframe(ledger, hide_index=True, width="stretch")

# ---------------------------------------------------------------- realizado × esperado
st.header("Realizado × esperado")
exp = data["expected"]
if exp.empty:
    st.info("Sem backtests registrados em `backtest_runs` para as estratégias habilitadas.")
else:
    st.dataframe(exp[["sharpe", "cagr", "max_drawdown", "dd_p95", "run_id"]], width="stretch")
health = data["health"]
st.subheader("Saúde das estratégias (regra de desligamento)")
if health.empty:
    st.caption("Ainda sem avaliações — `swing-quant health` roda mensalmente.")
else:
    st.dataframe(
        health[
            ["as_of", "strategy", "status", "reason", "sharpe_6m", "drawdown", "consecutive_alerts"]
        ],
        hide_index=True,
        width="stretch",
    )

with st.expander("Backtests registrados"):
    st.dataframe(data["runs"], hide_index=True, width="stretch")
