"""Página de backtests: compara os runs já rodados e abre o resultado ação por ação.

Os números vêm de `backtest_runs` (métricas) e dos CSVs de trades em `reports/` — nada é
recalculado aqui. Para gerar novos runs: `swing-quant bench -s donchian,momentum -m all`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swing_quant.backtest.explore import (  # noqa: E402
    benchmark_yearly,
    by_exit_reason,
    by_ticker,
    load_runs,
    load_trades,
    yearly_returns,
)
from swing_quant.config import load_config  # noqa: E402
from swing_quant.data.riskfree import (  # noqa: E402
    RISK_FREE_LABEL,
    annual_returns,
    risk_free_daily,
)
from swing_quant.data.store import MarketStore  # noqa: E402

st.set_page_config(page_title="Backtests — Swing Quant", page_icon="🔬", layout="wide")
cfg = load_config(ROOT / "config.yaml")
REPORTS = ROOT / "reports"


@st.cache_data(ttl=300)
def _runs() -> pd.DataFrame:
    with MarketStore(ROOT / cfg.data.db_path) as store:
        return load_runs(store)


@st.cache_data(ttl=300)
def _trades(run_id: str) -> pd.DataFrame:
    return load_trades(run_id, REPORTS)


st.title("🔬 Backtests")
st.caption(
    "Cada linha é uma execução do protocolo (docs/04). **Veredito** é o checklist de 10 itens; "
    "as métricas são do período de teste (OOS), nunca do treino. "
    "Para rodar mais: `swing-quant bench -s donchian,momentum -m all`."
)
st.caption(
    "⏱️ **Timeframe: candles diários (D1)** — é o único que a base tem. O sinal nasce no "
    "fechamento e é executado na abertura do pregão seguinte; toda coluna de permanência "
    "(`perm.`, `pregões`) conta **pregões**, não horas nem semanas."
)

runs = _runs()
if runs.empty:
    st.warning("Nenhum backtest registrado ainda. Rode `swing-quant bench` e recarregue.")
    st.stop()

# ---------------------------------------------------------------- filtros e comparação
c1, c2, c3 = st.columns([1, 2, 1])
mercados = c1.multiselect("Mercado", sorted(runs["mercado"].unique()), default=None)
estrategias = c2.multiselect("Estratégia", sorted(runs["estrategia"].unique()), default=None)
so_aprovadas = c3.checkbox("Só aprovadas", value=False)

view = runs.copy()
if mercados:
    view = view[view["mercado"].isin(mercados)]
if estrategias:
    view = view[view["estrategia"].isin(estrategias)]
if so_aprovadas:
    view = view[view["aprovada"]]

st.subheader(f"Runs registrados ({len(view)})")
show = view.assign(
    quando=view["quando"].dt.strftime("%d/%m %H:%M"),
    sharpe_oos=view["sharpe_oos"].round(2),
    cagr_oos=view["cagr_oos"] * 100,
    mdd_oos=view["mdd_oos"] * 100,
    dd_p95_1a=view["dd_p95_1a"] * 100,
    win_rate=view["win_rate"] * 100,
    profit_factor=view["profit_factor"].round(2),
    permanencia_media=view["permanencia_media"].round(1),
)
st.dataframe(
    show[
        [
            "quando",
            "estrategia",
            "mercado",
            "aprovada",
            "sharpe_oos",
            "cagr_oos",
            "mdd_oos",
            "dd_p95_1a",
            "trades",
            "win_rate",
            "profit_factor",
            "permanencia_media",
            "params",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "quando": "quando",
        "aprovada": st.column_config.CheckboxColumn("aprovada"),
        "sharpe_oos": st.column_config.NumberColumn("Sharpe OOS", format="%.2f"),
        "cagr_oos": st.column_config.NumberColumn("CAGR", format="%.1f%%"),
        "mdd_oos": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
        "dd_p95_1a": st.column_config.NumberColumn("DD p95 1a", format="%.1f%%"),
        "trades": st.column_config.NumberColumn("trades", format="%d"),
        "win_rate": st.column_config.NumberColumn("acerto", format="%.0f%%"),
        "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
        "permanencia_media": st.column_config.NumberColumn("perm. média", format="%.1f"),
        "params": "parâmetros escolhidos",
    },
)

# ---------------------------------------------------------------- retorno por ano
st.header("Retorno por ano")
st.caption(
    "Cada estratégia (o run mais recente dela naquele mercado), o índice e a renda fixa local, "
    "ano a ano. O ano da estratégia é aproximado: o P&L de cada trade cai no ano em que a "
    "posição **fechou** — o acumulado do período bate com o backtest, os anos isolados são "
    "para comparar, não para reportar. O ano corrente é parcial."
)
mkt_ano = st.radio("Mercado", sorted(runs["mercado"].unique()), horizontal=True, key="mercado_ano")


@st.cache_data(ttl=300)
def _por_ano(mkt: str) -> pd.DataFrame:
    todos = _runs()
    ultimos = todos[todos["mercado"] == mkt].sort_values("quando").groupby("estrategia").last()
    colunas: dict[str, pd.Series] = {}
    for nome, linha in ultimos.iterrows():
        serie = yearly_returns(load_trades(str(linha["run_id"]), REPORTS))
        if not serie.empty:
            colunas[str(nome)] = serie
    with MarketStore(ROOT / cfg.data.db_path) as store:
        indice = cfg.market_universe(mkt).benchmark
        colunas[indice] = benchmark_yearly(store, indice)
        rf = risk_free_daily(store, mkt)
    if not rf.empty:
        colunas[RISK_FREE_LABEL[mkt]] = annual_returns(rf)
    tab = pd.DataFrame(colunas)
    return tab.sort_index()


tab_ano = _por_ano(mkt_ano)
if tab_ano.empty:
    st.info("Sem runs com CSV de trades para este mercado.")
else:
    if RISK_FREE_LABEL[mkt_ano] not in tab_ano.columns:
        st.warning(
            f"Renda fixa ({RISK_FREE_LABEL[mkt_ano]}) ainda não baixada — rode "
            "`swing-quant update-riskfree -m all`."
        )
    st.dataframe(
        (tab_ano * 100).round(1),
        width="stretch",
        height=min(640, 40 + 35 * len(tab_ano)),
        column_config={
            c: st.column_config.NumberColumn(c, format="%.1f%%") for c in tab_ano.columns
        },
    )
    resumo = pd.DataFrame(
        {
            "acumulado": (1 + tab_ano.fillna(0.0)).prod() - 1,
            "melhor ano": tab_ano.max(),
            "pior ano": tab_ano.min(),
            "anos positivos": (tab_ano > 0).sum(),
            "anos negativos": (tab_ano < 0).sum(),
        }
    )
    st.dataframe(
        resumo.assign(
            acumulado=resumo["acumulado"] * 100,
            **{"melhor ano": resumo["melhor ano"] * 100, "pior ano": resumo["pior ano"] * 100},
        ),
        width="stretch",
        column_config={
            "acumulado": st.column_config.NumberColumn("acumulado", format="%.0f%%"),
            "melhor ano": st.column_config.NumberColumn("melhor ano", format="%.1f%%"),
            "pior ano": st.column_config.NumberColumn("pior ano", format="%.1f%%"),
            "anos positivos": st.column_config.NumberColumn(format="%d"),
            "anos negativos": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption("100.000 investidos no início da série, ano a ano:")
    st.line_chart((1 + tab_ano.fillna(0.0)).cumprod() * 100_000, height=360)

# ---------------------------------------------------------------- resultado por ação
st.header("Resultado por ação")
labels = {
    f"{r.estrategia}/{r.mercado} — {r.quando:%d/%m %H:%M} "
    f"(Sharpe {r.sharpe_oos:.2f}{', aprovada' if r.aprovada else ''})": r.run_id
    for r in view.itertuples()
}
if not labels:
    st.info("Nenhum run com esses filtros.")
    st.stop()
escolhido = st.selectbox("Run", list(labels), index=0)
run_id = labels[escolhido]
trades = _trades(run_id)

if trades.empty:
    st.warning(
        f"O CSV de trades do run `{run_id}` não está em `reports/` — só as métricas agregadas "
        "sobrevivem no banco. Rode o backtest de novo para reconstruí-lo."
    )
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trades", f"{len(trades):,}")
k2.metric("Ações distintas", f"{trades['ticker'].nunique()}")
k3.metric("P&L total", f"{trades['pnl'].sum():,.0f}")
k4.metric("Acerto", f"{(trades['pnl'] > 0).mean():.0%}")
k5.metric("Permanência mediana", f"{trades['bars_held'].median():.0f} pregões")

tabela = by_ticker(trades)
st.subheader(f"Por ação ({len(tabela)})")
st.caption(
    "`contribuicao` é a fatia do P&L total do run. Clique no cabeçalho para ordenar — "
    "o topo mostra quem sustentou o resultado, o fim mostra quem o consumiu."
)
st.dataframe(
    tabela.assign(
        win_rate=tabela["win_rate"] * 100,
        contribuicao=tabela["contribuicao"] * 100,
        ret_medio=tabela["ret_medio"] * 100,
        ret_mediano=tabela["ret_mediano"] * 100,
        melhor=tabela["melhor"] * 100,
        pior=tabela["pior"] * 100,
    ),
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "trades": st.column_config.NumberColumn(format="%d"),
        "win_rate": st.column_config.NumberColumn("acerto", format="%.0f%%"),
        "pnl": st.column_config.NumberColumn("P&L", format="%.0f"),
        "contribuicao": st.column_config.NumberColumn("contribuição", format="%.1f%%"),
        "ret_medio": st.column_config.NumberColumn("ret. médio", format="%.2f%%"),
        "ret_mediano": st.column_config.NumberColumn("ret. mediano", format="%.2f%%"),
        "permanencia_mediana": st.column_config.NumberColumn("perm. mediana", format="%.0f"),
        "melhor": st.column_config.NumberColumn("melhor", format="%.1f%%"),
        "pior": st.column_config.NumberColumn("pior", format="%.1f%%"),
    },
)

n = st.slider("Quantas ações no gráfico (cada ponta)", 5, 30, 12, step=1)
extremos = pd.concat([tabela.head(n), tabela.tail(n)]).drop_duplicates(subset="ticker")
st.bar_chart(extremos.set_index("ticker")["pnl"], height=340)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Como as posições morreram")
    saidas = by_exit_reason(trades)
    st.dataframe(
        saidas.assign(share=saidas["share"] * 100, ret_medio=saidas["ret_medio"] * 100),
        hide_index=True,
        width="stretch",
        column_config={
            "exit_reason": "motivo",
            "trades": st.column_config.NumberColumn(format="%d"),
            "share": st.column_config.NumberColumn("fatia", format="%.0f%%"),
            "ret_medio": st.column_config.NumberColumn("ret. médio", format="%.2f%%"),
            "bars_mediano": st.column_config.NumberColumn("perm. mediana", format="%.0f"),
        },
    )
with col_b:
    st.subheader("Permanência (pregões)")
    st.bar_chart(trades["bars_held"].value_counts().sort_index().head(80), height=300)

st.subheader("Trades de uma ação")
ticker = st.selectbox("Ação", tabela["ticker"].tolist())
detalhe = trades[trades["ticker"] == ticker].sort_values("entry_date")
st.dataframe(
    detalhe.assign(ret=detalhe["ret"] * 100)[
        [
            "entry_date",
            "exit_date",
            "bars_held",
            "entry_price",
            "exit_price",
            "qty",
            "pnl",
            "ret",
            "exit_reason",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "entry_date": st.column_config.DateColumn("entrada", format="DD/MM/YYYY"),
        "exit_date": st.column_config.DateColumn("saída", format="DD/MM/YYYY"),
        "bars_held": st.column_config.NumberColumn("pregões", format="%d"),
        "entry_price": st.column_config.NumberColumn("preço entrada", format="%.2f"),
        "exit_price": st.column_config.NumberColumn("preço saída", format="%.2f"),
        "qty": st.column_config.NumberColumn("qtd", format="%d"),
        "pnl": st.column_config.NumberColumn("P&L", format="%.0f"),
        "ret": st.column_config.NumberColumn("retorno", format="%.2f%%"),
        "exit_reason": "motivo da saída",
    },
)
st.line_chart(detalhe.set_index("exit_date")["pnl"].cumsum(), height=240)

# ---------------------------------------------------------------- comparação entre dois runs
st.header("Comparar dois runs por ação")
st.caption("Mesma ação, dois backtests — quem melhorou e quem piorou quando o parâmetro mudou.")
c1, c2 = st.columns(2)
esq = c1.selectbox("Run A", list(labels), index=0, key="run_a")
dir_ = c2.selectbox("Run B", list(labels), index=min(1, len(labels) - 1), key="run_b")
if labels[esq] == labels[dir_]:
    st.info("Escolha dois runs diferentes.")
else:
    a, b = by_ticker(_trades(labels[esq])), by_ticker(_trades(labels[dir_]))
    if a.empty or b.empty:
        st.warning("Um dos runs está sem CSV de trades em `reports/`.")
    else:
        merged = a.merge(b, on="ticker", how="outer", suffixes=("_a", "_b")).fillna(
            {"pnl_a": 0.0, "pnl_b": 0.0, "trades_a": 0, "trades_b": 0}
        )
        merged["delta_pnl"] = merged["pnl_b"] - merged["pnl_a"]
        merged = merged.sort_values("delta_pnl", ascending=False)
        st.dataframe(
            merged[["ticker", "trades_a", "pnl_a", "trades_b", "pnl_b", "delta_pnl"]],
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "trades_a": st.column_config.NumberColumn("trades A", format="%d"),
                "pnl_a": st.column_config.NumberColumn("P&L A", format="%.0f"),
                "trades_b": st.column_config.NumberColumn("trades B", format="%d"),
                "pnl_b": st.column_config.NumberColumn("P&L B", format="%.0f"),
                "delta_pnl": st.column_config.NumberColumn("B − A", format="%.0f"),
            },
        )
