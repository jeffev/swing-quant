"""Exploração dos backtests já rodados: comparação entre runs e resultado por ação.

`backtest_runs` guarda as métricas agregadas de cada execução do protocolo; os trades ficam no
CSV que o relatório escreveu (`reports/<prefixo>_trades.csv`, o `run_id` sem o hash final).
Este módulo junta as duas pontas para o dashboard e para quem quiser abrir no notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swing_quant.data.store import MarketStore

RUN_COLUMNS = [
    "run_id",
    "estrategia",
    "mercado",
    "quando",
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

TICKER_COLUMNS = [
    "ticker",
    "trades",
    "win_rate",
    "pnl",
    "contribuicao",
    "ret_medio",
    "ret_mediano",
    "permanencia_mediana",
    "melhor",
    "pior",
]


def _as_dict(value: Any) -> dict[str, Any]:
    """Campos JSON do DuckDB voltam como `str` ou já desserializados, dependendo da versão."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


def _num(section: Any, key: str) -> float:
    if not isinstance(section, dict):
        return float("nan")
    value = section.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def load_runs(store: MarketStore, market: str | None = None, limit: int = 200) -> pd.DataFrame:
    """Uma linha por backtest registrado, com as métricas do teste OOS já achatadas.

    `strategy` no banco é `nome/mercado` (ex.: `donchian/b3`); aqui vira duas colunas.
    """
    rows = store.con.execute(
        "SELECT run_id, strategy, params, metrics, created_at "
        "FROM backtest_runs ORDER BY created_at DESC LIMIT ?",
        [limit],
    ).df()
    if rows.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)

    out: list[dict[str, Any]] = []
    for _, r in rows.iterrows():
        name, _, mkt = str(r["strategy"]).partition("/")
        metrics = _as_dict(r["metrics"])
        test = metrics.get("test", {})
        params = _as_dict(r["params"])
        out.append(
            {
                "run_id": str(r["run_id"]),
                "estrategia": name,
                "mercado": mkt,
                "quando": pd.Timestamp(r["created_at"]),
                "aprovada": bool(metrics.get("approved", False)),
                "sharpe_oos": _num(test, "sharpe"),
                "cagr_oos": _num(test, "cagr"),
                "mdd_oos": _num(test, "max_drawdown"),
                "dd_p95_1a": _num(metrics.get("dd_bootstrap", {}), "mdd_p95"),
                "trades": _num(test, "n_trades"),
                "win_rate": _num(test, "win_rate"),
                "profit_factor": _num(test, "profit_factor"),
                "permanencia_media": _num(test, "avg_hold_bars"),
                "params": ", ".join(f"{k}={_fmt_param(v)}" for k, v in params.items()),
            }
        )
    df = pd.DataFrame(out, columns=RUN_COLUMNS)
    if market:
        df = df[df["mercado"] == market].reset_index(drop=True)
    return df


def _fmt_param(value: Any) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def trades_path(run_id: str, reports_dir: Path) -> Path:
    """CSV de trades do run: mesmo prefixo do `run_id`, sem o hash do fim."""
    return Path(reports_dir) / f"{run_id.rsplit('_', 1)[0]}_trades.csv"


def load_trades(run_id: str, reports_dir: Path) -> pd.DataFrame:
    """Trades de um run; DataFrame vazio se o CSV do relatório não existir mais."""
    path = trades_path(run_id, reports_dir)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    df["run_id"] = run_id
    return df


def by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    """Agrega os trades por ação: quantos, quanto renderam e quanto tempo ficaram na carteira.

    `contribuicao` é a fatia do P&L total do run — negativa quando a ação destruiu resultado.
    """
    if trades.empty:
        return pd.DataFrame(columns=TICKER_COLUMNS)
    g = trades.groupby("ticker")
    total = float(trades["pnl"].sum())
    out = pd.DataFrame(
        {
            "trades": g["pnl"].size(),
            "win_rate": g["pnl"].apply(lambda s: float((s > 0).mean())),
            "pnl": g["pnl"].sum(),
            "ret_medio": g["ret"].mean(),
            "ret_mediano": g["ret"].median(),
            "permanencia_mediana": g["bars_held"].median(),
            "melhor": g["ret"].max(),
            "pior": g["ret"].min(),
        }
    ).reset_index()
    out["contribuicao"] = out["pnl"] / total if total else np.nan
    return out[TICKER_COLUMNS].sort_values("pnl", ascending=False).reset_index(drop=True)


def by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """Como as posições morreram: sinal, stop, tempo ou fim do período."""
    if trades.empty:
        return pd.DataFrame(columns=["exit_reason", "trades", "share", "ret_medio", "bars_mediano"])
    g = trades.groupby("exit_reason")
    out = pd.DataFrame(
        {
            "trades": g["pnl"].size(),
            "ret_medio": g["ret"].mean(),
            "bars_mediano": g["bars_held"].median(),
        }
    ).reset_index()
    out["share"] = out["trades"] / len(trades)
    return (
        out[["exit_reason", "trades", "share", "ret_medio", "bars_mediano"]]
        .sort_values("trades", ascending=False)
        .reset_index(drop=True)
    )


def yearly_returns(trades: pd.DataFrame, capital: float = 100_000.0) -> pd.Series:
    """Retorno por ano civil da carteira do run, a partir do P&L dos trades.

    O patrimônio é reconstruído como `capital` mais o P&L acumulado na data de **saída** —
    posições que atravessam a virada do ano têm o resultado inteiro contado no ano em que
    fecharam. O acumulado do período bate com o retorno total do backtest; os anos individuais
    são uma aproximação boa o bastante para comparar, não para reportar.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    pnl = trades.groupby(pd.to_datetime(trades["exit_date"]).dt.year)["pnl"].sum()
    anos = [int(a) for a in pnl.index.to_numpy(dtype=int)]
    valores = [float(v) for v in pnl.to_numpy(dtype=float)]
    equity = capital
    out: dict[int, float] = {}
    for ano, valor in zip(anos, valores, strict=True):
        out[ano] = valor / equity
        equity += valor
    return pd.Series(out)


def benchmark_yearly(
    store: MarketStore, ticker: str, start: str | None = None, end: str | None = None
) -> pd.Series:
    """Retorno por ano civil do índice (primeiro e último fechamento ajustado de cada ano)."""
    sql = "SELECT date, adj_close FROM prices WHERE ticker = ?"
    params: list[Any] = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    d = store.con.execute(sql + " ORDER BY date", params).df()
    if d.empty:
        return pd.Series(dtype=float)
    ano = pd.to_datetime(d["date"]).dt.year
    primeiro = d.groupby(ano)["adj_close"].first()
    ultimo = d.groupby(ano)["adj_close"].last()
    return (ultimo / primeiro - 1.0).rename(None)
