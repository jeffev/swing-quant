"""Desempenho realizado a partir do journal: marcação a mercado, ledger de trades, slippage e
aderência sinal → execução (docs/04 §4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from swing_quant.backtest.metrics import Metrics, compute_metrics

LEDGER_COLUMNS = [
    "signal_id",
    "ticker",
    "strategy",
    "market",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "qty",
    "fees",
    "pnl",
    "ret",
    "bars_held",
    "ref_price",
    "slippage_pct",
    "open",
]


def mark_to_market(
    executions: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Curva diária da carteira real: caixa + posições × fechamento.

    `executions` = `Journal.executions()` (colunas executed_at, side, price, qty, fees, ticker).
    `prices` = formato longo do store (ticker, date, close). Sem execuções → curva plana.
    Execuções em dias fora do calendário de `dates` são atribuídas ao próximo pregão disponível.
    """
    if dates is None:
        if prices.empty:
            raise ValueError("sem preços nem datas para marcar a mercado")
        dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    dates = pd.DatetimeIndex(dates)
    if executions.empty:
        return pd.DataFrame(
            {"equity": initial_capital, "cash": initial_capital, "invested": 0.0, "n_positions": 0},
            index=dates,
        )
    ex = executions.copy()
    days = pd.to_datetime(ex["executed_at"]).dt.normalize().to_numpy()
    idx = np.minimum(dates.searchsorted(days), len(dates) - 1)
    ex["day"] = dates.to_numpy()[idx]
    side = ex["side"].to_numpy()
    qty = ex["qty"].to_numpy(dtype=float)
    price = ex["price"].to_numpy(dtype=float)
    fees = ex["fees"].to_numpy(dtype=float)
    ex["signed_qty"] = np.where(side == "buy", qty, -qty)
    # compra consome caixa, venda devolve; taxas sempre saem
    ex["cash_flow"] = np.where(side == "sell", price * qty, -price * qty) - fees
    ex["ticker"] = ex["ticker"].astype(str)
    last_px = {str(k): float(v) for k, v in ex.groupby("ticker")["price"].last().items()}

    close = (
        prices.pivot_table(index="date", columns="ticker", values="close").reindex(dates).ffill()
    )
    close_np = close.to_numpy(dtype=float)
    col_idx = {str(c): i for i, c in enumerate(close.columns)}

    rows = []
    cash = initial_capital
    held: dict[str, float] = {}
    ex_by_day = {pd.Timestamp(str(d)): g for d, g in ex.groupby("day")}
    for i, day in enumerate(dates):
        g = ex_by_day.get(pd.Timestamp(day))
        if g is not None:
            cash += float(g["cash_flow"].sum())
            for t, q in g.groupby("ticker")["signed_qty"].sum().items():
                held[str(t)] = held.get(str(t), 0.0) + float(q)
        invested = 0.0
        n_pos = 0
        for t, q in held.items():
            if q <= 0:
                continue
            n_pos += 1
            j = col_idx.get(t)
            px = float(close_np[i, j]) if j is not None else math.nan
            if math.isnan(px):  # sem cotação: usa último preço executado
                px = last_px[t]
            invested += q * px
        rows.append(
            {"equity": cash + invested, "cash": cash, "invested": invested, "n_positions": n_pos}
        )
    return pd.DataFrame(rows, index=dates)


def trade_ledger(signals: pd.DataFrame, executions: pd.DataFrame) -> pd.DataFrame:
    """Um registro por sinal de compra executado: preço médio de entrada/saída, P&L, slippage
    (execução vs preço de referência do sinal) e se ainda está aberto."""
    if executions.empty or signals.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    sig = {int(str(r["id"])): r for r in signals.to_dict("records")}
    rows = []
    for sid_raw, g in executions.groupby("signal_id"):
        sid = int(str(sid_raw))
        s = sig.get(sid)
        if s is None or s["side"] != "buy":
            continue
        buys, sells = g[g["side"] == "buy"], g[g["side"] == "sell"]
        if buys.empty:
            continue
        qty_buy = float(buys["qty"].sum())
        avg_buy = float((buys["price"] * buys["qty"]).sum() / qty_buy)
        qty_sell = float(sells["qty"].sum()) if not sells.empty else 0.0
        avg_sell = float((sells["price"] * sells["qty"]).sum() / qty_sell) if qty_sell else math.nan
        fees = float(g["fees"].sum())
        is_open = qty_sell < qty_buy
        pnl = (avg_sell - avg_buy) * qty_sell - fees if qty_sell else -fees
        cost = avg_buy * qty_buy + float(buys["fees"].sum())
        entry_date = pd.Timestamp(pd.to_datetime(buys["executed_at"]).min()).normalize()
        if qty_sell:
            exit_date = pd.Timestamp(pd.to_datetime(sells["executed_at"]).max()).normalize()
            bars = int(pd.bdate_range(entry_date, exit_date).size - 1)
        else:
            exit_date, bars = pd.Timestamp("NaT"), 0
        ref_raw = s["ref_price"]
        ref = float(ref_raw) if ref_raw is not None and not pd.isna(ref_raw) else math.nan
        rows.append(
            {
                "signal_id": sid,
                "ticker": s["ticker"],
                "strategy": s["strategy"],
                "market": s["market"],
                "entry_date": entry_date,
                "entry_price": avg_buy,
                "exit_date": exit_date,
                "exit_price": avg_sell,
                "qty": qty_buy,
                "fees": fees,
                "pnl": pnl,
                "ret": pnl / cost if cost else 0.0,
                "bars_held": bars,
                "ref_price": ref,
                "slippage_pct": (avg_buy / ref - 1.0) if ref > 0 else math.nan,
                "open": is_open,
            }
        )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def adherence(signals: pd.DataFrame, executions: pd.DataFrame) -> dict[str, float]:
    """Fração de sinais de compra que viraram execução, e idem para vendas."""
    out: dict[str, float] = {}
    executed = set(executions["signal_id"]) if not executions.empty else set()
    for side in ("buy", "sell"):
        s = signals[signals["side"] == side] if not signals.empty else signals
        n = len(s)
        out[f"{side}_signals"] = float(n)
        out[f"{side}_executed_pct"] = float(s["id"].isin(executed).mean()) if n else math.nan
    return out


def realized_metrics(equity: pd.Series, ledger: pd.DataFrame) -> Metrics:
    closed = ledger[~ledger["open"]] if not ledger.empty else ledger
    trades = (
        closed.loc[:, ["pnl", "ret", "bars_held", "fees"]]
        if not closed.empty
        else pd.DataFrame(columns=["pnl", "ret", "bars_held", "fees"])
    )
    return compute_metrics(equity, trades)


def strategy_daily_returns(ledger: pd.DataFrame, equity: pd.Series) -> dict[str, pd.Series]:
    """Retornos diários aproximados por estratégia: P&L dos trades fechados distribuído
    uniformemente entre entrada e saída, dividido pelo patrimônio do dia anterior."""
    out: dict[str, pd.Series] = {}
    if ledger.empty:
        return out
    base = equity.shift(1).fillna(equity.iloc[0])
    for strat, g in ledger[~ledger["open"]].groupby("strategy"):
        pnl = pd.Series(0.0, index=equity.index)
        for p, e, x in zip(
            g["pnl"].to_numpy(dtype=float),
            pd.to_datetime(g["entry_date"]),
            pd.to_datetime(g["exit_date"]),
            strict=True,
        ):
            span = equity.index[(equity.index >= e) & (equity.index <= x)]
            if len(span) == 0:
                continue
            pnl.loc[span] += float(p) / len(span)
        out[str(strat)] = (pnl / base).fillna(0.0)
    return out
