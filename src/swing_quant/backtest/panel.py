"""Constrói o painel (datas x tickers) de preços ajustados e sinais para o engine.

Ajuste total (ADR-009): fator = adj_close / close aplicado a open/high/low, de modo que
todo o OHLC fique na mesma base de retorno total. Volume financeiro usa preços brutos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swing_quant.indicators import atr, dollar_volume
from swing_quant.strategies.base import OHLCV_COLUMNS, Strategy, validate_signals


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe linhas de um ticker (formato longo do store) e devolve OHLCV ajustado por data."""
    out = df.set_index("date").sort_index()
    factor = (
        (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    )
    adj = pd.DataFrame(
        {
            "open": out["open"] * factor,
            "high": out["high"] * factor,
            "low": out["low"] * factor,
            "close": out["adj_close"],
            "volume": out["volume"].astype(float),
            "raw_close": out["close"],
        },
        index=out.index,
    )
    adj.index.name = "date"
    return adj


@dataclass
class Panel:
    """Matrizes alinhadas (índice = datas, colunas = tickers)."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    atr: pd.DataFrame
    dollar_vol: pd.DataFrame
    entry: pd.DataFrame
    exit: pd.DataFrame
    stop: pd.DataFrame
    score: pd.DataFrame
    max_hold: pd.DataFrame
    #: preço-alvo de cada entrada (NaN = sem alvo). Vazio vira um frame de NaN com a forma
    #: do `close`, para que estratégias e painéis anteriores ao alvo continuem válidos.
    target: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict[str, object] = field(default_factory=dict)
    #: ticker subjacente de cada coluna (== tickers num painel simples; "PETR4.SA" para
    #: a coluna "PETR4.SA@rsi2" num painel combinado). Uma posição por subjacente.
    underlying: list[str] = field(default_factory=list)
    #: estratégia de origem de cada coluna ("default" num painel simples).
    strategy_of: list[str] = field(default_factory=list)
    #: setor por subjacente (para o limite de exposição setorial); vazio = sem limite.
    sectors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target.empty:
            self.target = pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        if not self.underlying:
            self.underlying = list(self.tickers)
        if not self.strategy_of:
            self.strategy_of = ["default"] * len(self.tickers)

    def slice(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> Panel:
        mask = pd.Series(True, index=self.dates)
        if start is not None:
            mask &= self.dates >= start
        if end is not None:
            mask &= self.dates <= end
        idx = self.dates[mask.to_numpy()]
        return Panel(
            dates=idx,
            tickers=self.tickers,
            open=self.open.loc[idx],
            high=self.high.loc[idx],
            low=self.low.loc[idx],
            close=self.close.loc[idx],
            atr=self.atr.loc[idx],
            dollar_vol=self.dollar_vol.loc[idx],
            entry=self.entry.loc[idx],
            exit=self.exit.loc[idx],
            stop=self.stop.loc[idx],
            target=self.target.loc[idx],
            score=self.score.loc[idx],
            max_hold=self.max_hold.loc[idx],
            meta=dict(self.meta),
            underlying=list(self.underlying),
            strategy_of=list(self.strategy_of),
            sectors=dict(self.sectors),
        )


def build_panel(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    atr_period: int = 14,
    dollar_volume_window: int = 20,
    min_rows: int | None = None,
) -> Panel:
    """Gera indicadores + sinais por ticker e alinha tudo num calendário comum.

    `prices` é o formato longo do `MarketStore.get_prices`. Tickers com menos linhas que
    `min_rows` (padrão: warmup da estratégia + 20) são descartados.
    """
    min_rows = min_rows or strategy.warmup + 20
    frames: dict[str, dict[str, pd.Series]] = {}
    for ticker, g in prices.groupby("ticker", sort=True):
        if len(g) < min_rows:
            continue
        ohlc = adjust_ohlc(g)
        sig = validate_signals(strategy.generate(ohlc.loc[:, list(OHLCV_COLUMNS)]), ohlc.index)
        frames[str(ticker)] = {
            "open": ohlc["open"],
            "high": ohlc["high"],
            "low": ohlc["low"],
            "close": ohlc["close"],
            "atr": atr(ohlc["high"], ohlc["low"], ohlc["close"], atr_period),
            "dollar_vol": dollar_volume(ohlc["raw_close"], ohlc["volume"], dollar_volume_window),
            "entry": sig["entry"],
            "exit": sig["exit"],
            "stop": sig["stop"],
            "target": sig["target"],
            "score": sig["score"],
            "max_hold": sig["max_hold"],
        }
    if not frames:
        raise ValueError("nenhum ticker com histórico suficiente para o painel")

    tickers = sorted(frames)

    def wide(key: str) -> pd.DataFrame:
        return pd.concat({t: frames[t][key] for t in tickers}, axis=1).sort_index()

    entry = wide("entry").fillna(False).astype(bool)
    exit_ = wide("exit").fillna(False).astype(bool)
    max_hold = wide("max_hold").fillna(0).astype(int)
    close = wide("close")
    return Panel(
        dates=pd.DatetimeIndex(close.index),
        tickers=tickers,
        open=wide("open"),
        high=wide("high"),
        low=wide("low"),
        close=close,
        atr=wide("atr"),
        dollar_vol=wide("dollar_vol"),
        entry=entry,
        exit=exit_,
        stop=wide("stop"),
        target=wide("target"),
        score=wide("score"),
        max_hold=max_hold,
        meta={"strategy": repr(strategy), "n_tickers": len(tickers)},
    )
