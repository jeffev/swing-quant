"""Renda fixa de referência por mercado: CDI (BCB SGS série 12) e T-bills dos EUA (ETF BIL).

Serve de **baseline** nas comparações — "a estratégia rendeu mais do que deixar o dinheiro
parado?" — e só isso: o engine continua sem remunerar o caixa das carteiras simuladas, então
nenhum backtest muda por causa desta tabela.

O CDI vem em taxa diária (% ao dia, dias úteis); o T-bill é aproximado pelo retorno total do
ETF BIL (letras de 1–3 meses, com dividendos reinvestidos). Os dois viram a mesma coisa aqui:
uma série de retornos diários que se acumula por multiplicação.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from typing import Literal

import numpy as np
import pandas as pd

from swing_quant.data.calendar import Market
from swing_quant.data.store import MarketStore

RISK_FREE_COLUMNS = ("market", "date", "daily_return", "source")
RISK_FREE_LABEL: dict[str, str] = {"b3": "CDI", "us": "T-bills (BIL)"}

_SGS_CDI = 12
_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"
# a API do BCB recusa o User-Agent padrão do urllib (HTTP 406) e limita a janela por requisição
_SGS_HEADERS = {"User-Agent": "swing-quant/1.0"}
_SGS_MAX_YEARS = 10


def _sgs(serie: int, start: dt.date, end: dt.date) -> list[dict[str, str]]:
    """Série do SGS em blocos de até 10 anos (limite da API)."""
    out: list[dict[str, str]] = []
    ini = start
    while ini <= end:
        fim = min(end, dt.date(ini.year + _SGS_MAX_YEARS - 1, 12, 31))
        url = _SGS_URL.format(serie=serie, ini=f"{ini:%d/%m/%Y}", fim=f"{fim:%d/%m/%Y}")
        req = urllib.request.Request(url, headers=_SGS_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            out += json.loads(resp.read())
        ini = fim + dt.timedelta(days=1)
    return out


def fetch_cdi(start: dt.date, end: dt.date) -> pd.DataFrame:
    """CDI diário do BCB, já convertido de "% ao dia" para retorno decimal."""
    rows = _sgs(_SGS_CDI, start, end)
    if not rows:
        return pd.DataFrame(columns=list(RISK_FREE_COLUMNS))
    df = pd.DataFrame(rows).drop_duplicates(subset="data")
    return pd.DataFrame(
        {
            "market": "b3",
            "date": pd.to_datetime(df["data"], format="%d/%m/%Y").dt.date,
            "daily_return": df["valor"].astype(float) / 100.0,
            "source": "bcb_sgs_12",
        }
    )


def fetch_us_tbill(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Retorno diário das letras curtas dos EUA, via retorno total do ETF BIL."""
    import yfinance as yf  # import tardio: pesado e só usado com rede

    raw = yf.download(
        "BIL",
        start=str(start),
        end=str(end + dt.timedelta(days=1)),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=list(RISK_FREE_COLUMNS))
    close = raw["Close"].squeeze()
    rets = close.pct_change().dropna()
    return pd.DataFrame(
        {
            "market": "us",
            "date": pd.DatetimeIndex(rets.index).date,
            "daily_return": rets.to_numpy(dtype=float),
            "source": "yfinance_BIL",
        }
    )


def save_risk_free(store: MarketStore, df: pd.DataFrame) -> int:
    """Upsert idempotente por (mercado, data)."""
    if df.empty:
        return 0
    df = df.loc[:, list(RISK_FREE_COLUMNS)]
    store.con.register("_rf_in", df)
    store.con.execute("INSERT OR REPLACE INTO risk_free SELECT * FROM _rf_in")
    store.con.unregister("_rf_in")
    return len(df)


def update_risk_free(
    store: MarketStore,
    market: Literal["b3", "us"],
    start: dt.date,
    end: dt.date | None = None,
) -> int:
    """Baixa e grava a série do mercado; devolve quantas linhas foram gravadas."""
    end = end or dt.date.today()
    df = fetch_cdi(start, end) if market == "b3" else fetch_us_tbill(start, end)
    return save_risk_free(store, df)


def risk_free_daily(store: MarketStore, market: Market) -> pd.DataFrame:
    """Série diária gravada (colunas date, daily_return), ordenada."""
    return store.con.execute(
        "SELECT date, daily_return FROM risk_free WHERE market = ? ORDER BY date",
        [market],
    ).df()


def annual_returns(daily: pd.DataFrame) -> pd.Series:
    """Retorno acumulado por ano civil a partir da série diária."""
    if daily.empty:
        return pd.Series(dtype=float)
    anos = pd.to_datetime(daily["date"]).dt.year.to_numpy(dtype=int)
    rets = daily["daily_return"].to_numpy(dtype=float)
    acumulado = {int(ano): float(np.prod(1.0 + rets[anos == ano]) - 1.0) for ano in np.unique(anos)}
    return pd.Series(acumulado)
