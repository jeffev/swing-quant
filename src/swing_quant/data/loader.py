"""Download de OHLCV via yfinance e atualização incremental do `MarketStore`."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from swing_quant.data.store import PRICE_COLUMNS, MarketStore

log = logging.getLogger(__name__)

_YF_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
BATCH_SIZE = 50
# Rebaixar alguns dias no incremental: yfinance pode revisar o último pregão e o
# adj_close muda em ex-data de proventos (o histórico inteiro só é refeito com full=True).
INCREMENTAL_LOOKBACK_DAYS = 7


def to_long(raw: pd.DataFrame, tickers: Sequence[str], source: str = "yfinance") -> pd.DataFrame:
    """Converte o DataFrame largo do yfinance (group_by='ticker') para o formato longo."""
    if raw.empty:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = pd.DataFrame(raw.xs(t, axis=1, level=0)).copy()
            sub["ticker"] = t
            frames.append(sub)
    else:  # ticker único: colunas planas
        single = raw.copy()
        single["ticker"] = tickers[0]
        frames.append(single)

    if not frames:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))

    df = pd.concat(frames)
    df.index.name = "date"
    df = df.reset_index().rename(columns=_YF_FIELDS)
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["source"] = source
    df = df.dropna(subset=["close"])
    return df.loc[:, list(PRICE_COLUMNS)].sort_values(["ticker", "date"]).reset_index(drop=True)


def download(
    tickers: Sequence[str],
    start: dt.date | str,
    end: dt.date | str | None = None,
) -> pd.DataFrame:
    """Baixa OHLCV (não ajustado + adj_close) para os tickers, em formato longo."""
    import yfinance as yf  # import tardio: pesado e só usado com rede

    if not tickers:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    raw = yf.download(
        tickers=list(tickers),
        start=str(pd.Timestamp(start).date()),
        end=str((end_ts + pd.Timedelta(days=1)).date()),  # yfinance: `end` é exclusivo
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return to_long(raw, tickers)


@dataclass
class UpdateResult:
    requested: int
    downloaded_rows: int = 0
    tickers_updated: int = 0
    tickers_failed: list[str] = field(default_factory=list)
    up_to_date: int = 0
    repaired_rows: int = 0


def update_prices(
    store: MarketStore,
    tickers: Sequence[str],
    history_start: dt.date | str,
    *,
    full: bool = False,
    as_of: dt.date | None = None,
    downloader: Callable[[Sequence[str], dt.date | str, dt.date | str | None], pd.DataFrame]
    | None = None,
) -> UpdateResult:
    """Atualiza o store de forma incremental (ou completa com `full=True`).

    `downloader` é injetável para testes (assinatura igual a `download`).
    """
    dl = downloader or download
    as_of = as_of or dt.date.today()
    result = UpdateResult(requested=len(tickers))
    history_start = pd.Timestamp(history_start).date()

    last = {} if full else store.last_dates(tickers)
    # Agrupa tickers pela data de início necessária para minimizar requisições.
    groups: dict[dt.date, list[str]] = {}
    for t in tickers:
        if t in last:
            if last[t] >= as_of:
                result.up_to_date += 1
                continue
            start = max(history_start, last[t] - dt.timedelta(days=INCREMENTAL_LOOKBACK_DAYS))
        else:
            start = history_start
        groups.setdefault(start, []).append(t)

    for start, group in groups.items():
        for i in range(0, len(group), BATCH_SIZE):
            batch = group[i : i + BATCH_SIZE]
            try:
                df = dl(batch, start, as_of)
            except Exception as exc:
                log.warning("falha ao baixar lote %s: %s", batch[:3], exc)
                result.tickers_failed.extend(batch)
                continue
            got = set(df["ticker"].unique()) if not df.empty else set()
            result.tickers_failed.extend(t for t in batch if t not in got)
            result.downloaded_rows += store.upsert_prices(df)
            result.tickers_updated += len(got)

    # Erro conhecido da fonte: barras com high < low em alguns dias (ex.: B3 em 2012-10-10).
    result.repaired_rows = store.repair_high_low()
    if result.repaired_rows:
        log.warning("%d barras com high<low reparadas (source += '+repair')", result.repaired_rows)
    return result
