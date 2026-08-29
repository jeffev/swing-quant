"""Calendário de pregões (B3 e NYSE) via pandas_market_calendars."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Literal

import pandas as pd
import pandas_market_calendars as mcal

Market = Literal["b3", "us"]

_CAL_NAMES: dict[Market, tuple[str, ...]] = {
    "b3": ("B3", "BMF"),  # nome varia entre versões da biblioteca
    "us": ("NYSE",),
}


@lru_cache(maxsize=4)
def _calendar(market: Market) -> mcal.MarketCalendar:
    last_err: Exception | None = None
    for name in _CAL_NAMES[market]:
        try:
            return mcal.get_calendar(name)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"calendário indisponível para {market}: {last_err}")


def trading_days(market: Market, start: dt.date | str, end: dt.date | str) -> pd.DatetimeIndex:
    """Dias de pregão em [start, end], normalizados (sem timezone, meia-noite)."""
    sched = _calendar(market).schedule(start_date=str(start), end_date=str(end))
    return pd.DatetimeIndex(sched.index).normalize()


def is_trading_day(market: Market, day: dt.date | str) -> bool:
    return len(trading_days(market, day, day)) == 1


def last_trading_day(market: Market, ref: dt.date | None = None) -> dt.date:
    """Último pregão ≤ `ref` (padrão: hoje)."""
    ref = ref or dt.date.today()
    days = trading_days(market, ref - dt.timedelta(days=15), ref)
    if len(days) == 0:
        raise RuntimeError(f"nenhum pregão nos 15 dias até {ref}")
    return days[-1].date()


def next_trading_day(market: Market, ref: dt.date) -> dt.date:
    """Primeiro pregão > `ref`."""
    days = trading_days(market, ref + dt.timedelta(days=1), ref + dt.timedelta(days=15))
    if len(days) == 0:
        raise RuntimeError(f"nenhum pregão nos 15 dias após {ref}")
    return days[0].date()
