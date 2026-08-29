"""Fixtures compartilhadas: dados sintéticos sem acesso à rede."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swing_quant.data.calendar import trading_days
from swing_quant.data.store import PRICE_COLUMNS, MarketStore


def make_prices(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    market: str = "b3",
    seed: int = 7,
) -> pd.DataFrame:
    """Random walk OHLCV alinhado ao calendário real do mercado."""
    rng = np.random.default_rng(seed)
    days = trading_days(market, start, end)  # type: ignore[arg-type]
    frames = []
    for i, t in enumerate(tickers):
        n = len(days)
        rets = rng.normal(0.0005, 0.02, n)
        close = 20.0 * (1 + i) * np.cumprod(1 + rets)
        open_ = close * (1 + rng.normal(0, 0.005, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": days,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n),
                    "source": "synthetic",
                }
            )
        )
    return pd.concat(frames, ignore_index=True).loc[:, list(PRICE_COLUMNS)]


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    return make_prices(["AAA3.SA", "BBB4.SA"], dt.date(2024, 1, 2), dt.date(2024, 12, 30))


@pytest.fixture
def store() -> MarketStore:
    s = MarketStore(":memory:")
    yield s  # type: ignore[misc]
    s.close()
