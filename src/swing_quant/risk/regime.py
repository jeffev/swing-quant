"""Filtros de regime de mercado (docs/02 §C, docs/03 §1).

Todas as séries são indexadas por data e calculadas com dados até o fechamento de D; o engine
aplica `allow_entries[D]` aos sinais gerados em D (executados em D+1) e `size_factor[D]` ao
sizing das entradas de D+1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from swing_quant.indicators import sma


@dataclass(frozen=True)
class RegimeConfig:
    trend_sma: int = 200
    vol_window: int = 20
    vol_percentile: float = 0.90
    vol_lookback: int = 756  # ~3 anos para o percentil rolante
    high_vol_size_factor: float = 0.5
    use_trend: bool = True  # False -> allow_entries sempre True (ADR-013)
    use_vol: bool = True  # False -> size_factor sempre 1.0


@dataclass
class Regime:
    allow_entries: pd.Series  # bool por data
    size_factor: pd.Series  # float por data
    trend_on: pd.Series
    high_vol: pd.Series
    realized_vol: pd.Series

    def summary(self) -> dict[str, float]:
        return {
            "pct_days_trend_on": float(self.trend_on.mean()),
            "pct_days_high_vol": float(self.high_vol.mean()),
        }


def trend_filter(bench_close: pd.Series, n: int = 200) -> pd.Series:
    """True quando o benchmark fecha acima da sua SMA(n). NaN (aquecimento) -> False."""
    return (bench_close > sma(bench_close, n)).fillna(False).astype(bool)


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Volatilidade anualizada dos retornos diários (janela móvel)."""
    return close.pct_change().rolling(window, min_periods=window).std(ddof=1) * math.sqrt(252)


def high_volatility(
    close: pd.Series, window: int = 20, percentile: float = 0.90, lookback: int = 756
) -> pd.Series:
    """True quando a vol realizada está acima do percentil histórico rolante (sem look-ahead).

    O percentil é calculado sobre a janela `lookback` **anterior** (shift 1), com mínimo de
    `window*5` observações.
    """
    vol = realized_volatility(close, window)
    thr = vol.shift(1).rolling(lookback, min_periods=window * 5).quantile(percentile)
    return (vol > thr).fillna(False).astype(bool)


def build_regime(bench_close: pd.Series, cfg: RegimeConfig | None = None) -> Regime:
    cfg = cfg or RegimeConfig()
    close = bench_close.dropna().sort_index()
    trend = trend_filter(close, cfg.trend_sma)
    hv = high_volatility(close, cfg.vol_window, cfg.vol_percentile, cfg.vol_lookback)
    factor = pd.Series(1.0, index=close.index).where(~hv, cfg.high_vol_size_factor)
    allow = trend if cfg.use_trend else pd.Series(True, index=close.index)
    size = factor if cfg.use_vol else pd.Series(1.0, index=close.index)
    return Regime(
        allow_entries=allow,
        size_factor=size,
        trend_on=trend,
        high_vol=hv,
        realized_vol=realized_volatility(close, cfg.vol_window),
    )
