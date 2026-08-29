"""Validação de qualidade dos dados de preços (docs/07-armadilhas.md §1).

`run_checks` devolve um DataFrame de problemas com colunas
(ticker, date, check, severity, detail). Severidades: info < warning < critical.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from swing_quant.data.calendar import Market, trading_days

SEVERITIES = ("info", "warning", "critical")
ISSUE_COLUMNS = ["ticker", "date", "check", "severity", "detail"]


@dataclass(frozen=True)
class QualityThresholds:
    max_abs_return: float = 0.30  # |retorno diário| acima disso → possível split não ajustado
    max_gap_ratio: float = 0.05  # % de pregões faltando → critical acima disso
    stale_days: int = 3  # pregões sem dado → critical
    min_history_rows: int = 250  # menos que ~1 ano → warning
    relisting_block_share: float = 0.90  # ≥ 90% dos dias faltantes num só bloco → relisting_gap


def _issue(ticker: str, date: object, check: str, severity: str, detail: str) -> dict[str, object]:
    return {"ticker": ticker, "date": date, "check": check, "severity": severity, "detail": detail}


def check_ohlc_integrity(df: pd.DataFrame) -> list[dict[str, object]]:
    """Preços não positivos, high < low, close/open fora de [low, high]."""
    issues: list[dict[str, object]] = []
    bad_nonpos = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    for r in bad_nonpos.itertuples():
        issues.append(_issue(str(r.ticker), r.date, "non_positive_price", "critical", "preço <= 0"))
    bad_hl = df[df["high"] < df["low"]]
    for r in bad_hl.itertuples():
        issues.append(_issue(str(r.ticker), r.date, "high_lt_low", "critical", "high < low"))
    tol = 1e-6
    out_of_range = df[
        (df["close"] > df["high"] * (1 + tol))
        | (df["close"] < df["low"] * (1 - tol))
        | (df["open"] > df["high"] * (1 + tol))
        | (df["open"] < df["low"] * (1 - tol))
    ]
    for r in out_of_range.itertuples():
        issues.append(
            _issue(
                str(r.ticker),
                r.date,
                "close_out_of_range",
                "warning",
                "open/close fora de [low,high]",
            )
        )
    return issues


def check_zero_volume(df: pd.DataFrame) -> list[dict[str, object]]:
    zero = df[df["volume"].fillna(0) <= 0]
    return [
        _issue(str(r.ticker), r.date, "zero_volume", "info", "volume zero")
        for r in zero.itertuples()
    ]


def check_outlier_returns(df: pd.DataFrame, thr: QualityThresholds) -> list[dict[str, object]]:
    """Variações extremas em `close` E `adj_close` (se só o close variar, é provento ajustado)."""
    issues: list[dict[str, object]] = []
    for ticker, g in df.sort_values("date").groupby("ticker", sort=False):
        ret = g["close"].pct_change().abs()
        adj_ret = g["adj_close"].pct_change().abs()
        mask = (ret > thr.max_abs_return) & (adj_ret > thr.max_abs_return)
        for date, r, a in zip(g.loc[mask, "date"], ret[mask], adj_ret[mask], strict=True):
            issues.append(
                _issue(
                    str(ticker),
                    date,
                    "extreme_return",
                    "warning",
                    f"|ret close|={r:.1%} |ret adj|={a:.1%} — verificar split/erro",
                )
            )
    return issues


def check_gaps(df: pd.DataFrame, market: Market, thr: QualityThresholds) -> list[dict[str, object]]:
    """Pregões do calendário sem linha entre a primeira e a última data de cada ticker."""
    issues: list[dict[str, object]] = []
    if df.empty:
        return issues
    all_days = trading_days(market, df["date"].min().date(), df["date"].max().date())
    for ticker, g in df.groupby("ticker", sort=False):
        dates = pd.DatetimeIndex(g["date"]).normalize()
        expected = all_days[(all_days >= dates.min()) & (all_days <= dates.max())]
        missing = expected.difference(dates)
        if len(missing) == 0:
            continue
        ratio = len(missing) / max(len(expected), 1)
        block = _largest_contiguous_block(missing, expected)
        if ratio > thr.max_gap_ratio and block / len(missing) >= thr.relisting_block_share:
            # Um único buraco longo = deslistagem/relistagem, não corrupção de dados.
            issues.append(
                _issue(
                    str(ticker),
                    missing[-1],
                    "relisting_gap",
                    "warning",
                    f"bloco contíguo de {block} pregões sem dados (de {len(missing)} faltando); "
                    f"histórico útil começa após {missing[-1].date()}",
                )
            )
            continue
        severity = "critical" if ratio > thr.max_gap_ratio else "warning"
        issues.append(
            _issue(
                str(ticker),
                missing[-1],
                "missing_days",
                severity,
                f"{len(missing)} pregões faltando ({ratio:.1%}); último: {missing[-1].date()}",
            )
        )
    return issues


def _largest_contiguous_block(missing: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> int:
    """Maior sequência de pregões consecutivos (no calendário) ausentes."""
    if len(missing) == 0:
        return 0
    pos = pd.Series(expected.get_indexer(missing)).sort_values()
    breaks = pos.diff().ne(1).cumsum()
    return int(pos.groupby(breaks).size().max())


def check_stale(
    df: pd.DataFrame, market: Market, thr: QualityThresholds, as_of: dt.date
) -> list[dict[str, object]]:
    """Último dado muito antigo em relação ao último pregão."""
    issues: list[dict[str, object]] = []
    if df.empty:
        return issues
    recent = trading_days(market, as_of - dt.timedelta(days=30), as_of)
    if len(recent) == 0:
        return issues
    cutoff = recent[-min(thr.stale_days, len(recent))]
    last = df.groupby("ticker")["date"].max()
    for ticker, d in last[last < cutoff].items():
        issues.append(
            _issue(
                str(ticker),
                d,
                "stale",
                "critical",
                f"último dado {pd.Timestamp(d).date()} < {cutoff.date()}",
            )
        )
    return issues


def check_short_history(df: pd.DataFrame, thr: QualityThresholds) -> list[dict[str, object]]:
    counts = df.groupby("ticker").size()
    return [
        _issue(str(t), pd.NaT, "short_history", "warning", f"apenas {n} linhas")
        for t, n in counts[counts < thr.min_history_rows].items()
    ]


def run_checks(
    df: pd.DataFrame,
    market: Market,
    *,
    as_of: dt.date | None = None,
    thresholds: QualityThresholds | None = None,
) -> pd.DataFrame:
    """Executa todas as verificações e devolve DataFrame(ISSUE_COLUMNS) ordenado por severidade."""
    thr = thresholds or QualityThresholds()
    as_of = as_of or dt.date.today()
    if df.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    issues: list[dict[str, object]] = []
    issues += check_ohlc_integrity(df)
    issues += check_zero_volume(df)
    issues += check_outlier_returns(df, thr)
    issues += check_gaps(df, market, thr)
    issues += check_stale(df, market, thr, as_of)
    issues += check_short_history(df, thr)

    out = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    if out.empty:
        return out
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    out["_rank"] = out["severity"].map(rank)
    out = out.sort_values(["_rank", "ticker", "date"], ascending=[False, True, True])
    return out.drop(columns="_rank").reset_index(drop=True)


def summarize(issues: pd.DataFrame) -> dict[str, int]:
    """Contagem por severidade (sempre com as três chaves)."""
    counts = issues["severity"].value_counts() if not issues.empty else pd.Series(dtype=int)
    return {s: int(counts.get(s, 0)) for s in SEVERITIES}


def has_critical(issues: pd.DataFrame) -> bool:
    return bool(not issues.empty and (issues["severity"] == "critical").any())


__all__ = [
    "ISSUE_COLUMNS",
    "QualityThresholds",
    "has_critical",
    "run_checks",
    "summarize",
]
