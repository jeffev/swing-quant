import datetime as dt

import pandas as pd

from swing_quant.data.quality import (
    ISSUE_COLUMNS,
    QualityThresholds,
    has_critical,
    run_checks,
    summarize,
)

AS_OF = dt.date(2024, 12, 30)  # última data do fixture sample_prices


def test_clean_data_has_no_critical(sample_prices: pd.DataFrame) -> None:
    issues = run_checks(sample_prices, "b3", as_of=AS_OF)
    assert list(issues.columns) == ISSUE_COLUMNS
    assert not has_critical(issues)
    assert summarize(issues)["critical"] == 0


def test_empty_input() -> None:
    issues = run_checks(pd.DataFrame(), "b3", as_of=AS_OF)
    assert issues.empty
    assert summarize(issues) == {"info": 0, "warning": 0, "critical": 0}


def test_detects_non_positive_and_high_lt_low(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    df.loc[0, "close"] = -1.0
    df.loc[1, ["high", "low"]] = [1.0, 2.0]
    issues = run_checks(df, "b3", as_of=AS_OF)
    checks = set(issues["check"])
    assert {"non_positive_price", "high_lt_low"} <= checks
    assert has_critical(issues)


def test_detects_zero_volume(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    df.loc[5, "volume"] = 0
    issues = run_checks(df, "b3", as_of=AS_OF)
    zero = issues[issues["check"] == "zero_volume"]
    assert len(zero) == 1 and zero["severity"].iloc[0] == "info"


def test_extreme_return_only_when_adj_also_moves(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    i = df.index[df["ticker"] == "AAA3.SA"][100]
    # split não ajustado: close e adj_close caem 50%
    df.loc[i:, "close"] = df.loc[i:, "close"] * 0.5
    df.loc[i:, "adj_close"] = df.loc[i:, "adj_close"] * 0.5
    issues = run_checks(df, "b3", as_of=AS_OF)
    assert (issues["check"] == "extreme_return").sum() == 1

    # provento ajustado corretamente: só o close cai, adj_close segue suave -> sem alerta
    df2 = sample_prices.copy()
    df2.loc[i:, "close"] = df2.loc[i:, "close"] * 0.5
    issues2 = run_checks(df2, "b3", as_of=AS_OF)
    assert (issues2["check"] == "extreme_return").sum() == 0


def test_detects_missing_days(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    aaa = df.index[df["ticker"] == "AAA3.SA"]
    df = df.drop(aaa[50:53])  # 3 pregões faltando (~1,2%) -> warning
    issues = run_checks(df, "b3", as_of=AS_OF)
    gap = issues[(issues["check"] == "missing_days") & (issues["ticker"] == "AAA3.SA")]
    assert len(gap) == 1 and gap["severity"].iloc[0] == "warning"

    df = df.drop(aaa[60:200:3])  # ~47 dias espalhados (> 5%, sem bloco dominante) -> critical
    issues = run_checks(df, "b3", as_of=AS_OF)
    gap = issues[(issues["check"] == "missing_days") & (issues["ticker"] == "AAA3.SA")]
    assert gap["severity"].iloc[0] == "critical"


def test_detects_stale(sample_prices: pd.DataFrame) -> None:
    df = sample_prices[sample_prices["date"] < "2024-12-01"]
    issues = run_checks(df, "b3", as_of=AS_OF)
    stale = issues[issues["check"] == "stale"]
    assert set(stale["ticker"]) == {"AAA3.SA", "BBB4.SA"}
    assert has_critical(issues)


def test_short_history_warning(sample_prices: pd.DataFrame) -> None:
    df = sample_prices[sample_prices["date"] >= "2024-10-01"]
    issues = run_checks(df, "b3", as_of=AS_OF, thresholds=QualityThresholds(min_history_rows=250))
    assert (issues["check"] == "short_history").sum() == 2


def test_sorted_by_severity(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    df.loc[5, "volume"] = 0  # info
    df.loc[0, "close"] = -1.0  # critical
    issues = run_checks(df, "b3", as_of=AS_OF)
    assert issues["severity"].iloc[0] == "critical"
    assert issues["severity"].iloc[-1] == "info"


def test_relisting_gap_is_warning_not_critical(sample_prices: pd.DataFrame) -> None:
    df = sample_prices.copy()
    aaa = df.index[df["ticker"] == "AAA3.SA"]
    df = df.drop(aaa[20:140])  # um único buraco contíguo de 120 pregões (~48%)
    issues = run_checks(df, "b3", as_of=AS_OF)
    mine = issues[issues["ticker"] == "AAA3.SA"]
    assert "relisting_gap" in set(mine["check"])
    assert "missing_days" not in set(mine["check"])
    assert not has_critical(mine)
