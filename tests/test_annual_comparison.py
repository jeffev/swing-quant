"""A tabela ano a ano (notebooks/annual_comparison.py) e o cálculo anual de `asset_classes`.

O que erra em silêncio aqui é a janela: uma linha que começa tarde encolhe a janela comum de
todo mundo, e um CAGR calculado em janelas diferentes ordena a tabela errado sem avisar. O resto
do módulo é montagem que precisa do banco — fica para o notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import annual_comparison as ac  # noqa: E402
import asset_classes as acl  # noqa: E402


def _months(n: int, start: str = "2010-01") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="M")


def _flat(value: float, n: int, start: str = "2010-01") -> pd.Series:
    return pd.Series([value] * n, index=_months(n, start))


# --------------------------------------------------------------------------- cálculo anual
def test_annual_returns_compound_inside_the_year() -> None:
    r = _flat(0.01, 24)
    out = acl.annual_returns(r)
    assert out.index.tolist() == [2010, 2011]
    assert out.iloc[0] == pytest.approx(1.01**12 - 1)


def test_annual_returns_deflate_when_given_inflation() -> None:
    """Retorno real não é retorno menos inflação: é a razão entre os dois."""
    nominal = _flat(0.01, 12)
    inflation = _flat(0.004, 12)
    real = acl.annual_returns(nominal, inflation)
    assert real.iloc[0] == pytest.approx((1.01 / 1.004) ** 12 - 1)
    assert real.iloc[0] != pytest.approx(acl.annual_returns(nominal).iloc[0] - (1.004**12 - 1))


def test_cagr_annualises_a_partial_year() -> None:
    assert acl.cagr(_flat(0.01, 6)) == pytest.approx(1.01**12 - 1)


# --------------------------------------------------------------------------- janela comum
def test_common_window_is_where_every_line_exists() -> None:
    lines = [
        ac.Line("a", ac.SWING, _flat(0.01, 36, "2010-01")),
        ac.Line("b", ac.CLASSES, _flat(0.01, 24, "2011-01")),
    ]
    start, end = ac.common_window(lines, _flat(0.003, 60, "2010-01"))
    assert (start, end) == (pd.Period("2011-01", "M"), pd.Period("2012-12", "M"))


def test_late_start_lines_do_not_shrink_the_window_for_everyone() -> None:
    """O bitcoin começa quatro anos e meio depois; quem paga por isso não pode ser o resto."""
    lines = [
        ac.Line("a", ac.SWING, _flat(0.01, 36, "2010-01")),
        ac.Line("cripto", ac.CLASSES, _flat(0.02, 12, "2012-01"), tags=("late_start",)),
    ]
    start, _ = ac.common_window(lines, _flat(0.003, 60, "2010-01"))
    assert start == pd.Period("2010-01", "M")


def test_common_window_is_bounded_by_the_deflator() -> None:
    """Sem índice de preços não há retorno real — a janela termina onde o IPCA termina."""
    lines = [ac.Line("a", ac.SWING, _flat(0.01, 36, "2010-01"))]
    _, end = ac.common_window(lines, _flat(0.003, 24, "2010-01"))
    assert end == pd.Period("2011-12", "M")


# --------------------------------------------------------------------------- a tabela
def test_annual_frame_keeps_each_row_history_but_ranks_on_the_common_window() -> None:
    """A linha curta mostra os anos que teve e um traço nos que não teve.

    O CAGR das duas é medido só onde as duas existem, senão a que viveu num período melhor
    ganharia a tabela por ter nascido na hora certa.
    """
    inflation = _flat(0.0, 36, "2010-01")
    lines = [
        ac.Line("longa", ac.SWING, _flat(0.01, 36, "2010-01")),
        ac.Line("curta", ac.CLASSES, _flat(0.01, 24, "2011-01")),
    ]
    frame = ac.annual_frame(lines, inflation)
    assert frame.attrs["window"] == (pd.Period("2011-01", "M"), pd.Period("2012-12", "M"))
    assert pd.isna(frame.loc["curta", "2010"])
    assert frame.loc["curta", "desde"] == 2011
    assert frame.loc["longa", "CAGR"] == pytest.approx(frame.loc["curta", "CAGR"])


def test_groups_come_out_in_order_and_sorted_by_cagr() -> None:
    inflation = _flat(0.0, 36)
    lines = [
        ac.Line("classe fraca", ac.CLASSES, _flat(0.001, 36)),
        ac.Line("classe forte", ac.CLASSES, _flat(0.02, 36)),
        ac.Line("swing", ac.SWING, _flat(0.005, 36)),
        ac.Line("carteira", ac.PORTFOLIOS, _flat(0.005, 36)),
    ]
    frame = ac.annual_frame(lines, inflation)
    assert frame["grupo"].tolist() == [ac.SWING, ac.PORTFOLIOS, ac.CLASSES, ac.CLASSES]
    assert frame.index.tolist()[-2:] == ["classe forte", "classe fraca"]


def test_monthly_from_equity_uses_month_end_closes() -> None:
    """A curva do swing é diária; o resto da tabela é mensal, e a ponte é o fechamento do mês."""
    days = pd.bdate_range("2010-01-01", "2010-03-31")
    equity = pd.Series(range(100, 100 + len(days)), index=days, dtype=float)
    monthly = ac._monthly_from_equity(equity)
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert monthly.index[0] == pd.Period("2010-02", "M")
    jan_close = float(equity[equity.index.month == 1].iloc[-1])
    feb_close = float(equity[equity.index.month == 2].iloc[-1])
    assert monthly.iloc[0] == pytest.approx(feb_close / jan_close - 1.0)
