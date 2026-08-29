import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from swing_quant.backtest.engine import CostModel, RiskModel
from swing_quant.backtest.protocol import ProtocolResult, run_protocol
from swing_quant.backtest.report import render_markdown, save_report
from swing_quant.backtest.validation import default_panel_factory
from swing_quant.data.store import MarketStore
from swing_quant.strategies import RSI2
from tests.conftest import make_prices


@pytest.fixture(scope="module")
def result() -> ProtocolResult:
    prices = make_prices(
        ["AAA3.SA", "BBB4.SA", "CCC3.SA"], dt.date(2015, 1, 2), dt.date(2020, 12, 30), seed=5
    )
    return run_protocol(
        RSI2(),
        default_panel_factory(prices),
        market="b3",
        costs=CostModel(0.0, 0.0003, 0.001),
        risk=RiskModel(initial_capital=100_000, board_lot=1),
        grid={"rsi_entry": [10.0, 15.0]},
        train_years=2,
        test_years=1,
        mc_runs=20,
        boot_runs=20,
        baseline_runs=2,
        min_trades_select=1,
    )


def test_render_markdown_sections(result: ProtocolResult) -> None:
    md = render_markdown(result)
    for section in (
        "# Backtest — rsi2 / B3",
        "## Checklist de aprovação",
        "## Split temporal",
        "## Métricas",
        "## Robustez de parâmetros",
        "## Walk-forward",
        "## Drawdown simulado",
        "Bootstrap 1 ano (gate)",
        "Bootstrap horizonte completo",
        "MC ordem dos trades",
        "**Calibração**",
        "## Bootstrap do Sharpe",
        "## Sensibilidade a custos",
        "## Baseline aleatória",
        "## Observações",
        "## Saídas por motivo",
    ):
        assert section in md
    assert ("APROVADA" in md) or ("REPROVADA" in md)
    assert "sobrevivência" in md


def test_save_report_writes_files_and_persists(result: ProtocolResult, tmp_path: Path) -> None:
    with MarketStore(":memory:") as store:
        md_path = save_report(result, tmp_path, store)
        assert md_path.exists() and md_path.suffix == ".md"
        files = {p.suffix for p in tmp_path.iterdir()}
        assert ".csv" in files
        assert ".png" in files  # matplotlib instalado
        rows = store.con.execute("SELECT strategy, params, metrics FROM backtest_runs").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "rsi2/b3"
        assert "rsi_entry" in rows[0][1]
        assert "approved" in rows[0][2]
    trades = pd.read_csv(next(tmp_path.glob("*_trades.csv")))
    assert {"ticker", "entry_date", "pnl", "exit_reason"} <= set(trades.columns)
