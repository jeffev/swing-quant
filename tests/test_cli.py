"""Testes de fumaça do CLI."""

from pathlib import Path

from typer.testing import CliRunner

from swing_quant import __version__
from swing_quant.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_show_config() -> None:
    result = runner.invoke(app, ["show-config", "-c", str(ROOT / "config.yaml")])
    assert result.exit_code == 0
    assert "donchian" in result.stdout  # única estratégia habilitada (ADR-013)


def test_backtest_unknown_strategy() -> None:
    result = runner.invoke(
        app, ["backtest", "--strategy", "nao-existe", "-c", str(ROOT / "config.yaml")]
    )
    assert result.exit_code == 1
