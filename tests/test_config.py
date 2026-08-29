"""Testes do carregamento/validação de configuração."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swing_quant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.capital.initial > 0
    assert 0 < cfg.risk.risk_per_trade <= 0.05
    assert cfg.data.execution == "next_open"
    assert "rsi2" in cfg.strategies


def test_split_must_sum_to_one(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["validation"]["split"] = [0.5, 0.3, 0.3]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="split"):
        load_config(bad)


def test_risk_per_trade_upper_bound(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["risk"]["risk_per_trade"] = 0.10
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_enabled_strategies_by_market_and_capital() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert "donchian" in cfg.enabled_strategies("b3")
    assert "donchian" not in cfg.enabled_strategies("us")
    assert "momentum" in cfg.enabled_strategies("us")
    assert "momentum" not in cfg.enabled_strategies("b3")
    assert cfg.capital.for_market("us") == 20000
    assert cfg.capital.for_market("b3") == 100000
    assert cfg.capital.for_market("xx") == cfg.capital.initial


def test_risk_per_trade_by_market() -> None:
    """ADR-018: cada sleeve tem seu risco, calibrado pelo orçamento de DD."""
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.risk.risk_for_market("us") == pytest.approx(0.0035)
    assert cfg.risk.risk_for_market("b3") == pytest.approx(0.005)
    assert cfg.risk.risk_for_market("xx") == cfg.risk.risk_per_trade


def test_risk_per_trade_by_market_bounds(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["risk"]["risk_per_trade_by_market"]["us"] = 0.20
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="risk_per_trade_by_market"):
        load_config(bad)
