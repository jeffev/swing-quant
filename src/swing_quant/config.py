"""Carregamento e validação da configuração (config.yaml) via pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class CapitalConfig(BaseModel):
    initial: float = Field(gt=0)
    currency: str = "BRL"
    #: capital dedicado por mercado (sleeves independentes); ausente -> `initial`
    initial_by_market: dict[str, float] = Field(default_factory=dict)

    #: caixa não investido rende a renda fixa do mercado (ADR-020). Desligar volta ao
    #: comportamento antigo (caixa a 0%) — e aí o Sharpe também deve voltar a ser contra zero.
    cash_earns_risk_free: bool = True

    def for_market(self, market: str) -> float:
        return float(self.initial_by_market.get(market, self.initial))


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(gt=0, le=0.05)
    #: risco por trade dedicado por mercado (ADR-018); ausente -> `risk_per_trade`
    risk_per_trade_by_market: dict[str, float] = Field(default_factory=dict)
    atr_multiple_default: float = Field(gt=0)
    max_position_pct: float = Field(gt=0, le=1)
    max_positions: int = Field(ge=1)
    max_gross_exposure: float = Field(gt=0)
    max_sector_pct: float = Field(gt=0, le=1)
    max_correlation: float = Field(ge=0, le=1)
    max_strategy_pct: float = Field(gt=0, le=1)
    max_volume_participation: float = Field(gt=0, le=1)
    monthly_dd_reduce: float = Field(gt=0, le=1)
    circuit_breaker_dd: float = Field(gt=0, le=1)
    board_lot: int = Field(ge=1)

    def risk_for_market(self, market: str) -> float:
        """Risco por trade da sleeve; cada mercado tem seu orçamento de drawdown (ADR-018)."""
        return float(self.risk_per_trade_by_market.get(market, self.risk_per_trade))

    @field_validator("risk_per_trade_by_market")
    @classmethod
    def _per_market_within_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        for market, value in v.items():
            if not 0 < value <= 0.05:
                raise ValueError(f"risk_per_trade_by_market[{market}] fora de (0; 0,05]: {value}")
        return v


class MarketCosts(BaseModel):
    commission_per_order: float = Field(ge=0)
    fees_pct: float = Field(ge=0)
    slippage_pct_liquid: float = Field(ge=0)
    slippage_pct_illiquid: float = Field(ge=0)


class CostsConfig(BaseModel):
    b3: MarketCosts
    us: MarketCosts


class MarketUniverse(BaseModel):
    index: str
    benchmark: str
    suffix: str = ""
    min_avg_dollar_volume_20d: float = Field(ge=0)


class UniverseConfig(BaseModel):
    b3: MarketUniverse
    us: MarketUniverse


class RegimeConfig(BaseModel):
    benchmark_sma: int = Field(ge=1)
    high_vol_percentile: float = Field(gt=0, lt=1)
    high_vol_size_factor: float = Field(gt=0, le=1)
    trend_filter: bool = False  # ADR-013: prejudica trend-following na B3
    vol_filter: bool = True


class DataConfig(BaseModel):
    db_path: Path
    parquet_dir: Path
    history_start: str
    execution: Literal["next_open", "next_close"] = "next_open"


class WalkForwardConfig(BaseModel):
    train_years: int = Field(ge=1)
    test_years: int = Field(ge=1)
    anchored: bool = False


class ValidationConfig(BaseModel):
    split: list[float]
    walkforward: WalkForwardConfig
    min_trades: int = Field(ge=1)
    min_test_trades: int = Field(ge=1)
    cost_multipliers: list[float]
    monte_carlo_runs: int = Field(ge=1)

    @field_validator("split")
    @classmethod
    def _split_sums_to_one(cls, v: list[float]) -> list[float]:
        if len(v) != 3 or abs(sum(v) - 1.0) > 1e-9:
            raise ValueError("split deve ter 3 frações que somam 1.0")
        return v


class TelegramConfig(BaseModel):
    enabled: bool = False
    top_n: int = Field(ge=1)


class AlertsConfig(BaseModel):
    telegram: TelegramConfig


class Config(BaseModel):
    capital: CapitalConfig
    risk: RiskConfig
    costs: CostsConfig
    universe: UniverseConfig
    regime: RegimeConfig
    data: DataConfig
    validation: ValidationConfig
    # Parâmetros de estratégia são livres (cada Strategy valida os seus).
    strategies: dict[str, dict[str, object]]
    alerts: AlertsConfig

    def market_costs(self, market: Literal["b3", "us"]) -> MarketCosts:
        return getattr(self.costs, market)  # type: ignore[no-any-return]

    def market_universe(self, market: Literal["b3", "us"]) -> MarketUniverse:
        return getattr(self.universe, market)  # type: ignore[no-any-return]

    def enabled_strategies(self, market: str) -> list[str]:
        """Estratégias com `enabled: true` cujo `markets` (padrão: todos) inclui `market`."""
        out = []
        for name, params in self.strategies.items():
            if not params.get("enabled"):
                continue
            markets = params.get("markets")
            if markets is None or (
                isinstance(markets, (list, tuple)) and market in [str(m) for m in markets]
            ):
                out.append(name)
        return out


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lê o YAML e devolve um `Config` validado."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
