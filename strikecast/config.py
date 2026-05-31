from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class DataConfig(BaseModel):
    source: str = "coinbase"
    start: str = "2025-12-01"
    end: str = "2026-05-30"
    data_dir: str = "data/"
    rate_limit_req_per_sec: int = 10


class PolymarketConfig(BaseModel):
    enabled: bool = True
    mode: Literal["historical"] = "historical"


class ModelConfig(BaseModel):
    checkpoint: str = "NeoQuasar/Kronos-small"
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "auto"
    max_context: int = 512
    max_batch: int = 256


class EstimatorsConfig(BaseModel):
    sample_count: int = 1000
    temperature: float = 1.0
    top_p: float = 0.9
    garch_lookback: int = 2016

    @model_validator(mode="after")
    def validate_sample_count(self) -> EstimatorsConfig:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return self


class CalibrationConfig(BaseModel):
    method: Literal["isotonic", "platt"] = "isotonic"


class EvalConfig(BaseModel):
    train_frac: float = 0.60
    val_frac: float = 0.20
    purge_windows: int = 1
    embargo_windows: int = 1
    edge_threshold: float = 0.02
    bootstrap_samples: int = 10000
    moneyness_near_threshold: float = 0.001
    moneyness_far_threshold: float = 0.01
    label_source: Literal["coinbase", "chainlink"] = "coinbase"
    max_test_windows: int | None = None

    @model_validator(mode="after")
    def validate_split_fracs(self) -> EvalConfig:
        total = self.train_frac + self.val_frac
        if total >= 1.0:
            raise ValueError(
                f"train_frac + val_frac must sum to < 1.0, got {total}"
            )
        return self


class StrikecastConfig(BaseModel):
    seed: int = 42
    symbol: str = "BTC/USD"
    granularity: int = 300
    data: DataConfig = DataConfig()
    polymarket: PolymarketConfig = PolymarketConfig()
    model: ModelConfig = ModelConfig()
    estimators: EstimatorsConfig = EstimatorsConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def validate_granularity(self) -> StrikecastConfig:
        if self.granularity <= 0:
            raise ValueError("granularity must be positive")
        return self


def load_config(path: str | Path) -> StrikecastConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return StrikecastConfig(**raw)
