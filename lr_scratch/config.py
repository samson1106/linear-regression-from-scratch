"""Configuration helpers for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int = 200
    noise: float = 2.5
    true_slope: float = 2.5
    true_intercept: float = 5.0
    train_ratio: float = 0.8
    seed: int = 42
    learning_rate: float = 0.05
    max_epochs: int = 2_000
    tolerance: float = 1e-8
    patience: int = 25
    early_stopping: bool = True
    scale_features: bool = True

    def __post_init__(self) -> None:
        if self.n_samples < 5:
            raise ValueError("n_samples must be at least 5.")
        if self.noise < 0:
            raise ValueError("noise must be non-negative.")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from JSON and validate known fields only."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config field(s): {sorted(unknown)}")
    return ExperimentConfig(**raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
