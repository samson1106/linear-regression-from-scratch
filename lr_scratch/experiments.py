"""Experiment utilities for comparison, benchmarking, and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
import time
from typing import TypeAlias

import numpy as np

from .config import ExperimentConfig
from .data import generate_linear_data, train_test_split
from .metrics import evaluate_regression
from .model import LinearRegressionGD
from .preprocessing import StandardScalerScratch

LOGGER = logging.getLogger(__name__)
ExperimentRow: TypeAlias = dict[str, float | int | str | bool]


@dataclass(frozen=True)
class TrainingBundle:
    model: LinearRegressionGD
    scaler: StandardScalerScratch | None
    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    elapsed_seconds: float


def prepare_split(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = generate_linear_data(
        n_samples=config.n_samples,
        noise=config.noise,
        true_slope=config.true_slope,
        true_intercept=config.true_intercept,
        seed=config.seed,
    )
    return train_test_split(x, y, config.train_ratio, config.seed)


def train_from_config(config: ExperimentConfig) -> TrainingBundle:
    x_train_raw, x_test_raw, y_train, y_test = prepare_split(config)
    scaler = StandardScalerScratch() if config.scale_features else None
    if scaler is None:
        x_train, x_test = x_train_raw, x_test_raw
    else:
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

    model = LinearRegressionGD(
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        tolerance=config.tolerance,
        patience=config.patience,
        early_stopping=config.early_stopping,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train, x_test, y_test)
    elapsed = time.perf_counter() - start
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    return TrainingBundle(
        model=model,
        scaler=scaler,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        train_metrics=evaluate_regression(y_train, y_train_pred),
        test_metrics=evaluate_regression(y_test, y_test_pred),
        elapsed_seconds=elapsed,
    )


def run_learning_rate_experiment(
    config: ExperimentConfig,
    learning_rates: Iterable[float],
) -> list[ExperimentRow]:
    results: list[ExperimentRow] = []
    for learning_rate in learning_rates:
        trial_config = ExperimentConfig(**{**config.to_dict(), "learning_rate": learning_rate})
        try:
            bundle = train_from_config(trial_config)
            assert bundle.model.cost_history_ is not None
            results.append(
                {
                    "learning_rate": learning_rate,
                    "epochs_run": bundle.model.epochs_run_,
                    "final_train_cost": float(bundle.model.cost_history_[-1]),
                    "test_rmse": bundle.test_metrics["rmse"],
                    "test_r2": bundle.test_metrics["r2"],
                    "stopped_reason": bundle.model.stopped_reason_,
                    "elapsed_ms": bundle.elapsed_seconds * 1000.0,
                }
            )
        except Exception as exc:
            LOGGER.exception("Learning-rate experiment failed for %s", learning_rate)
            results.append(
                {
                    "learning_rate": learning_rate,
                    "epochs_run": 0,
                    "final_train_cost": float("nan"),
                    "test_rmse": float("nan"),
                    "test_r2": float("nan"),
                    "stopped_reason": f"failed: {exc}",
                    "elapsed_ms": 0.0,
                }
            )
    return results


def run_scaling_comparison(config: ExperimentConfig) -> list[dict[str, float | int | str | bool]]:
    rows: list[ExperimentRow] = []
    for scale_features in (False, True):
        trial_config = ExperimentConfig(**{**config.to_dict(), "scale_features": scale_features})
        bundle = train_from_config(trial_config)
        assert bundle.model.cost_history_ is not None
        rows.append(
            {
                "scaled": scale_features,
                "epochs_run": bundle.model.epochs_run_,
                "final_train_cost": float(bundle.model.cost_history_[-1]),
                "test_rmse": bundle.test_metrics["rmse"],
                "test_r2": bundle.test_metrics["r2"],
                "stopped_reason": bundle.model.stopped_reason_,
                "elapsed_ms": bundle.elapsed_seconds * 1000.0,
            }
        )
    return rows


def benchmark(config: ExperimentConfig, sample_sizes: Iterable[int]) -> list[dict[str, float | int]]:
    rows = []
    for n_samples in sample_sizes:
        trial_config = ExperimentConfig(**{**config.to_dict(), "n_samples": int(n_samples)})
        bundle = train_from_config(trial_config)
        rows.append(
            {
                "n_samples": int(n_samples),
                "epochs_run": bundle.model.epochs_run_,
                "elapsed_ms": bundle.elapsed_seconds * 1000.0,
                "microseconds_per_epoch": (bundle.elapsed_seconds * 1_000_000.0)
                / max(bundle.model.epochs_run_, 1),
                "test_rmse": bundle.test_metrics["rmse"],
            }
        )
    return rows
