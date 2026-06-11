"""Batch gradient-descent linear regression from scratch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import as_1d_target, as_2d_features
from .metrics import mse
from .preprocessing import StandardScalerScratch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitResult:
    theta: NDArray[np.float64]
    cost_history: NDArray[np.float64]
    validation_cost_history: NDArray[np.float64]
    epochs_run: int
    converged: bool
    stopped_reason: str


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.05
    max_epochs: int = 2_000
    tolerance: float = 1e-8
    patience: int = 25
    early_stopping: bool = True
    fit_intercept: bool = True
    theta_: NDArray[np.float64] | None = None
    cost_history_: NDArray[np.float64] | None = None
    validation_cost_history_: NDArray[np.float64] | None = None
    epochs_run_: int = 0
    converged_: bool = False
    stopped_reason_: str = "not_fitted"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")

    def _design_matrix(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = as_2d_features(x)
        if not self.fit_intercept:
            return x_arr
        return np.column_stack([np.ones(x_arr.shape[0]), x_arr])

    @staticmethod
    def compute_cost(design_x: ArrayLike, y: ArrayLike, theta: ArrayLike) -> float:
        x_arr = np.asarray(design_x, dtype=float)
        y_arr = as_1d_target(y)
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[1] != theta_arr.shape[0]:
            raise ValueError("Incompatible shapes for X, y, and theta.")
        residuals = x_arr @ theta_arr - y_arr
        return float((residuals @ residuals) / (2.0 * len(y_arr)))

    @staticmethod
    def gradient(design_x: ArrayLike, y: ArrayLike, theta: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(design_x, dtype=float)
        y_arr = as_1d_target(y)
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        return (x_arr.T @ (x_arr @ theta_arr - y_arr)) / len(y_arr)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
    ) -> FitResult:
        design_x = self._design_matrix(x)
        y_arr = as_1d_target(y)
        if design_x.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        val_design = None
        val_target = None
        if x_val is not None or y_val is not None:
            if x_val is None or y_val is None:
                raise ValueError("x_val and y_val must be provided together.")
            val_design = self._design_matrix(x_val)
            val_target = as_1d_target(y_val)

        theta = np.zeros(design_x.shape[1], dtype=float)
        costs: list[float] = []
        validation_costs: list[float] = []
        best_validation_cost = np.inf
        epochs_without_improvement = 0
        stopped_reason = "max_epochs"
        converged = False
        previous_cost = np.inf

        for epoch in range(self.max_epochs):
            grad = self.gradient(design_x, y_arr, theta)
            with np.errstate(over="ignore", invalid="ignore"):
                theta = theta - self.learning_rate * grad
                train_cost = self.compute_cost(design_x, y_arr, theta)

            if not np.isfinite(train_cost) or not np.all(np.isfinite(theta)):
                stopped_reason = "diverged"
                LOGGER.warning("Training diverged at epoch %s", epoch + 1)
                break

            costs.append(train_cost)

            if val_design is not None and val_target is not None:
                val_cost = self.compute_cost(val_design, val_target, theta)
                validation_costs.append(val_cost)
                if val_cost + self.tolerance < best_validation_cost:
                    best_validation_cost = val_cost
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if self.early_stopping and epochs_without_improvement >= self.patience:
                    stopped_reason = "early_stopping"
                    converged = True
                    break

            improvement = previous_cost - train_cost
            if self.early_stopping and 0 <= improvement < self.tolerance:
                stopped_reason = "tolerance"
                converged = True
                break
            previous_cost = train_cost

        self.theta_ = theta
        self.cost_history_ = np.asarray(costs, dtype=float)
        self.validation_cost_history_ = np.asarray(validation_costs, dtype=float)
        self.epochs_run_ = len(costs)
        self.converged_ = converged
        self.stopped_reason_ = stopped_reason
        LOGGER.info("Finished training after %s epochs: %s", self.epochs_run_, stopped_reason)
        return FitResult(
            theta=theta.copy(),
            cost_history=self.cost_history_.copy(),
            validation_cost_history=self.validation_cost_history_.copy(),
            epochs_run=self.epochs_run_,
            converged=converged,
            stopped_reason=stopped_reason,
        )

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before predict.")
        return self._design_matrix(x) @ self.theta_

    def score(self, x: ArrayLike, y: ArrayLike) -> float:
        return mse(y, self.predict(x))

    def to_dict(self, scaler: StandardScalerScratch | None = None) -> dict[str, Any]:
        if self.theta_ is None:
            raise RuntimeError("Cannot serialize an unfitted model.")
        payload: dict[str, Any] = {
            "model_type": "LinearRegressionGD",
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "patience": self.patience,
            "early_stopping": self.early_stopping,
            "fit_intercept": self.fit_intercept,
            "theta": self.theta_.tolist(),
            "epochs_run": self.epochs_run_,
            "converged": self.converged_,
            "stopped_reason": self.stopped_reason_,
        }
        if scaler is not None:
            payload["scaler"] = scaler.to_dict()
        return payload

    def save_json(self, path: str | Path, scaler: StandardScalerScratch | None = None) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(scaler), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple[LinearRegressionGD, StandardScalerScratch | None]:
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            max_epochs=int(payload["max_epochs"]),
            tolerance=float(payload["tolerance"]),
            patience=int(payload["patience"]),
            early_stopping=bool(payload["early_stopping"]),
            fit_intercept=bool(payload["fit_intercept"]),
        )
        model.theta_ = np.asarray(payload["theta"], dtype=float)
        model.epochs_run_ = int(payload.get("epochs_run", 0))
        model.converged_ = bool(payload.get("converged", False))
        model.stopped_reason_ = str(payload.get("stopped_reason", "loaded"))
        scaler = None
        if "scaler" in payload:
            scaler = StandardScalerScratch.from_dict(payload["scaler"])
        return model, scaler

    @classmethod
    def load_json(cls, path: str | Path) -> tuple[LinearRegressionGD, StandardScalerScratch | None]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "LinearRegressionGD":
            raise ValueError("Unsupported model_type in JSON file.")
        return cls.from_dict(payload)

    def parameters(self) -> dict[str, Any]:
        return asdict(self)
