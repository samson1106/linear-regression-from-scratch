"""Regression metrics implemented without scikit-learn."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return actual, predicted


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _pair(y_true, y_pred)
    return float(np.mean((actual - predicted) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _pair(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _pair(y_true, y_pred)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
