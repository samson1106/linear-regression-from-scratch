"""Data generation and splitting without scikit-learn."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_2d_features(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("Features must be a 1D or 2D array.")


def as_1d_target(y: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Target array must not be empty.")
    return arr


def generate_linear_data(
    n_samples: int = 200,
    noise: float = 2.5,
    true_slope: float = 2.5,
    true_intercept: float = 5.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a reproducible one-feature linear regression dataset."""
    if n_samples < 5:
        raise ValueError("n_samples must be at least 5.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=n_samples)
    y = true_intercept + true_slope * x + rng.normal(0.0, noise, size=n_samples)
    return x.reshape(-1, 1), y


def train_test_split(
    x: ArrayLike,
    y: ArrayLike,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Shuffle with a local RNG and return X_train, X_test, y_train, y_test."""
    x_arr = as_2d_features(x)
    y_arr = as_1d_target(y)
    if len(x_arr) != len(y_arr):
        raise ValueError("Features and target must contain the same number of rows.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if len(y_arr) < 5:
        raise ValueError("At least 5 samples are required for a meaningful split.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y_arr))
    split = int(round(len(y_arr) * train_ratio))
    split = min(max(split, 1), len(y_arr) - 1)
    train_idx, test_idx = indices[:split], indices[split:]
    return x_arr[train_idx], x_arr[test_idx], y_arr[train_idx], y_arr[test_idx]
