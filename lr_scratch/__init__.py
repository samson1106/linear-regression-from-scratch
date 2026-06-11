"""Pure NumPy linear regression utilities."""

from .config import ExperimentConfig, load_config
from .data import generate_linear_data, train_test_split
from .metrics import evaluate_regression, mae, mse, r2_score, rmse
from .model import FitResult, LinearRegressionGD
from .preprocessing import StandardScalerScratch

__all__ = [
    "ExperimentConfig",
    "FitResult",
    "LinearRegressionGD",
    "StandardScalerScratch",
    "evaluate_regression",
    "generate_linear_data",
    "load_config",
    "mae",
    "mse",
    "r2_score",
    "rmse",
    "train_test_split",
]
