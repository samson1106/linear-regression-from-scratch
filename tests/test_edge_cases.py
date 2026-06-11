from __future__ import annotations

import math
import unittest

import numpy as np

from lr_scratch.config import ExperimentConfig
from lr_scratch.data import generate_linear_data, train_test_split
from lr_scratch.experiments import run_learning_rate_experiment, run_scaling_comparison, train_from_config
from lr_scratch.metrics import r2_score
from lr_scratch.model import LinearRegressionGD


class EdgeCaseTests(unittest.TestCase):
    def test_bad_train_ratio_is_rejected(self) -> None:
        x = np.arange(10)
        y = np.arange(10)
        with self.assertRaises(ValueError):
            train_test_split(x, y, train_ratio=1.0)

    def test_mismatched_feature_target_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_test_split(np.arange(10), np.arange(9))

    def test_invalid_config_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(learning_rate=0.0)

    def test_constant_target_r2_is_well_defined(self) -> None:
        perfect = r2_score([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
        imperfect = r2_score([3.0, 3.0, 3.0], [2.0, 3.0, 4.0])
        self.assertEqual(perfect, 1.0)
        self.assertEqual(imperfect, 0.0)

    def test_divergent_learning_rate_stops_without_crashing(self) -> None:
        x, y = generate_linear_data(n_samples=80, seed=11)
        model = LinearRegressionGD(learning_rate=1e308, max_epochs=50, early_stopping=False)
        result = model.fit(x, y)
        self.assertLessEqual(result.epochs_run, 50)
        self.assertIn(result.stopped_reason, {"diverged", "max_epochs"})

    def test_scaling_comparison_returns_scaled_and_unscaled_rows(self) -> None:
        config = ExperimentConfig(n_samples=120, max_epochs=100, tolerance=1e-6)
        rows = run_scaling_comparison(config)
        self.assertEqual({row["scaled"] for row in rows}, {False, True})
        self.assertTrue(all(math.isfinite(float(row["test_rmse"])) for row in rows))

    def test_learning_rate_experiment_reports_each_learning_rate(self) -> None:
        config = ExperimentConfig(n_samples=120, max_epochs=100, tolerance=1e-6)
        rates = [0.001, 0.01, 0.05]
        rows = run_learning_rate_experiment(config, rates)
        self.assertEqual([row["learning_rate"] for row in rows], rates)

    def test_noiseless_data_gets_near_perfect_fit(self) -> None:
        config = ExperimentConfig(n_samples=250, noise=0.0, learning_rate=0.05, max_epochs=5000)
        bundle = train_from_config(config)
        self.assertGreater(bundle.test_metrics["r2"], 0.999999)
        self.assertLess(bundle.test_metrics["rmse"], 1e-3)


if __name__ == "__main__":
    unittest.main()
