from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lr_scratch.config import ExperimentConfig
from lr_scratch.data import generate_linear_data, train_test_split
from lr_scratch.experiments import train_from_config
from lr_scratch.metrics import evaluate_regression, r2_score
from lr_scratch.model import LinearRegressionGD
from lr_scratch.preprocessing import StandardScalerScratch


class LinearRegressionTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self) -> None:
        x = np.array([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
        y = np.array([0.0, 2.0, 4.0])
        theta = np.array([0.3, 1.7])
        analytic = LinearRegressionGD.gradient(x, y, theta)
        numeric = np.zeros_like(theta)
        eps = 1e-6
        for i in range(len(theta)):
            plus = theta.copy()
            minus = theta.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric[i] = (
                LinearRegressionGD.compute_cost(x, y, plus)
                - LinearRegressionGD.compute_cost(x, y, minus)
            ) / (2.0 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)

    def test_model_recovers_low_noise_linear_signal(self) -> None:
        config = ExperimentConfig(n_samples=500, noise=0.2, learning_rate=0.05, max_epochs=5000)
        bundle = train_from_config(config)
        self.assertGreater(bundle.test_metrics["r2"], 0.99)
        self.assertLess(bundle.test_metrics["rmse"], 0.35)

    def test_split_is_reproducible(self) -> None:
        x, y = generate_linear_data(seed=7)
        split_a = train_test_split(x, y, seed=123)
        split_b = train_test_split(x, y, seed=123)
        for left, right in zip(split_a, split_b, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_scaler_handles_constant_feature(self) -> None:
        scaler = StandardScalerScratch().fit(np.ones((5, 1)))
        transformed = scaler.transform(np.ones((5, 1)))
        self.assertTrue(np.all(np.isfinite(transformed)))
        np.testing.assert_array_equal(transformed, np.zeros((5, 1)))

    def test_metrics_are_from_first_principles(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        metrics = evaluate_regression(y_true, y_pred)
        self.assertAlmostEqual(metrics["mse"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.5)

    def test_json_model_round_trip_preserves_predictions(self) -> None:
        config = ExperimentConfig(n_samples=120, noise=1.0)
        bundle = train_from_config(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            bundle.model.save_json(path, bundle.scaler)
            loaded_model, loaded_scaler = LinearRegressionGD.load_json(path)
            x_test = loaded_scaler.transform(bundle.x_test_raw) if loaded_scaler else bundle.x_test_raw
            np.testing.assert_allclose(loaded_model.predict(x_test), bundle.y_test_pred)


if __name__ == "__main__":
    unittest.main()
