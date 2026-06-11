"""Command-line experiment runner for linear regression from scratch."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lr_scratch.config import ExperimentConfig, load_config
from lr_scratch.experiments import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure NumPy linear regression.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to JSON config.")
    parser.add_argument("--save-model", type=Path, default=Path("artifacts/model.json"), help="Output model JSON path.")
    parser.add_argument("--no-save", action="store_true", help="Skip JSON model persistence.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    config = load_config(args.config) if args.config.exists() else ExperimentConfig()
    bundle = train_from_config(config)

    print("\nConfiguration")
    print(json.dumps(config.to_dict(), indent=2))
    print("\nLearned parameters")
    print(bundle.model.theta_)
    print(f"epochs_run={bundle.model.epochs_run_} stopped_reason={bundle.model.stopped_reason_}")

    print("\nMetrics")
    print(f"{'metric':<8} {'train':>12} {'test':>12}")
    print("-" * 34)
    for metric in ("mse", "rmse", "mae", "r2"):
        print(f"{metric:<8} {bundle.train_metrics[metric]:>12.6f} {bundle.test_metrics[metric]:>12.6f}")

    if not args.no_save:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        bundle.model.save_json(args.save_model, bundle.scaler)
        print(f"\nSaved model JSON: {args.save_model}")


if __name__ == "__main__":
    main()
