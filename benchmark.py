"""Performance benchmark for the vectorized NumPy implementation."""

from __future__ import annotations

import argparse
import pandas as pd

from lr_scratch.config import load_config
from lr_scratch.experiments import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark linear regression training runtime.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--samples", nargs="+", type=int, default=[100, 500, 1_000, 5_000, 10_000])
    args = parser.parse_args()
    config = load_config(args.config)
    rows = benchmark(config, args.samples)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
