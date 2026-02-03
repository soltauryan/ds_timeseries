#!/usr/bin/env python
"""Benchmark baseline models on M5 sample data.

Establishes performance baselines for Phase 1 models using proper
time series cross-validation.

Usage:
    python scripts/benchmark_baselines.py [--n-series N] [--n-folds N]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import pandas as pd

from ds_timeseries.data import download_toy_dataset
from ds_timeseries.evaluation import cross_validate, cv_score
from ds_timeseries.evaluation.metrics import mae, wape
from ds_timeseries.models import (
    ETSForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    SeasonalNaiveStatsForecast,
)


def benchmark_model(
    model: Any,
    df: pd.DataFrame,
    n_folds: int = 3,
    horizon: int = 4,
) -> dict[str, Any]:
    """Benchmark a single model with cross-validation."""
    model_name = model.__class__.__name__

    print(f"  Benchmarking {model_name}...", end=" ", flush=True)
    start = time.time()

    try:
        # Run CV
        results = cross_validate(
            model,
            df,
            n_folds=n_folds,
            horizon=horizon,
            min_train_size=52 * 2,  # At least 2 years of training
        )

        # Compute metrics
        overall_wape = wape(results["y"], results["yhat"])
        overall_mae = mae(results["y"], results["yhat"])

        # Per-fold metrics
        fold_wapes = []
        fold_maes = []
        for fold_id in results["fold_id"].unique():
            fold_data = results[results["fold_id"] == fold_id]
            fold_wapes.append(wape(fold_data["y"], fold_data["yhat"]))
            fold_maes.append(mae(fold_data["y"], fold_data["yhat"]))

        elapsed = time.time() - start
        print(f"WAPE={overall_wape:.2%}, MAE={overall_mae:.2f} ({elapsed:.1f}s)")

        return {
            "model": model_name,
            "wape": overall_wape,
            "mae": overall_mae,
            "wape_std": pd.Series(fold_wapes).std(),
            "mae_std": pd.Series(fold_maes).std(),
            "n_folds": n_folds,
            "elapsed_seconds": elapsed,
            "status": "success",
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED: {e}")
        return {
            "model": model_name,
            "wape": None,
            "mae": None,
            "status": f"failed: {e}",
            "elapsed_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline models")
    parser.add_argument("--n-series", type=int, default=100, help="Number of series to use")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon (weeks)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1 Baseline Benchmarks")
    print("=" * 60)
    print(f"Settings: {args.n_series} series, {args.n_folds} folds, {args.horizon}-week horizon")
    print()

    # Load data
    print("Loading M5 sample data...")
    df = download_toy_dataset("m5_sample", n_series=args.n_series)
    print(f"  Loaded {df['unique_id'].nunique()} series, {len(df)} rows")
    print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print()

    # Define models to benchmark
    models = [
        NaiveForecaster(),
        MovingAverageForecaster(window=4),
        MovingAverageForecaster(window=8),
        SeasonalNaiveForecaster(season_length=52),
        SeasonalNaiveStatsForecast(season_length=52),
        ETSForecaster(season_length=52),
    ]

    # Run benchmarks
    print("Running benchmarks:")
    results = []
    for model in models:
        result = benchmark_model(model, df, n_folds=args.n_folds, horizon=args.horizon)
        results.append(result)

    # Summary table
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("wape")

    print(f"\n{'Model':<35} {'WAPE':>10} {'MAE':>10} {'Time':>8}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        if row["status"] == "success":
            print(
                f"{row['model']:<35} {row['wape']:>9.2%} {row['mae']:>10.2f} {row['elapsed_seconds']:>7.1f}s"
            )
        else:
            print(f"{row['model']:<35} {'FAILED':>10}")

    # Best model
    best = results_df[results_df["status"] == "success"].iloc[0]
    print()
    print(f"Best baseline: {best['model']} with WAPE={best['wape']:.2%}")

    # Save results
    results_df.to_csv("data/raw/baseline_results.csv", index=False)
    print(f"\nResults saved to data/raw/baseline_results.csv")


if __name__ == "__main__":
    main()
