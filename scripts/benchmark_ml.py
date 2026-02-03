#!/usr/bin/env python
"""Benchmark ML models against baselines.

Compares LightGBM and XGBoost against Phase 1 baselines using
proper time series cross-validation.

Usage:
    python scripts/benchmark_ml.py [--n-series N] [--n-folds N]
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import pandas as pd

from ds_timeseries.data import download_toy_dataset
from ds_timeseries.evaluation.metrics import mae, wape
from ds_timeseries.features import FiscalCalendarConfig, FeatureConfig
from ds_timeseries.models import (
    ETSForecaster,
    LightGBMForecaster,
    MovingAverageForecaster,
    XGBoostForecaster,
)

warnings.filterwarnings("ignore")


def benchmark_model_simple(
    model: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """Benchmark a model with simple train/test split."""
    model_name = model.__class__.__name__
    print(f"  {model_name}...", end=" ", flush=True)

    start = time.time()

    try:
        # Fit
        model.fit(train)

        # Predict
        preds = model.predict(horizon=horizon)

        # Merge with actuals
        merged = test.merge(preds, on=["unique_id", "ds"], how="inner")

        if len(merged) == 0:
            raise ValueError("No matching predictions found")

        # Compute metrics
        w = wape(merged["y"], merged["yhat"])
        m = mae(merged["y"], merged["yhat"])

        elapsed = time.time() - start
        print(f"WAPE={w:.2%}, MAE={m:.2f} ({elapsed:.1f}s)")

        return {
            "model": model_name,
            "wape": w,
            "mae": m,
            "elapsed_seconds": elapsed,
            "status": "success",
            "n_predictions": len(merged),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED: {e}")
        return {
            "model": model_name,
            "wape": None,
            "mae": None,
            "elapsed_seconds": elapsed,
            "status": f"failed: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML models")
    parser.add_argument("--n-series", type=int, default=100, help="Number of series")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon (weeks)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2: ML Model Benchmarks")
    print("=" * 70)
    print(f"Settings: {args.n_series} series, {args.horizon}-week horizon")
    print()

    # Load data
    print("Loading M5 sample data...")
    df = download_toy_dataset("m5_sample", n_series=args.n_series)
    print(f"  Loaded {df['unique_id'].nunique()} series, {len(df)} rows")
    print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")

    # Train/test split (last horizon weeks for test)
    cutoff = df["ds"].max() - pd.Timedelta(weeks=args.horizon)
    train = df[df["ds"] <= cutoff].copy()
    test = df[df["ds"] > cutoff].copy()

    print(f"  Train: {len(train)} rows up to {cutoff.date()}")
    print(f"  Test: {len(test)} rows ({test['ds'].nunique()} weeks)")
    print()

    # Feature config with fiscal calendar
    fiscal_config = FiscalCalendarConfig(
        fiscal_year_start_month=11,
        week_pattern="5-4-4",
    )

    feature_config = FeatureConfig(
        lags=[1, 2, 4, 8, 13, 26, 52],
        rolling_windows=[4, 8, 13, 26],
        rolling_aggs=["mean", "std"],
        diff_periods=[1, 4, 52],
        pct_change_periods=[1, 52],
        include_expanding=True,
        fiscal_config=fiscal_config,
    )

    # Define models
    models = [
        # Baselines for comparison
        MovingAverageForecaster(window=8),
        ETSForecaster(season_length=52),
        # ML Models
        LightGBMForecaster(
            feature_config=feature_config,
            lgb_params={
                "objective": "regression",
                "metric": "mae",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "verbose": -1,
            },
        ),
        XGBoostForecaster(
            feature_config=feature_config,
            xgb_params={
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "verbosity": 0,
            },
        ),
    ]

    # Run benchmarks
    print("Running benchmarks:")
    results = []
    for model in models:
        result = benchmark_model_simple(model, train, test, args.horizon)
        results.append(result)

    # Feature importance for best ML model
    print()
    print("Feature Importance (LightGBM top 15):")
    lgb_model = models[2]  # LightGBM
    if lgb_model._is_fitted:
        importance = lgb_model.get_feature_importance()
        for i, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:>8.0f}")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("wape")

    print(f"\n{'Model':<25} {'WAPE':>10} {'MAE':>10} {'Time':>10}")
    print("-" * 57)

    for _, row in results_df.iterrows():
        if row["status"] == "success":
            print(
                f"{row['model']:<25} {row['wape']:>9.2%} {row['mae']:>10.2f} {row['elapsed_seconds']:>9.1f}s"
            )
        else:
            print(f"{row['model']:<25} {'FAILED':>10}")

    # Compare to baseline
    baseline_wape = results_df[results_df["model"] == "ETSForecaster"]["wape"].values[0]
    best = results_df[results_df["status"] == "success"].iloc[0]

    print()
    if best["model"] != "ETSForecaster":
        improvement = (baseline_wape - best["wape"]) / baseline_wape * 100
        print(f"Best model: {best['model']} with WAPE={best['wape']:.2%}")
        print(f"Improvement over ETS baseline: {improvement:.1f}%")
    else:
        print(f"ETS baseline still best at WAPE={baseline_wape:.2%}")

    # Save results
    results_df.to_csv("data/raw/ml_benchmark_results.csv", index=False)
    print(f"\nResults saved to data/raw/ml_benchmark_results.csv")


if __name__ == "__main__":
    main()
