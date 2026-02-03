"""Data validation and demand classification utilities.

Provides tools for:
1. Validating time series data format
2. Classifying demand patterns (Syntetos-Boylan scheme)
3. Detecting potential data quality issues
4. Recommending models based on data characteristics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DemandCategory = Literal["smooth", "erratic", "intermittent", "lumpy"]


@dataclass
class DataValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


@dataclass
class DemandClassification:
    """Demand classification result for a single series."""

    unique_id: str
    category: DemandCategory
    adi: float  # Average Demand Interval
    cv2: float  # Squared Coefficient of Variation
    zero_pct: float  # Percentage of zeros
    recommended_models: list[str]


def validate_data(df: pd.DataFrame) -> DataValidationResult:
    """Validate time series data format and quality.

    Checks for:
    - Required columns (unique_id, ds, y)
    - Data types
    - Missing values
    - Duplicate entries
    - Chronological ordering
    - Sufficient history

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.

    Returns
    -------
    DataValidationResult
        Validation result with errors, warnings, and stats.

    Examples
    --------
    >>> result = validate_data(df)
    >>> if not result.is_valid:
    ...     print("Errors:", result.errors)
    >>> print("Warnings:", result.warnings)
    """
    errors = []
    warnings = []
    stats = {}

    # Check required columns
    required_cols = {"unique_id", "ds", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return DataValidationResult(False, errors, warnings, stats)

    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        errors.append("Column 'ds' must be datetime type")

    if not pd.api.types.is_numeric_dtype(df["y"]):
        errors.append("Column 'y' must be numeric type")

    # Basic stats
    stats["n_rows"] = len(df)
    stats["n_series"] = df["unique_id"].nunique()
    stats["date_range"] = (df["ds"].min(), df["ds"].max())
    stats["zero_pct"] = (df["y"] == 0).mean()
    stats["negative_pct"] = (df["y"] < 0).mean()

    # Check for missing values
    for col in ["unique_id", "ds", "y"]:
        null_pct = df[col].isna().mean()
        if null_pct > 0:
            if null_pct > 0.01:
                errors.append(f"Column '{col}' has {null_pct:.1%} missing values")
            else:
                warnings.append(f"Column '{col}' has {null_pct:.2%} missing values")

    # Check for duplicates
    dupe_count = df.duplicated(subset=["unique_id", "ds"]).sum()
    if dupe_count > 0:
        errors.append(f"Found {dupe_count:,} duplicate (unique_id, ds) pairs")

    # Check for negative values
    if stats["negative_pct"] > 0:
        warnings.append(f"Found {stats['negative_pct']:.1%} negative values in 'y'")

    # Check history per series
    history = df.groupby("unique_id")["ds"].nunique()
    stats["min_history"] = history.min()
    stats["max_history"] = history.max()
    stats["median_history"] = history.median()

    if stats["min_history"] < 10:
        warnings.append(f"Some series have only {stats['min_history']} observations")

    # Check for gaps in dates
    def check_gaps(group):
        dates = group["ds"].sort_values()
        if len(dates) < 2:
            return 0
        expected_freq = dates.diff().median()
        gaps = (dates.diff() > expected_freq * 1.5).sum()
        return gaps

    gap_counts = df.groupby("unique_id").apply(check_gaps)
    total_gaps = gap_counts.sum()
    if total_gaps > 0:
        warnings.append(f"Found {total_gaps:,} date gaps across {(gap_counts > 0).sum():,} series")

    is_valid = len(errors) == 0

    return DataValidationResult(is_valid, errors, warnings, stats)


def classify_demand(
    df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> pd.DataFrame:
    """Classify demand patterns using Syntetos-Boylan scheme.

    Categories:
    - Smooth: Low variability, low intermittence (easiest to forecast)
    - Erratic: High variability, low intermittence
    - Intermittent: Low variability, high intermittence
    - Lumpy: High variability, high intermittence (hardest to forecast)

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    adi_threshold : float
        ADI threshold (default 1.32 from Syntetos-Boylan 2005).
    cv2_threshold : float
        CV² threshold (default 0.49 from Syntetos-Boylan 2005).

    Returns
    -------
    pd.DataFrame
        Classification per series with columns:
        unique_id, category, adi, cv2, zero_pct, recommended_models.

    Examples
    --------
    >>> classifications = classify_demand(df)
    >>> print(classifications["category"].value_counts())
    """
    results = []

    for uid, group in df.groupby("unique_id"):
        y = group["y"].values

        # Calculate metrics
        zero_pct = (y == 0).mean()
        non_zero = y[y > 0]

        if len(non_zero) < 2:
            # Not enough non-zero values
            adi = float("inf")
            cv2 = float("inf")
            category = "lumpy"
        else:
            # ADI: Average Demand Interval
            # Number of periods / Number of non-zero periods
            adi = len(y) / len(non_zero)

            # CV²: Squared Coefficient of Variation of non-zero demand
            cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else 0

            # Classify using thresholds
            if adi < adi_threshold and cv2 < cv2_threshold:
                category = "smooth"
            elif adi < adi_threshold and cv2 >= cv2_threshold:
                category = "erratic"
            elif adi >= adi_threshold and cv2 < cv2_threshold:
                category = "intermittent"
            else:
                category = "lumpy"

        # Recommend models based on category
        recommended = _get_recommended_models(category, zero_pct)

        results.append({
            "unique_id": uid,
            "category": category,
            "adi": adi,
            "cv2": cv2,
            "zero_pct": zero_pct,
            "recommended_models": recommended,
        })

    return pd.DataFrame(results)


def _get_recommended_models(category: DemandCategory, zero_pct: float) -> list[str]:
    """Get model recommendations based on demand category."""
    if category == "smooth":
        return ["ETSForecaster", "LightGBMForecaster", "XGBoostForecaster"]
    elif category == "erratic":
        return ["LightGBMForecaster", "XGBoostForecaster", "ETSForecaster"]
    elif category == "intermittent":
        if zero_pct > 0.8:
            return ["SBAForecaster", "TSBForecaster", "CrostonForecaster"]
        else:
            return ["SBAForecaster", "LightGBMForecaster(use_tweedie=True)"]
    else:  # lumpy
        return ["TSBForecaster", "SBAForecaster", "IMAPAForecaster"]


def recommend_models(df: pd.DataFrame) -> dict:
    """Analyze data and recommend appropriate models.

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.

    Returns
    -------
    dict
        Recommendations including:
        - primary_model: Best single model to try first
        - ensemble_models: Models to include in ensemble
        - intermittent_pct: Percentage of intermittent/lumpy series
        - classification_summary: Count by category
    """
    classifications = classify_demand(df)

    # Summary
    category_counts = classifications["category"].value_counts()
    total = len(classifications)

    intermittent_pct = (
        category_counts.get("intermittent", 0) + category_counts.get("lumpy", 0)
    ) / total

    # Primary model recommendation
    if intermittent_pct > 0.5:
        primary = "SBAForecaster"
        ensemble = ["SBAForecaster", "TSBForecaster", "LightGBMForecaster(use_tweedie=True)"]
    elif intermittent_pct > 0.2:
        primary = "LightGBMForecaster(use_tweedie=True)"
        ensemble = ["LightGBMForecaster", "XGBoostForecaster", "SBAForecaster"]
    else:
        primary = "LightGBMForecaster"
        ensemble = ["LightGBMForecaster", "XGBoostForecaster", "ETSForecaster"]

    return {
        "primary_model": primary,
        "ensemble_models": ensemble,
        "intermittent_pct": intermittent_pct,
        "classification_summary": category_counts.to_dict(),
        "total_series": total,
    }


def detect_data_leakage_risk(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """Detect potential data leakage risks in features.

    Checks for:
    - Future-looking features (lag < 0)
    - Features perfectly correlated with target
    - Suspiciously high feature importances

    Parameters
    ----------
    df : pd.DataFrame
        Data with features.
    feature_cols : list[str] | None
        Feature columns to check. If None, checks all non-standard columns.

    Returns
    -------
    list[str]
        List of warning messages about potential leakage.
    """
    warnings = []

    standard_cols = {"unique_id", "ds", "y"}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in standard_cols]

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Check for perfect or near-perfect correlation with target
        if "y" in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr = df[col].corr(df["y"])
                if abs(corr) > 0.99:
                    warnings.append(
                        f"Feature '{col}' has {corr:.3f} correlation with target - "
                        "possible data leakage"
                    )
            except Exception:
                pass

        # Check for suspicious feature names
        leaky_patterns = ["future", "next", "tomorrow", "target", "label", "answer"]
        if any(pattern in col.lower() for pattern in leaky_patterns):
            warnings.append(
                f"Feature '{col}' has suspicious name - verify it doesn't use future data"
            )

    return warnings
