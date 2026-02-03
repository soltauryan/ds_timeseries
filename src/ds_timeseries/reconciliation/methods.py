"""Hierarchical Reconciliation Methods.

Implements reconciliation algorithms to ensure forecasts are coherent
across hierarchy levels (sum of children = parent).

Methods:
- Bottom-Up: Aggregate bottom forecasts upward
- Top-Down: Disaggregate top forecast using proportions
- Middle-Out: Combine bottom-up and top-down from middle level
- MinTrace (OLS/WLS): Optimal reconciliation minimizing trace of error covariance
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ds_timeseries.reconciliation.hierarchy import HierarchySpec


ReconciliationMethod = Literal["bottom_up", "top_down", "middle_out", "ols", "wls", "mint_shrink"]


def reconcile_forecasts(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
    method: ReconciliationMethod = "bottom_up",
) -> pd.DataFrame:
    """Reconcile forecasts to be coherent across hierarchy.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Base forecasts with unique_id, ds, yhat columns.
        Should include forecasts for all hierarchy levels.
    actuals : pd.DataFrame
        Historical actuals for computing proportions.
    hierarchy : HierarchySpec
        Hierarchy specification.
    method : str
        Reconciliation method:
        - "bottom_up": Aggregate from bottom level
        - "top_down": Disaggregate from top using historical proportions
        - "ols": Ordinary least squares (MinTrace)
        - "wls": Weighted least squares (MinTrace with variance weights)
        - "mint_shrink": MinTrace with shrinkage estimator

    Returns
    -------
    pd.DataFrame
        Reconciled forecasts with unique_id, ds, yhat columns.

    Examples
    --------
    >>> reconciled = reconcile_forecasts(
    ...     forecasts, actuals, hierarchy, method="bottom_up"
    ... )
    """
    if method == "bottom_up":
        return bottom_up_reconcile(forecasts, actuals, hierarchy)
    elif method == "top_down":
        return top_down_reconcile(forecasts, actuals, hierarchy)
    elif method in ("ols", "wls", "mint_shrink"):
        return mintrace_reconcile(forecasts, actuals, hierarchy, method)
    else:
        raise ValueError(f"Unknown method: {method}")


def bottom_up_reconcile(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
) -> pd.DataFrame:
    """Bottom-Up reconciliation.

    Uses only bottom-level forecasts and aggregates upward.
    Simple and unbiased, but ignores information from aggregate forecasts.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecasts for bottom-level series.
    actuals : pd.DataFrame
        Historical data with hierarchy columns.
    hierarchy : HierarchySpec
        Hierarchy specification.

    Returns
    -------
    pd.DataFrame
        Reconciled forecasts for all levels.
    """
    # Build summing matrix
    S = hierarchy.build_aggregation_matrix(actuals)

    # Get bottom-level forecasts
    bottom_series = actuals[hierarchy.bottom_level].unique()
    bottom_forecasts = forecasts[forecasts["unique_id"].isin(bottom_series)]

    # Pivot to matrix form (dates x series)
    forecast_pivot = bottom_forecasts.pivot(
        index="ds", columns="unique_id", values="yhat"
    ).fillna(0)

    # Reorder to match S matrix
    forecast_matrix = forecast_pivot[bottom_series].values

    # Reconcile: y_tilde = S @ y_bottom
    reconciled_matrix = forecast_matrix @ S.T

    # Convert back to DataFrame
    return _matrix_to_dataframe(
        reconciled_matrix, forecast_pivot.index, actuals, hierarchy
    )


def top_down_reconcile(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
    proportion_method: str = "average_historical",
) -> pd.DataFrame:
    """Top-Down reconciliation.

    Disaggregates top-level forecast using historical proportions.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecasts (must include top-level "Total" forecast).
    actuals : pd.DataFrame
        Historical data for computing proportions.
    hierarchy : HierarchySpec
        Hierarchy specification.
    proportion_method : str
        Method for computing proportions:
        - "average_historical": Average of historical proportions
        - "proportion_of_historical_average": Proportion of averages

    Returns
    -------
    pd.DataFrame
        Reconciled bottom-level forecasts.
    """
    # Get total forecast
    total_forecast = forecasts[forecasts["unique_id"] == "Total"]

    if total_forecast.empty:
        # Compute total from bottom-level
        bottom_series = actuals[hierarchy.bottom_level].unique()
        bottom_forecasts = forecasts[forecasts["unique_id"].isin(bottom_series)]
        total_forecast = (
            bottom_forecasts
            .groupby("ds")["yhat"]
            .sum()
            .reset_index()
        )
        total_forecast["unique_id"] = "Total"

    # Compute historical proportions
    bottom_series = actuals[hierarchy.bottom_level].unique()
    proportions = _compute_proportions(actuals, hierarchy, proportion_method)

    # Disaggregate
    results = []
    for _, row in total_forecast.iterrows():
        total_value = row["yhat"]
        ds = row["ds"]

        for series_id, prop in proportions.items():
            results.append({
                "unique_id": series_id,
                "ds": ds,
                "yhat": total_value * prop,
            })

    return pd.DataFrame(results)


def mintrace_reconcile(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
    method: str = "ols",
) -> pd.DataFrame:
    """MinTrace reconciliation (optimal combination).

    Finds reconciled forecasts that minimize trace of forecast error
    covariance matrix subject to aggregation constraints.

    Methods:
    - "ols": Assumes equal variance (identity weight matrix)
    - "wls": Weights by variance of each series
    - "mint_shrink": Shrinkage estimator for covariance

    Parameters
    ----------
    forecasts : pd.DataFrame
        Base forecasts for all levels.
    actuals : pd.DataFrame
        Historical data.
    hierarchy : HierarchySpec
        Hierarchy specification.
    method : str
        Covariance estimation method.

    Returns
    -------
    pd.DataFrame
        Optimally reconciled forecasts.
    """
    # Build summing matrix
    S = hierarchy.build_aggregation_matrix(actuals)
    n_all, n_bottom = S.shape

    # Get all forecasts in matrix form
    all_series = _get_all_series_ids(actuals, hierarchy)
    forecast_pivot = forecasts.pivot(
        index="ds", columns="unique_id", values="yhat"
    ).reindex(columns=all_series).fillna(0)

    forecast_matrix = forecast_pivot.values  # T x n_all

    # Compute weight matrix W based on method
    if method == "ols":
        W = np.eye(n_all)
    elif method == "wls":
        # Diagonal weights based on variance
        variances = _compute_variances(actuals, hierarchy)
        W = np.diag(1.0 / (variances + 1e-6))
    elif method == "mint_shrink":
        # Shrinkage covariance estimator
        W = _compute_shrinkage_weights(actuals, hierarchy)
    else:
        raise ValueError(f"Unknown method: {method}")

    # MinTrace reconciliation matrix: G = (S'WS)^{-1} S'W
    # Reconciled = S @ G @ base_forecasts
    SWS = S.T @ W @ S
    SWS_inv = np.linalg.pinv(SWS)
    G = SWS_inv @ S.T @ W

    # P = S @ G is the reconciliation projection matrix
    P = S @ G

    # Reconcile: y_tilde = P @ y_hat
    reconciled_matrix = forecast_matrix @ P.T

    return _matrix_to_dataframe(
        reconciled_matrix, forecast_pivot.index, actuals, hierarchy
    )


def _compute_proportions(
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
    method: str = "average_historical",
) -> dict[str, float]:
    """Compute historical proportions for top-down disaggregation."""
    bottom_series = actuals[hierarchy.bottom_level].unique()

    if method == "average_historical":
        # Average of (series / total) over time
        totals = actuals.groupby("ds")["y"].sum()
        proportions = {}

        for series_id in bottom_series:
            series_values = actuals[actuals[hierarchy.bottom_level] == series_id].set_index("ds")["y"]
            aligned_totals = totals.reindex(series_values.index)
            props = series_values / (aligned_totals + 1e-6)
            proportions[series_id] = props.mean()

    elif method == "proportion_of_historical_average":
        # (average of series) / (average of total)
        total_avg = actuals["y"].sum()
        proportions = {}

        for series_id in bottom_series:
            series_avg = actuals[actuals[hierarchy.bottom_level] == series_id]["y"].sum()
            proportions[series_id] = series_avg / (total_avg + 1e-6)

    else:
        raise ValueError(f"Unknown proportion method: {method}")

    # Normalize to sum to 1
    total_prop = sum(proportions.values())
    proportions = {k: v / total_prop for k, v in proportions.items()}

    return proportions


def _compute_variances(
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
) -> np.ndarray:
    """Compute variance for each series in the hierarchy."""
    all_series = _get_all_series_ids(actuals, hierarchy)
    variances = []

    # Total variance
    total_var = actuals.groupby("ds")["y"].sum().var()
    variances.append(total_var if total_var > 0 else 1.0)

    # Level variances
    for level in reversed(hierarchy.levels):
        if level not in actuals.columns:
            continue
        for val in actuals[level].unique():
            level_var = (
                actuals[actuals[level] == val]
                .groupby("ds")["y"]
                .sum()
                .var()
            )
            variances.append(level_var if level_var > 0 else 1.0)

    # Bottom-level variances
    for series_id in actuals[hierarchy.bottom_level].unique():
        series_var = actuals[actuals[hierarchy.bottom_level] == series_id]["y"].var()
        variances.append(series_var if series_var > 0 else 1.0)

    return np.array(variances)


def _compute_shrinkage_weights(
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
) -> np.ndarray:
    """Compute shrinkage covariance estimator."""
    variances = _compute_variances(actuals, hierarchy)
    # Simple diagonal shrinkage
    return np.diag(1.0 / (variances + 1e-6))


def _get_all_series_ids(
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
) -> list[str]:
    """Get all series IDs in hierarchy order."""
    all_ids = ["Total"]

    for level in reversed(hierarchy.levels):
        if level not in actuals.columns:
            continue
        for val in actuals[level].unique():
            all_ids.append(f"{level}_{val}")

    for series_id in actuals[hierarchy.bottom_level].unique():
        all_ids.append(series_id)

    return all_ids


def _matrix_to_dataframe(
    reconciled_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    actuals: pd.DataFrame,
    hierarchy: HierarchySpec,
) -> pd.DataFrame:
    """Convert reconciled matrix back to DataFrame."""
    all_series = _get_all_series_ids(actuals, hierarchy)
    results = []

    for date_idx, ds in enumerate(dates):
        for series_idx, series_id in enumerate(all_series):
            results.append({
                "unique_id": series_id,
                "ds": ds,
                "yhat": max(0, reconciled_matrix[date_idx, series_idx]),
            })

    return pd.DataFrame(results)
