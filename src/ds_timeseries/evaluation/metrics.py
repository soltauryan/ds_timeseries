"""Evaluation metrics for time series forecasting.

Primary metrics designed to handle intermittent demand (zeros in data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = sum(|actual - forecast|) / sum(actual)

    Preferred metric for intermittent demand as it avoids division by zero
    issues that plague MAPE. Weights errors by actual values.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        WAPE value (0 = perfect, higher = worse).
        Returns inf if sum of actuals is zero.

    Examples
    --------
    >>> wape([100, 200, 0, 150], [90, 210, 10, 140])
    0.0889  # ~8.9% weighted error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_actual = np.sum(np.abs(y_true))

    if total_actual == 0:
        return float("inf")

    return np.sum(np.abs(y_true - y_pred)) / total_actual


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error.

    Simple, interpretable metric in the same units as the target variable.
    Preferred over MSE/RMSE when outliers should not be over-penalized.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        MAE value (0 = perfect, higher = worse).

    Examples
    --------
    >>> mae([100, 200, 150], [90, 210, 140])
    10.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error.

    Penalizes large errors more heavily than MAE.
    Use when large errors are particularly undesirable.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        RMSE value (0 = perfect, higher = worse).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales errors relative to a naive seasonal forecast on training data.
    MASE < 1 means the model beats the seasonal naive baseline.

    Parameters
    ----------
    y_true : array-like
        Actual values for the test period.
    y_pred : array-like
        Predicted/forecast values.
    y_train : array-like
        Training data (for computing the naive baseline scale).
    seasonality : int
        Seasonal period (1 for non-seasonal, 52 for weekly with yearly seasonality).

    Returns
    -------
    float
        MASE value (< 1 beats naive baseline, > 1 worse than naive).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Compute naive seasonal error on training data
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return float("inf")

    forecast_errors = np.abs(y_true - y_pred)

    return np.mean(forecast_errors) / scale


def bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Forecast Bias (Mean Error).

    Measures systematic over/under forecasting.
    - Positive bias = over-forecasting (predicting too high)
    - Negative bias = under-forecasting (predicting too low)

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        Bias value (0 = unbiased).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_pred - y_true)


def coverage(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float:
    """Prediction Interval Coverage.

    Percentage of actual values falling within the prediction interval.
    For a 95% interval, expect ~95% coverage.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_lower : array-like
        Lower bound of prediction interval.
    y_upper : array-like
        Upper bound of prediction interval.

    Returns
    -------
    float
        Coverage percentage (0 to 1).
    """
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    within_interval = (y_true >= y_lower) & (y_true <= y_upper)

    return np.mean(within_interval)


def evaluate_forecast(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 52,
) -> dict[str, float]:
    """Compute all standard forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.
    y_train : array-like, optional
        Training data for MASE calculation.
    seasonality : int
        Seasonal period for MASE (default 52 for weekly data).

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    metrics = {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }

    if y_train is not None and len(y_train) > seasonality:
        metrics["mase"] = mase(y_true, y_pred, y_train, seasonality)

    return metrics
