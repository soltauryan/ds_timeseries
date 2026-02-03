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


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Percentage Error.

    CAUTION: MAPE has known issues:
    - Division by zero when y_true = 0
    - Asymmetric (penalizes over-forecasts less than under-forecasts)
    - Not recommended for intermittent demand

    Use WAPE or MASE instead for most applications.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        MAPE value (0 = perfect, higher = worse).
        Returns inf if any y_true = 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle zeros
    mask = y_true != 0
    if not mask.any():
        return float("inf")

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Attempts to fix MAPE's asymmetry by using average of actual and forecast
    in denominator. Used in M3 competition.

    CAUTION: sMAPE has issues near zero values and Hyndman recommends
    avoiding it. Use WAPE or MASE instead.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.

    Returns
    -------
    float
        sMAPE value (0 = perfect, 2 = worst).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator > 0

    if not mask.any():
        return float("inf")

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def rmsse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    seasonality: int = 1,
) -> float:
    """Root Mean Squared Scaled Error.

    Used as the official metric in M5 competition.
    Like MASE but uses squared errors instead of absolute errors.

    Parameters
    ----------
    y_true : array-like
        Actual values for test period.
    y_pred : array-like
        Predicted/forecast values.
    y_train : array-like
        Training data for computing naive baseline scale.
    seasonality : int
        Seasonal period (1 for non-seasonal).

    Returns
    -------
    float
        RMSSE value (< 1 beats naive, > 1 worse than naive).

    References
    ----------
    - M5 Forecasting Competition evaluation metric
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Compute naive seasonal MSE on training data
    naive_errors_sq = (y_train[seasonality:] - y_train[:-seasonality]) ** 2
    scale = np.mean(naive_errors_sq)

    if scale == 0:
        return float("inf")

    forecast_errors_sq = (y_true - y_pred) ** 2

    return np.sqrt(np.mean(forecast_errors_sq) / scale)


def interval_width(
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float:
    """Average width of prediction intervals.

    Narrower intervals are better (if coverage is maintained).

    Parameters
    ----------
    y_lower : array-like
        Lower bound of prediction interval.
    y_upper : array-like
        Upper bound of prediction interval.

    Returns
    -------
    float
        Average interval width.
    """
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    return np.mean(y_upper - y_lower)


def winkler_score(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> float:
    """Winkler Score for prediction interval evaluation.

    Measures both interval width and coverage. Lower is better.
    Penalizes intervals that don't contain the actual value.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_lower : array-like
        Lower bound of prediction interval.
    y_upper : array-like
        Upper bound of prediction interval.
    alpha : float
        Significance level (0.05 for 95% interval).

    Returns
    -------
    float
        Winkler score (lower = better).

    References
    ----------
    - Winkler, R. L. (1972). "A Decision-Theoretic Approach to
      Interval Estimation"
    """
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    width = y_upper - y_lower
    penalty = np.zeros_like(y_true, dtype=float)

    # Penalty for below lower bound
    below = y_true < y_lower
    penalty[below] = (2 / alpha) * (y_lower[below] - y_true[below])

    # Penalty for above upper bound
    above = y_true > y_upper
    penalty[above] = (2 / alpha) * (y_true[above] - y_upper[above])

    return np.mean(width + penalty)


def scaled_pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    quantile: float = 0.5,
    seasonality: int = 1,
) -> float:
    """Scaled Pinball Loss for quantile forecasts.

    Used in M5 Uncertainty competition for probabilistic forecasts.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted quantile values.
    y_train : array-like
        Training data for scaling.
    quantile : float
        Quantile being predicted (0 to 1).
    seasonality : int
        Seasonal period.

    Returns
    -------
    float
        Scaled pinball loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Pinball loss
    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)

    # Scale
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return float("inf")

    return np.mean(loss) / scale


def evaluate_forecast(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 52,
    include_all: bool = False,
) -> dict[str, float]:
    """Compute all standard forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted/forecast values.
    y_train : array-like, optional
        Training data for MASE/RMSSE calculation.
    seasonality : int
        Seasonal period for scaled metrics (default 52 for weekly data).
    include_all : bool
        If True, include MAPE/sMAPE (not recommended, but available).

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
        metrics["rmsse"] = rmsse(y_true, y_pred, y_train, seasonality)

    if include_all:
        metrics["mape"] = mape(y_true, y_pred)
        metrics["smape"] = smape(y_true, y_pred)

    return metrics


def evaluate_intervals(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Evaluate prediction interval quality.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_lower : array-like
        Lower bound of prediction interval.
    y_upper : array-like
        Upper bound of prediction interval.
    alpha : float
        Significance level (0.05 for 95% interval).

    Returns
    -------
    dict[str, float]
        Metrics: coverage, width, winkler_score.
    """
    return {
        "coverage": coverage(y_true, y_lower, y_upper),
        "interval_width": interval_width(y_lower, y_upper),
        "winkler_score": winkler_score(y_true, y_lower, y_upper, alpha),
    }
