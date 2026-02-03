"""Lag and Rolling Features for Time Series.

Creates lag features and rolling statistics commonly used in
time series forecasting with ML models.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    target_col: str = "y",
    group_col: str = "unique_id",
) -> pd.DataFrame:
    """Add lag features to time series data.

    Creates lagged versions of the target variable. For weekly data,
    common lags are 1 (last week), 4 (last month), 13 (last quarter),
    52 (last year).

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    lags : list[int] | None
        Lag periods to create. Defaults to [1, 2, 4, 8, 13, 26, 52].
    target_col : str
        Column to create lags from.
    group_col : str
        Column to group by (each time series).

    Returns
    -------
    pd.DataFrame
        Data with lag columns added (e.g., y_lag_1, y_lag_4, etc.).

    Examples
    --------
    >>> df = add_lag_features(df, lags=[1, 4, 52])
    >>> df[["ds", "y", "y_lag_1", "y_lag_4", "y_lag_52"]].head()
    """
    if lags is None:
        lags = [1, 2, 4, 8, 13, 26, 52]

    df = df.copy()
    df = df.sort_values([group_col, "ds"])

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    target_col: str = "y",
    group_col: str = "unique_id",
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling window statistics to time series data.

    Creates rolling mean, std, min, max over specified windows.
    Uses a "shift by 1" to avoid data leakage (current value excluded).

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    windows : list[int] | None
        Rolling window sizes. Defaults to [4, 8, 13, 26, 52].
    target_col : str
        Column to compute rolling stats from.
    group_col : str
        Column to group by (each time series).
    agg_funcs : list[str] | None
        Aggregation functions. Defaults to ["mean", "std", "min", "max"].

    Returns
    -------
    pd.DataFrame
        Data with rolling feature columns added.

    Examples
    --------
    >>> df = add_rolling_features(df, windows=[4, 13])
    >>> df[["ds", "y", "y_roll_4_mean", "y_roll_13_mean"]].head()
    """
    if windows is None:
        windows = [4, 8, 13, 26, 52]

    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max"]

    df = df.copy()
    df = df.sort_values([group_col, "ds"])

    for window in windows:
        # Shift by 1 to avoid including current value (data leakage)
        shifted = df.groupby(group_col)[target_col].shift(1)

        for func in agg_funcs:
            col_name = f"{target_col}_roll_{window}_{func}"

            if func == "mean":
                df[col_name] = shifted.groupby(df[group_col]).transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            elif func == "std":
                df[col_name] = shifted.groupby(df[group_col]).transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
            elif func == "min":
                df[col_name] = shifted.groupby(df[group_col]).transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
            elif func == "max":
                df[col_name] = shifted.groupby(df[group_col]).transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
            elif func == "sum":
                df[col_name] = shifted.groupby(df[group_col]).transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )

    return df


def add_diff_features(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    target_col: str = "y",
    group_col: str = "unique_id",
) -> pd.DataFrame:
    """Add difference features (change from previous periods).

    Creates features representing the change in value from N periods ago.
    Useful for capturing trends and momentum.

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    periods : list[int] | None
        Differencing periods. Defaults to [1, 4, 52].
    target_col : str
        Column to difference.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Data with difference columns added.

    Examples
    --------
    >>> df = add_diff_features(df, periods=[1, 52])
    >>> # y_diff_1 = change from last week
    >>> # y_diff_52 = change from same week last year
    """
    if periods is None:
        periods = [1, 4, 52]

    df = df.copy()
    df = df.sort_values([group_col, "ds"])

    for period in periods:
        col_name = f"{target_col}_diff_{period}"
        df[col_name] = df.groupby(group_col)[target_col].diff(period)

    return df


def add_pct_change_features(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    target_col: str = "y",
    group_col: str = "unique_id",
) -> pd.DataFrame:
    """Add percentage change features.

    Creates features representing percent change from N periods ago.

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    periods : list[int] | None
        Periods for pct change. Defaults to [1, 4, 52].
    target_col : str
        Column to compute pct change from.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Data with pct change columns added.
    """
    if periods is None:
        periods = [1, 4, 52]

    df = df.copy()
    df = df.sort_values([group_col, "ds"])

    for period in periods:
        col_name = f"{target_col}_pct_{period}"
        df[col_name] = df.groupby(group_col)[target_col].pct_change(period)
        # Handle inf values from division by zero
        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

    return df


def add_expanding_features(
    df: pd.DataFrame,
    target_col: str = "y",
    group_col: str = "unique_id",
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Add expanding window statistics (all history up to current point).

    Creates cumulative/expanding statistics that use all available
    historical data up to (but not including) the current period.

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    target_col : str
        Column to compute expanding stats from.
    group_col : str
        Column to group by.
    agg_funcs : list[str] | None
        Aggregation functions. Defaults to ["mean", "std"].

    Returns
    -------
    pd.DataFrame
        Data with expanding feature columns.
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    df = df.copy()
    df = df.sort_values([group_col, "ds"])

    # Shift by 1 to avoid data leakage
    shifted = df.groupby(group_col)[target_col].shift(1)

    for func in agg_funcs:
        col_name = f"{target_col}_expanding_{func}"

        if func == "mean":
            df[col_name] = shifted.groupby(df[group_col]).transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
        elif func == "std":
            df[col_name] = shifted.groupby(df[group_col]).transform(
                lambda x: x.expanding(min_periods=2).std()
            )
        elif func == "sum":
            df[col_name] = shifted.groupby(df[group_col]).transform(
                lambda x: x.expanding(min_periods=1).sum()
            )
        elif func == "count":
            df[col_name] = shifted.groupby(df[group_col]).transform(
                lambda x: x.expanding(min_periods=1).count()
            )

    return df
