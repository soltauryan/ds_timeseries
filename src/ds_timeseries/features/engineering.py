"""Feature Engineering Pipeline.

Combines calendar, lag, and rolling features into a unified pipeline
for preparing data for ML models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from ds_timeseries.features.calendar import (
    FiscalCalendarConfig,
    add_fiscal_features,
    generate_fiscal_calendar,
)
from ds_timeseries.features.lags import (
    add_lag_features,
    add_rolling_features,
    add_diff_features,
    add_pct_change_features,
    add_expanding_features,
)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline.

    Attributes
    ----------
    lags : list[int]
        Lag periods for lag features.
    rolling_windows : list[int]
        Window sizes for rolling statistics.
    rolling_aggs : list[str]
        Aggregation functions for rolling stats.
    diff_periods : list[int]
        Periods for difference features.
    pct_change_periods : list[int]
        Periods for percent change features.
    include_expanding : bool
        Whether to include expanding window features.
    fiscal_config : FiscalCalendarConfig | None
        Fiscal calendar configuration.
    include_calendar_features : bool
        Whether to include basic calendar features (day, month, etc.).
    """

    lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 13, 26, 52])
    rolling_windows: list[int] = field(default_factory=lambda: [4, 13, 26, 52])
    rolling_aggs: list[str] = field(default_factory=lambda: ["mean", "std"])
    diff_periods: list[int] = field(default_factory=lambda: [1, 4, 52])
    pct_change_periods: list[int] = field(default_factory=lambda: [1, 52])
    include_expanding: bool = True
    fiscal_config: FiscalCalendarConfig | None = None
    include_calendar_features: bool = True

    def __post_init__(self):
        if self.fiscal_config is None:
            # Default: November start, 5-4-4 pattern
            self.fiscal_config = FiscalCalendarConfig(
                fiscal_year_start_month=11,
                week_pattern="5-4-4",
            )


def engineer_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    target_col: str = "y",
    group_col: str = "unique_id",
    fiscal_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply full feature engineering pipeline.

    Creates lag, rolling, calendar, and fiscal features for ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Data with unique_id, ds, y columns.
    config : FeatureConfig | None
        Feature engineering configuration.
    target_col : str
        Name of target column.
    group_col : str
        Name of grouping column.
    fiscal_calendar : pd.DataFrame | None
        Pre-generated fiscal calendar. Generated if not provided.

    Returns
    -------
    pd.DataFrame
        Data with all engineered features.

    Examples
    --------
    >>> config = FeatureConfig(lags=[1, 4, 52], rolling_windows=[4, 13])
    >>> df_features = engineer_features(df, config)
    >>> print(df_features.columns.tolist())
    """
    config = config or FeatureConfig()
    df = df.copy()

    # 1. Add fiscal calendar features
    if config.fiscal_config is not None:
        if fiscal_calendar is None:
            fiscal_calendar = generate_fiscal_calendar(
                df["ds"].min() - pd.Timedelta(days=7),
                df["ds"].max() + pd.Timedelta(days=7),
                config.fiscal_config,
            )
        df = add_fiscal_features(df, fiscal_calendar, config.fiscal_config)

    # 2. Add basic calendar features
    if config.include_calendar_features:
        df = add_calendar_features(df)

    # 3. Add lag features
    if config.lags:
        df = add_lag_features(df, config.lags, target_col, group_col)

    # 4. Add rolling features
    if config.rolling_windows:
        df = add_rolling_features(
            df, config.rolling_windows, target_col, group_col, config.rolling_aggs
        )

    # 5. Add difference features
    if config.diff_periods:
        df = add_diff_features(df, config.diff_periods, target_col, group_col)

    # 6. Add pct change features
    if config.pct_change_periods:
        df = add_pct_change_features(df, config.pct_change_periods, target_col, group_col)

    # 7. Add expanding features
    if config.include_expanding:
        df = add_expanding_features(df, target_col, group_col, ["mean", "std"])

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic calendar features from datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'ds' datetime column.

    Returns
    -------
    pd.DataFrame
        Data with calendar features added.
    """
    df = df.copy()

    df["calendar_year"] = df["ds"].dt.year
    df["calendar_month"] = df["ds"].dt.month
    df["calendar_week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["calendar_day_of_year"] = df["ds"].dt.dayofyear
    df["calendar_quarter"] = df["ds"].dt.quarter

    # Cyclical encoding for week of year (captures yearly seasonality)
    df["week_sin"] = np.sin(2 * np.pi * df["calendar_week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["calendar_week"] / 52)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["calendar_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["calendar_month"] / 12)

    return df


def prepare_ml_dataset(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    target_col: str = "y",
    drop_na: bool = True,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare dataset for ML training.

    Applies feature engineering and returns feature/target splits
    with metadata about feature types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with unique_id, ds, y.
    config : FeatureConfig | None
        Feature configuration.
    target_col : str
        Target column name.
    drop_na : bool
        Whether to drop rows with NaN (from lag features).
    categorical_cols : list[str] | None
        Columns to treat as categorical. Auto-detected if None.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        (dataframe, numeric_features, categorical_features)

    Examples
    --------
    >>> df_ml, num_feats, cat_feats = prepare_ml_dataset(df)
    >>> X = df_ml[num_feats + cat_feats]
    >>> y = df_ml["y"]
    """
    config = config or FeatureConfig()

    # Engineer features
    df_feat = engineer_features(df, config, target_col)

    # Identify feature columns
    exclude_cols = {"unique_id", "ds", target_col}
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

    # Separate numeric and categorical
    if categorical_cols is None:
        categorical_cols = []
        for col in feature_cols:
            if df_feat[col].dtype == "object" or df_feat[col].dtype.name == "category":
                categorical_cols.append(col)
            # Also treat hierarchy columns as categorical
            elif col in ["item_id", "store_id", "cat_id", "dept_id", "state_id",
                        "customer_id", "material_id", "category", "location"]:
                categorical_cols.append(col)

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Drop rows with NaN if requested
    if drop_na:
        df_feat = df_feat.dropna(subset=numeric_cols)

    return df_feat, numeric_cols, categorical_cols


def get_feature_importance_groups() -> dict[str, list[str]]:
    """Get feature name patterns grouped by type.

    Useful for feature importance analysis.

    Returns
    -------
    dict[str, list[str]]
        Mapping of feature group to name patterns.
    """
    return {
        "lag": ["_lag_"],
        "rolling": ["_roll_"],
        "diff": ["_diff_"],
        "pct_change": ["_pct_"],
        "expanding": ["_expanding_"],
        "fiscal": ["fiscal_", "is_fiscal_"],
        "calendar": ["calendar_", "week_sin", "week_cos", "month_sin", "month_cos"],
        "hierarchy": ["item_id", "store_id", "cat_id", "dept_id", "state_id"],
    }
