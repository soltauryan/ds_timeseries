"""Feature engineering modules for time series."""

from ds_timeseries.features.calendar import (
    FiscalCalendarConfig,
    add_fiscal_features,
    create_mock_fiscal_calendar,
    generate_fiscal_calendar,
    get_fiscal_period_summary,
)
from ds_timeseries.features.engineering import (
    FeatureConfig,
    engineer_features,
    prepare_ml_dataset,
)
from ds_timeseries.features.lags import (
    add_diff_features,
    add_expanding_features,
    add_lag_features,
    add_pct_change_features,
    add_rolling_features,
)

__all__ = [
    # Calendar
    "FiscalCalendarConfig",
    "generate_fiscal_calendar",
    "create_mock_fiscal_calendar",
    "add_fiscal_features",
    "get_fiscal_period_summary",
    # Lags
    "add_lag_features",
    "add_rolling_features",
    "add_diff_features",
    "add_pct_change_features",
    "add_expanding_features",
    # Engineering
    "FeatureConfig",
    "engineer_features",
    "prepare_ml_dataset",
]
