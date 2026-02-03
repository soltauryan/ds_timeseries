"""Forecasting models."""

from ds_timeseries.models.base import BaseForecaster
from ds_timeseries.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WeightedMovingAverageForecaster,
)
from ds_timeseries.models.ensemble import (
    HierarchicalEnsemble,
    SimpleEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
)
from ds_timeseries.models.ml import (
    LightGBMForecaster,
    ProphetForecaster,
    ProphetXGBoostHybrid,
    XGBoostForecaster,
)
from ds_timeseries.models.statistical import (
    ETSForecaster,
    NaiveStatsForecast,
    SeasonalNaiveStatsForecast,
)
from ds_timeseries.models.tuning import (
    TuningResult,
    grid_search_cv,
    random_search_cv,
    tune_lightgbm,
    tune_xgboost,
)

__all__ = [
    "BaseForecaster",
    # Baselines
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "WeightedMovingAverageForecaster",
    # Statistical
    "ETSForecaster",
    "NaiveStatsForecast",
    "SeasonalNaiveStatsForecast",
    # ML
    "LightGBMForecaster",
    "XGBoostForecaster",
    "ProphetForecaster",
    "ProphetXGBoostHybrid",
    # Ensembles
    "SimpleEnsemble",
    "WeightedEnsemble",
    "StackingEnsemble",
    "HierarchicalEnsemble",
    # Tuning
    "TuningResult",
    "grid_search_cv",
    "random_search_cv",
    "tune_lightgbm",
    "tune_xgboost",
]
