"""Forecasting models."""

from ds_timeseries.models.base import BaseForecaster
from ds_timeseries.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WeightedMovingAverageForecaster,
)
from ds_timeseries.models.ensemble import (
    DRFAMEnsemble,
    HierarchicalEnsemble,
    M5WinnerEnsemble,
    MultiLevelPoolingEnsemble,
    SimpleEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
)
from ds_timeseries.models.ml import (
    CatBoostForecaster,
    LightGBMForecaster,
    ProphetForecaster,
    ProphetXGBoostHybrid,
    XGBoostForecaster,
)
from ds_timeseries.models.statistical import (
    ETSForecaster,
    NaiveStatsForecast,
    SeasonalNaiveStatsForecast,
    # Intermittent demand models
    CrostonForecaster,
    SBAForecaster,
    TSBForecaster,
    ADIDAForecaster,
    IMAPAForecaster,
)
from ds_timeseries.models.tuning import (
    TuningResult,
    grid_search_cv,
    random_search_cv,
    tune_lightgbm,
    tune_xgboost,
)

# Neural models (require optional 'neural' dependencies)
try:
    from ds_timeseries.models.neural import (
        AutoNeuralForecaster,
        DeepARForecaster,
        LSTMForecaster,
        NBEATSForecaster,
        NeuralConfig,
        NeuralEnsembleForecaster,
        NHITSForecaster,
        TFTForecaster,
    )
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

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
    # Intermittent Demand (sparse/sporadic demand)
    "CrostonForecaster",
    "SBAForecaster",
    "TSBForecaster",
    "ADIDAForecaster",
    "IMAPAForecaster",
    # ML (Gradient Boosting)
    "LightGBMForecaster",
    "XGBoostForecaster",
    "CatBoostForecaster",
    "ProphetForecaster",
    "ProphetXGBoostHybrid",
    # Ensembles
    "SimpleEnsemble",
    "WeightedEnsemble",
    "StackingEnsemble",
    "HierarchicalEnsemble",
    "DRFAMEnsemble",
    "MultiLevelPoolingEnsemble",
    "M5WinnerEnsemble",
    # Tuning
    "TuningResult",
    "grid_search_cv",
    "random_search_cv",
    "tune_lightgbm",
    "tune_xgboost",
]

# Add neural models if available
if _NEURAL_AVAILABLE:
    __all__.extend([
        # Neural Networks (require 'neural' extra)
        "NeuralConfig",
        "NBEATSForecaster",
        "NHITSForecaster",
        "DeepARForecaster",
        "TFTForecaster",
        "LSTMForecaster",
        "AutoNeuralForecaster",
        "NeuralEnsembleForecaster",
    ])
