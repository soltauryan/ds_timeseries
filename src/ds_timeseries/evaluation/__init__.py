"""Evaluation metrics and cross-validation."""

from ds_timeseries.evaluation.cross_validation import (
    CVFold,
    cross_validate,
    cv_score,
    time_series_cv,
)
from ds_timeseries.evaluation.metrics import (
    bias,
    coverage,
    evaluate_forecast,
    evaluate_intervals,
    interval_width,
    mae,
    mape,
    mase,
    rmse,
    rmsse,
    scaled_pinball_loss,
    smape,
    wape,
    winkler_score,
)
from ds_timeseries.evaluation.plots import (
    plot_cv_performance,
    plot_feature_importance,
    plot_forecast,
    plot_forecast_grid,
    plot_metrics_comparison,
    plot_model_comparison,
    plot_residuals,
)

__all__ = [
    # Primary Metrics (recommended)
    "wape",
    "mae",
    "rmse",
    "mase",
    "rmsse",
    "bias",
    # Secondary Metrics (use with caution)
    "mape",
    "smape",
    # Interval Metrics
    "coverage",
    "interval_width",
    "winkler_score",
    "scaled_pinball_loss",
    # Evaluation Helpers
    "evaluate_forecast",
    "evaluate_intervals",
    # Cross-validation
    "time_series_cv",
    "cross_validate",
    "cv_score",
    "CVFold",
    # Plots
    "plot_forecast",
    "plot_forecast_grid",
    "plot_model_comparison",
    "plot_metrics_comparison",
    "plot_residuals",
    "plot_feature_importance",
    "plot_cv_performance",
]
