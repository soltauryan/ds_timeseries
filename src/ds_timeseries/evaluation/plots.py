"""Visualization utilities for time series forecasting.

Provides plots for:
- Predictions vs actuals
- Model comparison
- Residual analysis
- Feature importance
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_forecast(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    series_id: str | None = None,
    n_history: int = 52,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> Any:
    """Plot forecasts vs actuals for a single series.

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data with unique_id, ds, y.
    forecasts : pd.DataFrame
        Forecast data with unique_id, ds, yhat.
    series_id : str | None
        Series to plot. If None, uses first series.
    n_history : int
        Number of historical periods to show before forecast.
    title : str | None
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure. If None, displays plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> fig = plot_forecast(test_df, forecasts, series_id="FOODS_1_001_CA_1")
    >>> fig.savefig("forecast.png")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    if series_id is None:
        series_id = actuals["unique_id"].iloc[0]

    # Filter data
    actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")
    forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")

    # Get forecast start date
    forecast_start = forecast_series["ds"].min()

    # Get history (n_history periods before forecast)
    history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)

    # Get actuals during forecast period
    forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot history
    ax.plot(history["ds"], history["y"], "b-", label="Historical", linewidth=1.5)

    # Plot actuals during forecast period
    if not forecast_actuals.empty:
        ax.plot(
            forecast_actuals["ds"],
            forecast_actuals["y"],
            "b--",
            label="Actual",
            linewidth=1.5,
            alpha=0.7,
        )

    # Plot forecast
    ax.plot(
        forecast_series["ds"],
        forecast_series["yhat"],
        "r-",
        label="Forecast",
        linewidth=2,
    )

    # Add vertical line at forecast start
    ax.axvline(x=forecast_start, color="gray", linestyle="--", alpha=0.5)

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title or f"Forecast vs Actual: {series_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_forecast_grid(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    series_ids: list[str] | None = None,
    n_series: int = 9,
    n_history: int = 26,
    figsize: tuple[int, int] = (15, 12),
    save_path: str | None = None,
) -> Any:
    """Plot forecasts for multiple series in a grid.

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data.
    forecasts : pd.DataFrame
        Forecast data.
    series_ids : list[str] | None
        Series to plot. If None, samples n_series randomly.
    n_series : int
        Number of series to plot if series_ids not provided.
    n_history : int
        Historical periods to show.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    if series_ids is None:
        all_ids = forecasts["unique_id"].unique()
        n_series = min(n_series, len(all_ids))
        series_ids = np.random.choice(all_ids, size=n_series, replace=False)

    n_cols = 3
    n_rows = (len(series_ids) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, series_id in enumerate(series_ids):
        ax = axes[idx]

        actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")
        forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")

        if forecast_series.empty:
            continue

        forecast_start = forecast_series["ds"].min()
        history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)
        forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]

        ax.plot(history["ds"], history["y"], "b-", linewidth=1)
        if not forecast_actuals.empty:
            ax.plot(forecast_actuals["ds"], forecast_actuals["y"], "b--", alpha=0.7)
        ax.plot(forecast_series["ds"], forecast_series["yhat"], "r-", linewidth=1.5)

        ax.set_title(series_id[:25], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(series_ids), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Forecast vs Actual (Blue=Actual, Red=Forecast)", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    actuals: pd.DataFrame,
    model_forecasts: dict[str, pd.DataFrame],
    series_id: str | None = None,
    n_history: int = 26,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> Any:
    """Compare forecasts from multiple models.

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data.
    model_forecasts : dict[str, pd.DataFrame]
        Dictionary mapping model names to their forecasts.
    series_id : str | None
        Series to plot.
    n_history : int
        Historical periods to show.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> forecasts = {
    ...     "LightGBM": lgb_preds,
    ...     "XGBoost": xgb_preds,
    ...     "ETS": ets_preds,
    ... }
    >>> fig = plot_model_comparison(test_df, forecasts, series_id="FOODS_1_001_CA_1")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    if series_id is None:
        series_id = actuals["unique_id"].iloc[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Get actual series
    actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")

    # Find forecast start
    first_forecast = list(model_forecasts.values())[0]
    first_forecast_series = first_forecast[first_forecast["unique_id"] == series_id]
    forecast_start = first_forecast_series["ds"].min()

    # Plot history
    history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)
    ax.plot(history["ds"], history["y"], "k-", label="Historical", linewidth=2)

    # Plot actuals during forecast
    forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]
    if not forecast_actuals.empty:
        ax.plot(
            forecast_actuals["ds"],
            forecast_actuals["y"],
            "k--",
            label="Actual",
            linewidth=2,
            alpha=0.8,
        )

    # Plot each model's forecast
    colors = plt.cm.tab10.colors
    for idx, (model_name, forecasts) in enumerate(model_forecasts.items()):
        forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")
        ax.plot(
            forecast_series["ds"],
            forecast_series["yhat"],
            color=colors[idx % len(colors)],
            label=model_name,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.axvline(x=forecast_start, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Model Comparison: {series_id}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "wape",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> Any:
    """Bar chart comparing model metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with 'model' column and metric columns.
    metric : str
        Metric to plot.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> metrics = pd.DataFrame([
    ...     {"model": "LightGBM", "wape": 0.37, "mae": 5.60},
    ...     {"model": "XGBoost", "wape": 0.36, "mae": 5.55},
    ... ])
    >>> fig = plot_metrics_comparison(metrics, metric="wape")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric
    df = metrics_df.sort_values(metric)

    colors = ["green" if i == 0 else "steelblue" for i in range(len(df))]

    bars = ax.barh(df["model"], df[metric], color=colors)

    # Add value labels
    for bar, val in zip(bars, df[metric]):
        if metric == "wape":
            label = f"{val:.1%}"
        else:
            label = f"{val:.2f}"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison by {metric.upper()}")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_residuals(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> Any:
    """Plot residual analysis.

    Shows:
    - Residuals over time
    - Residual distribution
    - Residuals by series

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data with unique_id, ds, y.
    forecasts : pd.DataFrame
        Forecast data with unique_id, ds, yhat.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    # Merge and compute residuals
    merged = actuals.merge(forecasts, on=["unique_id", "ds"])
    merged["residual"] = merged["y"] - merged["yhat"]
    merged["pct_error"] = merged["residual"] / (merged["y"] + 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Residuals over time
    ax = axes[0, 0]
    ax.scatter(merged["ds"], merged["residual"], alpha=0.5, s=10)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals Over Time")
    ax.grid(True, alpha=0.3)

    # 2. Residual distribution
    ax = axes[0, 1]
    ax.hist(merged["residual"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution (mean={merged['residual'].mean():.2f})")
    ax.grid(True, alpha=0.3)

    # 3. Residuals vs predicted
    ax = axes[1, 0]
    ax.scatter(merged["yhat"], merged["residual"], alpha=0.5, s=10)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)

    # 4. WAPE by series
    ax = axes[1, 1]
    series_wape = (
        merged.groupby("unique_id")
        .apply(lambda x: np.sum(np.abs(x["residual"])) / (np.sum(np.abs(x["y"])) + 1e-6))
        .sort_values(ascending=False)
        .head(20)
    )
    ax.barh(range(len(series_wape)), series_wape.values)
    ax.set_yticks(range(len(series_wape)))
    ax.set_yticklabels([s[:15] for s in series_wape.index], fontsize=8)
    ax.set_xlabel("WAPE")
    ax.set_title("Worst 20 Series by WAPE")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | None = None,
) -> Any:
    """Plot feature importance from ML model.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns.
    top_n : int
        Number of top features to show.
    figsize : tuple
        Figure size.
    save_path : str | None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> importance = model.get_feature_importance()
    >>> fig = plot_feature_importance(importance, top_n=15)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    df = importance_df.head(top_n).sort_values("importance")

    # Color by feature type
    colors = []
    for feat in df["feature"]:
        if "lag" in feat:
            colors.append("steelblue")
        elif "roll" in feat:
            colors.append("forestgreen")
        elif "fiscal" in feat:
            colors.append("coral")
        elif "calendar" in feat:
            colors.append("gold")
        else:
            colors.append("gray")

    ax.barh(df["feature"], df["importance"], color=colors)

    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Lag"),
        Patch(facecolor="forestgreen", label="Rolling"),
        Patch(facecolor="coral", label="Fiscal"),
        Patch(facecolor="gold", label="Calendar"),
        Patch(facecolor="gray", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
