"""Interactive visualization utilities for time series forecasting.

Modern Plotly-based visualizations with:
- Clean, professional styling
- Confidence intervals
- Interactive zooming/hovering
- Responsive date formatting
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Modern color palette
COLORS = {
    "historical": "#2E86AB",      # Steel blue
    "actual": "#A23B72",          # Berry
    "forecast": "#F18F01",        # Orange
    "ci_fill": "rgba(241, 143, 1, 0.2)",  # Orange transparent
    "grid": "#E5E5E5",
    "text": "#2D3436",
}

# Model comparison colors (colorblind-friendly)
MODEL_COLORS = [
    "#2E86AB",  # Steel blue
    "#F18F01",  # Orange
    "#C73E1D",  # Red
    "#3A7D44",  # Forest green
    "#7B2D8E",  # Purple
    "#E8871E",  # Gold
    "#1B998B",  # Teal
    "#5C4D7D",  # Violet
]


def _check_plotly() -> None:
    """Check if Plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly required for interactive visualizations. "
            "Install with: pip install plotly"
        )


def plot_forecast(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    series_id: str | None = None,
    n_history: int = 52,
    title: str | None = None,
    show_ci: bool = True,
    ci_cols: tuple[str, str] | None = None,
    height: int = 500,
    save_path: str | None = None,
) -> Any:
    """Plot forecasts vs actuals for a single series.

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data with unique_id, ds, y.
    forecasts : pd.DataFrame
        Forecast data with unique_id, ds, yhat.
        Optionally includes confidence interval columns.
    series_id : str | None
        Series to plot. If None, uses first series.
    n_history : int
        Number of historical periods to show before forecast.
    title : str | None
        Plot title.
    show_ci : bool
        If True and CI columns present, show confidence intervals.
    ci_cols : tuple[str, str] | None
        Column names for (lower, upper) bounds. Auto-detects if None.
    height : int
        Plot height in pixels.
    save_path : str | None
        Path to save HTML file. If None, displays plot.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure object.

    Examples
    --------
    >>> fig = plot_forecast(test_df, forecasts, series_id="FOODS_1_001_CA_1")
    >>> fig.show()
    """
    _check_plotly()

    if series_id is None:
        series_id = actuals["unique_id"].iloc[0]

    # Filter data
    actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")
    forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")

    if forecast_series.empty:
        raise ValueError(f"No forecasts found for series: {series_id}")

    # Get forecast start date
    forecast_start = forecast_series["ds"].min()

    # Get history
    history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)
    forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]

    # Auto-detect CI columns
    if ci_cols is None and show_ci:
        lower_candidates = ["yhat_lower", "lo", "lower", "yhat_lo"]
        upper_candidates = ["yhat_upper", "hi", "upper", "yhat_hi"]
        for low, high in zip(lower_candidates, upper_candidates):
            if low in forecast_series.columns and high in forecast_series.columns:
                ci_cols = (low, high)
                break

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=history["ds"],
        y=history["y"],
        mode="lines",
        name="Historical",
        line=dict(color=COLORS["historical"], width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>",
    ))

    # Confidence interval (add before forecast line so it's behind)
    if show_ci and ci_cols is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_series["ds"], forecast_series["ds"][::-1]]),
            y=pd.concat([forecast_series[ci_cols[1]], forecast_series[ci_cols[0]][::-1]]),
            fill="toself",
            fillcolor=COLORS["ci_fill"],
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_series["ds"],
        y=forecast_series["yhat"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=COLORS["forecast"], width=2.5),
        marker=dict(size=6),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra></extra>",
    ))

    # Actual values during forecast period
    if not forecast_actuals.empty:
        fig.add_trace(go.Scatter(
            x=forecast_actuals["ds"],
            y=forecast_actuals["y"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=COLORS["actual"], width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.1f}<extra></extra>",
        ))

    # Vertical line at forecast start
    fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        line_color="#888888",
        annotation_text="Forecast Start",
        annotation_position="top",
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=title or f"Forecast: {series_id}",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis=dict(
            title="Date",
            gridcolor=COLORS["grid"],
            tickformat="%b %Y",
            dtick="M2",
        ),
        yaxis=dict(
            title="Value",
            gridcolor=COLORS["grid"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        height=height,
        template="plotly_white",
        margin=dict(t=80, b=60),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_forecast_grid(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    series_ids: list[str] | None = None,
    n_series: int = 9,
    n_history: int = 26,
    n_cols: int = 3,
    height: int = 800,
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
    n_cols : int
        Number of columns in grid.
    height : int
        Total figure height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.
    """
    _check_plotly()

    if series_ids is None:
        all_ids = forecasts["unique_id"].unique()
        n_series = min(n_series, len(all_ids))
        series_ids = list(np.random.choice(all_ids, size=n_series, replace=False))

    n_rows = (len(series_ids) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[s[:30] for s in series_ids],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for idx, series_id in enumerate(series_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")
        forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")

        if forecast_series.empty:
            continue

        forecast_start = forecast_series["ds"].min()
        history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)
        forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]

        # Historical
        fig.add_trace(
            go.Scatter(
                x=history["ds"],
                y=history["y"],
                mode="lines",
                line=dict(color=COLORS["historical"], width=1.5),
                showlegend=(idx == 0),
                name="Historical",
                hovertemplate="%{y:.1f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_series["ds"],
                y=forecast_series["yhat"],
                mode="lines",
                line=dict(color=COLORS["forecast"], width=2),
                showlegend=(idx == 0),
                name="Forecast",
                hovertemplate="%{y:.1f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Actuals in forecast period
        if not forecast_actuals.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast_actuals["ds"],
                    y=forecast_actuals["y"],
                    mode="lines",
                    line=dict(color=COLORS["actual"], width=1.5, dash="dot"),
                    showlegend=(idx == 0),
                    name="Actual",
                    hovertemplate="%{y:.1f}<extra></extra>",
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(
            text="Forecast Grid",
            font=dict(size=16, color=COLORS["text"]),
        ),
        height=height,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(tickformat="%b %y", gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_model_comparison(
    actuals: pd.DataFrame,
    model_forecasts: dict[str, pd.DataFrame],
    series_id: str | None = None,
    n_history: int = 26,
    height: int = 500,
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
    height : int
        Plot height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.

    Examples
    --------
    >>> forecasts = {"LightGBM": lgb_preds, "XGBoost": xgb_preds}
    >>> fig = plot_model_comparison(test_df, forecasts)
    """
    _check_plotly()

    if series_id is None:
        series_id = actuals["unique_id"].iloc[0]

    actual_series = actuals[actuals["unique_id"] == series_id].sort_values("ds")

    # Get forecast start from first model
    first_forecast = list(model_forecasts.values())[0]
    first_series = first_forecast[first_forecast["unique_id"] == series_id]
    forecast_start = first_series["ds"].min()

    history = actual_series[actual_series["ds"] < forecast_start].tail(n_history)
    forecast_actuals = actual_series[actual_series["ds"] >= forecast_start]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=history["ds"],
        y=history["y"],
        mode="lines",
        name="Historical",
        line=dict(color=COLORS["text"], width=2.5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>",
    ))

    # Actuals during forecast
    if not forecast_actuals.empty:
        fig.add_trace(go.Scatter(
            x=forecast_actuals["ds"],
            y=forecast_actuals["y"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=COLORS["text"], width=2.5, dash="dot"),
            marker=dict(size=6),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.1f}<extra></extra>",
        ))

    # Each model's forecast
    for idx, (model_name, forecasts) in enumerate(model_forecasts.items()):
        forecast_series = forecasts[forecasts["unique_id"] == series_id].sort_values("ds")
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]

        fig.add_trace(go.Scatter(
            x=forecast_series["ds"],
            y=forecast_series["yhat"],
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"{model_name}<br>Date: %{{x|%Y-%m-%d}}<br>Forecast: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_vline(x=forecast_start, line_dash="dash", line_color="#888888")

    fig.update_layout(
        title=dict(
            text=f"Model Comparison: {series_id}",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis=dict(title="Date", gridcolor=COLORS["grid"], tickformat="%b %Y"),
        yaxis=dict(title="Value", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=height,
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "wape",
    height: int = 400,
    save_path: str | None = None,
) -> Any:
    """Bar chart comparing model metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with 'model' column and metric columns.
    metric : str
        Metric to plot.
    height : int
        Plot height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.
    """
    _check_plotly()

    df = metrics_df.sort_values(metric, ascending=True)

    # Best model gets highlight color
    colors = [COLORS["forecast"] if i == 0 else COLORS["historical"]
              for i in range(len(df))]

    # Format values for display
    if metric.lower() == "wape":
        text = [f"{v:.1%}" for v in df[metric]]
    else:
        text = [f"{v:.2f}" for v in df[metric]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["model"],
        x=df[metric],
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        hovertemplate="%{y}<br>" + metric.upper() + ": %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Model Comparison: {metric.upper()}",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis=dict(
            title=metric.upper(),
            gridcolor=COLORS["grid"],
            tickformat=".1%" if metric.lower() == "wape" else ".2f",
        ),
        yaxis=dict(title="", gridcolor=COLORS["grid"]),
        height=height,
        template="plotly_white",
        margin=dict(l=150),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_residuals(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    height: int = 700,
    save_path: str | None = None,
) -> Any:
    """Plot residual analysis in a 2x2 grid.

    Shows:
    - Residuals over time
    - Residual distribution
    - Residuals vs predicted
    - WAPE by series (worst performers)

    Parameters
    ----------
    actuals : pd.DataFrame
        Actual data with unique_id, ds, y.
    forecasts : pd.DataFrame
        Forecast data with unique_id, ds, yhat.
    height : int
        Figure height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.
    """
    _check_plotly()

    # Compute residuals
    merged = actuals.merge(forecasts, on=["unique_id", "ds"])
    merged["residual"] = merged["y"] - merged["yhat"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residuals Over Time",
            "Residual Distribution",
            "Residuals vs Predicted",
            "Worst 20 Series (WAPE)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Residuals over time
    fig.add_trace(
        go.Scatter(
            x=merged["ds"],
            y=merged["residual"],
            mode="markers",
            marker=dict(color=COLORS["historical"], size=4, opacity=0.5),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Residual: %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["actual"], row=1, col=1)

    # 2. Residual distribution
    fig.add_trace(
        go.Histogram(
            x=merged["residual"],
            marker_color=COLORS["historical"],
            opacity=0.7,
            hovertemplate="Residual: %{x:.1f}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["actual"], row=1, col=2)

    # Add mean annotation
    mean_res = merged["residual"].mean()
    fig.add_annotation(
        x=mean_res, y=0.95,
        text=f"Mean: {mean_res:.2f}",
        showarrow=False,
        xref="x2", yref="y2 domain",
        font=dict(size=11),
    )

    # 3. Residuals vs predicted
    fig.add_trace(
        go.Scatter(
            x=merged["yhat"],
            y=merged["residual"],
            mode="markers",
            marker=dict(color=COLORS["historical"], size=4, opacity=0.5),
            hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["actual"], row=2, col=1)

    # 4. WAPE by series (worst 20)
    series_wape = (
        merged.groupby("unique_id")
        .apply(lambda x: np.sum(np.abs(x["residual"])) / (np.sum(np.abs(x["y"])) + 1e-6), include_groups=False)
        .sort_values(ascending=True)
        .tail(20)
    )

    fig.add_trace(
        go.Bar(
            y=[s[:20] for s in series_wape.index],
            x=series_wape.values,
            orientation="h",
            marker_color=COLORS["actual"],
            hovertemplate="%{y}<br>WAPE: %{x:.1%}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title=dict(
            text="Residual Analysis",
            font=dict(size=16, color=COLORS["text"]),
        ),
        height=height,
        template="plotly_white",
        showlegend=False,
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Residual", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="Residual", row=1, col=2, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Count", row=1, col=2, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="Predicted", row=2, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Residual", row=2, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="WAPE", row=2, col=2, gridcolor=COLORS["grid"], tickformat=".0%")
    fig.update_yaxes(title_text="", row=2, col=2, gridcolor=COLORS["grid"])

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    height: int = 500,
    save_path: str | None = None,
) -> Any:
    """Plot feature importance from ML model.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns.
    top_n : int
        Number of top features to show.
    height : int
        Plot height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.
    """
    _check_plotly()

    df = importance_df.head(top_n).sort_values("importance", ascending=True)

    # Color by feature type
    def get_color(feat: str) -> str:
        feat_lower = feat.lower()
        if "lag" in feat_lower:
            return "#2E86AB"  # Blue - Lag
        elif "roll" in feat_lower:
            return "#3A7D44"  # Green - Rolling
        elif "fiscal" in feat_lower:
            return "#F18F01"  # Orange - Fiscal
        elif "calendar" in feat_lower or "week" in feat_lower or "month" in feat_lower:
            return "#E8871E"  # Gold - Calendar
        else:
            return "#636E72"  # Gray - Other

    colors = [get_color(f) for f in df["feature"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["importance"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))

    # Add legend manually with dummy traces
    for name, color in [
        ("Lag", "#2E86AB"),
        ("Rolling", "#3A7D44"),
        ("Fiscal", "#F18F01"),
        ("Calendar", "#E8871E"),
        ("Other", "#636E72"),
    ]:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker_color=color,
            name=name,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importance",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis=dict(title="Importance", gridcolor=COLORS["grid"]),
        yaxis=dict(title="", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        template="plotly_white",
        margin=dict(l=200),
        barmode="overlay",
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_cv_performance(
    cv_results: pd.DataFrame,
    metric: str = "wape",
    height: int = 400,
    save_path: str | None = None,
) -> Any:
    """Plot cross-validation performance over folds.

    Parameters
    ----------
    cv_results : pd.DataFrame
        CV results with 'fold' and metric columns.
    metric : str
        Metric to plot.
    height : int
        Plot height.
    save_path : str | None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object.
    """
    _check_plotly()

    fig = go.Figure()

    # Check if multiple models
    if "model" in cv_results.columns:
        for idx, model in enumerate(cv_results["model"].unique()):
            model_data = cv_results[cv_results["model"] == model]
            color = MODEL_COLORS[idx % len(MODEL_COLORS)]

            fig.add_trace(go.Scatter(
                x=model_data["fold"],
                y=model_data[metric],
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=cv_results["fold"],
            y=cv_results[metric],
            mode="lines+markers",
            name=metric.upper(),
            line=dict(color=COLORS["forecast"], width=2),
            marker=dict(size=8),
        ))

    # Add mean line
    mean_val = cv_results[metric].mean()
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="#888888",
        annotation_text=f"Mean: {mean_val:.3f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=dict(
            text=f"Cross-Validation Performance: {metric.upper()}",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis=dict(title="Fold", gridcolor=COLORS["grid"], dtick=1),
        yaxis=dict(
            title=metric.upper(),
            gridcolor=COLORS["grid"],
            tickformat=".1%" if metric.lower() == "wape" else ".3f",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)

    return fig
