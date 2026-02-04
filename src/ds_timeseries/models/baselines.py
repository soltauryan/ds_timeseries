"""Baseline forecasting models.

Simple heuristic models to establish benchmarks. These should be the first
models run to set a minimum bar for more complex approaches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ds_timeseries.models.base import BaseForecaster
from ds_timeseries.utils.config import DEFAULT_FREQ


class NaiveForecaster(BaseForecaster):
    """Naive forecaster - predicts last observed value.

    The simplest possible baseline. Predicts that future values
    will equal the most recent observed value.

    Examples
    --------
    >>> model = NaiveForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def fit(self, df: pd.DataFrame) -> "NaiveForecaster":
        """Store the last value for each time series."""
        self._validate_input(df)

        self.last_values_ = (
            df.sort_values("ds")
            .groupby("unique_id")
            .agg({"y": "last", "ds": "max"})
        )
        self._is_fitted = True
        return self

    def predict(self, horizon: int, freq: str = DEFAULT_FREQ, **kwargs: Any) -> pd.DataFrame:
        """Predict last value for all future periods."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        records = []
        for uid, row in self.last_values_.iterrows():
            future_dates = pd.date_range(
                start=row["ds"] + pd.Timedelta(days=7),
                periods=horizon,
                freq=freq,
            )
            for ds in future_dates:
                records.append({
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": row["y"],
                })

        return pd.DataFrame(records)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive forecaster - predicts value from same period last year.

    For weekly data, predicts this week's value as the value from the
    same week last year (52 weeks ago). This is a surprisingly strong
    baseline for retail data with yearly seasonality.

    Parameters
    ----------
    season_length : int
        Number of periods in one seasonal cycle (default 52 for weekly).

    Examples
    --------
    >>> model = SeasonalNaiveForecaster(season_length=52)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(self, season_length: int = 52, **kwargs: Any) -> None:
        super().__init__(season_length=season_length, **kwargs)
        self.season_length = season_length

    def fit(self, df: pd.DataFrame) -> "SeasonalNaiveForecaster":
        """Store the last season of values for each time series."""
        self._validate_input(df)

        # Store last `season_length` values per series
        self.history_ = (
            df.sort_values("ds")
            .groupby("unique_id")
            .tail(self.season_length)
            .copy()
        )
        self.last_dates_ = df.groupby("unique_id")["ds"].max()
        self._is_fitted = True
        return self

    def predict(self, horizon: int, freq: str = DEFAULT_FREQ, **kwargs: Any) -> pd.DataFrame:
        """Predict using values from same period last season."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        records = []

        for uid in self.last_dates_.index:
            last_date = self.last_dates_[uid]
            series_history = self.history_[self.history_["unique_id"] == uid].sort_values("ds")

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=horizon,
                freq=freq,
            )

            values = series_history["y"].values
            n_values = len(values)

            for i, ds in enumerate(future_dates):
                # Cycle through seasonal values
                idx = i % n_values
                records.append({
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": values[idx] if n_values > 0 else 0.0,
                })

        return pd.DataFrame(records)


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecaster.

    Predicts future values as the average of the last `window` observations.
    Simple but effective for stable time series without strong trends.

    Parameters
    ----------
    window : int
        Number of periods to average (default 4).

    Examples
    --------
    >>> model = MovingAverageForecaster(window=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(self, window: int = 4, **kwargs: Any) -> None:
        super().__init__(window=window, **kwargs)
        self.window = window

    def fit(self, df: pd.DataFrame) -> "MovingAverageForecaster":
        """Compute moving average for each time series."""
        self._validate_input(df)

        # Compute mean of last `window` values
        self.means_ = (
            df.sort_values("ds")
            .groupby("unique_id")
            .tail(self.window)
            .groupby("unique_id")["y"]
            .mean()
        )
        self.last_dates_ = df.groupby("unique_id")["ds"].max()
        self._is_fitted = True
        return self

    def predict(self, horizon: int, freq: str = DEFAULT_FREQ, **kwargs: Any) -> pd.DataFrame:
        """Predict moving average for all future periods."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        records = []

        for uid in self.last_dates_.index:
            last_date = self.last_dates_[uid]
            mean_value = self.means_.get(uid, 0.0)

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=horizon,
                freq=freq,
            )

            for ds in future_dates:
                records.append({
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": mean_value,
                })

        return pd.DataFrame(records)


class WeightedMovingAverageForecaster(BaseForecaster):
    """Weighted Moving Average forecaster.

    More recent observations have higher weights.
    Uses linearly decaying weights by default.

    Parameters
    ----------
    window : int
        Number of periods to average (default 4).
    weights : list[float], optional
        Custom weights (most recent first). If None, uses linear decay.

    Examples
    --------
    >>> model = WeightedMovingAverageForecaster(window=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        window: int = 4,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(window=window, **kwargs)
        self.window = window

        if weights is None:
            # Linear decay: most recent has highest weight
            self.weights = np.arange(1, window + 1, dtype=float)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def fit(self, df: pd.DataFrame) -> "WeightedMovingAverageForecaster":
        """Compute weighted moving average for each time series."""
        self._validate_input(df)

        self.wmeans_ = {}
        self.last_dates_ = {}

        for uid, group in df.groupby("unique_id"):
            group = group.sort_values("ds").tail(self.window)
            values = group["y"].values

            # Pad with zeros if not enough history
            if len(values) < self.window:
                values = np.pad(values, (self.window - len(values), 0), mode="edge")

            # Apply weights (weights are for most recent first, so reverse values)
            wmean = np.dot(self.weights, values)
            self.wmeans_[uid] = wmean
            self.last_dates_[uid] = group["ds"].max()

        self._is_fitted = True
        return self

    def predict(self, horizon: int, freq: str = DEFAULT_FREQ, **kwargs: Any) -> pd.DataFrame:
        """Predict weighted moving average for all future periods."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        records = []

        for uid, last_date in self.last_dates_.items():
            wmean = self.wmeans_.get(uid, 0.0)

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=horizon,
                freq=freq,
            )

            for ds in future_dates:
                records.append({
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": wmean,
                })

        return pd.DataFrame(records)
