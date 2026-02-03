"""Statistical forecasting models.

Wraps statsforecast models to provide consistent API with our BaseForecaster.
Uses the Nixtla statsforecast library for fast, optimized implementations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoETS,
    SeasonalNaive as SFSeasonalNaive,
    Naive as SFNaive,
)

from ds_timeseries.models.base import BaseForecaster


class StatsForecastWrapper(BaseForecaster):
    """Base wrapper for statsforecast models.

    Provides consistent interface while leveraging statsforecast's
    optimized implementations.
    """

    def __init__(self, sf_model: Any, season_length: int = 52, **kwargs: Any) -> None:
        """Initialize with a statsforecast model instance.

        Parameters
        ----------
        sf_model : statsforecast model
            An instantiated statsforecast model (e.g., AutoETS()).
        season_length : int
            Seasonal period (52 for weekly with yearly seasonality).
        """
        super().__init__(season_length=season_length, **kwargs)
        self.sf_model = sf_model
        self.season_length = season_length
        self._sf: StatsForecast | None = None

    def fit(self, df: pd.DataFrame) -> "StatsForecastWrapper":
        """Fit the model using statsforecast."""
        self._validate_input(df)

        self._sf = StatsForecast(
            models=[self.sf_model],
            freq="W-MON",  # Weekly frequency, Monday start
            n_jobs=1,
        )
        self._sf.fit(df[["unique_id", "ds", "y"]])
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._sf is None:
            raise RuntimeError("Model must be fitted before prediction")

        forecasts = self._sf.predict(h=horizon)

        # Standardize output column name to 'yhat'
        model_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={model_col: "yhat"})

        return forecasts[["unique_id", "ds", "yhat"]].reset_index(drop=True)


class ETSForecaster(StatsForecastWrapper):
    """Exponential Smoothing (ETS) forecaster.

    Uses automatic model selection to find the best ETS specification:
    - Error: Additive (A) or Multiplicative (M)
    - Trend: None (N), Additive (A), Additive Damped (Ad)
    - Seasonal: None (N), Additive (A), Multiplicative (M)

    Great for capturing trends and seasonality without heavy compute.

    Parameters
    ----------
    season_length : int
        Seasonal period (default 52 for weekly data with yearly seasonality).
    damped : bool | None
        If True, use damped trend. If None, auto-select.

    Examples
    --------
    >>> model = ETSForecaster(season_length=52)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        season_length: int = 52,
        damped: bool | None = None,
        **kwargs: Any,
    ) -> None:
        sf_model = AutoETS(season_length=season_length, damped=damped)
        super().__init__(sf_model, season_length=season_length, damped=damped, **kwargs)


class NaiveStatsForecast(StatsForecastWrapper):
    """Naive forecaster using statsforecast.

    Predicts the last observed value for all future periods.
    Faster than our pure-Python implementation for large datasets.

    Examples
    --------
    >>> model = NaiveStatsForecast()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(self, **kwargs: Any) -> None:
        sf_model = SFNaive()
        super().__init__(sf_model, **kwargs)


class SeasonalNaiveStatsForecast(StatsForecastWrapper):
    """Seasonal Naive forecaster using statsforecast.

    Predicts value from the same period in the previous season.
    For weekly data with season_length=52, predicts this week as
    the same week last year.

    Parameters
    ----------
    season_length : int
        Seasonal period (default 52 for weekly with yearly seasonality).

    Examples
    --------
    >>> model = SeasonalNaiveStatsForecast(season_length=52)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(self, season_length: int = 52, **kwargs: Any) -> None:
        sf_model = SFSeasonalNaive(season_length=season_length)
        super().__init__(sf_model, season_length=season_length, **kwargs)
