"""Statistical forecasting models.

Wraps statsforecast models to provide consistent API with our BaseForecaster.
Uses the Nixtla statsforecast library for fast, optimized implementations.

Includes specialized models for intermittent demand:
- Croston: Classic method for sporadic demand
- SBA: Syntetos-Boylan Approximation (bias-corrected Croston)
- TSB: Teunter-Syntetos-Babai (handles obsolescence)
- ADIDA: Aggregate-Disaggregate Intermittent Demand Approach
- IMAPA: Intermittent Multiple Aggregation Prediction Algorithm
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoETS,
    SeasonalNaive as SFSeasonalNaive,
    Naive as SFNaive,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    TSB,
    ADIDA,
    IMAPA,
)

from ds_timeseries.models.base import BaseForecaster
from ds_timeseries.utils.config import DEFAULT_FREQ


class StatsForecastWrapper(BaseForecaster):
    """Base wrapper for statsforecast models.

    Provides consistent interface while leveraging statsforecast's
    optimized implementations.
    """

    def __init__(self, sf_model: Any, season_length: int = 52, freq: str = DEFAULT_FREQ, **kwargs: Any) -> None:
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
        self.freq = freq
        self._sf: StatsForecast | None = None

    def fit(self, df: pd.DataFrame) -> "StatsForecastWrapper":
        """Fit the model using statsforecast."""
        self._validate_input(df)

        self._sf = StatsForecast(
            models=[self.sf_model],
            freq=self.freq,
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


# =============================================================================
# Intermittent Demand Models
# =============================================================================


class CrostonForecaster(StatsForecastWrapper):
    """Croston's method for intermittent demand forecasting.

    Designed for time series with many zeros (sporadic/intermittent demand).
    Separates the forecasting into:
    1. Non-zero demand size (exponential smoothing)
    2. Inter-demand interval (exponential smoothing)

    Best for: Spare parts, slow-moving items, products with >80% zero values.

    Parameters
    ----------
    optimized : bool
        If True, use optimized version that selects best alpha.
        If False, use classic version with fixed alpha.

    Examples
    --------
    >>> model = CrostonForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    Notes
    -----
    Known limitation: Croston's method is biased (tends to over-forecast).
    Consider using SBA (Syntetos-Boylan Approximation) for bias correction.

    References
    ----------
    - Croston, J. D. (1972). "Forecasting and Stock Control for
      Intermittent Demands"
    """

    def __init__(self, optimized: bool = True, **kwargs: Any) -> None:
        if optimized:
            sf_model = CrostonOptimized()
        else:
            sf_model = CrostonClassic()
        super().__init__(sf_model, optimized=optimized, **kwargs)


class SBAForecaster(StatsForecastWrapper):
    """Syntetos-Boylan Approximation (SBA) for intermittent demand.

    Bias-corrected version of Croston's method. Applies a deflating factor
    to reduce the systematic over-forecasting of Croston.

    Recommended over Croston for most intermittent demand cases.

    Parameters
    ----------
    None (uses default statsforecast parameters)

    Examples
    --------
    >>> model = SBAForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    Notes
    -----
    SBA has been shown in multiple empirical studies to outperform
    Croston's method in terms of both forecasting and inventory performance.

    References
    ----------
    - Syntetos, A. A., & Boylan, J. E. (2005). "The accuracy of intermittent
      demand estimates"
    """

    def __init__(self, **kwargs: Any) -> None:
        sf_model = CrostonSBA()
        super().__init__(sf_model, **kwargs)


class TSBForecaster(StatsForecastWrapper):
    """Teunter-Syntetos-Babai (TSB) method for intermittent demand.

    Improves on Croston and SBA by updating the demand probability
    (rather than interval) in every period, including zero-demand periods.

    Key advantage: Handles obsolescence (when an item stops selling).
    TSB will decrease forecasts during extended periods of zero demand.

    Parameters
    ----------
    alpha_d : float
        Smoothing parameter for demand size (0 < alpha < 1).
    alpha_p : float
        Smoothing parameter for demand probability (0 < alpha < 1).

    Examples
    --------
    >>> model = TSBForecaster(alpha_d=0.1, alpha_p=0.1)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    When to use:
    - Products with risk of obsolescence
    - Items that may be discontinued
    - Slow-moving inventory with declining demand

    References
    ----------
    - Teunter, R., Syntetos, A., & Babai, M. Z. (2011). "Intermittent demand:
      Linking forecasting to inventory obsolescence"
    """

    def __init__(self, alpha_d: float = 0.1, alpha_p: float = 0.1, **kwargs: Any) -> None:
        sf_model = TSB(alpha_d=alpha_d, alpha_p=alpha_p)
        super().__init__(sf_model, alpha_d=alpha_d, alpha_p=alpha_p, **kwargs)


class ADIDAForecaster(StatsForecastWrapper):
    """Aggregate-Disaggregate Intermittent Demand Approach (ADIDA).

    Temporal aggregation approach for intermittent demand:
    1. Aggregates data to reduce intermittence
    2. Forecasts at aggregated level
    3. Disaggregates back to original frequency

    Parameters
    ----------
    aggregation_level : int
        Number of periods to aggregate. Higher = more smoothing.

    Examples
    --------
    >>> model = ADIDAForecaster(aggregation_level=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    References
    ----------
    - Nikolopoulos, K., et al. (2011). "An aggregate-disaggregate
      intermittent demand approach (ADIDA) to forecasting"
    """

    def __init__(self, aggregation_level: int = 2, **kwargs: Any) -> None:
        sf_model = ADIDA(aggregation_level=aggregation_level)
        super().__init__(sf_model, aggregation_level=aggregation_level, **kwargs)


class IMAPAForecaster(StatsForecastWrapper):
    """Intermittent Multiple Aggregation Prediction Algorithm (IMAPA).

    Extension of ADIDA that uses multiple aggregation levels and
    combines forecasts from each level.

    More robust than ADIDA as it doesn't require choosing a single
    aggregation level.

    Examples
    --------
    >>> model = IMAPAForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    References
    ----------
    - Petropoulos, F., & Kourentzes, N. (2015). "Forecast combinations
      for intermittent demand"
    """

    def __init__(self, **kwargs: Any) -> None:
        sf_model = IMAPA()
        super().__init__(sf_model, **kwargs)
