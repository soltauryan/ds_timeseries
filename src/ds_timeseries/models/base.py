"""Base forecaster interface.

All models should inherit from BaseForecaster to ensure consistent API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models.

    All forecasters must implement fit, predict, and optionally fit_predict.
    The data format follows the Nixtla convention:
    - unique_id: str - Time series identifier
    - ds: datetime - Timestamp
    - y: float - Target value

    Examples
    --------
    >>> class MyModel(BaseForecaster):
    ...     def fit(self, df):
    ...         self.last_value_ = df.groupby('unique_id')['y'].last()
    ...         return self
    ...     def predict(self, horizon):
    ...         return pd.DataFrame(...)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the forecaster with optional parameters."""
        self._is_fitted = False
        self._params = kwargs

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        """Fit the model to training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with columns: unique_id, ds, y.
            May contain additional columns for hierarchy or features.

        Returns
        -------
        BaseForecaster
            The fitted model (self).
        """
        pass

    @abstractmethod
    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts for future periods.

        Parameters
        ----------
        horizon : int
            Number of periods to forecast.
        **kwargs
            Additional arguments (e.g., future exogenous features).

        Returns
        -------
        pd.DataFrame
            Forecasts with columns: unique_id, ds, yhat.
            May include prediction intervals: yhat_lower, yhat_upper.
        """
        pass

    def fit_predict(self, df: pd.DataFrame, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Fit the model and generate forecasts in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        horizon : int
            Number of periods to forecast.

        Returns
        -------
        pd.DataFrame
            Forecasts.
        """
        return self.fit(df).predict(horizon, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        required = {"unique_id", "ds", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            raise TypeError("Column 'ds' must be datetime type")

    def __repr__(self) -> str:
        """String representation of the model."""
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        fitted_str = " (fitted)" if self._is_fitted else ""
        return f"{self.__class__.__name__}({params_str}){fitted_str}"
