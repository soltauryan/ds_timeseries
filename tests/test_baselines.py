"""Tests for baseline forecasting models."""

import pandas as pd
import pytest

from ds_timeseries.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2023-01-02", periods=10, freq="W-MON")
    return pd.DataFrame({
        "unique_id": ["A"] * 10 + ["B"] * 10,
        "ds": list(dates) * 2,
        "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] +  # Series A: trending up
             [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],  # Series B: flat
    })


class TestNaiveForecaster:
    """Tests for NaiveForecaster."""

    def test_predicts_last_value(self, sample_data):
        """Naive should predict the last observed value."""
        model = NaiveForecaster()
        model.fit(sample_data)
        forecasts = model.predict(horizon=3)

        # Series A last value was 100
        a_forecasts = forecasts[forecasts["unique_id"] == "A"]
        assert all(a_forecasts["yhat"] == 100)

        # Series B last value was 100
        b_forecasts = forecasts[forecasts["unique_id"] == "B"]
        assert all(b_forecasts["yhat"] == 100)

    def test_forecast_dates_are_future(self, sample_data):
        """Forecast dates should be after training data."""
        model = NaiveForecaster()
        model.fit(sample_data)
        forecasts = model.predict(horizon=3)

        last_train_date = sample_data["ds"].max()
        assert all(forecasts["ds"] > last_train_date)

    def test_forecast_horizon(self, sample_data):
        """Should produce correct number of forecasts per series."""
        model = NaiveForecaster()
        model.fit(sample_data)
        forecasts = model.predict(horizon=5)

        for uid in ["A", "B"]:
            uid_forecasts = forecasts[forecasts["unique_id"] == uid]
            assert len(uid_forecasts) == 5


class TestMovingAverageForecaster:
    """Tests for MovingAverageForecaster."""

    def test_moving_average_calculation(self, sample_data):
        """MA should predict mean of last window values."""
        model = MovingAverageForecaster(window=4)
        model.fit(sample_data)
        forecasts = model.predict(horizon=1)

        # Series A: last 4 values are 70, 80, 90, 100 -> mean = 85
        a_forecasts = forecasts[forecasts["unique_id"] == "A"]
        assert a_forecasts["yhat"].iloc[0] == pytest.approx(85.0)

        # Series B: all values are 100 -> mean = 100
        b_forecasts = forecasts[forecasts["unique_id"] == "B"]
        assert b_forecasts["yhat"].iloc[0] == pytest.approx(100.0)


class TestSeasonalNaiveForecaster:
    """Tests for SeasonalNaiveForecaster."""

    def test_requires_fit_before_predict(self, sample_data):
        """Should raise error if predict called before fit."""
        model = SeasonalNaiveForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(horizon=1)

    def test_seasonal_cycle(self):
        """Should cycle through seasonal values."""
        # Create data with clear seasonal pattern
        dates = pd.date_range(start="2023-01-02", periods=4, freq="W-MON")
        df = pd.DataFrame({
            "unique_id": ["A"] * 4,
            "ds": dates,
            "y": [10, 20, 30, 40],  # Distinct values
        })

        model = SeasonalNaiveForecaster(season_length=4)
        model.fit(df)
        forecasts = model.predict(horizon=6)

        # Should cycle: 10, 20, 30, 40, 10, 20
        expected = [10, 20, 30, 40, 10, 20]
        assert list(forecasts["yhat"]) == expected
