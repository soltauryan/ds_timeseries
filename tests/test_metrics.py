"""Tests for evaluation metrics."""

import numpy as np
import pytest

from ds_timeseries.evaluation.metrics import bias, mae, rmse, wape


class TestWAPE:
    """Tests for WAPE metric."""

    def test_perfect_forecast(self):
        """WAPE should be 0 for perfect forecast."""
        y_true = [100, 200, 150]
        y_pred = [100, 200, 150]
        assert wape(y_true, y_pred) == 0.0

    def test_simple_case(self):
        """Test WAPE with known values."""
        y_true = [100, 200, 100]  # sum = 400
        y_pred = [110, 190, 100]  # errors = 10, 10, 0 = 20
        expected = 20 / 400  # 0.05
        assert wape(y_true, y_pred) == pytest.approx(expected)

    def test_handles_zeros_in_actuals(self):
        """WAPE should handle zeros in actuals gracefully."""
        y_true = [100, 0, 100]  # sum = 200
        y_pred = [90, 10, 110]  # errors = 10, 10, 10 = 30
        expected = 30 / 200  # 0.15
        assert wape(y_true, y_pred) == pytest.approx(expected)

    def test_all_zeros_returns_inf(self):
        """WAPE should return inf when all actuals are zero."""
        y_true = [0, 0, 0]
        y_pred = [10, 10, 10]
        assert wape(y_true, y_pred) == float("inf")

    def test_numpy_arrays(self):
        """WAPE should work with numpy arrays."""
        y_true = np.array([100, 200, 150])
        y_pred = np.array([100, 200, 150])
        assert wape(y_true, y_pred) == 0.0


class TestMAE:
    """Tests for MAE metric."""

    def test_perfect_forecast(self):
        """MAE should be 0 for perfect forecast."""
        y_true = [100, 200, 150]
        y_pred = [100, 200, 150]
        assert mae(y_true, y_pred) == 0.0

    def test_simple_case(self):
        """Test MAE with known values."""
        y_true = [100, 200, 150]
        y_pred = [90, 210, 140]  # errors = 10, 10, 10
        assert mae(y_true, y_pred) == 10.0

    def test_asymmetric_errors(self):
        """MAE should average absolute errors regardless of sign."""
        y_true = [100, 100]
        y_pred = [110, 80]  # errors = -10, +20
        assert mae(y_true, y_pred) == 15.0  # (10 + 20) / 2


class TestRMSE:
    """Tests for RMSE metric."""

    def test_perfect_forecast(self):
        """RMSE should be 0 for perfect forecast."""
        y_true = [100, 200, 150]
        y_pred = [100, 200, 150]
        assert rmse(y_true, y_pred) == 0.0

    def test_simple_case(self):
        """Test RMSE with known values."""
        y_true = [100, 100]
        y_pred = [110, 90]  # squared errors = 100, 100
        expected = np.sqrt(100)  # 10.0
        assert rmse(y_true, y_pred) == pytest.approx(expected)

    def test_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        y_true = [100, 100, 100, 100]
        y_pred = [100, 100, 100, 140]  # one large error of 40

        # MAE = 10 (40/4)
        # RMSE = sqrt(1600/4) = 20
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)


class TestBias:
    """Tests for Bias metric."""

    def test_unbiased(self):
        """Bias should be 0 for unbiased forecast."""
        y_true = [100, 200]
        y_pred = [110, 190]  # errors cancel out
        assert bias(y_true, y_pred) == 0.0

    def test_positive_bias(self):
        """Positive bias indicates over-forecasting."""
        y_true = [100, 100]
        y_pred = [110, 120]  # consistently high
        assert bias(y_true, y_pred) == 15.0  # mean of 10, 20

    def test_negative_bias(self):
        """Negative bias indicates under-forecasting."""
        y_true = [100, 100]
        y_pred = [90, 80]  # consistently low
        assert bias(y_true, y_pred) == -15.0  # mean of -10, -20
