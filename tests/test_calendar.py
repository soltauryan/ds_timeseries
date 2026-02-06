"""Tests for fiscal calendar features."""

import pandas as pd
import pytest

from ds_timeseries.features.calendar import (
    FiscalCalendarConfig,
    generate_fiscal_calendar,
    create_mock_fiscal_calendar,
    add_fiscal_features,
    get_fiscal_period_summary,
    rollup_to_fiscal_month,
)


class TestFiscalCalendarConfig:
    """Tests for FiscalCalendarConfig."""

    def test_default_config(self):
        """Default config should be Nov start, 5-4-4."""
        config = FiscalCalendarConfig()
        assert config.fiscal_year_start_month == 11
        assert config.week_pattern == "5-4-4"
        assert config.weeks_per_month == [5, 4, 4]

    def test_custom_config(self):
        """Custom configs should work."""
        config = FiscalCalendarConfig(fiscal_year_start_month=2, week_pattern="4-4-5")
        assert config.fiscal_year_start_month == 2
        assert config.weeks_per_month == [4, 4, 5]

    def test_invalid_month(self):
        """Should reject invalid month."""
        with pytest.raises(ValueError, match="fiscal_year_start_month"):
            FiscalCalendarConfig(fiscal_year_start_month=13)

    def test_invalid_pattern(self):
        """Should reject invalid week pattern."""
        with pytest.raises(ValueError, match="week_pattern"):
            FiscalCalendarConfig(week_pattern="3-3-7")


class TestGenerateFiscalCalendar:
    """Tests for generate_fiscal_calendar."""

    @pytest.fixture
    def config_544(self):
        """5-4-4 config starting in November."""
        return FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")

    def test_generates_weekly_data(self, config_544):
        """Should generate weekly data points."""
        cal = generate_fiscal_calendar("2024-01-01", "2024-03-31", config_544)

        assert "ds" in cal.columns
        assert len(cal) > 0

        # Check weekly frequency
        date_diffs = cal["ds"].diff().dropna()
        assert all(date_diffs == pd.Timedelta(days=7))

    def test_has_all_columns(self, config_544):
        """Should have all expected columns."""
        cal = generate_fiscal_calendar("2024-01-01", "2024-12-31", config_544)

        expected_cols = [
            "ds", "fiscal_year", "fiscal_quarter", "fiscal_month",
            "fiscal_week", "fiscal_week_in_month", "fiscal_week_in_quarter",
            "is_fiscal_month_end", "is_fiscal_quarter_end", "is_fiscal_year_end",
        ]
        for col in expected_cols:
            assert col in cal.columns, f"Missing column: {col}"

    def test_544_pattern_weeks_per_quarter(self, config_544):
        """5-4-4 pattern should have 13 weeks per quarter."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-10-31", config_544)

        # Filter to FY2024
        fy2024 = cal[cal["fiscal_year"] == 2024]

        # Each quarter should have 13 weeks
        for q in [1, 2, 3, 4]:
            q_weeks = fy2024[fy2024["fiscal_quarter"] == q]
            assert len(q_weeks) == 13, f"Quarter {q} should have 13 weeks, got {len(q_weeks)}"

    def test_544_pattern_weeks_per_month(self, config_544):
        """5-4-4 pattern: months should have 5, 4, 4 weeks."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-10-31", config_544)
        fy2024 = cal[cal["fiscal_year"] == 2024]

        # Check first quarter (months 1, 2, 3)
        for month, expected_weeks in [(1, 5), (2, 4), (3, 4)]:
            month_weeks = fy2024[fy2024["fiscal_month"] == month]
            assert len(month_weeks) == expected_weeks, \
                f"Month {month} should have {expected_weeks} weeks, got {len(month_weeks)}"

    def test_month_end_flags(self, config_544):
        """Month end flags should be set correctly."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-10-31", config_544)
        fy2024 = cal[cal["fiscal_year"] == 2024]

        # Should have 12 month ends per year
        month_ends = fy2024["is_fiscal_month_end"].sum()
        assert month_ends == 12, f"Should have 12 month ends, got {month_ends}"

    def test_quarter_end_flags(self, config_544):
        """Quarter end flags should be set correctly."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-10-31", config_544)
        fy2024 = cal[cal["fiscal_year"] == 2024]

        # Should have 4 quarter ends per year
        quarter_ends = fy2024["is_fiscal_quarter_end"].sum()
        assert quarter_ends == 4, f"Should have 4 quarter ends, got {quarter_ends}"

    def test_year_end_flag(self, config_544):
        """Year end flag should be on week 52."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-10-31", config_544)
        fy2024 = cal[cal["fiscal_year"] == 2024]

        year_end_row = fy2024[fy2024["is_fiscal_year_end"]]
        assert len(year_end_row) == 1
        assert year_end_row.iloc[0]["fiscal_week"] == 52

    def test_fiscal_year_assignment_november_start(self, config_544):
        """Fiscal year should be named after ending year."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-02-28", config_544)

        # November 2023 should be FY2024 (ends Oct 2024)
        nov_2023 = cal[(cal["ds"] >= "2023-11-01") & (cal["ds"] < "2023-12-01")]
        assert all(nov_2023["fiscal_year"] == 2024)

        # January 2024 should also be FY2024
        jan_2024 = cal[(cal["ds"] >= "2024-01-01") & (cal["ds"] < "2024-02-01")]
        assert all(jan_2024["fiscal_year"] == 2024)


class TestMockFiscalCalendar:
    """Tests for create_mock_fiscal_calendar."""

    def test_creates_multi_year_calendar(self):
        """Should create calendar spanning multiple years."""
        cal = create_mock_fiscal_calendar(2020, 2025)

        # Dates align to Monday, so may start slightly before requested year
        assert cal["ds"].min().year >= 2019  # Dec 30 2019 is Monday for Jan 1 2020
        assert cal["ds"].max().year == 2025
        assert len(cal) > 250  # Multiple years of weekly data

    def test_uses_custom_config(self):
        """Should use provided config."""
        config = FiscalCalendarConfig(fiscal_year_start_month=7, week_pattern="4-5-4")
        cal = create_mock_fiscal_calendar(2023, 2024, config)

        # Check 4-5-4 pattern for a quarter
        fy = cal[cal["fiscal_year"] == 2024].copy()
        q1 = fy[fy["fiscal_quarter"] == 1]

        m1_weeks = len(q1[q1["fiscal_month"] == 1])
        m2_weeks = len(q1[q1["fiscal_month"] == 2])
        m3_weeks = len(q1[q1["fiscal_month"] == 3])

        assert m1_weeks == 4, f"Month 1 should have 4 weeks in 4-5-4, got {m1_weeks}"
        assert m2_weeks == 5, f"Month 2 should have 5 weeks in 4-5-4, got {m2_weeks}"
        assert m3_weeks == 4, f"Month 3 should have 4 weeks in 4-5-4, got {m3_weeks}"


class TestAddFiscalFeatures:
    """Tests for add_fiscal_features."""

    def test_adds_features_to_dataframe(self):
        """Should merge fiscal features into data."""
        # Create sample data
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "y": [10, 20, 30, 40, 50],
        })

        result = add_fiscal_features(df)

        assert "fiscal_year" in result.columns
        assert "is_fiscal_month_end" in result.columns
        assert len(result) == len(df)

    def test_preserves_original_columns(self):
        """Should keep all original columns."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3, freq="W-MON"),
            "y": [10, 20, 30],
            "custom_col": ["x", "y", "z"],
        })

        result = add_fiscal_features(df)

        assert "unique_id" in result.columns
        assert "y" in result.columns
        assert "custom_col" in result.columns


class TestFiscalPeriodSummary:
    """Tests for get_fiscal_period_summary."""

    def test_summary_structure(self):
        """Summary should have expected structure."""
        config = FiscalCalendarConfig()
        cal = create_mock_fiscal_calendar(2024, 2024, config)

        summary = get_fiscal_period_summary(cal)

        assert "fiscal_year" in summary.columns
        assert "fiscal_quarter" in summary.columns
        assert "fiscal_month" in summary.columns
        assert "weeks" in summary.columns
        assert "month_ends" in summary.columns


class TestRollupToFiscalMonth:
    """Tests for rollup_to_fiscal_month."""

    @pytest.fixture
    def config_544(self):
        return FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")

    @pytest.fixture
    def weekly_data(self, config_544):
        """Create 13 weeks of data (one full quarter) for two series."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-02-28", config_544, freq="W-SAT")
        fy2024 = cal[cal["fiscal_year"] == 2024].head(13)  # Q1: 13 weeks

        rows = []
        for uid in ["A", "B"]:
            for _, row in fy2024.iterrows():
                rows.append({"unique_id": uid, "ds": row["ds"], "y": 10.0})
        return pd.DataFrame(rows)

    def test_basic_rollup(self, weekly_data, config_544):
        """Should sum weekly values into fiscal months."""
        monthly = rollup_to_fiscal_month(weekly_data, config_544, value_cols="y")

        # 2 series * 3 months (5-4-4 quarter) = 6 rows
        assert len(monthly) == 6

        # 5-week month should sum to 50, 4-week months to 40
        series_a = monthly[monthly["unique_id"] == "A"]
        assert series_a.iloc[0]["y"] == 50.0   # 5 weeks * 10
        assert series_a.iloc[1]["y"] == 40.0   # 4 weeks * 10
        assert series_a.iloc[2]["y"] == 40.0   # 4 weeks * 10

    def test_complete_flag(self, weekly_data, config_544):
        """All months should be flagged as complete when data covers full month."""
        monthly = rollup_to_fiscal_month(weekly_data, config_544, value_cols="y")
        assert monthly["is_complete"].all()

    def test_partial_month_detected(self, config_544):
        """Should flag incomplete months when not all weeks are present."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-02-28", config_544, freq="W-SAT")
        fy2024 = cal[cal["fiscal_year"] == 2024].head(13)

        # Take only first 3 weeks of month 1 (which expects 5)
        partial = fy2024.head(3)[["ds"]].copy()
        partial["unique_id"] = "A"
        partial["yhat"] = 10.0

        monthly = rollup_to_fiscal_month(partial, config_544, value_cols="yhat")
        row = monthly.iloc[0]
        assert row["weeks_expected"] == 5
        assert row["weeks_present"] == 3
        assert not row["is_complete"]

    def test_multiple_value_cols(self, config_544):
        """Should aggregate multiple value columns."""
        cal = generate_fiscal_calendar("2023-11-01", "2024-02-28", config_544, freq="W-SAT")
        fy2024 = cal[cal["fiscal_year"] == 2024].head(13)

        df = fy2024[["ds"]].copy()
        df["unique_id"] = "A"
        df["y"] = 10.0
        df["yhat"] = 12.0

        monthly = rollup_to_fiscal_month(df, config_544, value_cols=["y", "yhat"])

        assert "y" in monthly.columns
        assert "yhat" in monthly.columns
        # First month (5 weeks): y=50, yhat=60
        assert monthly.iloc[0]["y"] == 50.0
        assert monthly.iloc[0]["yhat"] == 60.0

    def test_weeks_expected_matches_pattern(self, weekly_data, config_544):
        """weeks_expected should reflect the 5-4-4 pattern."""
        monthly = rollup_to_fiscal_month(weekly_data, config_544, value_cols="y")
        series_a = monthly[monthly["unique_id"] == "A"]

        # 5-4-4 pattern: month 1=5, month 2=4, month 3=4
        assert series_a.iloc[0]["weeks_expected"] == 5
        assert series_a.iloc[1]["weeks_expected"] == 4
        assert series_a.iloc[2]["weeks_expected"] == 4

    def test_output_has_expected_columns(self, weekly_data, config_544):
        """Should have all documented output columns."""
        monthly = rollup_to_fiscal_month(weekly_data, config_544, value_cols="y")
        expected = [
            "unique_id", "fiscal_year", "fiscal_quarter", "fiscal_month",
            "y", "weeks_expected", "weeks_present", "is_complete",
            "month_start", "month_end",
        ]
        for col in expected:
            assert col in monthly.columns, f"Missing column: {col}"

    def test_445_pattern(self):
        """Should work with 4-4-5 pattern."""
        config = FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="4-4-5")
        cal = generate_fiscal_calendar("2023-11-01", "2024-02-28", config, freq="W-SAT")
        fy2024 = cal[cal["fiscal_year"] == 2024].head(13)

        df = fy2024[["ds"]].copy()
        df["unique_id"] = "A"
        df["y"] = 10.0

        monthly = rollup_to_fiscal_month(df, config, value_cols="y")

        # 4-4-5 pattern
        assert monthly.iloc[0]["weeks_expected"] == 4
        assert monthly.iloc[1]["weeks_expected"] == 4
        assert monthly.iloc[2]["weeks_expected"] == 5

    def test_external_fiscal_calendar(self):
        """Should accept a user-provided fiscal calendar instead of generating one."""
        # Build a custom calendar (e.g., from a company's own fiscal system)
        my_cal = pd.DataFrame({
            "ds": pd.date_range("2024-01-06", periods=9, freq="W-SAT"),
            "fiscal_year": [2024] * 9,
            "fiscal_quarter": [1] * 9,
            "fiscal_month": [1, 1, 1, 1, 1, 2, 2, 2, 2],  # 5-4 split
        })

        df = my_cal[["ds"]].copy()
        df["unique_id"] = "X"
        df["yhat"] = 5.0

        monthly = rollup_to_fiscal_month(df, fiscal_calendar=my_cal)

        assert len(monthly) == 2
        # weeks_expected derived from the calendar: month 1 has 5 weeks, month 2 has 4
        assert monthly.iloc[0]["weeks_expected"] == 5
        assert monthly.iloc[0]["yhat"] == 25.0
        assert monthly.iloc[1]["weeks_expected"] == 4
        assert monthly.iloc[1]["yhat"] == 20.0
        assert monthly["is_complete"].all()
