"""Fiscal Calendar Features.

Generates fiscal calendar data and extracts features for time series forecasting.
Supports custom fiscal year start months and week patterns (4-4-5, 5-4-4, 4-5-4).

Critical for capturing "hockey stick" sales patterns at fiscal period ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np


WeekPattern = Literal["4-4-5", "4-5-4", "5-4-4"]


@dataclass
class FiscalCalendarConfig:
    """Configuration for fiscal calendar generation.

    Attributes
    ----------
    fiscal_year_start_month : int
        Month when fiscal year begins (1=Jan, 11=Nov, etc.).
    week_pattern : WeekPattern
        Pattern of weeks per month within each quarter.
        "5-4-4" means 5 weeks in month 1, 4 in month 2, 4 in month 3.
    """

    fiscal_year_start_month: int = 11  # November
    week_pattern: WeekPattern = "5-4-4"

    def __post_init__(self):
        if not 1 <= self.fiscal_year_start_month <= 12:
            raise ValueError("fiscal_year_start_month must be between 1 and 12")
        if self.week_pattern not in ("4-4-5", "4-5-4", "5-4-4"):
            raise ValueError("week_pattern must be '4-4-5', '4-5-4', or '5-4-4'")

    @property
    def weeks_per_month(self) -> list[int]:
        """Get weeks per month pattern for a quarter."""
        return [int(x) for x in self.week_pattern.split("-")]


def generate_fiscal_calendar(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: FiscalCalendarConfig | None = None,
) -> pd.DataFrame:
    """Generate a fiscal calendar DataFrame.

    Creates a weekly fiscal calendar with all fiscal period assignments.

    Parameters
    ----------
    start_date : str | pd.Timestamp
        Start date for the calendar.
    end_date : str | pd.Timestamp
        End date for the calendar.
    config : FiscalCalendarConfig | None
        Fiscal calendar configuration. Defaults to Nov start, 5-4-4 pattern.

    Returns
    -------
    pd.DataFrame
        Fiscal calendar with columns:
        - ds: datetime (week start, Monday)
        - fiscal_year: int
        - fiscal_quarter: int (1-4)
        - fiscal_month: int (1-12)
        - fiscal_week: int (1-52/53)
        - fiscal_week_in_month: int (1-5)
        - fiscal_week_in_quarter: int (1-13)
        - is_fiscal_month_end: bool
        - is_fiscal_quarter_end: bool
        - is_fiscal_year_end: bool

    Examples
    --------
    >>> config = FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")
    >>> cal = generate_fiscal_calendar("2023-01-01", "2025-12-31", config)
    >>> cal[cal["is_fiscal_month_end"]].head()
    """
    config = config or FiscalCalendarConfig()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate weekly dates (Monday start)
    # Align to Monday
    start_monday = start - pd.Timedelta(days=start.weekday())
    weeks = pd.date_range(start=start_monday, end=end, freq="W-MON")

    records = []
    weeks_per_month = config.weeks_per_month  # e.g., [5, 4, 4]

    for week_date in weeks:
        # Determine fiscal year
        # If fiscal year starts in November (month 11), then:
        # - Nov 2023 to Oct 2024 is FY2024
        fiscal_year = _get_fiscal_year(week_date, config.fiscal_year_start_month)

        # Get fiscal year start date
        fy_start = _get_fiscal_year_start(fiscal_year, config.fiscal_year_start_month)

        # Calculate week number within fiscal year (1-based)
        days_since_fy_start = (week_date - fy_start).days
        fiscal_week = (days_since_fy_start // 7) + 1

        # Handle edge cases (week might be before/after FY bounds)
        if fiscal_week < 1:
            fiscal_week = 52 + fiscal_week  # Previous FY
            fiscal_year -= 1
        elif fiscal_week > 52:
            fiscal_week = fiscal_week - 52
            fiscal_year += 1

        # Determine quarter, month, week-in-month from fiscal_week
        fiscal_quarter, fiscal_month, fiscal_week_in_month, fiscal_week_in_quarter = (
            _week_to_fiscal_periods(fiscal_week, weeks_per_month)
        )

        # Determine period-end flags
        is_fiscal_month_end = _is_month_end(
            fiscal_week_in_month, fiscal_month, weeks_per_month
        )
        is_fiscal_quarter_end = fiscal_week_in_quarter == 13
        is_fiscal_year_end = fiscal_week == 52

        records.append({
            "ds": week_date,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "fiscal_month": fiscal_month,
            "fiscal_week": fiscal_week,
            "fiscal_week_in_month": fiscal_week_in_month,
            "fiscal_week_in_quarter": fiscal_week_in_quarter,
            "is_fiscal_month_end": is_fiscal_month_end,
            "is_fiscal_quarter_end": is_fiscal_quarter_end,
            "is_fiscal_year_end": is_fiscal_year_end,
        })

    return pd.DataFrame(records)


def _get_fiscal_year(date: pd.Timestamp, fy_start_month: int) -> int:
    """Determine fiscal year for a given date.

    Fiscal year is named after the year it ends in.
    E.g., if FY starts Nov 2023, it's FY2024 (ends Oct 2024).
    """
    if date.month >= fy_start_month:
        return date.year + 1
    return date.year


def _get_fiscal_year_start(fiscal_year: int, fy_start_month: int) -> pd.Timestamp:
    """Get the start date of a fiscal year.

    Returns the first Monday on or after the fiscal year start.
    """
    # FY2024 with Nov start means it starts Nov 2023
    calendar_year = fiscal_year - 1 if fy_start_month > 6 else fiscal_year
    fy_start = pd.Timestamp(year=calendar_year, month=fy_start_month, day=1)

    # Align to Monday (fiscal weeks start on Monday)
    days_to_monday = (7 - fy_start.weekday()) % 7
    if days_to_monday == 0 and fy_start.weekday() != 0:
        days_to_monday = 7
    if fy_start.weekday() == 0:
        days_to_monday = 0

    return fy_start + pd.Timedelta(days=days_to_monday)


def _week_to_fiscal_periods(
    fiscal_week: int, weeks_per_month: list[int]
) -> tuple[int, int, int, int]:
    """Convert fiscal week to quarter, month, week-in-month, week-in-quarter.

    Parameters
    ----------
    fiscal_week : int
        Week number in fiscal year (1-52).
    weeks_per_month : list[int]
        Weeks per month pattern, e.g., [5, 4, 4].

    Returns
    -------
    tuple
        (fiscal_quarter, fiscal_month, fiscal_week_in_month, fiscal_week_in_quarter)
    """
    # 13 weeks per quarter
    fiscal_quarter = ((fiscal_week - 1) // 13) + 1
    fiscal_week_in_quarter = ((fiscal_week - 1) % 13) + 1

    # Determine month within quarter
    # weeks_per_month = [5, 4, 4] for 5-4-4 pattern
    cumulative_weeks = 0
    month_in_quarter = 0

    for i, weeks_in_month in enumerate(weeks_per_month):
        if fiscal_week_in_quarter <= cumulative_weeks + weeks_in_month:
            month_in_quarter = i + 1
            fiscal_week_in_month = fiscal_week_in_quarter - cumulative_weeks
            break
        cumulative_weeks += weeks_in_month

    # Convert to fiscal month (1-12)
    fiscal_month = (fiscal_quarter - 1) * 3 + month_in_quarter

    return fiscal_quarter, fiscal_month, fiscal_week_in_month, fiscal_week_in_quarter


def _is_month_end(
    fiscal_week_in_month: int, fiscal_month: int, weeks_per_month: list[int]
) -> bool:
    """Check if this week is the last week of a fiscal month."""
    month_in_quarter = ((fiscal_month - 1) % 3) + 1
    max_weeks_in_month = weeks_per_month[month_in_quarter - 1]
    return fiscal_week_in_month == max_weeks_in_month


def create_mock_fiscal_calendar(
    start_year: int = 2011,
    end_year: int = 2026,
    config: FiscalCalendarConfig | None = None,
) -> pd.DataFrame:
    """Create a mock fiscal calendar for testing.

    Generates a fiscal calendar spanning multiple years with the
    configured pattern.

    Parameters
    ----------
    start_year : int
        Calendar year to start from.
    end_year : int
        Calendar year to end at.
    config : FiscalCalendarConfig | None
        Fiscal calendar config. Defaults to Nov start, 5-4-4.

    Returns
    -------
    pd.DataFrame
        Complete fiscal calendar.

    Examples
    --------
    >>> # Create 5-4-4 calendar starting in November
    >>> config = FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")
    >>> cal = create_mock_fiscal_calendar(2020, 2025, config)
    >>> cal.to_parquet("data/raw/fiscal_calendar.parquet")
    """
    config = config or FiscalCalendarConfig()

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    return generate_fiscal_calendar(start_date, end_date, config)


def add_fiscal_features(
    df: pd.DataFrame,
    fiscal_calendar: pd.DataFrame | None = None,
    config: FiscalCalendarConfig | None = None,
) -> pd.DataFrame:
    """Add fiscal calendar features to a time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'ds' column (datetime).
    fiscal_calendar : pd.DataFrame | None
        Pre-generated fiscal calendar. If None, generates one.
    config : FiscalCalendarConfig | None
        Config for generating calendar if not provided.

    Returns
    -------
    pd.DataFrame
        Original data with fiscal features added.

    Examples
    --------
    >>> df = download_toy_dataset("m5_sample", n_series=100)
    >>> df_with_fiscal = add_fiscal_features(df)
    >>> df_with_fiscal[df_with_fiscal["is_fiscal_month_end"]].head()
    """
    if fiscal_calendar is None:
        config = config or FiscalCalendarConfig()
        fiscal_calendar = generate_fiscal_calendar(
            df["ds"].min() - pd.Timedelta(days=7),
            df["ds"].max() + pd.Timedelta(days=7),
            config,
        )

    # Merge on ds
    result = df.merge(fiscal_calendar, on="ds", how="left")

    return result


def get_fiscal_period_summary(fiscal_calendar: pd.DataFrame) -> pd.DataFrame:
    """Get a summary of fiscal periods for validation.

    Parameters
    ----------
    fiscal_calendar : pd.DataFrame
        Generated fiscal calendar.

    Returns
    -------
    pd.DataFrame
        Summary with weeks per fiscal month/quarter/year.
    """
    summary = (
        fiscal_calendar
        .groupby(["fiscal_year", "fiscal_quarter", "fiscal_month"])
        .agg(
            weeks=("ds", "count"),
            start_date=("ds", "min"),
            end_date=("ds", "max"),
            month_ends=("is_fiscal_month_end", "sum"),
        )
        .reset_index()
    )

    return summary
