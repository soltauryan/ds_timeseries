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

from ds_timeseries.utils.config import DEFAULT_FREQ


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
    freq: str = DEFAULT_FREQ,
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

    # Generate weekly dates aligned to frequency
    # Extract day from freq (e.g., "W-SAT" -> Saturday = 5)
    freq_day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    freq_day = freq_day_map.get(freq.split("-")[-1], 5)  # Default to Saturday

    # Align start date to the frequency day
    days_to_freq_day = (freq_day - start.weekday()) % 7
    start_aligned = start + pd.Timedelta(days=days_to_freq_day)
    weeks = pd.date_range(start=start_aligned, end=end, freq=freq)

    records = []
    weeks_per_month = config.weeks_per_month  # e.g., [5, 4, 4]

    for week_date in weeks:
        # Determine fiscal year
        # If fiscal year starts in November (month 11), then:
        # - Nov 2023 to Oct 2024 is FY2024
        fiscal_year = _get_fiscal_year(week_date, config.fiscal_year_start_month)

        # Get fiscal year start date
        fy_start = _get_fiscal_year_start(fiscal_year, config.fiscal_year_start_month, freq)

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


def _get_fiscal_year_start(
    fiscal_year: int, fy_start_month: int, freq: str = DEFAULT_FREQ
) -> pd.Timestamp:
    """Get the start date of a fiscal year.

    Returns the first week-end date on or after the fiscal year start.
    """
    # FY2024 with Nov start means it starts Nov 2023
    calendar_year = fiscal_year - 1 if fy_start_month > 6 else fiscal_year
    fy_start = pd.Timestamp(year=calendar_year, month=fy_start_month, day=1)

    # Align to frequency day (e.g., Saturday for W-SAT)
    freq_day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    target_day = freq_day_map.get(freq.split("-")[-1], 5)  # Default to Saturday

    days_to_target = (target_day - fy_start.weekday()) % 7
    return fy_start + pd.Timedelta(days=days_to_target)


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
    freq: str = DEFAULT_FREQ,
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
    freq : str
        Week frequency (e.g., "W-SAT", "W-MON").

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

    return generate_fiscal_calendar(start_date, end_date, config, freq)


def add_fiscal_features(
    df: pd.DataFrame,
    fiscal_calendar: pd.DataFrame | None = None,
    config: FiscalCalendarConfig | None = None,
    freq: str = DEFAULT_FREQ,
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
    freq : str
        Week frequency (e.g., "W-SAT", "W-MON").

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
            freq,
        )

    # Drop any pre-existing fiscal columns to avoid _x/_y merge collisions
    fiscal_cols = [c for c in fiscal_calendar.columns if c != "ds"]
    overlap = [c for c in fiscal_cols if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    # Merge on ds
    result = df.merge(fiscal_calendar, on="ds", how="left")

    return result


def rollup_to_fiscal_month(
    df: pd.DataFrame,
    fiscal_config: FiscalCalendarConfig | None = None,
    value_cols: str | list[str] = "yhat",
    freq: str = DEFAULT_FREQ,
    fiscal_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate weekly data to fiscal monthly totals.

    Maps each week to its fiscal month (4 or 5 weeks depending on position
    in the quarter pattern) and sums the specified value columns. Flags
    months where not all expected weeks are present.

    Works for both forecasts (value_col="yhat") and actuals (value_col="y").

    You can either pass a ``fiscal_config`` to auto-generate the calendar, or
    pass your own ``fiscal_calendar`` DataFrame. See the README section
    "Bringing Your Own Fiscal Calendar" for the required schema.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly data with columns: unique_id, ds, and one or more value columns.
    fiscal_config : FiscalCalendarConfig | None
        Fiscal calendar configuration. Defaults to 5-4-4, Nov start.
        Ignored when ``fiscal_calendar`` is provided.
    value_cols : str | list[str]
        Column(s) to aggregate. Default "yhat" for forecasts; use "y" for
        actuals, or ["y", "yhat"] to roll up both together.
    freq : str
        Week frequency (e.g., "W-SAT"). Only used when generating a calendar
        (ignored when ``fiscal_calendar`` is provided).
    fiscal_calendar : pd.DataFrame | None
        Pre-built fiscal calendar with at least columns:
        ``ds``, ``fiscal_year``, ``fiscal_quarter``, ``fiscal_month``.
        If provided, ``weeks_expected`` is computed from the calendar itself
        (count of rows per fiscal_year/fiscal_month) rather than from the
        config pattern, so it works with any custom week counts.

    Returns
    -------
    pd.DataFrame
        Monthly totals with columns:
        - unique_id: str
        - fiscal_year: int
        - fiscal_quarter: int (1-4)
        - fiscal_month: int (1-12)
        - <value_col>: float (summed)
        - weeks_expected: int (4 or 5 per the pattern)
        - weeks_present: int (how many weeks had data)
        - is_complete: bool (weeks_present == weeks_expected)
        - month_start: datetime (first week date in the month)
        - month_end: datetime (last week date in the month)

    Examples
    --------
    >>> # Roll up weekly forecasts to fiscal months
    >>> monthly = rollup_to_fiscal_month(forecasts, value_cols="yhat")
    >>> monthly[monthly["is_complete"]]  # Only complete months

    >>> # Roll up with your own fiscal calendar
    >>> my_cal = pd.read_csv("our_fiscal_calendar.csv", parse_dates=["ds"])
    >>> monthly = rollup_to_fiscal_month(forecasts, fiscal_calendar=my_cal)

    >>> # Roll up actuals and forecasts together for monthly evaluation
    >>> merged = actuals.merge(forecasts, on=["unique_id", "ds"])
    >>> monthly = rollup_to_fiscal_month(merged, value_cols=["y", "yhat"])
    >>> from ds_timeseries.evaluation.metrics import wape
    >>> wape(monthly["y"], monthly["yhat"])  # Monthly-level WAPE
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # Use provided calendar or generate one
    if fiscal_calendar is not None:
        fiscal_cal = fiscal_calendar
        use_external_calendar = True
    else:
        fiscal_config = fiscal_config or FiscalCalendarConfig()
        fiscal_cal = generate_fiscal_calendar(
            df["ds"].min() - pd.Timedelta(days=7),
            df["ds"].max() + pd.Timedelta(days=7),
            fiscal_config,
            freq,
        )
        use_external_calendar = False

    # Select merge columns — use what's available in the calendar
    merge_cols = ["ds", "fiscal_year", "fiscal_quarter", "fiscal_month"]
    available = [c for c in merge_cols if c in fiscal_cal.columns]
    if not {"ds", "fiscal_year", "fiscal_month"}.issubset(fiscal_cal.columns):
        raise ValueError(
            "fiscal_calendar must have at least columns: ds, fiscal_year, "
            "fiscal_month. See README 'Bringing Your Own Fiscal Calendar'."
        )

    # Merge fiscal periods onto data
    df_fiscal = df.merge(fiscal_cal[available], on="ds", how="left")

    # Warn about unmatched weeks
    unmatched = df_fiscal["fiscal_month"].isna().sum()
    if unmatched > 0:
        import warnings
        warnings.warn(
            f"{unmatched} rows could not be mapped to a fiscal month. "
            f"Check that 'ds' dates align to the weekly frequency."
        )
        df_fiscal = df_fiscal.dropna(subset=["fiscal_month"])

    # Aggregate to fiscal month
    group_cols = [c for c in ["unique_id", "fiscal_year", "fiscal_quarter", "fiscal_month"]
                  if c in df_fiscal.columns]
    agg_dict = {col: "sum" for col in value_cols}
    agg_dict["ds"] = ["min", "max", "count"]

    monthly = df_fiscal.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    monthly.columns = [
        f"{a}_{b}" if b and b != "" else a
        for a, b in monthly.columns
    ]
    monthly = monthly.rename(columns={
        "ds_min": "month_start",
        "ds_max": "month_end",
        "ds_count": "weeks_present",
    })
    # Fix value col names (they got _sum suffix)
    for col in value_cols:
        monthly = monthly.rename(columns={f"{col}_sum": col})

    # Compute expected weeks per month
    if use_external_calendar:
        # Derive from the calendar itself: count weeks per fiscal year/month
        expected = (
            fiscal_cal.groupby(["fiscal_year", "fiscal_month"])["ds"]
            .count()
            .reset_index()
            .rename(columns={"ds": "weeks_expected"})
        )
        monthly = monthly.merge(expected, on=["fiscal_year", "fiscal_month"], how="left")
    else:
        weeks_per_month_pattern = fiscal_config.weeks_per_month  # e.g. [5, 4, 4]

        def _expected_weeks(fiscal_month: int) -> int:
            month_in_quarter = ((int(fiscal_month) - 1) % 3)
            return weeks_per_month_pattern[month_in_quarter]

        monthly["weeks_expected"] = monthly["fiscal_month"].apply(_expected_weeks)

    monthly["is_complete"] = monthly["weeks_present"] == monthly["weeks_expected"]

    # Cast fiscal period columns back to int (groupby preserves them but be safe)
    for col in ["fiscal_year", "fiscal_quarter", "fiscal_month"]:
        if col in monthly.columns:
            monthly[col] = monthly[col].astype(int)

    # Order columns cleanly
    ordered = (
        [c for c in ["unique_id", "fiscal_year", "fiscal_quarter", "fiscal_month"]
         if c in monthly.columns]
        + value_cols
        + ["weeks_expected", "weeks_present", "is_complete", "month_start", "month_end"]
    )
    monthly = monthly[ordered].sort_values(
        ["unique_id", "fiscal_year", "fiscal_month"]
    ).reset_index(drop=True)

    return monthly


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
