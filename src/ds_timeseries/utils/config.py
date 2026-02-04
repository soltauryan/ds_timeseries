"""Global configuration for ds_timeseries.

Centralized configuration values that can be modified at runtime.
"""

# Default frequency for weekly time series
# Common values: "W-MON" (Monday), "W-SUN" (Sunday), "W-SAT" (Saturday)
DEFAULT_FREQ = "W-SAT"

# Season length for weekly data (52 weeks = 1 year)
DEFAULT_SEASON_LENGTH = 52
