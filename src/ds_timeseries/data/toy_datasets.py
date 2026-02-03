"""Toy dataset downloaders for development and testing.

Provides functions to download sample retail/sales datasets that mimic
hierarchical Customer-Material forecasting scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

# Default data directory
DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "raw"

DATASETS = {
    "m5_sample": {
        "description": "M5 Forecasting - Walmart hierarchical sales (via datasetsforecast)",
        "format": "datasetsforecast",
        "hierarchy": ["item_id", "store_id", "cat_id", "dept_id", "state_id"],
        "date_col": "ds",
        "target_col": "y",
    },
    "tourism": {
        "description": "Australian Tourism - Hierarchical visitor nights",
        "url": "https://raw.githubusercontent.com/Nixtla/hierarchicalforecast/main/nbs/data/tourism.parquet",
        "format": "parquet",
        "hierarchy": ["Region", "State", "Purpose"],
        "date_col": "ds",
        "target_col": "y",
    },
}

DatasetName = Literal["m5_sample", "tourism"]


def list_toy_datasets() -> dict[str, str]:
    """List available toy datasets with descriptions.

    Returns
    -------
    dict[str, str]
        Dictionary mapping dataset names to descriptions.

    Examples
    --------
    >>> list_toy_datasets()
    {'store_sales': 'Kaggle Store Sales - Time Series Forecasting (Favorita stores)', ...}
    """
    return {name: info["description"] for name, info in DATASETS.items()}


def download_toy_dataset(
    name: DatasetName,
    data_dir: Path | str | None = None,
    force: bool = False,
    aggregate_weekly: bool = True,
    n_series: int | None = None,
) -> pd.DataFrame:
    """Download a toy dataset for development and testing.

    Parameters
    ----------
    name : str
        Name of the dataset to download. Use `list_toy_datasets()` to see options.
    data_dir : Path | str | None
        Directory to save the data. Defaults to `data/raw/` in project root.
    force : bool
        If True, re-download even if file exists.
    aggregate_weekly : bool
        If True, aggregate daily data to weekly (recommended for most use cases).
    n_series : int | None
        Limit to first N series (useful for quick testing). None = all series.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - unique_id: str - Hierarchical identifier
        - ds: datetime - Date (week start if aggregated)
        - y: float - Target value

    Examples
    --------
    >>> df = download_toy_dataset("m5_sample", n_series=100)
    >>> df.head()
       unique_id         ds       y
    0  FOODS_1_001_CA_1 2011-01-29  3.0
    1  FOODS_1_001_CA_1 2011-02-05  0.0
    ...
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    config = DATASETS[name]
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_n{n_series}" if n_series else ""
    cache_file = data_dir / f"{name}{suffix}.parquet"

    if cache_file.exists() and not force:
        print(f"Loading cached dataset from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Downloading {name}: {config['description']}")
    df = _download_and_process(name, config, n_series=n_series)

    if aggregate_weekly and name == "m5_sample":
        df = _aggregate_to_weekly(df)

    # Cache processed data
    df.to_parquet(cache_file, index=False)
    print(f"Saved to {cache_file}")

    return df


def _download_and_process(name: str, config: dict, n_series: int | None = None) -> pd.DataFrame:
    """Download and standardize dataset format."""
    fmt = config["format"]

    if fmt == "datasetsforecast":
        df = _download_from_datasetsforecast(name, n_series=n_series)
    elif fmt == "parquet":
        url = config["url"]
        df = pd.read_parquet(url)
        df = _standardize_format(df, name, config)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return df


def _download_from_datasetsforecast(name: str, n_series: int | None = None) -> pd.DataFrame:
    """Download data using the datasetsforecast library."""
    if name == "m5_sample":
        from datasetsforecast.m5 import M5

        # Download M5 data (this caches automatically)
        Y_df, X_df, S_df = M5.load(directory=str(DATA_DIR))

        # Merge with static features to get hierarchy columns
        df = Y_df.merge(S_df, on="unique_id", how="left")

        # Limit series if requested
        if n_series is not None:
            series_ids = df["unique_id"].unique()[:n_series]
            df = df[df["unique_id"].isin(series_ids)]

        # Ensure proper types
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        # Reorder columns
        cols = ["unique_id", "ds", "y"]
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            if col in df.columns:
                cols.append(col)
        df = df[cols]

        return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    else:
        raise ValueError(f"Dataset {name} not supported via datasetsforecast")


def _standardize_format(df: pd.DataFrame, name: str, config: dict) -> pd.DataFrame:
    """Convert dataset to standard format with unique_id, ds, y columns."""
    date_col = config["date_col"]
    target_col = config["target_col"]

    if name == "tourism":
        # Already in hierarchical format
        if "unique_id" not in df.columns:
            df["unique_id"] = (
                df["Region"].astype(str) + "_" + df["State"].astype(str) + "_" + df["Purpose"].astype(str)
            )

        df["ds"] = pd.to_datetime(df[date_col])
        df["y"] = df[target_col].astype(float)

        keep_cols = ["unique_id", "ds", "y"]
        for col in ["Region", "State", "Purpose"]:
            if col in df.columns:
                keep_cols.append(col)

        result = df[keep_cols].copy()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Sort by unique_id and date
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return result


def _aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to weekly frequency.

    Uses week starting Monday (ISO week).
    """
    df = df.copy()

    # Get non-time columns for grouping (hierarchy columns)
    id_cols = [c for c in df.columns if c not in ["ds", "y"]]

    # Aggregate to weekly
    df["ds"] = df["ds"].dt.to_period("W-SUN").dt.start_time

    agg_dict = {"y": "sum"}
    for col in id_cols:
        if col != "unique_id":
            agg_dict[col] = "first"

    result = df.groupby(["unique_id", "ds"]).agg(agg_dict).reset_index()

    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def create_synthetic_customer_material(
    n_customers: int = 50,
    n_materials: int = 100,
    n_weeks: int = 104,
    fiscal_pattern: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic Customer-Material sales data for testing.

    Generates realistic hierarchical sales data with:
    - Trend, seasonality, and noise
    - Fiscal month-end "hockey stick" patterns (optional)
    - Intermittent demand (zeros)

    Parameters
    ----------
    n_customers : int
        Number of unique customers.
    n_materials : int
        Number of unique materials.
    n_weeks : int
        Number of weeks of history.
    fiscal_pattern : bool
        If True, add month-end sales spikes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic sales data with unique_id, ds, y, customer_id, material_id.
    """
    import numpy as np

    np.random.seed(seed)

    # Create all customer-material combinations (sparse - not all combos exist)
    n_combos = int(n_customers * n_materials * 0.3)  # 30% sparsity

    customers = np.random.randint(1, n_customers + 1, size=n_combos)
    materials = np.random.randint(1, n_materials + 1, size=n_combos)

    # Remove duplicates
    combos = list(set(zip(customers, materials)))
    n_combos = len(combos)

    # Generate date range
    start_date = pd.Timestamp("2023-01-02")  # Monday
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

    records = []

    for cust, mat in combos:
        # Base demand level (varies by customer-material)
        base_demand = np.random.exponential(scale=50)

        # Some combinations have intermittent demand
        is_intermittent = np.random.random() < 0.3

        for i, date in enumerate(dates):
            # Trend component
            trend = 1 + 0.001 * i

            # Yearly seasonality
            week_of_year = date.isocalendar()[1]
            seasonality = 1 + 0.2 * np.sin(2 * np.pi * week_of_year / 52)

            # Fiscal month-end spike (hockey stick)
            fiscal_multiplier = 1.0
            if fiscal_pattern:
                # Assume 4-4-5 pattern: weeks 4, 8, 13 of each quarter are month-ends
                week_in_quarter = (week_of_year - 1) % 13 + 1
                if week_in_quarter in [4, 8, 13]:
                    fiscal_multiplier = 1.5 + np.random.random() * 0.5

            # Calculate demand
            demand = base_demand * trend * seasonality * fiscal_multiplier

            # Add noise
            demand *= np.random.lognormal(0, 0.3)

            # Intermittent demand: sometimes zero
            if is_intermittent and np.random.random() < 0.4:
                demand = 0

            records.append(
                {
                    "customer_id": f"CUST{cust:04d}",
                    "material_id": f"MAT{mat:05d}",
                    "ds": date,
                    "y": max(0, round(demand, 2)),
                }
            )

    df = pd.DataFrame(records)
    df["unique_id"] = df["customer_id"] + "_" + df["material_id"]

    return df[["unique_id", "ds", "y", "customer_id", "material_id"]].sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)
