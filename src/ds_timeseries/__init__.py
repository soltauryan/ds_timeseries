"""DS Time Series - Hierarchical Time Series Forecasting Library."""

__version__ = "0.1.0"

from ds_timeseries.data.toy_datasets import download_toy_dataset, list_toy_datasets
from ds_timeseries.evaluation.metrics import mae, wape

__all__ = [
    "download_toy_dataset",
    "list_toy_datasets",
    "wape",
    "mae",
]
