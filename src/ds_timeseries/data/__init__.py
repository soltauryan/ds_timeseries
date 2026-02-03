"""Data loading and preprocessing modules."""

from ds_timeseries.data.toy_datasets import download_toy_dataset, list_toy_datasets
from ds_timeseries.data.validation import (
    classify_demand,
    detect_data_leakage_risk,
    recommend_models,
    validate_data,
    DataValidationResult,
    DemandClassification,
)

__all__ = [
    # Data loading
    "download_toy_dataset",
    "list_toy_datasets",
    # Validation & classification
    "validate_data",
    "classify_demand",
    "recommend_models",
    "detect_data_leakage_risk",
    "DataValidationResult",
    "DemandClassification",
]
