"""Hierarchical Structure Definition.

Defines the hierarchy for reconciliation. Maps bottom-level series
to aggregated levels (e.g., item → category → total).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HierarchySpec:
    """Specification for a hierarchical time series structure.

    Defines how bottom-level series aggregate to higher levels.

    Parameters
    ----------
    levels : list[str]
        Column names defining hierarchy from bottom to top.
        E.g., ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    bottom_level : str
        Column name for the most granular level (typically "unique_id").

    Examples
    --------
    >>> spec = HierarchySpec(
    ...     levels=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    ...     bottom_level="unique_id",
    ... )
    """

    levels: list[str]
    bottom_level: str = "unique_id"
    _aggregation_matrix: np.ndarray | None = field(default=None, repr=False)
    _level_mapping: dict[str, list[str]] | None = field(default=None, repr=False)

    def build_aggregation_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build the summing matrix S for hierarchical aggregation.

        S maps bottom-level forecasts to all hierarchical levels.
        y_all = S @ y_bottom

        Parameters
        ----------
        df : pd.DataFrame
            Data with hierarchy columns.

        Returns
        -------
        np.ndarray
            Summing matrix of shape (n_all_series, n_bottom_series).
        """
        # Get unique bottom-level series
        bottom_series = df[self.bottom_level].unique()
        n_bottom = len(bottom_series)
        bottom_to_idx = {s: i for i, s in enumerate(bottom_series)}

        # Build aggregation levels
        all_rows = []
        self._level_mapping = {}

        # 1. Total level (sum of everything)
        total_row = np.ones(n_bottom)
        all_rows.append(total_row)
        self._level_mapping["Total"] = ["Total"]

        # 2. Each hierarchy level (progressively more granular)
        for level in reversed(self.levels):
            if level not in df.columns:
                continue

            level_values = df[level].unique()
            self._level_mapping[level] = list(level_values)

            for val in level_values:
                # Find which bottom series belong to this aggregate
                mask = df[df[level] == val][self.bottom_level].unique()
                row = np.zeros(n_bottom)
                for s in mask:
                    if s in bottom_to_idx:
                        row[bottom_to_idx[s]] = 1
                all_rows.append(row)

        # 3. Bottom level (identity for each series)
        for s in bottom_series:
            row = np.zeros(n_bottom)
            row[bottom_to_idx[s]] = 1
            all_rows.append(row)

        self._aggregation_matrix = np.array(all_rows)
        return self._aggregation_matrix

    def get_summing_matrix(self) -> np.ndarray:
        """Get the summing matrix S."""
        if self._aggregation_matrix is None:
            raise RuntimeError("Call build_aggregation_matrix first")
        return self._aggregation_matrix

    def aggregate_forecasts(
        self,
        bottom_forecasts: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate bottom-level forecasts to all hierarchy levels.

        Parameters
        ----------
        bottom_forecasts : pd.DataFrame
            Forecasts for bottom-level series with unique_id, ds, yhat.
        df : pd.DataFrame
            Original data with hierarchy columns.

        Returns
        -------
        pd.DataFrame
            Forecasts for all hierarchy levels.
        """
        if self._aggregation_matrix is None:
            self.build_aggregation_matrix(df)

        # Pivot forecasts to matrix form
        forecast_pivot = bottom_forecasts.pivot(
            index="ds", columns="unique_id", values="yhat"
        ).fillna(0)

        # Get series order matching the aggregation matrix
        bottom_series = df[self.bottom_level].unique()
        forecast_matrix = forecast_pivot[bottom_series].values

        # Aggregate: all_forecasts = S @ bottom_forecasts
        all_forecasts = forecast_matrix @ self._aggregation_matrix.T

        # Convert back to DataFrame
        results = []
        col_idx = 0

        # Total
        for ds_idx, ds in enumerate(forecast_pivot.index):
            results.append({
                "unique_id": "Total",
                "ds": ds,
                "yhat": all_forecasts[ds_idx, col_idx],
                "level": "Total",
            })
        col_idx += 1

        # Each level
        for level in reversed(self.levels):
            if level not in self._level_mapping:
                continue
            for val in self._level_mapping[level]:
                for ds_idx, ds in enumerate(forecast_pivot.index):
                    results.append({
                        "unique_id": f"{level}_{val}",
                        "ds": ds,
                        "yhat": all_forecasts[ds_idx, col_idx],
                        "level": level,
                    })
                col_idx += 1

        # Bottom level
        for s in bottom_series:
            for ds_idx, ds in enumerate(forecast_pivot.index):
                results.append({
                    "unique_id": s,
                    "ds": ds,
                    "yhat": all_forecasts[ds_idx, col_idx],
                    "level": self.bottom_level,
                })
            col_idx += 1

        return pd.DataFrame(results)


def create_hierarchy_from_data(
    df: pd.DataFrame,
    hierarchy_cols: list[str] | None = None,
) -> HierarchySpec:
    """Auto-detect hierarchy structure from data.

    Parameters
    ----------
    df : pd.DataFrame
        Data with hierarchy columns.
    hierarchy_cols : list[str] | None
        Columns defining hierarchy. Auto-detected if None.

    Returns
    -------
    HierarchySpec
        Hierarchy specification.
    """
    if hierarchy_cols is None:
        # Auto-detect common hierarchy columns
        potential_cols = [
            "state_id", "store_id", "cat_id", "dept_id", "item_id",
            "region", "location", "category", "customer_id", "material_id",
        ]
        hierarchy_cols = [c for c in potential_cols if c in df.columns]

    return HierarchySpec(levels=hierarchy_cols)
