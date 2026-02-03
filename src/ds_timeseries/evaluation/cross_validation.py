"""Time Series Cross-Validation.

Implements rolling window and expanding window cross-validation
with strict chronological ordering to prevent data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import pandas as pd

if TYPE_CHECKING:
    from ds_timeseries.models.base import BaseForecaster


@dataclass
class CVFold:
    """A single cross-validation fold."""

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def time_series_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = 4,
    step_size: int | None = None,
    min_train_size: int | None = None,
    expanding: bool = True,
) -> Iterator[CVFold]:
    """Generate time series cross-validation folds.

    Creates chronologically ordered train/test splits for time series.
    NO random shuffling - strictly temporal ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: unique_id, ds, y.
    n_folds : int
        Number of CV folds to generate.
    horizon : int
        Number of periods in each test set.
    step_size : int | None
        Number of periods to step between folds. Defaults to horizon.
    min_train_size : int | None
        Minimum periods in training set. Defaults to 2 * horizon.
    expanding : bool
        If True, use expanding window (training grows).
        If False, use rolling window (fixed training size).

    Yields
    ------
    CVFold
        Named tuple with (fold_id, train, test, cutoff).

    Examples
    --------
    >>> for fold in time_series_cv(df, n_folds=3, horizon=4):
    ...     print(f"Fold {fold.fold_id}: train={len(fold.train)}, test={len(fold.test)}")
    ...     model.fit(fold.train)
    ...     preds = model.predict(horizon=4)

    Notes
    -----
    Fold layout (expanding window, 3 folds, horizon=4):

    Time:  [----train----][test]
    Fold1: [=============][****]................
    Fold2: [================][****]............
    Fold3: [===================][****].........
    """
    step_size = step_size or horizon
    min_train_size = min_train_size or (2 * horizon)

    # Get unique sorted dates
    dates = df["ds"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(dates)

    # Calculate fold cutoff positions (working backwards from end)
    # Last fold ends at n_dates, test is [n_dates - horizon, n_dates)
    # Each previous fold steps back by step_size
    fold_configs = []

    for i in range(n_folds):
        test_end_idx = n_dates - (i * step_size)
        test_start_idx = test_end_idx - horizon
        cutoff_idx = test_start_idx - 1  # Last training index

        if cutoff_idx < min_train_size - 1:
            break  # Not enough training data

        if expanding:
            train_start_idx = 0
        else:
            train_start_idx = max(0, cutoff_idx - min_train_size + 1)

        fold_configs.append({
            "train_start_idx": train_start_idx,
            "cutoff_idx": cutoff_idx,
            "test_start_idx": test_start_idx,
            "test_end_idx": test_end_idx,
        })

    # Reverse to go from oldest to newest
    fold_configs = fold_configs[::-1]

    for fold_id, config in enumerate(fold_configs):
        train_dates = dates.iloc[config["train_start_idx"] : config["cutoff_idx"] + 1]
        test_dates = dates.iloc[config["test_start_idx"] : config["test_end_idx"]]
        cutoff = dates.iloc[config["cutoff_idx"]]

        train = df[df["ds"].isin(train_dates)]
        test = df[df["ds"].isin(test_dates)]

        yield CVFold(
            fold_id=fold_id,
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            cutoff=cutoff,
        )


def cross_validate(
    model: "BaseForecaster",
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = 4,
    step_size: int | None = None,
    min_train_size: int | None = None,
    expanding: bool = True,
) -> pd.DataFrame:
    """Run cross-validation and return predictions with actuals.

    Parameters
    ----------
    model : BaseForecaster
        Model to evaluate (will be cloned for each fold).
    df : pd.DataFrame
        Data with columns: unique_id, ds, y.
    n_folds : int
        Number of CV folds.
    horizon : int
        Forecast horizon for each fold.
    step_size : int | None
        Periods between fold cutoffs.
    min_train_size : int | None
        Minimum training periods.
    expanding : bool
        Expanding (True) or rolling (False) window.

    Returns
    -------
    pd.DataFrame
        Predictions merged with actuals:
        unique_id, ds, y (actual), yhat (prediction), fold_id, cutoff.

    Examples
    --------
    >>> from ds_timeseries.models.baselines import SeasonalNaiveForecaster
    >>> model = SeasonalNaiveForecaster(season_length=52)
    >>> results = cross_validate(model, df, n_folds=3, horizon=4)
    >>> print(results.groupby('fold_id').apply(lambda x: wape(x['y'], x['yhat'])))
    """
    all_results = []

    for fold in time_series_cv(
        df,
        n_folds=n_folds,
        horizon=horizon,
        step_size=step_size,
        min_train_size=min_train_size,
        expanding=expanding,
    ):
        # Create fresh model instance (clone params)
        fold_model = model.__class__(**model._params)

        # Fit and predict
        fold_model.fit(fold.train)
        preds = fold_model.predict(horizon=horizon)

        # Merge with actuals
        merged = fold.test.merge(
            preds,
            on=["unique_id", "ds"],
            how="inner",
        )
        merged["fold_id"] = fold.fold_id
        merged["cutoff"] = fold.cutoff

        all_results.append(merged)

    return pd.concat(all_results, ignore_index=True)


def cv_score(
    model: "BaseForecaster",
    df: pd.DataFrame,
    metric_fn: callable,
    n_folds: int = 5,
    horizon: int = 4,
    **cv_kwargs,
) -> dict[str, float]:
    """Run cross-validation and compute aggregate metrics.

    Parameters
    ----------
    model : BaseForecaster
        Model to evaluate.
    df : pd.DataFrame
        Data with columns: unique_id, ds, y.
    metric_fn : callable
        Metric function(y_true, y_pred) -> float.
    n_folds : int
        Number of CV folds.
    horizon : int
        Forecast horizon.
    **cv_kwargs
        Additional arguments to cross_validate.

    Returns
    -------
    dict[str, float]
        Dictionary with 'mean', 'std', 'min', 'max' of metric across folds.
    """
    results = cross_validate(model, df, n_folds=n_folds, horizon=horizon, **cv_kwargs)

    fold_scores = []
    for fold_id in results["fold_id"].unique():
        fold_data = results[results["fold_id"] == fold_id]
        score = metric_fn(fold_data["y"], fold_data["yhat"])
        fold_scores.append(score)

    import numpy as np

    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
        "n_folds": len(fold_scores),
    }
