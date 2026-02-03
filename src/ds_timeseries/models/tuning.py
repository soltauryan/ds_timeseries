"""Hyperparameter Tuning for Time Series Models.

Provides tuning utilities using time series cross-validation
to prevent data leakage during hyperparameter optimization.

Supports:
- Grid Search
- Random Search
- Optuna (if installed)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ds_timeseries.evaluation.cross_validation import time_series_cv
from ds_timeseries.evaluation.metrics import wape, mae
from ds_timeseries.models.base import BaseForecaster


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    best_model: BaseForecaster | None = None


def grid_search_cv(
    model_class: type,
    param_grid: dict[str, list[Any]],
    df: pd.DataFrame,
    n_folds: int = 3,
    horizon: int = 4,
    metric: str = "wape",
    fixed_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> TuningResult:
    """Grid search with time series cross-validation.

    Exhaustively searches all parameter combinations.

    Parameters
    ----------
    model_class : type
        Forecaster class to tune.
    param_grid : dict
        Dictionary of parameters to search.
        E.g., {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.05, 0.1]}
    df : pd.DataFrame
        Training data.
    n_folds : int
        Number of CV folds.
    horizon : int
        Forecast horizon.
    metric : str
        Metric to optimize ("wape" or "mae").
    fixed_params : dict
        Parameters to keep fixed (not tuned).
    verbose : bool
        Print progress.

    Returns
    -------
    TuningResult
        Best parameters and all results.

    Examples
    --------
    >>> from ds_timeseries.models import LightGBMForecaster
    >>> result = grid_search_cv(
    ...     LightGBMForecaster,
    ...     param_grid={
    ...         "lgb_params": [
    ...             {"num_leaves": 15, "learning_rate": 0.05},
    ...             {"num_leaves": 31, "learning_rate": 0.05},
    ...             {"num_leaves": 31, "learning_rate": 0.1},
    ...         ]
    ...     },
    ...     df=train_df,
    ... )
    >>> print(f"Best params: {result.best_params}")
    """
    metric_fn = wape if metric == "wape" else mae
    fixed_params = fixed_params or {}

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    if verbose:
        print(f"Grid search: {len(combinations)} combinations, {n_folds} folds")

    results = []
    best_score = float("inf")
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        all_params = {**fixed_params, **params}

        # CV evaluation
        fold_scores = []

        for fold in time_series_cv(df, n_folds=n_folds, horizon=horizon):
            try:
                model = model_class(**all_params)
                model.fit(fold.train)
                preds = model.predict(horizon=horizon)

                merged = fold.test.merge(preds, on=["unique_id", "ds"], how="inner")
                if len(merged) > 0:
                    score = metric_fn(merged["y"], merged["yhat"])
                    fold_scores.append(score)
            except Exception as e:
                if verbose:
                    print(f"  Error with {params}: {e}")
                break

        if fold_scores:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            results.append({
                **params,
                "mean_score": mean_score,
                "std_score": std_score,
                "n_folds": len(fold_scores),
            })

            if verbose:
                print(f"  [{i+1}/{len(combinations)}] {params} -> {metric}={mean_score:.4f} ± {std_score:.4f}")

            if mean_score < best_score:
                best_score = mean_score
                best_params = all_params

    results_df = pd.DataFrame(results).sort_values("mean_score")

    # Train best model on full data
    best_model = None
    if best_params:
        best_model = model_class(**best_params)
        best_model.fit(df)

    return TuningResult(
        best_params=best_params or {},
        best_score=best_score,
        all_results=results_df,
        best_model=best_model,
    )


def random_search_cv(
    model_class: type,
    param_distributions: dict[str, Any],
    df: pd.DataFrame,
    n_iter: int = 20,
    n_folds: int = 3,
    horizon: int = 4,
    metric: str = "wape",
    fixed_params: dict[str, Any] | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> TuningResult:
    """Random search with time series cross-validation.

    Samples random parameter combinations from distributions.

    Parameters
    ----------
    model_class : type
        Forecaster class to tune.
    param_distributions : dict
        Dictionary of parameter distributions.
        Values can be:
        - list: Sample uniformly
        - tuple (low, high): Uniform float range
        - tuple (low, high, "log"): Log-uniform range
        - callable: Function that returns a value
    df : pd.DataFrame
        Training data.
    n_iter : int
        Number of random combinations to try.
    n_folds : int
        Number of CV folds.
    horizon : int
        Forecast horizon.
    metric : str
        Metric to optimize.
    fixed_params : dict
        Parameters to keep fixed.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    TuningResult
        Best parameters and all results.

    Examples
    --------
    >>> result = random_search_cv(
    ...     LightGBMForecaster,
    ...     param_distributions={
    ...         "lgb_params.num_leaves": [15, 31, 63, 127],
    ...         "lgb_params.learning_rate": (0.01, 0.2, "log"),
    ...         "lgb_params.n_estimators": [50, 100, 200],
    ...     },
    ...     df=train_df,
    ...     n_iter=20,
    ... )
    """
    np.random.seed(random_state)
    metric_fn = wape if metric == "wape" else mae
    fixed_params = fixed_params or {}

    if verbose:
        print(f"Random search: {n_iter} iterations, {n_folds} folds")

    results = []
    best_score = float("inf")
    best_params = None

    for i in range(n_iter):
        # Sample parameters
        params = _sample_params(param_distributions)
        all_params = {**fixed_params, **_unflatten_params(params)}

        # CV evaluation
        fold_scores = []

        for fold in time_series_cv(df, n_folds=n_folds, horizon=horizon):
            try:
                model = model_class(**all_params)
                model.fit(fold.train)
                preds = model.predict(horizon=horizon)

                merged = fold.test.merge(preds, on=["unique_id", "ds"], how="inner")
                if len(merged) > 0:
                    score = metric_fn(merged["y"], merged["yhat"])
                    fold_scores.append(score)
            except Exception as e:
                if verbose:
                    print(f"  Error with {params}: {e}")
                break

        if fold_scores:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            results.append({
                **params,
                "mean_score": mean_score,
                "std_score": std_score,
            })

            if verbose:
                print(f"  [{i+1}/{n_iter}] {metric}={mean_score:.4f} ± {std_score:.4f}")

            if mean_score < best_score:
                best_score = mean_score
                best_params = all_params

    results_df = pd.DataFrame(results).sort_values("mean_score")

    best_model = None
    if best_params:
        best_model = model_class(**best_params)
        best_model.fit(df)

    return TuningResult(
        best_params=best_params or {},
        best_score=best_score,
        all_results=results_df,
        best_model=best_model,
    )


def _sample_params(distributions: dict[str, Any]) -> dict[str, Any]:
    """Sample parameters from distributions."""
    params = {}

    for name, dist in distributions.items():
        if isinstance(dist, list):
            params[name] = np.random.choice(dist)
        elif isinstance(dist, tuple):
            if len(dist) == 2:
                low, high = dist
                params[name] = np.random.uniform(low, high)
            elif len(dist) == 3 and dist[2] == "log":
                low, high, _ = dist
                params[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            elif len(dist) == 3 and dist[2] == "int":
                low, high, _ = dist
                params[name] = np.random.randint(low, high + 1)
        elif callable(dist):
            params[name] = dist()
        else:
            params[name] = dist

    return params


def _unflatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert flat params like 'lgb_params.num_leaves' to nested dict."""
    result = {}

    for key, value in params.items():
        if "." in key:
            parts = key.split(".")
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        else:
            result[key] = value

    return result


def tune_lightgbm(
    df: pd.DataFrame,
    feature_config: Any = None,
    n_iter: int = 20,
    n_folds: int = 3,
    verbose: bool = True,
) -> TuningResult:
    """Convenience function to tune LightGBM hyperparameters.

    Uses sensible parameter ranges based on M5 competition winners.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    feature_config : FeatureConfig
        Feature engineering config.
    n_iter : int
        Number of random search iterations.
    n_folds : int
        Number of CV folds.
    verbose : bool
        Print progress.

    Returns
    -------
    TuningResult
        Best parameters and tuned model.

    Examples
    --------
    >>> result = tune_lightgbm(train_df, n_iter=30)
    >>> model = result.best_model
    >>> forecasts = model.predict(horizon=4)
    """
    from ds_timeseries.models.ml import LightGBMForecaster

    param_distributions = {
        "lgb_params.num_leaves": [15, 31, 63, 127],
        "lgb_params.learning_rate": (0.01, 0.1, "log"),
        "lgb_params.n_estimators": [50, 100, 200, 300],
        "lgb_params.feature_fraction": (0.6, 1.0),
        "lgb_params.bagging_fraction": (0.6, 1.0),
        "lgb_params.min_child_samples": [5, 10, 20, 50],
    }

    fixed_params = {
        "feature_config": feature_config,
        "use_tweedie": True,
    }

    return random_search_cv(
        LightGBMForecaster,
        param_distributions,
        df,
        n_iter=n_iter,
        n_folds=n_folds,
        fixed_params=fixed_params,
        verbose=verbose,
    )


def tune_xgboost(
    df: pd.DataFrame,
    feature_config: Any = None,
    n_iter: int = 20,
    n_folds: int = 3,
    verbose: bool = True,
) -> TuningResult:
    """Convenience function to tune XGBoost hyperparameters.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    feature_config : FeatureConfig
        Feature engineering config.
    n_iter : int
        Number of random search iterations.
    n_folds : int
        Number of CV folds.
    verbose : bool
        Print progress.

    Returns
    -------
    TuningResult
        Best parameters and tuned model.
    """
    from ds_timeseries.models.ml import XGBoostForecaster

    param_distributions = {
        "xgb_params.max_depth": [3, 4, 5, 6, 8],
        "xgb_params.learning_rate": (0.01, 0.2, "log"),
        "xgb_params.n_estimators": [50, 100, 200, 300],
        "xgb_params.subsample": (0.6, 1.0),
        "xgb_params.colsample_bytree": (0.6, 1.0),
        "xgb_params.min_child_weight": [1, 3, 5, 10],
    }

    fixed_params = {
        "feature_config": feature_config,
    }

    return random_search_cv(
        XGBoostForecaster,
        param_distributions,
        df,
        n_iter=n_iter,
        n_folds=n_folds,
        fixed_params=fixed_params,
        verbose=verbose,
    )


try:
    import optuna

    def optuna_tune(
        model_class: type,
        df: pd.DataFrame,
        param_space: Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 50,
        n_folds: int = 3,
        horizon: int = 4,
        metric: str = "wape",
        fixed_params: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> TuningResult:
        """Hyperparameter tuning with Optuna.

        Uses Bayesian optimization for efficient search.

        Parameters
        ----------
        model_class : type
            Forecaster class to tune.
        df : pd.DataFrame
            Training data.
        param_space : callable
            Function that takes an Optuna trial and returns params dict.
        n_trials : int
            Number of Optuna trials.
        n_folds : int
            Number of CV folds.
        horizon : int
            Forecast horizon.
        metric : str
            Metric to optimize.
        fixed_params : dict
            Fixed parameters.
        verbose : bool
            Print progress.

        Returns
        -------
        TuningResult
            Best parameters and model.

        Examples
        --------
        >>> def param_space(trial):
        ...     return {
        ...         "lgb_params": {
        ...             "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        ...             "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        ...         }
        ...     }
        >>> result = optuna_tune(LightGBMForecaster, df, param_space, n_trials=50)
        """
        metric_fn = wape if metric == "wape" else mae
        fixed_params = fixed_params or {}

        def objective(trial: optuna.Trial) -> float:
            params = param_space(trial)
            all_params = {**fixed_params, **params}

            fold_scores = []
            for fold in time_series_cv(df, n_folds=n_folds, horizon=horizon):
                try:
                    model = model_class(**all_params)
                    model.fit(fold.train)
                    preds = model.predict(horizon=horizon)

                    merged = fold.test.merge(preds, on=["unique_id", "ds"], how="inner")
                    if len(merged) > 0:
                        score = metric_fn(merged["y"], merged["yhat"])
                        fold_scores.append(score)
                except Exception:
                    return float("inf")

            return np.mean(fold_scores) if fold_scores else float("inf")

        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=verbose,
        )

        # Get best params
        best_params = {**fixed_params, **param_space(study.best_trial)}

        # Train final model
        best_model = model_class(**best_params)
        best_model.fit(df)

        # Create results DataFrame
        results_df = pd.DataFrame([
            {**t.params, "score": t.value}
            for t in study.trials
        ]).sort_values("score")

        return TuningResult(
            best_params=best_params,
            best_score=study.best_value,
            all_results=results_df,
            best_model=best_model,
        )

except ImportError:
    pass  # Optuna not installed
