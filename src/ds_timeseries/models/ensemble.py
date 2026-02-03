"""Ensemble Forecasting Models.

Combines multiple forecasters for improved accuracy and robustness.
Based on winning strategies from M5 Kaggle competition.

Ensemble methods:
- Simple Average: Equal weight to all models
- Weighted Average: Optimize weights based on CV performance
- Stacking: Meta-learner predicts optimal combination
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ds_timeseries.models.base import BaseForecaster


class SimpleEnsemble(BaseForecaster):
    """Simple average ensemble of multiple forecasters.

    Averages predictions from multiple models. Simple but effective,
    often matches or beats complex ensembles.

    Parameters
    ----------
    models : list[BaseForecaster]
        List of forecaster instances to ensemble.
    weights : list[float] | None
        Optional weights for each model. Defaults to equal weights.

    Examples
    --------
    >>> from ds_timeseries.models import LightGBMForecaster, XGBoostForecaster, ETSForecaster
    >>> ensemble = SimpleEnsemble([
    ...     LightGBMForecaster(),
    ...     XGBoostForecaster(),
    ...     ETSForecaster(),
    ... ])
    >>> ensemble.fit(train_df)
    >>> forecasts = ensemble.predict(horizon=4)
    """

    def __init__(
        self,
        models: list[BaseForecaster],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.weights = weights

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("weights must have same length as models")
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(self, df: pd.DataFrame, **kwargs) -> "SimpleEnsemble":
        """Fit all models in the ensemble."""
        self._validate_input(df)

        for model in self.models:
            model.fit(df, **kwargs)

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate ensemble predictions by averaging model forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Get predictions from all models
        all_preds = []
        for model in self.models:
            preds = model.predict(horizon, **kwargs)
            all_preds.append(preds)

        # Merge predictions
        base = all_preds[0][["unique_id", "ds"]].copy()

        for i, preds in enumerate(all_preds):
            base = base.merge(
                preds.rename(columns={"yhat": f"yhat_{i}"}),
                on=["unique_id", "ds"],
                how="outer",
            )

        # Average predictions
        pred_cols = [f"yhat_{i}" for i in range(len(self.models))]

        if self.weights:
            base["yhat"] = sum(
                base[col] * w for col, w in zip(pred_cols, self.weights)
            )
        else:
            base["yhat"] = base[pred_cols].mean(axis=1)

        base["yhat"] = base["yhat"].clip(lower=0)

        return base[["unique_id", "ds", "yhat"]]


class WeightedEnsemble(BaseForecaster):
    """Weighted ensemble with CV-optimized weights.

    Learns optimal weights for each model based on cross-validation
    performance on training data.

    Parameters
    ----------
    models : list[BaseForecaster]
        List of forecaster instances.
    cv_folds : int
        Number of CV folds for weight optimization.
    metric : str
        Metric to optimize ("wape" or "mae").

    Examples
    --------
    >>> ensemble = WeightedEnsemble([
    ...     LightGBMForecaster(),
    ...     XGBoostForecaster(),
    ... ], cv_folds=3)
    >>> ensemble.fit(train_df)
    >>> print(f"Learned weights: {ensemble.weights_}")
    """

    def __init__(
        self,
        models: list[BaseForecaster],
        cv_folds: int = 3,
        metric: str = "wape",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.cv_folds = cv_folds
        self.metric = metric
        self.weights_: list[float] | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "WeightedEnsemble":
        """Fit models and learn optimal weights via CV."""
        from ds_timeseries.evaluation.cross_validation import time_series_cv
        from ds_timeseries.evaluation.metrics import wape, mae

        self._validate_input(df)

        metric_fn = wape if self.metric == "wape" else mae

        # Collect CV predictions from each model
        model_cv_preds = {i: [] for i in range(len(self.models))}
        cv_actuals = []

        for fold in time_series_cv(df, n_folds=self.cv_folds, horizon=4):
            fold_actuals = fold.test[["unique_id", "ds", "y"]].copy()
            cv_actuals.append(fold_actuals)

            for i, model in enumerate(self.models):
                # Clone model
                model_clone = model.__class__(**model._params)
                model_clone.fit(fold.train)
                preds = model_clone.predict(horizon=4)

                merged = fold_actuals.merge(
                    preds, on=["unique_id", "ds"], how="inner"
                )
                model_cv_preds[i].append(merged["yhat"].values)

        # Concatenate CV predictions
        all_actuals = np.concatenate([
            fold["y"].values for fold in cv_actuals
        ])

        pred_matrix = np.column_stack([
            np.concatenate(model_cv_preds[i])
            for i in range(len(self.models))
        ])

        # Optimize weights using constrained optimization
        self.weights_ = self._optimize_weights(pred_matrix, all_actuals, metric_fn)

        # Fit final models on full data
        for model in self.models:
            model.fit(df, **kwargs)

        self._is_fitted = True
        return self

    def _optimize_weights(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metric_fn: callable,
    ) -> list[float]:
        """Find optimal weights via grid search."""
        n_models = predictions.shape[1]
        best_score = float("inf")
        best_weights = [1.0 / n_models] * n_models

        # Grid search over weight combinations
        if n_models == 2:
            for w1 in np.arange(0, 1.05, 0.1):
                w2 = 1 - w1
                weights = [w1, w2]
                ensemble_pred = predictions @ np.array(weights)
                score = metric_fn(actuals, ensemble_pred)
                if score < best_score:
                    best_score = score
                    best_weights = weights

        elif n_models == 3:
            for w1 in np.arange(0, 1.05, 0.1):
                for w2 in np.arange(0, 1.05 - w1, 0.1):
                    w3 = 1 - w1 - w2
                    weights = [w1, w2, w3]
                    ensemble_pred = predictions @ np.array(weights)
                    score = metric_fn(actuals, ensemble_pred)
                    if score < best_score:
                        best_score = score
                        best_weights = weights

        else:
            # For more models, use simple averaging as fallback
            # or scipy.optimize
            best_weights = [1.0 / n_models] * n_models

        return best_weights

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate weighted ensemble predictions."""
        if not self._is_fitted or self.weights_ is None:
            raise RuntimeError("Ensemble must be fitted before prediction")

        all_preds = []
        for model in self.models:
            preds = model.predict(horizon, **kwargs)
            all_preds.append(preds)

        base = all_preds[0][["unique_id", "ds"]].copy()

        for i, preds in enumerate(all_preds):
            base = base.merge(
                preds.rename(columns={"yhat": f"yhat_{i}"}),
                on=["unique_id", "ds"],
                how="outer",
            )

        pred_cols = [f"yhat_{i}" for i in range(len(self.models))]
        base["yhat"] = sum(
            base[col] * w for col, w in zip(pred_cols, self.weights_)
        )
        base["yhat"] = base["yhat"].clip(lower=0)

        return base[["unique_id", "ds", "yhat"]]


class StackingEnsemble(BaseForecaster):
    """Stacking ensemble with meta-learner.

    Level 0: Base models generate predictions
    Level 1: Meta-learner (Ridge regression) learns to combine them

    Based on winning approach from M5 competition where ensembles
    of LightGBM models were combined.

    Parameters
    ----------
    models : list[BaseForecaster]
        Base forecasters (Level 0).
    meta_learner : Any
        Sklearn-compatible regressor for Level 1.
        Defaults to Ridge regression.
    use_features : bool
        Whether to include original features in meta-learner.
    cv_folds : int
        Number of CV folds for generating meta-features.

    Examples
    --------
    >>> ensemble = StackingEnsemble([
    ...     LightGBMForecaster(),
    ...     XGBoostForecaster(),
    ...     ETSForecaster(),
    ... ])
    >>> ensemble.fit(train_df)
    >>> forecasts = ensemble.predict(horizon=4)
    """

    def __init__(
        self,
        models: list[BaseForecaster],
        meta_learner: Any = None,
        use_features: bool = False,
        cv_folds: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.use_features = use_features
        self.cv_folds = cv_folds
        self._train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "StackingEnsemble":
        """Fit base models and meta-learner."""
        from ds_timeseries.evaluation.cross_validation import time_series_cv

        self._validate_input(df)
        self._train_data = df.copy()

        # Generate out-of-fold predictions for meta-learner training
        oof_predictions = {i: [] for i in range(len(self.models))}
        oof_actuals = []
        oof_indices = []

        for fold in time_series_cv(df, n_folds=self.cv_folds, horizon=4):
            for i, model in enumerate(self.models):
                model_clone = model.__class__(**model._params)
                model_clone.fit(fold.train)
                preds = model_clone.predict(horizon=4)

                merged = fold.test.merge(
                    preds, on=["unique_id", "ds"], how="inner"
                )

                if i == 0:
                    oof_actuals.extend(merged["y"].values)
                    oof_indices.extend(merged.index.tolist())

                oof_predictions[i].extend(merged["yhat"].values)

        # Train meta-learner
        X_meta = np.column_stack([
            np.array(oof_predictions[i]) for i in range(len(self.models))
        ])
        y_meta = np.array(oof_actuals)

        self.meta_learner.fit(X_meta, y_meta)

        # Fit final base models on full data
        for model in self.models:
            model.fit(df, **kwargs)

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate stacked predictions."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Get base model predictions
        all_preds = []
        for model in self.models:
            preds = model.predict(horizon, **kwargs)
            all_preds.append(preds)

        # Combine predictions
        base = all_preds[0][["unique_id", "ds"]].copy()

        pred_arrays = []
        for i, preds in enumerate(all_preds):
            base = base.merge(
                preds.rename(columns={"yhat": f"yhat_{i}"}),
                on=["unique_id", "ds"],
                how="outer",
            )
            pred_arrays.append(preds["yhat"].values)

        # Meta-learner prediction
        X_meta = np.column_stack([
            base[f"yhat_{i}"].values for i in range(len(self.models))
        ])

        base["yhat"] = self.meta_learner.predict(X_meta)
        base["yhat"] = base["yhat"].clip(lower=0)

        return base[["unique_id", "ds", "yhat"]]


class HierarchicalEnsemble(BaseForecaster):
    """Ensemble of models trained at different hierarchy levels.

    Winner of M5 competition used this approach: train separate
    models for each store, category, department, and combine.

    Parameters
    ----------
    model_class : type
        Forecaster class to instantiate for each level.
    model_params : dict
        Parameters for the model class.
    levels : list[str]
        Hierarchy columns to train separate models for.

    Examples
    --------
    >>> ensemble = HierarchicalEnsemble(
    ...     model_class=LightGBMForecaster,
    ...     levels=["store_id", "cat_id"],
    ... )
    >>> ensemble.fit(train_df)
    >>> forecasts = ensemble.predict(horizon=4)
    """

    def __init__(
        self,
        model_class: type,
        model_params: dict | None = None,
        levels: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.levels = levels or ["store_id"]

        self._level_models: dict[str, dict[str, BaseForecaster]] = {}
        self._train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "HierarchicalEnsemble":
        """Fit models for each hierarchy level."""
        self._validate_input(df)
        self._train_data = df.copy()

        for level in self.levels:
            if level not in df.columns:
                continue

            self._level_models[level] = {}

            for level_value in df[level].unique():
                # Filter data for this level
                level_df = df[df[level] == level_value].copy()

                if len(level_df) < 52:  # Skip if too few observations
                    continue

                # Train model for this level
                model = self.model_class(**self.model_params)
                model.fit(level_df, **kwargs)
                self._level_models[level][level_value] = model

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate predictions and average across levels."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        all_preds = []

        for level, level_models in self._level_models.items():
            for level_value, model in level_models.items():
                preds = model.predict(horizon, **kwargs)
                preds["level"] = level
                preds["level_value"] = level_value
                all_preds.append(preds)

        if not all_preds:
            raise RuntimeError("No models were trained")

        # Combine predictions
        combined = pd.concat(all_preds, ignore_index=True)

        # Average predictions for each series across levels
        result = (
            combined
            .groupby(["unique_id", "ds"])["yhat"]
            .mean()
            .reset_index()
        )
        result["yhat"] = result["yhat"].clip(lower=0)

        return result
