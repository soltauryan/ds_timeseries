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


class DRFAMEnsemble(BaseForecaster):
    """Direct + Recursive Forecast Averaging Method (DRFAM).

    M5 Kaggle 1st place winning approach: averages predictions from
    both direct and recursive forecasting strategies across multiple
    hierarchy pooling levels.

    The key insight is that direct and recursive strategies have
    complementary error patterns, and averaging them reduces variance.

    Architecture (based on M5 winner):
    - 2 strategies: recursive, direct
    - 3 pooling levels: store, store-category, store-department
    - Total: 6 model configurations per series, averaged

    Parameters
    ----------
    model_class : type
        Forecaster class (must support strategy="recursive" and "direct").
    model_params : dict | None
        Parameters for the model class.
    pooling_levels : list[str] | None
        Hierarchy columns for pooling. Default: ["store_id", "cat_id", "dept_id"].
    use_direct : bool
        Whether to include direct forecasting models.
    use_recursive : bool
        Whether to include recursive forecasting models.

    Examples
    --------
    >>> from ds_timeseries.models import LightGBMForecaster
    >>> ensemble = DRFAMEnsemble(
    ...     model_class=LightGBMForecaster,
    ...     pooling_levels=["store_id", "dept_id"],
    ... )
    >>> ensemble.fit(train_df, horizon=4)
    >>> forecasts = ensemble.predict(horizon=4)

    References
    ----------
    - M5 Forecasting 1st place solution
    - Paper: "Simple averaging of direct and recursive forecasts via partial
      pooling using machine learning"
    """

    def __init__(
        self,
        model_class: type,
        model_params: dict | None = None,
        pooling_levels: list[str] | None = None,
        use_direct: bool = True,
        use_recursive: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.pooling_levels = pooling_levels or ["store_id"]
        self.use_direct = use_direct
        self.use_recursive = use_recursive

        if not use_direct and not use_recursive:
            raise ValueError("At least one of use_direct or use_recursive must be True")

        # Models dict: (strategy, level, level_value) -> model
        self._models: dict[tuple[str, str, str], BaseForecaster] = {}
        self._train_data: pd.DataFrame | None = None
        self._fitted_horizon: int | None = None

    def fit(self, df: pd.DataFrame, horizon: int = 4, **kwargs) -> "DRFAMEnsemble":
        """Fit direct and recursive models at multiple pooling levels.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with unique_id, ds, y columns.
        horizon : int
            Forecast horizon (required for direct strategy).
        """
        self._validate_input(df)
        self._train_data = df.copy()
        self._fitted_horizon = horizon

        strategies = []
        if self.use_recursive:
            strategies.append("recursive")
        if self.use_direct:
            strategies.append("direct")

        # Train models for each combination
        for strategy in strategies:
            for level in self.pooling_levels:
                if level not in df.columns:
                    # If level not in data, train on full data
                    model = self._create_model(strategy)
                    if strategy == "direct":
                        model.fit(df, horizon=horizon, **kwargs)
                    else:
                        model.fit(df, **kwargs)
                    self._models[(strategy, level, "__all__")] = model
                    continue

                for level_value in df[level].unique():
                    level_df = df[df[level] == level_value].copy()

                    if len(level_df) < 52:  # Skip if too few observations
                        continue

                    model = self._create_model(strategy)

                    if strategy == "direct":
                        model.fit(level_df, horizon=horizon, **kwargs)
                    else:
                        model.fit(level_df, **kwargs)

                    self._models[(strategy, level, level_value)] = model

        self._is_fitted = True
        return self

    def _create_model(self, strategy: str) -> BaseForecaster:
        """Create a model instance with the specified strategy."""
        params = self.model_params.copy()
        params["strategy"] = strategy
        return self.model_class(**params)

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate ensemble predictions by averaging all models."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        if self._fitted_horizon is not None and horizon > self._fitted_horizon:
            raise ValueError(
                f"Ensemble was fitted for horizon={self._fitted_horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        all_preds = []

        for (strategy, level, level_value), model in self._models.items():
            try:
                preds = model.predict(horizon, **kwargs)
                preds["strategy"] = strategy
                preds["level"] = level
                preds["level_value"] = level_value
                all_preds.append(preds)
            except Exception:
                # Skip failed predictions
                continue

        if not all_preds:
            raise RuntimeError("No models generated predictions")

        combined = pd.concat(all_preds, ignore_index=True)

        # Average predictions for each series across all models
        result = (
            combined
            .groupby(["unique_id", "ds"])["yhat"]
            .mean()
            .reset_index()
        )
        result["yhat"] = result["yhat"].clip(lower=0)

        return result

    def predict_with_components(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate predictions with individual model components.

        Returns predictions from each model configuration for analysis.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        all_preds = []

        for (strategy, level, level_value), model in self._models.items():
            try:
                preds = model.predict(horizon, **kwargs)
                preds["strategy"] = strategy
                preds["level"] = level
                preds["level_value"] = level_value
                all_preds.append(preds)
            except Exception:
                continue

        return pd.concat(all_preds, ignore_index=True)


class MultiLevelPoolingEnsemble(BaseForecaster):
    """Multi-level data pooling ensemble.

    Trains models at different aggregation levels of the hierarchy and
    combines their predictions. Based on M5 competition insights.

    Pooling approaches:
    - Store level: One model per store (10 models)
    - Store-Category level: One model per store-category combo (30 models)
    - Store-Department level: One model per store-dept combo (70 models)

    Parameters
    ----------
    model_class : type
        Forecaster class to use.
    model_params : dict | None
        Parameters for the model class.
    levels : list[list[str]] | None
        List of grouping levels. Each inner list is a set of columns
        to group by. E.g., [["store_id"], ["store_id", "cat_id"]].
    aggregation : str
        How to combine predictions: "mean" or "median".

    Examples
    --------
    >>> ensemble = MultiLevelPoolingEnsemble(
    ...     model_class=LightGBMForecaster,
    ...     levels=[["store_id"], ["store_id", "cat_id"], ["store_id", "dept_id"]],
    ... )
    >>> ensemble.fit(train_df)
    >>> forecasts = ensemble.predict(horizon=4)
    """

    def __init__(
        self,
        model_class: type,
        model_params: dict | None = None,
        levels: list[list[str]] | None = None,
        aggregation: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.levels = levels or [["store_id"]]
        self.aggregation = aggregation

        # Models: (level_tuple, group_value_tuple) -> model
        self._models: dict[tuple, BaseForecaster] = {}
        self._train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "MultiLevelPoolingEnsemble":
        """Fit models at each pooling level."""
        self._validate_input(df)
        self._train_data = df.copy()

        for level_cols in self.levels:
            # Check if all columns exist
            missing = [c for c in level_cols if c not in df.columns]
            if missing:
                continue

            level_key = tuple(level_cols)

            # Group data by the level columns
            for group_vals, group_df in df.groupby(level_cols):
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)

                if len(group_df) < 52:  # Skip if too few observations
                    continue

                model = self.model_class(**self.model_params)
                model.fit(group_df.copy(), **kwargs)
                self._models[(level_key, group_vals)] = model

        if not self._models:
            # Fallback: train on full data if no groupings possible
            model = self.model_class(**self.model_params)
            model.fit(df, **kwargs)
            self._models[(("__all__",), ("__all__",))] = model

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate predictions and aggregate across pooling levels."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        all_preds = []

        for (level_key, group_vals), model in self._models.items():
            try:
                preds = model.predict(horizon, **kwargs)
                preds["pool_level"] = str(level_key)
                preds["pool_group"] = str(group_vals)
                all_preds.append(preds)
            except Exception:
                continue

        if not all_preds:
            raise RuntimeError("No models generated predictions")

        combined = pd.concat(all_preds, ignore_index=True)

        # Aggregate predictions
        if self.aggregation == "mean":
            result = (
                combined
                .groupby(["unique_id", "ds"])["yhat"]
                .mean()
                .reset_index()
            )
        else:  # median
            result = (
                combined
                .groupby(["unique_id", "ds"])["yhat"]
                .median()
                .reset_index()
            )

        result["yhat"] = result["yhat"].clip(lower=0)
        return result

    def get_model_count(self) -> dict:
        """Get count of models at each pooling level."""
        counts = {}
        for (level_key, _), _ in self._models.items():
            key = str(level_key)
            counts[key] = counts.get(key, 0) + 1
        return counts


class M5WinnerEnsemble(BaseForecaster):
    """Full M5 1st place winning ensemble architecture.

    This implements the complete winning approach:
    1. Train LightGBM models with Tweedie objective
    2. Use both recursive and direct forecasting
    3. Pool data at 3 levels: store, store-cat, store-dept
    4. Average all 220 model predictions (10+30+70 pools x 2 strategies)

    Parameters
    ----------
    feature_config : Any | None
        Feature engineering configuration.
    pooling_levels : list[list[str]] | None
        Pooling levels. Default mimics M5 structure.
    lgb_params : dict | None
        LightGBM parameters.

    Examples
    --------
    >>> ensemble = M5WinnerEnsemble()
    >>> ensemble.fit(train_df, horizon=4)
    >>> forecasts = ensemble.predict(horizon=4)

    References
    ----------
    - M5 Forecasting Accuracy Competition 1st place
    - DRFAM: Direct + Recursive Forecast Averaging Method
    """

    def __init__(
        self,
        feature_config: Any = None,
        pooling_levels: list[list[str]] | None = None,
        lgb_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_config = feature_config
        self.pooling_levels = pooling_levels or [
            ["store_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
        ]
        self.lgb_params = lgb_params

        self._drfam_ensemble: DRFAMEnsemble | None = None
        self._fitted_horizon: int | None = None

    def fit(self, df: pd.DataFrame, horizon: int = 4, **kwargs) -> "M5WinnerEnsemble":
        """Fit the M5 winner ensemble."""
        from ds_timeseries.models.ml import LightGBMForecaster

        self._validate_input(df)
        self._fitted_horizon = horizon

        # Build model params
        model_params = {}
        if self.feature_config:
            model_params["feature_config"] = self.feature_config
        if self.lgb_params:
            model_params["lgb_params"] = self.lgb_params
        else:
            # M5 winner params
            model_params["lgb_params"] = {
                "boosting_type": "gbdt",
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "num_leaves": 127,
                "learning_rate": 0.03,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "n_estimators": 300,
                "verbose": -1,
            }

        # Flatten pooling levels for DRFAM
        flat_levels = []
        for level in self.pooling_levels:
            if len(level) == 1:
                flat_levels.append(level[0])
            else:
                # For multi-column levels, create composite column
                level_name = "_".join(level)
                if level_name not in df.columns:
                    df[level_name] = df[level].astype(str).agg("_".join, axis=1)
                flat_levels.append(level_name)

        self._drfam_ensemble = DRFAMEnsemble(
            model_class=LightGBMForecaster,
            model_params=model_params,
            pooling_levels=flat_levels,
            use_direct=True,
            use_recursive=True,
        )

        self._drfam_ensemble.fit(df, horizon=horizon, **kwargs)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate ensemble predictions."""
        if not self._is_fitted or self._drfam_ensemble is None:
            raise RuntimeError("Ensemble must be fitted before prediction")

        return self._drfam_ensemble.predict(horizon, **kwargs)

    def get_model_count(self) -> int:
        """Get total number of models in the ensemble."""
        if self._drfam_ensemble is None:
            return 0
        return len(self._drfam_ensemble._models)
