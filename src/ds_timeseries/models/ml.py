"""Machine Learning Forecasting Models.

LightGBM, XGBoost, CatBoost, Prophet, and hybrid model wrappers that transform
time series forecasting into supervised regression problems.

Based on winning approaches from M5 Kaggle competition:
- LightGBM with Tweedie objective for intermittent demand
- Feature engineering with lags at multiple aggregation levels
- Ensemble approaches
- Direct + Recursive forecasting (DRFAM: M5 1st place)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ds_timeseries.features.engineering import FeatureConfig, engineer_features
from ds_timeseries.models.base import BaseForecaster
from ds_timeseries.utils.config import DEFAULT_FREQ

# Type for forecasting strategy
ForecastStrategy = Literal["recursive", "direct"]


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based forecaster for time series.

    Transforms time series into regression using lag/rolling features.
    Uses Tweedie objective by default (better for intermittent demand).

    Parameters
    ----------
    feature_config : FeatureConfig | None
        Feature engineering configuration.
    lgb_params : dict | None
        LightGBM parameters. Defaults to Tweedie objective.
    use_tweedie : bool
        If True, use Tweedie objective (recommended for zeros in data).
    strategy : {"recursive", "direct"}
        Forecasting strategy. "recursive" predicts one step at a time
        and uses predictions as features. "direct" trains separate models
        for each horizon step (M5 winner used both).

    Examples
    --------
    >>> model = LightGBMForecaster(use_tweedie=True)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    >>> # Direct forecasting (separate model per horizon)
    >>> model = LightGBMForecaster(strategy="direct")
    >>> model.fit(train_df, horizon=4)  # Must specify horizon at fit time
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        lgb_params: dict | None = None,
        use_tweedie: bool = True,
        strategy: ForecastStrategy = "recursive",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_config = feature_config or FeatureConfig()
        self.use_tweedie = use_tweedie
        self.strategy = strategy

        # Default params based on M5 competition winners
        default_params = {
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
        }

        if use_tweedie:
            default_params["objective"] = "tweedie"
            default_params["tweedie_variance_power"] = 1.5
            default_params["metric"] = "tweedie"
        else:
            default_params["objective"] = "regression"
            default_params["metric"] = "mae"

        self.lgb_params = lgb_params or default_params

        self._model = None
        self._direct_models: dict[int, Any] = {}  # For direct strategy
        self._feature_cols: list[str] = []
        self._cat_features: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._last_train_data: pd.DataFrame | None = None
        self._fitted_horizon: int | None = None

    def fit(
        self,
        df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        horizon: int | None = None,
    ) -> "LightGBMForecaster":
        """Fit the model on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with unique_id, ds, y columns.
        valid_df : pd.DataFrame | None
            Optional validation data (not used currently).
        horizon : int | None
            Required for direct strategy. Number of steps to forecast.
        """
        import lightgbm as lgb

        self._validate_input(df)
        self._last_train_data = df.copy()

        if self.strategy == "direct":
            if horizon is None:
                raise ValueError("horizon is required for direct forecasting strategy")
            self._fitted_horizon = horizon
            return self._fit_direct(df, horizon)

        # Recursive strategy: single model
        df_feat = engineer_features(df, self.feature_config)

        # Identify feature columns
        exclude = {"unique_id", "ds", "y"}
        self._feature_cols = [c for c in df_feat.columns if c not in exclude]
        self._cat_features = self._get_categorical_features(df_feat)

        # Drop NaN rows
        df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                         if c not in self._cat_features])

        X = df_feat[self._feature_cols].copy()
        y = df_feat["y"]

        # Encode categoricals
        X = self._encode_categoricals(X, fit=True)

        # Train
        self._model = lgb.LGBMRegressor(**self.lgb_params)

        cat_feature_indices = [self._feature_cols.index(c) for c in self._cat_features
                               if c in self._feature_cols]

        self._model.fit(
            X, y,
            categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
        )
        self._is_fitted = True

        return self

    def _fit_direct(self, df: pd.DataFrame, horizon: int) -> "LightGBMForecaster":
        """Fit separate models for each horizon step (direct forecasting)."""
        import lightgbm as lgb

        # Create targets shifted by h steps for each horizon
        for h in range(1, horizon + 1):
            df_h = df.copy()

            # Shift target: y_t+h becomes target for features at time t
            df_h["y_target"] = (
                df_h.groupby("unique_id")["y"]
                .shift(-h)
            )

            # Drop rows without valid target
            df_h = df_h.dropna(subset=["y_target"])

            # Engineer features based on current time (not future)
            df_feat = engineer_features(df_h, self.feature_config)

            if h == 1:  # Set up feature columns once
                exclude = {"unique_id", "ds", "y", "y_target"}
                self._feature_cols = [c for c in df_feat.columns if c not in exclude]
                self._cat_features = self._get_categorical_features(df_feat)

            df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                             if c not in self._cat_features])

            X = df_feat[self._feature_cols].copy()
            y = df_feat["y_target"]

            if h == 1:
                X = self._encode_categoricals(X, fit=True)
            else:
                X = self._encode_categoricals(X, fit=False)

            model = lgb.LGBMRegressor(**self.lgb_params)

            cat_feature_indices = [self._feature_cols.index(c) for c in self._cat_features
                                   if c in self._feature_cols]

            model.fit(
                X, y,
                categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
            )

            self._direct_models[h] = model

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts using recursive or direct prediction."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.strategy == "direct":
            if self._fitted_horizon is None or horizon > self._fitted_horizon:
                raise ValueError(
                    f"Direct model was fitted for horizon={self._fitted_horizon}. "
                    f"Cannot predict horizon={horizon}."
                )
            return self._direct_predict(horizon)

        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self._recursive_predict(horizon)

    def _direct_predict(self, horizon: int) -> pd.DataFrame:
        """Direct multi-step forecasting (one model per step)."""
        last_dates = (
            self._last_train_data
            .groupby("unique_id")["ds"]
            .max()
            .reset_index()
        )

        all_forecasts = []

        # Engineer features once from the last training data
        df_feat = engineer_features(self._last_train_data, self.feature_config)

        # Get the most recent features for each series
        latest_feat = (
            df_feat
            .sort_values("ds")
            .groupby("unique_id")
            .tail(1)
            .copy()
        )

        for h in range(1, horizon + 1):
            model = self._direct_models[h]

            X = latest_feat[self._feature_cols].copy()
            X = self._encode_categoricals(X, fit=False)

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(0)

            preds = model.predict(X)
            preds = np.maximum(preds, 0)

            forecast_df = pd.DataFrame({
                "unique_id": latest_feat["unique_id"].values,
                "ds": last_dates["ds"] + pd.Timedelta(weeks=h),
                "yhat": preds,
            })
            all_forecasts.append(forecast_df)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _recursive_predict(self, horizon: int) -> pd.DataFrame:
        """Recursive multi-step forecasting."""
        last_dates = (
            self._last_train_data
            .groupby("unique_id")["ds"]
            .max()
            .reset_index()
        )

        all_forecasts = []
        history = self._last_train_data.copy()

        for h in range(horizon):
            future_rows = []
            for _, row in last_dates.iterrows():
                future_date = row["ds"] + pd.Timedelta(weeks=h + 1)
                future_rows.append({
                    "unique_id": row["unique_id"],
                    "ds": future_date,
                    "y": np.nan,
                })

            future_df = pd.DataFrame(future_rows)
            combined = pd.concat([history, future_df], ignore_index=True)
            combined = combined.sort_values(["unique_id", "ds"])

            combined_feat = engineer_features(combined, self.feature_config)
            future_feat = combined_feat[combined_feat["y"].isna()].copy()

            if future_feat.empty:
                break

            X_future = future_feat[self._feature_cols].copy()
            X_future = self._encode_categoricals(X_future, fit=False)

            # Fill NaN
            numeric_cols = X_future.select_dtypes(include=[np.number]).columns
            X_future[numeric_cols] = X_future[numeric_cols].fillna(0)

            preds = self._model.predict(X_future)
            preds = np.maximum(preds, 0)  # Ensure non-negative

            forecast_df = future_feat[["unique_id", "ds"]].copy()
            forecast_df["yhat"] = preds
            all_forecasts.append(forecast_df)

            future_df["y"] = preds
            history = pd.concat([history, future_df], ignore_index=True)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _get_categorical_features(self, df: pd.DataFrame) -> list[str]:
        """Identify categorical feature columns."""
        cat_cols = []
        hierarchy_cols = {"item_id", "store_id", "cat_id", "dept_id", "state_id",
                         "customer_id", "material_id", "category", "location"}

        for col in self._feature_cols:
            if col in hierarchy_cols or df[col].dtype == "object":
                cat_cols.append(col)

        return cat_cols

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical columns as integers."""
        X = X.copy()
        for col in self._cat_features:
            if col not in X.columns:
                continue

            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le:
                    # Handle unseen categories
                    X[col] = X[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return X

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class XGBoostForecaster(BaseForecaster):
    """XGBoost-based forecaster for time series.

    Parameters
    ----------
    feature_config : FeatureConfig | None
        Feature engineering configuration.
    xgb_params : dict | None
        XGBoost parameters.

    Examples
    --------
    >>> model = XGBoostForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        xgb_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_config = feature_config or FeatureConfig()

        self.xgb_params = xgb_params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
        }

        self._model = None
        self._feature_cols: list[str] = []
        self._cat_features: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._last_train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> "XGBoostForecaster":
        """Fit the model on training data."""
        import xgboost as xgb

        self._validate_input(df)

        df_feat = engineer_features(df, self.feature_config)
        self._last_train_data = df.copy()

        exclude = {"unique_id", "ds", "y"}
        self._feature_cols = [c for c in df_feat.columns if c not in exclude]
        self._cat_features = self._get_categorical_features(df_feat)

        df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                         if c not in self._cat_features])

        X = df_feat[self._feature_cols].copy()
        y = df_feat["y"]

        # Encode categoricals as integers for XGBoost
        X = self._encode_categoricals(X, fit=True)

        self._model = xgb.XGBRegressor(**self.xgb_params)
        self._model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts using recursive prediction."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self._recursive_predict(horizon)

    def _recursive_predict(self, horizon: int) -> pd.DataFrame:
        """Recursive multi-step forecasting."""
        last_dates = (
            self._last_train_data
            .groupby("unique_id")["ds"]
            .max()
            .reset_index()
        )

        all_forecasts = []
        history = self._last_train_data.copy()

        for h in range(horizon):
            future_rows = []
            for _, row in last_dates.iterrows():
                future_date = row["ds"] + pd.Timedelta(weeks=h + 1)
                future_rows.append({
                    "unique_id": row["unique_id"],
                    "ds": future_date,
                    "y": np.nan,
                })

            future_df = pd.DataFrame(future_rows)
            combined = pd.concat([history, future_df], ignore_index=True)
            combined = combined.sort_values(["unique_id", "ds"])

            combined_feat = engineer_features(combined, self.feature_config)
            future_feat = combined_feat[combined_feat["y"].isna()].copy()

            if future_feat.empty:
                break

            X_future = future_feat[self._feature_cols].copy()
            X_future = self._encode_categoricals(X_future, fit=False)

            numeric_cols = X_future.select_dtypes(include=[np.number]).columns
            X_future[numeric_cols] = X_future[numeric_cols].fillna(0)

            preds = self._model.predict(X_future)
            preds = np.maximum(preds, 0)

            forecast_df = future_feat[["unique_id", "ds"]].copy()
            forecast_df["yhat"] = preds
            all_forecasts.append(forecast_df)

            future_df["y"] = preds
            history = pd.concat([history, future_df], ignore_index=True)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _get_categorical_features(self, df: pd.DataFrame) -> list[str]:
        cat_cols = []
        hierarchy_cols = {"item_id", "store_id", "cat_id", "dept_id", "state_id",
                         "customer_id", "material_id", "category", "location"}

        for col in self._feature_cols:
            if col in hierarchy_cols or df[col].dtype == "object":
                cat_cols.append(col)

        return cat_cols

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()
        for col in self._cat_features:
            if col not in X.columns:
                continue

            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le:
                    X[col] = X[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return X

    def get_feature_importance(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class CatBoostForecaster(BaseForecaster):
    """CatBoost-based forecaster for time series.

    CatBoost handles categorical features natively and is robust to overfitting.
    Popular in Kaggle competitions alongside LightGBM and XGBoost.

    Parameters
    ----------
    feature_config : FeatureConfig | None
        Feature engineering configuration.
    cat_params : dict | None
        CatBoost parameters.
    use_tweedie : bool
        If True, use Tweedie objective (experimental in CatBoost).
    strategy : {"recursive", "direct"}
        Forecasting strategy.

    Examples
    --------
    >>> model = CatBoostForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        cat_params: dict | None = None,
        use_tweedie: bool = False,
        strategy: ForecastStrategy = "recursive",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_config = feature_config or FeatureConfig()
        self.use_tweedie = use_tweedie
        self.strategy = strategy

        # Default params for CatBoost
        default_params = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False,
        }

        if use_tweedie:
            default_params["loss_function"] = "Tweedie:variance_power=1.5"
        else:
            default_params["loss_function"] = "MAE"

        self.cat_params = cat_params or default_params

        self._model = None
        self._direct_models: dict[int, Any] = {}
        self._feature_cols: list[str] = []
        self._cat_features: list[str] = []
        self._last_train_data: pd.DataFrame | None = None
        self._fitted_horizon: int | None = None

    def fit(
        self,
        df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        horizon: int | None = None,
    ) -> "CatBoostForecaster":
        """Fit the model on training data."""
        from catboost import CatBoostRegressor

        self._validate_input(df)
        self._last_train_data = df.copy()

        if self.strategy == "direct":
            if horizon is None:
                raise ValueError("horizon is required for direct forecasting strategy")
            self._fitted_horizon = horizon
            return self._fit_direct(df, horizon)

        # Recursive strategy
        df_feat = engineer_features(df, self.feature_config)

        exclude = {"unique_id", "ds", "y"}
        self._feature_cols = [c for c in df_feat.columns if c not in exclude]
        self._cat_features = self._get_categorical_features(df_feat)

        df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                         if c not in self._cat_features])

        X = df_feat[self._feature_cols].copy()
        y = df_feat["y"]

        # Convert categoricals to string for CatBoost (native handling)
        for col in self._cat_features:
            if col in X.columns:
                X[col] = X[col].astype(str)

        self._model = CatBoostRegressor(**self.cat_params)

        cat_feature_indices = [self._feature_cols.index(c) for c in self._cat_features
                               if c in self._feature_cols]

        self._model.fit(
            X, y,
            cat_features=cat_feature_indices if cat_feature_indices else None,
        )
        self._is_fitted = True

        return self

    def _fit_direct(self, df: pd.DataFrame, horizon: int) -> "CatBoostForecaster":
        """Fit separate models for each horizon step."""
        from catboost import CatBoostRegressor

        for h in range(1, horizon + 1):
            df_h = df.copy()
            df_h["y_target"] = df_h.groupby("unique_id")["y"].shift(-h)
            df_h = df_h.dropna(subset=["y_target"])

            df_feat = engineer_features(df_h, self.feature_config)

            if h == 1:
                exclude = {"unique_id", "ds", "y", "y_target"}
                self._feature_cols = [c for c in df_feat.columns if c not in exclude]
                self._cat_features = self._get_categorical_features(df_feat)

            df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                             if c not in self._cat_features])

            X = df_feat[self._feature_cols].copy()
            y = df_feat["y_target"]

            for col in self._cat_features:
                if col in X.columns:
                    X[col] = X[col].astype(str)

            model = CatBoostRegressor(**self.cat_params)

            cat_feature_indices = [self._feature_cols.index(c) for c in self._cat_features
                                   if c in self._feature_cols]

            model.fit(
                X, y,
                cat_features=cat_feature_indices if cat_feature_indices else None,
            )

            self._direct_models[h] = model

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.strategy == "direct":
            if self._fitted_horizon is None or horizon > self._fitted_horizon:
                raise ValueError(
                    f"Direct model was fitted for horizon={self._fitted_horizon}. "
                    f"Cannot predict horizon={horizon}."
                )
            return self._direct_predict(horizon)

        return self._recursive_predict(horizon)

    def _direct_predict(self, horizon: int) -> pd.DataFrame:
        """Direct multi-step forecasting."""
        last_dates = (
            self._last_train_data
            .groupby("unique_id")["ds"]
            .max()
            .reset_index()
        )

        all_forecasts = []
        df_feat = engineer_features(self._last_train_data, self.feature_config)
        latest_feat = df_feat.sort_values("ds").groupby("unique_id").tail(1).copy()

        for h in range(1, horizon + 1):
            model = self._direct_models[h]

            X = latest_feat[self._feature_cols].copy()
            for col in self._cat_features:
                if col in X.columns:
                    X[col] = X[col].astype(str)

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(0)

            preds = model.predict(X)
            preds = np.maximum(preds, 0)

            forecast_df = pd.DataFrame({
                "unique_id": latest_feat["unique_id"].values,
                "ds": last_dates["ds"] + pd.Timedelta(weeks=h),
                "yhat": preds,
            })
            all_forecasts.append(forecast_df)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _recursive_predict(self, horizon: int) -> pd.DataFrame:
        """Recursive multi-step forecasting."""
        last_dates = (
            self._last_train_data
            .groupby("unique_id")["ds"]
            .max()
            .reset_index()
        )

        all_forecasts = []
        history = self._last_train_data.copy()

        for h in range(horizon):
            future_rows = []
            for _, row in last_dates.iterrows():
                future_date = row["ds"] + pd.Timedelta(weeks=h + 1)
                future_rows.append({
                    "unique_id": row["unique_id"],
                    "ds": future_date,
                    "y": np.nan,
                })

            future_df = pd.DataFrame(future_rows)
            combined = pd.concat([history, future_df], ignore_index=True)
            combined = combined.sort_values(["unique_id", "ds"])

            combined_feat = engineer_features(combined, self.feature_config)
            future_feat = combined_feat[combined_feat["y"].isna()].copy()

            if future_feat.empty:
                break

            X_future = future_feat[self._feature_cols].copy()
            for col in self._cat_features:
                if col in X_future.columns:
                    X_future[col] = X_future[col].astype(str)

            numeric_cols = X_future.select_dtypes(include=[np.number]).columns
            X_future[numeric_cols] = X_future[numeric_cols].fillna(0)

            preds = self._model.predict(X_future)
            preds = np.maximum(preds, 0)

            forecast_df = future_feat[["unique_id", "ds"]].copy()
            forecast_df["yhat"] = preds
            all_forecasts.append(forecast_df)

            future_df["y"] = preds
            history = pd.concat([history, future_df], ignore_index=True)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _get_categorical_features(self, df: pd.DataFrame) -> list[str]:
        cat_cols = []
        hierarchy_cols = {"item_id", "store_id", "cat_id", "dept_id", "state_id",
                         "customer_id", "material_id", "category", "location"}

        for col in self._feature_cols:
            if col in hierarchy_cols or df[col].dtype == "object":
                cat_cols.append(col)

        return cat_cols

    def get_feature_importance(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        model = self._model if self.strategy == "recursive" else self._direct_models[1]
        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class ProphetForecaster(BaseForecaster):
    """Prophet-based forecaster for time series.

    Uses Facebook Prophet for trend and seasonality decomposition.
    Best for series with strong seasonality and holiday effects.

    Parameters
    ----------
    prophet_params : dict | None
        Prophet model parameters.
    weekly_seasonality : bool
        Include weekly seasonality.
    yearly_seasonality : bool
        Include yearly seasonality.

    Examples
    --------
    >>> model = ProphetForecaster(yearly_seasonality=True)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        prophet_params: dict | None = None,
        weekly_seasonality: bool = False,
        yearly_seasonality: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prophet_params = prophet_params or {}
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

        self._models: dict[str, Any] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """Fit Prophet models for each time series."""
        from prophet import Prophet

        self._validate_input(df)

        # Fit one model per series
        for uid in df["unique_id"].unique():
            series_df = df[df["unique_id"] == uid][["ds", "y"]].copy()
            series_df = series_df.sort_values("ds")

            model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                **self.prophet_params,
            )

            # Suppress Prophet's verbose output
            model.fit(series_df)

            self._models[uid] = model
            self._last_dates[uid] = series_df["ds"].max()

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts for each series."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        all_forecasts = []

        for uid, model in self._models.items():
            last_date = self._last_dates[uid]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=horizon,
                freq=DEFAULT_FREQ,
            )

            future_df = pd.DataFrame({"ds": future_dates})
            forecast = model.predict(future_df)

            result = pd.DataFrame({
                "unique_id": uid,
                "ds": forecast["ds"],
                "yhat": forecast["yhat"].clip(lower=0),  # Non-negative
            })
            all_forecasts.append(result)

        return pd.concat(all_forecasts, ignore_index=True)


class ProphetXGBoostHybrid(BaseForecaster):
    """Hybrid Prophet + XGBoost model.

    Uses Prophet for trend/seasonality decomposition, then XGBoost
    to predict residuals using engineered features.

    This approach captures:
    1. Prophet: Strong trend and seasonality patterns
    2. XGBoost: Complex non-linear patterns and feature interactions

    Parameters
    ----------
    feature_config : FeatureConfig | None
        Feature config for XGBoost residual model.
    prophet_params : dict | None
        Prophet parameters.
    xgb_params : dict | None
        XGBoost parameters for residual model.

    Examples
    --------
    >>> model = ProphetXGBoostHybrid()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        prophet_params: dict | None = None,
        xgb_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_config = feature_config or FeatureConfig()
        self.prophet_params = prophet_params or {}
        self.xgb_params = xgb_params or {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "verbosity": 0,
        }

        self._prophet_models: dict[str, Any] = {}
        self._xgb_model = None
        self._feature_cols: list[str] = []
        self._cat_features: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._last_train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "ProphetXGBoostHybrid":
        """Fit Prophet models, then XGBoost on residuals."""
        from prophet import Prophet
        import xgboost as xgb

        self._validate_input(df)
        self._last_train_data = df.copy()

        # Step 1: Fit Prophet models and get in-sample predictions
        prophet_preds = []

        for uid in df["unique_id"].unique():
            series_df = df[df["unique_id"] == uid][["ds", "y"]].copy()
            series_df = series_df.sort_values("ds")

            model = Prophet(
                weekly_seasonality=False,
                yearly_seasonality=True,
                **self.prophet_params,
            )
            model.fit(series_df)

            # Get in-sample predictions
            in_sample = model.predict(series_df[["ds"]])

            pred_df = pd.DataFrame({
                "unique_id": uid,
                "ds": in_sample["ds"],
                "prophet_pred": in_sample["yhat"],
            })
            prophet_preds.append(pred_df)
            self._prophet_models[uid] = model

        prophet_preds_df = pd.concat(prophet_preds, ignore_index=True)

        # Step 2: Compute residuals
        df_with_prophet = df.merge(prophet_preds_df, on=["unique_id", "ds"], how="left")
        df_with_prophet["residual"] = df_with_prophet["y"] - df_with_prophet["prophet_pred"]

        # Step 3: Engineer features for residual prediction
        df_feat = engineer_features(df_with_prophet, self.feature_config)

        exclude = {"unique_id", "ds", "y", "prophet_pred", "residual"}
        self._feature_cols = [c for c in df_feat.columns if c not in exclude]
        self._cat_features = self._get_categorical_features(df_feat)

        # Drop NaN
        df_feat = df_feat.dropna(subset=[c for c in self._feature_cols
                                         if c not in self._cat_features])

        X = df_feat[self._feature_cols].copy()
        y_residual = df_feat["residual"]

        # Encode categoricals
        X = self._encode_categoricals(X, fit=True)

        # Step 4: Train XGBoost on residuals
        self._xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self._xgb_model.fit(X, y_residual)

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts: Prophet + XGBoost residual."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        all_forecasts = []
        history = self._last_train_data.copy()

        # Get last dates per series
        last_dates = history.groupby("unique_id")["ds"].max().to_dict()

        for h in range(horizon):
            step_forecasts = []

            for uid, prophet_model in self._prophet_models.items():
                last_date = last_dates[uid]
                future_date = last_date + pd.Timedelta(weeks=h + 1)

                # Prophet prediction
                future_df = pd.DataFrame({"ds": [future_date]})
                prophet_forecast = prophet_model.predict(future_df)
                prophet_pred = prophet_forecast["yhat"].values[0]

                step_forecasts.append({
                    "unique_id": uid,
                    "ds": future_date,
                    "prophet_pred": prophet_pred,
                    "y": np.nan,
                })

            future_df = pd.DataFrame(step_forecasts)

            # Combine with history for feature engineering
            combined = pd.concat([history, future_df[["unique_id", "ds", "y"]]],
                                 ignore_index=True)
            combined = combined.sort_values(["unique_id", "ds"])

            combined_feat = engineer_features(combined, self.feature_config)
            future_feat = combined_feat[combined_feat["y"].isna()].copy()

            if future_feat.empty:
                break

            # XGBoost residual prediction
            X_future = future_feat[self._feature_cols].copy()
            X_future = self._encode_categoricals(X_future, fit=False)

            numeric_cols = X_future.select_dtypes(include=[np.number]).columns
            X_future[numeric_cols] = X_future[numeric_cols].fillna(0)

            residual_preds = self._xgb_model.predict(X_future)

            # Combine Prophet + residual
            future_df = future_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            final_preds = future_df["prophet_pred"].values + residual_preds
            final_preds = np.maximum(final_preds, 0)  # Non-negative

            forecast_df = future_feat[["unique_id", "ds"]].copy()
            forecast_df["yhat"] = final_preds
            all_forecasts.append(forecast_df)

            # Add to history
            future_update = future_df[["unique_id", "ds"]].copy()
            future_update["y"] = final_preds
            history = pd.concat([history, future_update], ignore_index=True)

        return pd.concat(all_forecasts, ignore_index=True)[["unique_id", "ds", "yhat"]]

    def _get_categorical_features(self, df: pd.DataFrame) -> list[str]:
        cat_cols = []
        hierarchy_cols = {"item_id", "store_id", "cat_id", "dept_id", "state_id",
                         "customer_id", "material_id", "category", "location"}

        for col in self._feature_cols:
            if col in hierarchy_cols or df[col].dtype == "object":
                cat_cols.append(col)

        return cat_cols

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()
        for col in self._cat_features:
            if col not in X.columns:
                continue

            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le:
                    X[col] = X[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return X
