"""Neural Network Forecasting Models.

Wrappers for Nixtla's NeuralForecast library providing state-of-the-art
deep learning models for time series forecasting.

Models included:
- N-BEATS: Neural Basis Expansion Analysis (M4 competition winner)
- NHITS: Hierarchical interpolation for efficient long-horizon forecasting
- DeepAR: Amazon's probabilistic autoregressive model
- TFT: Temporal Fusion Transformer (Google, state-of-the-art interpretable)
- LSTM/GRU: Classic recurrent architectures

Based on research from:
- M5 Kaggle competition (2nd place used N-BEATS)
- Google's TFT paper (36-69% improvement over DeepAR)
- Nixtla NeuralForecast library
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ds_timeseries.models.base import BaseForecaster


@dataclass
class NeuralConfig:
    """Configuration for neural network models.

    Parameters
    ----------
    input_size : int
        Number of historical time steps to use as input.
    hidden_size : int
        Size of hidden layers.
    max_steps : int
        Maximum training steps (epochs).
    learning_rate : float
        Learning rate for optimizer.
    batch_size : int
        Training batch size.
    val_check_steps : int
        Steps between validation checks.
    early_stop_patience : int
        Patience for early stopping.
    accelerator : str
        Hardware accelerator ("cpu", "gpu", "auto").
    random_seed : int
        Random seed for reproducibility.
    """
    input_size: int = 52  # 1 year of weekly data
    hidden_size: int = 64
    max_steps: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 32
    val_check_steps: int = 50
    early_stop_patience: int = 5
    accelerator: str = "cpu"
    random_seed: int = 42
    scaler_type: str = "standard"
    num_workers: int = 0


class NBEATSForecaster(BaseForecaster):
    """N-BEATS Neural Basis Expansion forecaster.

    N-BEATS is a pure deep learning approach that achieved state-of-the-art
    results in the M4 competition. It uses backward and forward residual
    links with fully-connected layers.

    Features:
    - Interpretable: Decomposes into trend and seasonality
    - No feature engineering required
    - Handles multiple series via cross-learning

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Neural network configuration.
    stack_types : list[str]
        Stack types: "identity", "trend", "seasonality".

    Examples
    --------
    >>> model = NBEATSForecaster(horizon=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    References
    ----------
    - Original paper: https://arxiv.org/abs/1905.10437
    - Beat M4 winner by 3%
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        stack_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.stack_types = stack_types or ["identity", "trend", "seasonality"]

        self._model = None
        self._nf = None  # NeuralForecast instance

    def fit(self, df: pd.DataFrame) -> "NBEATSForecaster":
        """Fit N-BEATS model."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        self._validate_input(df)

        # Prepare data in NeuralForecast format
        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model = NBEATS(
            h=self.horizon,
            input_size=self.config.input_size,
            stack_types=self.stack_types,
            n_blocks=[1, 1, 1],
            mlp_units=[[self.config.hidden_size, self.config.hidden_size]] * 3,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            val_check_steps=self.config.val_check_steps,
            early_stop_patience_steps=self.config.early_stop_patience,
            scaler_type=self.config.scaler_type,
            accelerator=self.config.accelerator,
            random_seed=self.config.random_seed,
            num_workers_loader=self.config.num_workers,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"N-BEATS was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        # Rename prediction column
        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class NHITSForecaster(BaseForecaster):
    """NHITS (Neural Hierarchical Interpolation) forecaster.

    NHITS improves on N-BEATS with hierarchical interpolation for
    efficient long-horizon forecasting. Published at AAAI 2023.

    Features:
    - More efficient than N-BEATS for long horizons
    - Multi-rate data sampling
    - Hierarchical interpolation

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Neural network configuration.
    n_pool_kernel_size : list[int]
        Pooling kernel sizes for each stack.

    Examples
    --------
    >>> model = NHITSForecaster(horizon=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        n_pool_kernel_size: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.n_pool_kernel_size = n_pool_kernel_size or [2, 2, 1]

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "NHITSForecaster":
        """Fit NHITS model."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model = NHITS(
            h=self.horizon,
            input_size=self.config.input_size,
            n_pool_kernel_size=self.n_pool_kernel_size,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            val_check_steps=self.config.val_check_steps,
            early_stop_patience_steps=self.config.early_stop_patience,
            scaler_type=self.config.scaler_type,
            accelerator=self.config.accelerator,
            random_seed=self.config.random_seed,
            num_workers_loader=self.config.num_workers,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"NHITS was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class DeepARForecaster(BaseForecaster):
    """DeepAR probabilistic autoregressive forecaster.

    DeepAR is Amazon's autoregressive RNN model that produces
    probabilistic forecasts. It learns from multiple related time series.

    Features:
    - Probabilistic forecasts (prediction intervals)
    - Handles missing values
    - Cross-learning across series

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Neural network configuration.
    num_layers : int
        Number of LSTM layers.

    Examples
    --------
    >>> model = DeepARForecaster(horizon=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        num_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.num_layers = num_layers

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "DeepARForecaster":
        """Fit DeepAR model."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model = DeepAR(
            h=self.horizon,
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            n_layers=self.num_layers,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            val_check_steps=self.config.val_check_steps,
            early_stop_patience_steps=self.config.early_stop_patience,
            scaler_type=self.config.scaler_type,
            accelerator=self.config.accelerator,
            random_seed=self.config.random_seed,
            num_workers_loader=self.config.num_workers,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"DeepAR was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer forecaster.

    TFT is Google's state-of-the-art interpretable model that achieved
    36-69% improvement over DeepAR in benchmarks. It provides attention-
    based interpretability.

    Features:
    - Interpretable attention mechanism
    - Handles static, known, and unknown features
    - Multi-horizon forecasting
    - Variable selection networks

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Neural network configuration.
    num_heads : int
        Number of attention heads.

    Examples
    --------
    >>> model = TFTForecaster(horizon=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)

    References
    ----------
    - Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon
      Time Series Forecasting" (Google, 2019)
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.num_heads = num_heads

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "TFTForecaster":
        """Fit TFT model."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model = TFT(
            h=self.horizon,
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            n_head=self.num_heads,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            val_check_steps=self.config.val_check_steps,
            early_stop_patience_steps=self.config.early_stop_patience,
            scaler_type=self.config.scaler_type,
            accelerator=self.config.accelerator,
            random_seed=self.config.random_seed,
            num_workers_loader=self.config.num_workers,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"TFT was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class LSTMForecaster(BaseForecaster):
    """LSTM (Long Short-Term Memory) forecaster.

    Classic recurrent architecture for sequence modeling.

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Neural network configuration.
    num_layers : int
        Number of LSTM layers.

    Examples
    --------
    >>> model = LSTMForecaster(horizon=4)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        num_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.num_layers = num_layers

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "LSTMForecaster":
        """Fit LSTM model."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import LSTM

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model = LSTM(
            h=self.horizon,
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            n_layers=self.num_layers,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            val_check_steps=self.config.val_check_steps,
            early_stop_patience_steps=self.config.early_stop_patience,
            scaler_type=self.config.scaler_type,
            accelerator=self.config.accelerator,
            random_seed=self.config.random_seed,
            num_workers_loader=self.config.num_workers,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"LSTM was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class AutoNeuralForecaster(BaseForecaster):
    """Auto-tuning neural forecaster.

    Uses NeuralForecast's Auto models to automatically tune hyperparameters.
    Supports N-BEATS, NHITS, LSTM, and TFT with Optuna-based tuning.

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    model_type : str
        Model type: "nbeats", "nhits", "lstm", "deepar".
    config : NeuralConfig | None
        Base configuration (some params will be tuned).
    n_trials : int
        Number of Optuna trials for hyperparameter search.

    Examples
    --------
    >>> model = AutoNeuralForecaster(horizon=4, model_type="nhits", n_trials=20)
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        horizon: int = 4,
        model_type: Literal["nbeats", "nhits", "lstm", "deepar"] = "nhits",
        config: NeuralConfig | None = None,
        n_trials: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.model_type = model_type
        self.config = config or NeuralConfig()
        self.n_trials = n_trials

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "AutoNeuralForecaster":
        """Fit auto-tuned neural model."""
        from neuralforecast import NeuralForecast

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        # Import the appropriate Auto model
        if self.model_type == "nbeats":
            from neuralforecast.auto import AutoNBEATS as AutoModel
        elif self.model_type == "nhits":
            from neuralforecast.auto import AutoNHITS as AutoModel
        elif self.model_type == "lstm":
            from neuralforecast.auto import AutoLSTM as AutoModel
        elif self.model_type == "deepar":
            from neuralforecast.auto import AutoDeepAR as AutoModel
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        model = AutoModel(
            h=self.horizon,
            num_samples=self.n_trials,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"Model was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]
        forecasts = forecasts.rename(columns={pred_col: "yhat"})
        forecasts["yhat"] = forecasts["yhat"].clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]


class NeuralEnsembleForecaster(BaseForecaster):
    """Ensemble of multiple neural network models.

    Trains multiple neural architectures and averages their predictions.
    This follows the M5 competition insight that ensembles of diverse
    models outperform single models.

    Parameters
    ----------
    horizon : int
        Forecast horizon.
    config : NeuralConfig | None
        Shared neural network configuration.
    models : list[str] | None
        Models to include: "nbeats", "nhits", "lstm", "deepar".
        Defaults to ["nbeats", "nhits"].

    Examples
    --------
    >>> model = NeuralEnsembleForecaster(horizon=4, models=["nbeats", "nhits", "lstm"])
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=4)
    """

    def __init__(
        self,
        horizon: int = 4,
        config: NeuralConfig | None = None,
        models: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.config = config or NeuralConfig()
        self.model_names = models or ["nbeats", "nhits"]

        self._nf = None

    def fit(self, df: pd.DataFrame) -> "NeuralEnsembleForecaster":
        """Fit ensemble of neural models."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS, NHITS, LSTM, DeepAR

        self._validate_input(df)

        df_nf = df[["unique_id", "ds", "y"]].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"])

        model_instances = []

        model_classes = {
            "nbeats": lambda: NBEATS(
                h=self.horizon,
                input_size=self.config.input_size,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                scaler_type=self.config.scaler_type,
                accelerator=self.config.accelerator,
                random_seed=self.config.random_seed,
                num_workers_loader=self.config.num_workers,
            ),
            "nhits": lambda: NHITS(
                h=self.horizon,
                input_size=self.config.input_size,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                scaler_type=self.config.scaler_type,
                accelerator=self.config.accelerator,
                random_seed=self.config.random_seed,
                num_workers_loader=self.config.num_workers,
            ),
            "lstm": lambda: LSTM(
                h=self.horizon,
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                scaler_type=self.config.scaler_type,
                accelerator=self.config.accelerator,
                random_seed=self.config.random_seed,
                num_workers_loader=self.config.num_workers,
            ),
            "deepar": lambda: DeepAR(
                h=self.horizon,
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                scaler_type=self.config.scaler_type,
                accelerator=self.config.accelerator,
                random_seed=self.config.random_seed,
                num_workers_loader=self.config.num_workers,
            ),
        }

        for name in self.model_names:
            if name not in model_classes:
                raise ValueError(f"Unknown model: {name}. Choose from {list(model_classes.keys())}")
            model_instances.append(model_classes[name]())

        self._nf = NeuralForecast(
            models=model_instances,
            freq="W-MON",
        )

        self._nf.fit(df_nf)
        self._is_fitted = True

        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate ensemble forecasts (average of all models)."""
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon != self.horizon:
            raise ValueError(
                f"Ensemble was fitted for horizon={self.horizon}. "
                f"Cannot predict horizon={horizon}."
            )

        forecasts = self._nf.predict()
        forecasts = forecasts.reset_index()

        # Average all model predictions
        pred_cols = [c for c in forecasts.columns if c not in ["unique_id", "ds"]]
        forecasts["yhat"] = forecasts[pred_cols].mean(axis=1).clip(lower=0)

        return forecasts[["unique_id", "ds", "yhat"]]
