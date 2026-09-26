"""Independent NHITS forecaster implementation by Aditya."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS

from drift_lab.config import SEASON_LENGTH
from drift_lab.forecasting.base import Forecaster
from drift_lab.forecasting.nixtla_common_aditya import (
    as_series,
    roll_forecast,
    to_nixtla_frame,
)


class NHITSForecaster(Forecaster):
    """Frozen direct multi-step NHITS model with causal rolling inference."""

    name = "nhits_aditya"

    def __init__(
        self,
        horizon: int = SEASON_LENGTH * 7,
        input_size: int = SEASON_LENGTH * 14,
        n_pool_kernel_size: tuple[int, int, int] = (16, 8, 1),
        n_freq_downsample: tuple[int, int, int] = (48, 24, 1),
        freq: str = "30min",
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        val_size: int = SEASON_LENGTH * 14,
        early_stop_patience_steps: int = 5,
        val_check_steps: int = 50,
        scaler_type: str = "robust",
        random_seed: int = 1,
    ) -> None:
        if horizon < 1 or input_size < 1:
            raise ValueError("horizon and input_size must be >= 1")
        self.horizon = horizon
        self.input_size = input_size
        self.n_pool_kernel_size = list(n_pool_kernel_size)
        self.n_freq_downsample = list(n_freq_downsample)
        self.freq = freq
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.val_size = val_size
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self._neural_forecast: NeuralForecast | None = None
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NHITSForecaster:
        """Fit once on the supplied training window."""
        del X
        self._history = as_series(y)
        self._target_name = y.name
        use_validation = self.val_size > 0
        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,
            stack_types=["identity", "identity", "identity"],
            n_blocks=[1, 1, 1],
            mlp_units=[[512, 512], [512, 512], [512, 512]],
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            pooling_mode="MaxPool1d",
            interpolation_mode="linear",
            activation="ReLU",
            dropout_prob_theta=0.0,
            loss=MAE(),
            valid_loss=MAE(),
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            num_lr_decays=self.num_lr_decays,
            early_stop_patience_steps=(
                self.early_stop_patience_steps if use_validation else -1
            ),
            val_check_steps=self.val_check_steps,
            scaler_type=self.scaler_type,
            step_size=1,
            windows_batch_size=1024,
            training_data_availability_threshold=1.0,
            random_seed=self.random_seed,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        self._neural_forecast = NeuralForecast(
            models=[model], freq=self.freq
        )
        self._neural_forecast.fit(
            df=to_nixtla_frame(self._history), val_size=self.val_size
        )
        return self

    def observe(self, y: pd.Series) -> None:
        """Append already-observed values without retraining."""
        if self._history is None:
            raise RuntimeError("observe() called before fit()")
        extra = as_series(y, name=self._history.name)
        extra = extra[~extra.index.isin(self._history.index)]
        self._history = pd.concat([self._history, extra]).sort_index()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return causal rolling forecasts in the caller's row order."""
        if self._neural_forecast is None or self._history is None:
            raise RuntimeError("predict() called before fit()")

        original_index = pd.DatetimeIndex(X.index)
        sorted_index = original_index.sort_values()
        observed = None
        if self._target_name is not None and self._target_name in X.columns:
            observed = pd.Series(
                np.asarray(X[self._target_name], dtype=float),
                index=original_index,
            ).dropna()

        forecasts = roll_forecast(
            self._neural_forecast,
            self._history,
            sorted_index,
            self.horizon,
            self.input_size,
            observed,
        )
        return np.asarray([forecasts[timestamp] for timestamp in original_index])
