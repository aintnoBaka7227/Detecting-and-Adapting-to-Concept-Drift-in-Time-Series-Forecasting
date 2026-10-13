from __future__ import annotations

import os

# 避免 PyTorch / XGBoost OpenMP 冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS

from drift_lab.config import SEASON_LENGTH
from drift_lab.forecasting.base import Forecaster
from drift_lab.forecasting.nixtla_common import (
    as_series,
    roll_forecast,
    to_nixtla_frame,
)


# -----------------------------
# NHITS architecture
# -----------------------------

_STACK_TYPES = ["identity", "identity", "identity"]

_N_BLOCKS = [1, 1, 1]

_MLP_UNITS = [
    [512, 512],
    [512, 512],
    [512, 512],
]

_POOLING_MODE = "MaxPool1d"
_INTERPOLATION_MODE = "linear"
_ACTIVATION = "ReLU"

_DROPOUT_PROB_THETA = 0.0

_WINDOWS_BATCH_SIZE = 1024
_STEP_SIZE = 1

_TRAINING_DATA_AVAILABILITY_THRESHOLD = 1.0


class BingqiNHITS(Forecaster):
    """
    Bingqi's independent NHITS implementation.

    Main setup:
    - half-hourly AEMO demand
    - 7-day forecast horizon
    - 14-day input window
    - stochastic training
    - random seed is meaningful
    - frozen model during test prediction
    """

    name = "bingqi_nhits"

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

        self._nf: NeuralForecast | None = None
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BingqiNHITS:
        """
        Train NHITS on the supplied training window.
        """

        self._history = as_series(y)
        self._target_name = y.name

        use_val = self.val_size > 0

        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,

            stack_types=_STACK_TYPES,
            n_blocks=_N_BLOCKS,
            mlp_units=_MLP_UNITS,

            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,

            pooling_mode=_POOLING_MODE,
            interpolation_mode=_INTERPOLATION_MODE,
            activation=_ACTIVATION,

            dropout_prob_theta=_DROPOUT_PROB_THETA,

            loss=MAE(),
            valid_loss=MAE(),

            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            num_lr_decays=self.num_lr_decays,

            early_stop_patience_steps=(
                self.early_stop_patience_steps
                if use_val
                else -1
            ),

            val_check_steps=self.val_check_steps,

            scaler_type=self.scaler_type,

            step_size=_STEP_SIZE,
            windows_batch_size=_WINDOWS_BATCH_SIZE,

            training_data_availability_threshold=(
                _TRAINING_DATA_AVAILABILITY_THRESHOLD
            ),

            random_seed=self.random_seed,
            enable_progress_bar=False,

            accelerator="gpu",
            devices=1,
        )

        self._nf = NeuralForecast(
            models=[model],
            freq=self.freq,
        )

        self._nf.fit(
            df=to_nixtla_frame(self._history),
            val_size=self.val_size,
        )

        return self

    def observe(self, y: pd.Series) -> None:
        """
        Add already-observed real values to the history.

        The NHITS weights are NOT retrained here.
        """

        if self._history is None:
            raise RuntimeError(
                "observe() called before fit()"
            )

        extra = as_series(
            y,
            name=self._history.name,
        )

        # 避免重复 timestamp
        extra = extra[
            ~extra.index.isin(self._history.index)
        ]

        self._history = pd.concat(
            [self._history, extra]
        ).sort_index()

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Rolling block prediction.

        Predictions are produced in horizon-sized chunks.

        If X contains the target column, real observed values are used
        as context only after that forecast block has already been issued.
        """

        if self._nf is None or self._history is None:
            raise RuntimeError(
                "predict() called before fit()"
            )

        original_order = pd.DatetimeIndex(X.index)

        sorted_index = original_order.sort_values()

        observed = None

        if (
            self._target_name is not None
            and self._target_name in X.columns
        ):
            observed = pd.Series(
                np.asarray(
                    X[self._target_name],
                    dtype=float,
                ),
                index=original_order,
            ).dropna()

        forecasts = roll_forecast(
            self._nf,
            self._history,
            sorted_index,
            self.horizon,
            self.input_size,
            observed,
        )

        return np.array(
            [
                forecasts[timestamp]
                for timestamp in original_order
            ],
            dtype=float,
        )