"""NHITS baseline (Nixtla `neuralforecast`) behind the Forecaster interface.

Direct multi-step neural forecaster: a stack of MLP blocks at different
pooling resolutions, each backcasting-and-subtracting what it explains
before passing the residual to the next block, with per-block forecasts
summed at the end. Trained once by gradient descent on the frozen training
window, then rolled forward through calibration/test in `horizon`-sized
chunks with no further weight updates — see
`forecasting/nixtla_common.py::roll_forecast` for the chunking adapter this
class is built on.

Unlike the other three baselines, training here is stochastic (random
weight init + minibatch order), so `random_seed` is a real, meaningful
hyperparameter — record it as the run's `seed`, don't fabricate NaN.

`observe()` and the real-values-as-context path in `predict()` are frozen
and causal: the weights never change after `fit()`, and a real value only
ever becomes context for a *later* forecast, after the current one is
issued. That's a rolling forecast, not leakage. It only becomes leakage if
a value from inside a forecast block is used to change that same block's
predictions.
"""

from __future__ import annotations

import os

# XGBoost and PyTorch each bring their own OpenMP runtime; sharing a process
# with XGBoostForecaster (e.g. the test suite, or a future run_*.py that
# uses both) deadlocks — sometimes segfaults — once both actually run
# parallel regions. Must be set before numpy/torch/xgboost do any real work,
# which import order guarantees here (both direct scripts and pytest
# collection import this module before any model's fit() executes).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE as _MAE
from neuralforecast.models import NHITS as _NHITSModel

from drift_lab.config import SEASON_LENGTH
from drift_lab.forecasting.base import Forecaster
from drift_lab.forecasting.nixtla_common import (
    as_series,
    roll_forecast,
    to_nixtla_frame,
)

# Fixed architecture for this baseline. Tune horizon / input_size / training
# via the constructor; edit these if the architecture itself needs to change.
_STACK_TYPES = ["identity", "identity", "identity"]
_N_BLOCKS = [1, 1, 1]
_MLP_UNITS = [[512, 512], [512, 512], [512, 512]]
_POOLING_MODE = "MaxPool1d"
_INTERPOLATION_MODE = "linear"
_ACTIVATION = "ReLU"
_DROPOUT_PROB_THETA = 0.0
_WINDOWS_BATCH_SIZE = 1024
_STEP_SIZE = 1
_TRAINING_DATA_AVAILABILITY_THRESHOLD = 1.0  # skip training windows with any NaN


class NHITSForecaster(Forecaster):
    name = "nhits"

    def __init__(
        self,
        # 7-day horizon: matches config.ROLLING_WINDOW_DAYS / the 7-day rolling
        # MAE used everywhere else, so every F1 curve point is a rolling mean
        # over forecasts that are all <= 1 week ahead, re-anchored weekly. Long
        # enough to expose a stale learned relationship, short enough to stay a
        # real forecast (not a climatology) and learnable from ~2 years of one
        # series. h=720 (the paper's number) would be 15 days here and would
        # underfit -> a separate long-horizon variant, not the default.
        horizon: int = SEASON_LENGTH * 7,
        input_size: int = SEASON_LENGTH * 14,  # 2x horizon -> context spans 2 weekly cycles
        # Multi-rate stacks, scaled for a ~1-week horizon on half-hourly data
        # (the paper's [2,2,1]/[4,2,1] are for short hourly horizons). With
        # horizon=336: coarse stack -> 7 pts (daily trend across the week),
        # mid -> 14 pts (twice-daily / duck-curve), fine -> native. Each entry
        # must divide `horizon` (freq) / `input_size` (pool).
        n_pool_kernel_size: tuple[int, int, int] = (16, 8, 1),
        n_freq_downsample: tuple[int, int, int] = (48, 24, 1),
        freq: str = "30min",
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        val_size: int = SEASON_LENGTH * 14,  # held out from train for early stopping
        early_stop_patience_steps: int = 5,
        val_check_steps: int = 50,
        # "robust" (median / IQR per window): safe on raw MW magnitudes.
        # "identity" only if the input is already standardised.
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
        self.val_size = val_size  # >0 holds out the last val_size points for early stopping
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.scaler_type = scaler_type
        self.random_seed = random_seed

        self._nf: NeuralForecast | None = None
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NHITSForecaster:
        """Train on the training window only. Returns self."""
        self._history = as_series(y)
        self._target_name = y.name

        use_val = self.val_size > 0
        model = _NHITSModel(
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
            loss=_MAE(),
            valid_loss=_MAE(),
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            num_lr_decays=self.num_lr_decays,
            early_stop_patience_steps=self.early_stop_patience_steps if use_val else -1,
            val_check_steps=self.val_check_steps,
            scaler_type=self.scaler_type,
            step_size=_STEP_SIZE,
            windows_batch_size=_WINDOWS_BATCH_SIZE,
            training_data_availability_threshold=_TRAINING_DATA_AVAILABILITY_THRESHOLD,
            random_seed=self.random_seed,
            enable_progress_bar=False,
            # Force CPU rather than PyTorch Lightning's auto-detected MPS:
            # sharing a process with XGBoost's joblib/OpenMP thread pool
            # (e.g. the full test suite) deadlocks Lightning's MPS setup.
            accelerator="cpu",
        )
        self._nf = NeuralForecast(models=[model], freq=self.freq)
        self._nf.fit(df=to_nixtla_frame(self._history), val_size=self.val_size)
        return self

    def observe(self, y: pd.Series) -> None:
        """Extend the context history with already-observed values (e.g.
        the calibration window). The frozen network is not retrained."""
        if self._history is None:
            raise RuntimeError("observe() called before fit()")
        extra = as_series(y, name=self._history.name)
        extra = extra[~extra.index.isin(self._history.index)]
        self._history = pd.concat([self._history, extra]).sort_index()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts, one per row of X, produced by rolling the
        frozen model forward in `horizon`-sized chunks (see
        `nixtla_common.roll_forecast`). If `X` contains a column named
        like `y`, its non-NaN values are used as real context between
        chunks — a frozen, causal rolling forecast, not leakage."""
        if self._nf is None or self._history is None:
            raise RuntimeError("predict() called before fit()")

        original_order = pd.DatetimeIndex(X.index)
        sorted_index = original_order.sort_values()

        observed = None
        if self._target_name is not None and self._target_name in X.columns:
            observed = pd.Series(
                np.asarray(X[self._target_name], dtype=float), index=original_order
            ).dropna()

        forecasts = roll_forecast(
            self._nf,
            self._history,
            sorted_index,
            self.horizon,
            self.input_size,
            observed,
        )
        return np.array([forecasts[timestamp] for timestamp in original_order])
