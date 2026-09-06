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
from neuralforecast.models import NHITS as _NHITSModel

from drift_lab.config import SEASON_LENGTH
from drift_lab.forecasting.base import Forecaster
from drift_lab.forecasting.nixtla_common import (
    as_series,
    roll_forecast,
    to_nixtla_frame,
)


class NHITSForecaster(Forecaster):
    name = "nhits"

    def __init__(
        self,
        horizon: int = SEASON_LENGTH,
        input_size: int = SEASON_LENGTH * 7,
        freq: str = "30min",
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        random_seed: int = 1,
    ) -> None:
        self.horizon = horizon
        self.input_size = input_size
        self.freq = freq
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self._nf: NeuralForecast | None = None
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NHITSForecaster:
        """Train on the training window only. Returns self."""
        self._history = as_series(y)
        self._target_name = y.name

        model = _NHITSModel(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            enable_progress_bar=False,
            # Force CPU rather than PyTorch Lightning's auto-detected MPS:
            # sharing a process with XGBoost's joblib/OpenMP thread pool
            # (e.g. the full test suite) deadlocks Lightning's MPS setup.
            # Training cost isn't a concern here, so CPU-only sidesteps it.
            accelerator="cpu",
        )
        self._nf = NeuralForecast(models=[model], freq=self.freq)
        self._nf.fit(df=to_nixtla_frame(self._history))
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
        chunks (walk-forward) instead of the model's own forecasts."""
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
