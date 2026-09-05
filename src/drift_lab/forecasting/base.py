"""Forecaster contract every baseline and model must satisfy.

Not one of the three contracts fixed on the architecture slide, but
required by it: `adapter(changepoints, model, data) -> updated model`
only works if every model exposes the same fit/predict shape, regardless
of whether it's seasonal-naive, a classical model, or gradient boosting.
Frozen for the same reason and on the same schedule as the other three.

How to implement a model (Sprint 2, Step 3)
-----------------------------------------------
Add a new file next to this one (e.g. `seasonal_naive.py`), subclass
`Forecaster`, set `name`, and implement `fit` / `predict`. Override
`observe` if the model resolves lags from a running history (see
`seasonal_naive.py`, `xgboost_forecaster.py`).

Step 3 needs three of these (seasonal-naive, one classical model, one
gradient-boosted model on lag features), each fitted on the training
window only then rolled forward through the test stream untouched.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = ""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        """Fit on data available strictly before the forecast point. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts, one per row of X."""
        raise NotImplementedError

    def observe(self, y: pd.Series) -> None:
        """Add already-observed values to the model's history without
        retraining. Lets `predict` resolve lags across the train->test
        boundary (e.g. the calibration window). Default: no-op."""
