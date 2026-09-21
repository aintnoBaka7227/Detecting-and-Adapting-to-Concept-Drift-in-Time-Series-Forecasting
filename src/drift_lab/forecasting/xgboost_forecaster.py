"""Gradient-boosted forecaster on lag and calendar features (XGBoost).

Step 3 model: XGBoost regressor trained on features built from the
response series only. Lag values are fetched strictly before each
forecast point, so the model never looks ahead and can be fitted once on
the training window and then rolled forward through the test stream
untouched (see the leakage rule in docs/interfaces.md).
"""

import math
from collections import deque

import numpy as np
import pandas as pd
import xgboost as xgb

from drift_lab.forecasting.base import Forecaster


def _cyclical_columns(
    hour: np.ndarray, dow: np.ndarray, doy: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


class XGBoostForecaster(Forecaster):
    """Recursive XGBoost time-series forecaster.

    Features are built internally from the target series:

    - `lag_k`: the observed or previously forecast value exactly
      `k * interval` before the forecast point;
    - cyclical calendar columns (sin/cos of elapsed hour of day, day of
      week, and day of year) plus a weekend flag, derived purely from
      each row's timestamp;
    - `delta_*` momentum features (day-over-day / two-step changes);
    - `roll_mean`/`roll_std` over the most recent one-day window.

    `fit` receives the training window only. `predict` then forecasts the
    rows of `X` one at a time in chronological order, filling each lag
    from the stored training history where available and from its own
    earlier forecasts (recursive multi-step) otherwise. If `X` contains a
    column with the same name as `y` and that column holds non-NaN
    values, those observed values take precedence for lag filling — this
    supports one-step-ahead walk-forward evaluation with actual lags.

    Parameters
    ----------
    interval : str, default "30min"
        Temporal spacing between consecutive observations.
    max_lag : int, default 48
        Include all positive lags up to `max_lag` intervals back.
    extra_lags : tuple[int, ...], default (96, 144, 336)
        Additional lags beyond the contiguous range (e.g. 2-day, 3-day,
        weekly at 30-minute intervals).
    roll_window : int, default 48
        Rolling-mean/std context window in observations (default one day).
    n_estimators : int, default 1200
        Number of boosting rounds (early stopping may cut this short).
    max_depth : int, default 5
        Maximum tree depth.
    learning_rate : float, default 0.05
        Shrinkage applied to each round.
    subsample : float, default 0.8
        Fraction of rows sampled per boosting round.
    colsample_bytree : float, default 0.8
        Fraction of columns sampled per tree.
    min_child_weight : float, default 10
        Minimum sum of instance weights in a child node.
    gamma : float, default 0.3
        Minimum loss reduction required to make a further split.
    reg_alpha : float, default 0.5
        L1 regularization on leaf weights.
    reg_lambda : float, default 3
        L2 regularization on leaf weights.
    random_state : int, default 42
        Seed for the booster.
    early_stopping_rounds : int | None, default 30
        Stop boosting when the chronological validation slice stops
        improving. Disabled when the training window is too small.
    validation_fraction : float, default 0.15
        Fraction of the training window held out (from the end) for
        early stopping.
    """

    name = "xgboost"

    def __init__(
        self,
        interval: str = "30min",
        max_lag: int = 48,
        extra_lags: tuple[int, ...] = (96, 144, 336),
        roll_window: int = 48,
        n_estimators: int = 1200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 10,
        gamma: float = 0.3,
        reg_alpha: float = 0.5,
        reg_lambda: float = 3,
        random_state: int = 42,
        early_stopping_rounds: int | None = 30,
        validation_fraction: float = 0.15,
    ) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        if roll_window < 1:
            raise ValueError("roll_window must be >= 1")
        self.interval = interval
        self.max_lag = max_lag
        self.extra_lags = extra_lags
        self.roll_window = roll_window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

        self._interval = pd.Timedelta(interval)
        self._lags = sorted(set(range(1, max_lag + 1)) | set(extra_lags))
        self._momentum_lags = ("delta_1", "delta_48")
        self._calendar_lags = (
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "weekend",
        )
        self._rolling_lags = (f"roll_mean_{roll_window}", f"roll_std_{roll_window}")
        self._feature_columns = (
            [f"lag_{lag}" for lag in self._lags]
            + list(self._calendar_lags)
            + list(self._momentum_lags)
            + list(self._rolling_lags)
        )
        self._history: pd.Series | None = None
        self._target_name: str | None = None
        self._model: xgb.XGBRegressor | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "XGBoostForecaster":
        """Fit on the training window only. Returns self."""

        timestamps = self._as_datetime_index(X.index)

        y = y.reindex(timestamps)

        keep = y.notna()

        self._history = pd.Series(
            y[keep].to_numpy(),
            index=timestamps[keep],
            name=y.name,
        )
        self._target_name = y.name

        features = self._build_training_features(
            timestamps[keep],
            y[keep],
        )
        targets = y[keep].to_numpy()

        n = len(features)

        n_validation = int(n * self.validation_fraction)

        can_validate = (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and n_validation >= 20
            and (n - n_validation) >= n_validation
        )

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": 4,
        }

        if can_validate:
            params["callbacks"] = [
                xgb.callback.EarlyStopping(rounds=self.early_stopping_rounds)
            ]

        self._model = xgb.XGBRegressor(**params)

        if can_validate:
            split = n - n_validation
            self._model.fit(
                features.iloc[:split],
                targets[:split],
                eval_set=[
                    (
                        features.iloc[split:],
                        targets[split:],
                    )
                ],
                verbose=False,
            )
        else:
            self._model.fit(features, targets, verbose=False)

        return self

    def observe(self, y: pd.Series) -> None:
        """Extend the lag-lookup history with already-observed values (e.g. the
        calibration window) so the first test lags resolve across the
        train->test boundary. The frozen booster is not refitted."""

        if self._history is None:
            raise RuntimeError("observe() called before fit()")

        extra = pd.Series(
            np.asarray(y, dtype=float),
            index=self._as_datetime_index(y.index),
            name=self._history.name,
        )
        extra = extra[~extra.index.isin(self._history.index)]
        self._history = pd.concat([self._history, extra])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts, one per row of X, produced chronologically."""

        if self._model is None or self._history is None:
            raise RuntimeError("predict() called before fit()")

        original_order = self._as_datetime_index(X.index)

        timestamps = original_order.sort_values()

        actuals = None
        if self._target_name is not None and self._target_name in X.columns:
            actuals = dict(
                zip(
                    original_order,
                    X[self._target_name].to_numpy(),
                )
            )

        values = self._history.copy(deep=False)

        forecasts = {}

        feature_row = np.empty(
            (1, len(self._feature_columns)),
            dtype=np.float64,
        )

        columns = {name: index for index, name in enumerate(self._feature_columns)}

        recent = self._seed_rolling_window(values, timestamps)

        for timestamp in timestamps:
            cutoff = timestamp - self.roll_window * self._interval
            while recent and recent[0][0] <= cutoff:
                recent.popleft()

            lag_values = {}
            for lag in self._lags:
                lag_time = timestamp - lag * self._interval

                value = self._value_at(
                    lag_time,
                    actuals,
                    values,
                )
                lag_values[lag] = value

                feature_row[0, columns[f"lag_{lag}"]] = value

            hour = timestamp.hour + timestamp.minute / 60
            dow = timestamp.dayofweek + hour / 24
            doy = timestamp.timetuple().tm_yday - 1 + hour / 24

            for name, value in _cyclical_columns(
                np.array([hour]),
                np.array([dow]),
                np.array([doy]),
            ).items():
                feature_row[0, columns[name]] = value[0]

            feature_row[0, columns["weekend"]] = (
                1.0 if timestamp.dayofweek >= 5 else 0.0
            )

            lag_1 = lag_values.get(1, np.nan)
            lag_2 = lag_values.get(2, np.nan)
            lag_48 = lag_values.get(48, np.nan)
            lag_96 = lag_values.get(96, np.nan)

            feature_row[0, columns["delta_1"]] = lag_1 - lag_2
            feature_row[0, columns["delta_48"]] = lag_48 - lag_96

            roll_mean, roll_std = self._rolling_stats(recent)
            feature_row[0, columns[f"roll_mean_{self.roll_window}"]] = roll_mean
            feature_row[0, columns[f"roll_std_{self.roll_window}"]] = roll_std

            forecast = self._model.predict(feature_row)[0]

            forecasts[timestamp] = forecast

            if timestamp not in values.index:
                values.loc[timestamp] = forecast

            recent_value = forecast
            if actuals is not None:
                actual_value = actuals.get(timestamp)
                if actual_value is not None and not pd.isna(actual_value):
                    recent_value = float(actual_value)
            recent.append((timestamp, recent_value))

        return np.array([forecasts[timestamp] for timestamp in original_order])

    def _seed_rolling_window(
        self,
        values: pd.Series,
        timestamps: pd.DatetimeIndex,
    ) -> deque:
        if len(timestamps) == 0:
            return deque()
        first = timestamps[0]
        prefix = values.loc[: first - self._interval].tail(self.roll_window)
        return deque(
            (index, float(value))
            for index, value in prefix.items()
            if not pd.isna(value)
        )

    @staticmethod
    def _rolling_stats(recent: deque) -> tuple[float, float]:
        if not recent:
            return np.nan, np.nan
        mean = sum(value for _, value in recent) / len(recent)
        if len(recent) < 2:
            return mean, np.nan
        variance = sum((value - mean) ** 2 for _, value in recent) / (len(recent) - 1)
        return mean, math.sqrt(variance)

    def _build_training_features(
        self,
        timestamps: pd.DatetimeIndex,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Lag + calendar + momentum + rolling features for the training window."""

        lags = pd.DataFrame(index=timestamps)

        for lag in self._lags:
            offset = lag * self._interval

            lags[f"lag_{lag}"] = y.reindex(timestamps - offset).to_numpy()

        hour = timestamps.hour + timestamps.minute / 60
        dow = timestamps.dayofweek + hour / 24
        doy = timestamps.dayofyear - 1 + hour / 24

        for name, value in _cyclical_columns(hour, dow, doy).items():
            lags[name] = value

        lags["weekend"] = (timestamps.dayofweek >= 5).astype(int)

        lags["delta_1"] = lags["lag_1"].to_numpy() - lags["lag_2"].to_numpy()
        lags["delta_48"] = lags["lag_48"].to_numpy() - lags["lag_96"].to_numpy()

        arr = y.to_numpy()
        n = len(arr)
        cum_sum = np.concatenate([[0.0], np.cumsum(arr)])
        cum_sq = np.concatenate([[0.0], np.cumsum(arr**2)])
        window = self.roll_window
        indices = np.arange(n)
        count = np.minimum(indices + 1, window)
        win_sum = cum_sum[indices + 1] - cum_sum[np.maximum(indices + 1 - window, 0)]
        win_sq = cum_sq[indices + 1] - cum_sq[np.maximum(indices + 1 - window, 0)]
        roll_mean = win_sum / count
        roll_var = (win_sq - win_sum**2 / count) / np.maximum(count - 1, 1)
        roll_std = np.sqrt(np.where(count >= 2, roll_var, np.nan))

        lags[f"roll_mean_{window}"] = pd.Series(roll_mean, index=y.index).shift(1)
        lags[f"roll_std_{window}"] = pd.Series(roll_std, index=y.index).shift(1)

        return lags

    @staticmethod
    def _value_at(
        lag_time: pd.Timestamp,
        actuals: dict | None,
        values: pd.Series,
    ) -> float:
        """Value at `lag_time`, preferring observed to forecast values."""

        if actuals is not None:
            actual = actuals.get(lag_time)
            if actual is not None and not pd.isna(actual):
                return float(actual)

        if lag_time in values.index:
            return values.loc[lag_time]

        return np.nan

    @staticmethod
    def _as_datetime_index(index) -> pd.DatetimeIndex:
        numeric = pd.api.types.is_numeric_dtype(pd.Series(index))

        if numeric:
            raise ValueError(
                "An X index of datetime values is required to build lag features"
            )

        try:
            return pd.DatetimeIndex(index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "An X index of datetime values is required to build lag features"
            ) from error
