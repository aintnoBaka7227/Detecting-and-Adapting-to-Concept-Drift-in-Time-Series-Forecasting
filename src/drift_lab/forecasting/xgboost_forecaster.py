"""Gradient-boosted forecaster on lag and calendar features (XGBoost).

Step 3 model: XGBoost regressor trained on features built from the
response series only. Lag values are fetched strictly before each
forecast point, so the model never looks ahead and can be fitted once on
the training window and then rolled forward through the test stream
untouched (see the leakage rule in docs/interfaces.md).
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from drift_lab.forecasting.base import Forecaster

_CALENDAR_COLUMNS = (
    "hour",
    "minute",
    "day_of_week",
    "month",
)


class XGBoostForecaster(Forecaster):
    """Recursive XGBoost time-series forecaster.

    Features are built internally from the target series:

    - `lag_k`: the observed or previously forecast value exactly
      `k * interval` before the forecast point;
    - calendar columns derived from each row's timestamp.

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
    n_estimators : int, default 300
        Number of boosting rounds.
    max_depth : int, default 5
        Maximum tree depth.
    learning_rate : float, default 0.05
        Shrinkage applied to each round.
    random_state : int, default 42
        Seed for the booster.
    early_stopping_rounds : int | None, default 10
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
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
        early_stopping_rounds: int | None = 10,
        validation_fraction: float = 0.15,
    ) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        self.interval = interval
        self.max_lag = max_lag
        self.extra_lags = extra_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

        self._interval = pd.Timedelta(interval)
        self._lags = sorted(set(range(1, max_lag + 1)) | set(extra_lags))
        self._feature_columns = [f"lag_{lag}" for lag in self._lags] + list(
            _CALENDAR_COLUMNS
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
            "subsample": 0.8,
            "colsample_bytree": 0.8,
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

        for timestamp in timestamps:
            for lag in self._lags:
                lag_time = timestamp - lag * self._interval

                value = self._value_at(
                    lag_time,
                    actuals,
                    values,
                )

                feature_row[0, columns[f"lag_{lag}"]] = value

            feature_row[0, columns["hour"]] = timestamp.hour
            feature_row[0, columns["minute"]] = timestamp.minute
            feature_row[0, columns["day_of_week"]] = timestamp.dayofweek
            feature_row[0, columns["month"]] = timestamp.month

            forecast = self._model.predict(feature_row)[0]

            forecasts[timestamp] = forecast

            if timestamp not in values.index:
                values.loc[timestamp] = forecast

        return np.array([forecasts[timestamp] for timestamp in original_order])

    def _build_training_features(
        self,
        timestamps: pd.DatetimeIndex,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Lag + calendar features for the training window."""

        lags = pd.DataFrame(index=timestamps)

        for lag in self._lags:
            offset = lag * self._interval

            lags[f"lag_{lag}"] = y.reindex(timestamps - offset).to_numpy()

        calendar = pd.DataFrame(
            {
                "hour": timestamps.hour,
                "minute": timestamps.minute,
                "day_of_week": timestamps.dayofweek,
                "month": timestamps.month,
            },
            index=timestamps,
        )

        return pd.concat([lags, calendar], axis=1)

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
