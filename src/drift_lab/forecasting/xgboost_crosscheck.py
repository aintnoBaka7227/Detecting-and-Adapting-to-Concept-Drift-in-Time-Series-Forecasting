"""Independent XGBoost forecaster for baseline cross-checking.

Builds lag and calendar features from the demand series and supports
walk-forward prediction without using future observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from drift_lab.forecasting.base import Forecaster


class XGBoostCrosscheck(Forecaster):
    name = "xgboost_crosscheck"

    def __init__(
        self,
        max_lag: int = 48,
        extra_lags: tuple[int, ...] = (96, 144, 336),
        interval: str = "30min",
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
        early_stopping_rounds: int = 10,
        validation_fraction: float = 0.15,
    ) -> None:
        self.max_lag = max_lag
        self.extra_lags = extra_lags
        self.interval = pd.Timedelta(interval)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

        self.lags = sorted(set(range(1, max_lag + 1)) | set(extra_lags))

        self._model: xgb.XGBRegressor | None = None
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def _calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hour": index.hour,
                "minute": index.minute,
                "day_of_week": index.dayofweek,
                "month": index.month,
            },
            index=index,
        )

    def _training_features(
        self,
        index: pd.DatetimeIndex,
        y: pd.Series,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=index)

        for lag in self.lags:
            lag_index = index - lag * self.interval
            features[f"lag_{lag}"] = y.reindex(lag_index).to_numpy()

        return pd.concat(
            [features, self._calendar_features(index)],
            axis=1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostCrosscheck:
        index = pd.DatetimeIndex(X.index)

        if len(index) != len(y):
            raise ValueError("X and y must have the same number of observations.")

        target = pd.Series(
            np.asarray(y, dtype=float),
            index=index,
            name=y.name,
        )

        self._history = target.copy()
        self._target_name = y.name

        features = self._training_features(index, target)

        n_samples = len(features)
        n_validation = int(n_samples * self.validation_fraction)

        can_validate = (
            self.early_stopping_rounds > 0
            and n_validation >= 20
            and (n_samples - n_validation) >= n_validation
        )

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": 4,
        }

        if can_validate:
            params["callbacks"] = [
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                )
            ]

        self._model = xgb.XGBRegressor(**params)

        targets = target.to_numpy()

        if can_validate:
            split = n_samples - n_validation
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
        if self._history is None:
            raise RuntimeError("observe() called before fit()")

        observed = pd.Series(
            np.asarray(y, dtype=float),
            index=pd.DatetimeIndex(y.index),
            name=self._target_name,
        )

        observed = observed[~observed.index.isin(self._history.index)]
        self._history = pd.concat([self._history, observed])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._history is None:
            raise RuntimeError("predict() called before fit()")

        original_index = pd.DatetimeIndex(X.index)
        ordered_index = original_index.sort_values()

        observed_values: dict[pd.Timestamp, float] = {}

        if self._target_name is not None and self._target_name in X.columns:
            for timestamp, value in zip(
                original_index,
                X[self._target_name].to_numpy(),
            ):
                if not pd.isna(value):
                    observed_values[timestamp] = float(value)

        history = self._history.copy()
        forecasts: dict[pd.Timestamp, float] = {}

        feature_names = [f"lag_{lag}" for lag in self.lags] + [
            "hour",
            "minute",
            "day_of_week",
            "month",
        ]

        for timestamp in ordered_index:
            row: dict[str, float] = {}

            for lag in self.lags:
                lag_time = timestamp - lag * self.interval

                if lag_time in observed_values:
                    value = observed_values[lag_time]
                elif lag_time in history.index:
                    value = float(history.loc[lag_time])
                else:
                    value = np.nan

                row[f"lag_{lag}"] = value

            row["hour"] = timestamp.hour
            row["minute"] = timestamp.minute
            row["day_of_week"] = timestamp.dayofweek
            row["month"] = timestamp.month

            feature_row = pd.DataFrame(
                [[row[name] for name in feature_names]],
                columns=feature_names,
            )

            prediction = float(self._model.predict(feature_row)[0])

            forecasts[timestamp] = prediction

            if timestamp not in history.index:
                history.loc[timestamp] = prediction

        return np.asarray(
            [forecasts[timestamp] for timestamp in original_index],
            dtype=float,
        )


