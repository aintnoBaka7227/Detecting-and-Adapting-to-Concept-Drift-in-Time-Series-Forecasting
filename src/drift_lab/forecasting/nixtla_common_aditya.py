"""Small Nixtla adapter used by the independent Aditya NHITS forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIQUE_ID = "series"


def as_series(values: pd.Series, name: str | None = None) -> pd.Series:
    """Return numeric values on a datetime index."""
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.DatetimeIndex(values.index),
        name=name if name is not None else getattr(values, "name", None),
    )


def to_nixtla_frame(values: pd.Series) -> pd.DataFrame:
    """Convert a pandas series to NeuralForecast's long format."""
    return pd.DataFrame(
        {
            "unique_id": UNIQUE_ID,
            "ds": values.index,
            "y": values.to_numpy(),
        }
    )


def roll_forecast(
    model,
    history: pd.Series,
    timestamps: pd.DatetimeIndex,
    horizon: int,
    input_size: int,
    observed: pd.Series | None = None,
    chunk_size: int | None = None,
) -> dict[pd.Timestamp, float]:
    """Forecast with `horizon`-step outputs, refreshing context every chunk."""
    if chunk_size is None:
        chunk_size = horizon
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunk_size = min(chunk_size, horizon)

    history = history.sort_index()
    forecasts: dict[pd.Timestamp, float] = {}

    for start in range(0, len(timestamps), chunk_size):
        chunk = timestamps[start : start + chunk_size]
        context = to_nixtla_frame(history.iloc[-input_size:])
        prediction = model.predict(df=context)
        values = prediction.iloc[:, -1].to_numpy()[: len(chunk)]

        for timestamp, value in zip(chunk, values):
            forecasts[timestamp] = float(value)

        next_values = pd.Series(values, index=chunk)
        if observed is not None:
            actual = observed.reindex(chunk)
            next_values = actual.where(actual.notna(), next_values)
        new_values = next_values[~next_values.index.isin(history.index)]
        history = pd.concat([history, new_values]).sort_index()

    return forecasts
