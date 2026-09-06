"""Shared adapter between the `Forecaster` interface and Nixtla's
`neuralforecast` models (NHITS today, PatchTST or similar later).

Nixtla models forecast a fixed horizon `h` in one forward pass from a
fixed-length `input_size` context, and `NeuralForecast.predict(df=...)`
runs inference on a *new* context window without retraining. `Forecaster`,
by contrast, asks for one point forecast per row of an arbitrary `X` of any
length. `roll_forecast` bridges the two: it walks the requested timestamps
in `horizon`-sized chunks, calling `predict(df=...)` once per chunk against
the trailing `input_size` points of a running history, then extends that
history before the next chunk (with real observations where available,
otherwise with its own forecasts) — the frozen-weights equivalent of
`XGBoostForecaster`'s recursive lag-filling loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIQUE_ID = "series"


def to_nixtla_frame(y: pd.Series) -> pd.DataFrame:
    """`y` (DatetimeIndex-indexed) -> the long `unique_id, ds, y` frame
    every neuralforecast call expects."""
    return pd.DataFrame({"unique_id": UNIQUE_ID, "ds": y.index, "y": y.to_numpy()})


def roll_forecast(
    nf,
    history: pd.Series,
    index: pd.DatetimeIndex,
    horizon: int,
    input_size: int,
    observed: pd.Series | None = None,
) -> dict[pd.Timestamp, float]:
    """Forecast every timestamp in `index` (chronological) by rolling the
    frozen `nf` forward `horizon` steps at a time. Returns a
    timestamp -> value dict (mirrors `XGBoostForecaster.predict`'s internal
    convention) so the caller can re-index it back into any row order.

    `observed`, if given, is consulted after each chunk: timestamps it
    covers extend `history` with the real value (walk-forward); timestamps
    it doesn't (including all-NaN, or simply absent from `observed`'s
    index) fall back to the chunk's own forecast (frozen-blind). `observed`
    need not be dense — a *sparse* series (real values on some chunks,
    NaN/missing elsewhere) gives periodic re-grounding rather than either
    extreme: bounds how long the model can run purely on its own compounding
    forecasts without forcing daily feedback the way seasonal_naive gets
    (see run_aemo_nhits.py::reground_column for why fully blind, indefinite
    autoregression isn't safe to run this way — it can diverge).
    """
    history = history.sort_index()
    forecasts: dict[pd.Timestamp, float] = {}

    for start in range(0, len(index), horizon):
        chunk = index[start : start + horizon]
        context = to_nixtla_frame(history.iloc[-input_size:])
        prediction = nf.predict(df=context)
        value_column = prediction.columns[-1]
        values = prediction[value_column].to_numpy()[: len(chunk)]

        for timestamp, value in zip(chunk, values):
            forecasts[timestamp] = float(value)

        step = pd.Series(values, index=chunk)
        if observed is not None:
            actual = observed.reindex(chunk)
            step = actual.where(actual.notna(), step)
        new_points = step[~step.index.isin(history.index)]
        history = pd.concat([history, new_points]).sort_index()

    return forecasts


def as_series(y: pd.Series, name: str | None = None) -> pd.Series:
    """Coerce `y` to a float Series on a DatetimeIndex, matching the
    `_as_series` helpers already duplicated in seasonal_naive.py /
    xgboost_forecaster.py."""
    return pd.Series(
        np.asarray(y, dtype=float),
        index=pd.DatetimeIndex(y.index),
        name=name if name is not None else getattr(y, "name", None),
    )
