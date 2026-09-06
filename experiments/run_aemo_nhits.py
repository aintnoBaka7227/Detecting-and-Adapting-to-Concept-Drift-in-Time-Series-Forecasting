"""NHITS pilot on AEMO — run separately from run_aemo_baselines.py while the
neuralforecast adapter is still being validated.

Same frozen train/calibration/test split as the other three baselines
(config.SPLIT), but under its OWN split_id: split_id is meant to version
the *data partitioning*, which is unchanged here, but this run is kept
deliberately out of "aemo_frozen_v1" so these pilot rows never silently mix
into the reviewed baseline comparison in runs.csv.

Unlike the other three baselines, NHITS training is stochastic, so its
`seed` is the model's real `random_seed` — not the NaN used for the
deterministic baselines.

Evaluation protocol — periodic re-grounding, NOT fully frozen/blind:
A first attempt ran NHITS fully blind (no target column at all, same as
xgboost/dhr_arima) and it diverged: with nothing to correct it, its own
input_size=336 (7-day) context eventually fills up entirely with its own
forecasts, each chunk's error compounding into the next until the series
runs away exponentially (plausible ~1,000 MW in March 2020 to >1e37 MW by
~day 700, then NaN). That's a real property of running a direct multi-step
neural model autoregressively far past its trained horizon with zero
feedback — not a bug in roll_forecast, and not comparable to how such a
model would actually be operated.

So instead: every ROLLING_WINDOW_DAYS (7d), NHITS is re-grounded on the
real demand for that whole day; the days in between stay blind, using only
its own chunk-by-chunk forecasts (see nixtla_common.roll_forecast — it
already falls back to the model's own forecast wherever `observed` is NaN,
so a *sparse* target column is all periodic re-grounding needs). This is a
materially different condition from xgboost/dhr_arima's fully frozen
predict_input() (see run_aemo_baselines.py) — keep that distinction in
mind when comparing curves; it's a caveat on the F1 plot, not an
apples-to-apples baseline yet.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, ROLLING_WINDOW_DAYS
from drift_lab.forecasting.nhits_forecaster import NHITSForecaster
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_frozen_v1_nhits_pilot"
REGROUND_EVERY_DAYS = ROLLING_WINDOW_DAYS  # bounds blind drift to <= 1 window


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def reground_column(test_y: pd.Series, horizon: int, every_days: int) -> pd.Series:
    """`test_y` with every day's demand replaced by NaN except every
    `every_days`-th day (0-indexed from the start of the test window).

    Assumes chunks are exactly one day (`horizon == SEASON_LENGTH`), so
    `roll_forecast` re-grounds on a whole real day's worth of context at a
    time rather than a lone point mid-chunk.
    """
    day_index = np.arange(len(test_y)) // horizon
    sparse = test_y.to_numpy().astype(float).copy()
    sparse[day_index % every_days != 0] = np.nan
    return pd.Series(sparse, index=test_y.index, name=test_y.name)


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

        model = NHITSForecaster()
        t0 = time.perf_counter()
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)
        reground = reground_column(test_y, model.horizon, REGROUND_EVERY_DAYS)
        preds = model.predict(
            pd.DataFrame({"TOTALDEMAND": reground}, index=test_y.index)
        )
        wall_clock_s = time.perf_counter() - t0

        record_run(
            method=model.name,
            dataset="aemo",
            region=region,
            seed=model.random_seed,
            config=config_of(model),
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=len(train_y),
            forecast=(test_y.to_numpy(), preds, test_y.index),
        )
        print(f"{region} nhits: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()
