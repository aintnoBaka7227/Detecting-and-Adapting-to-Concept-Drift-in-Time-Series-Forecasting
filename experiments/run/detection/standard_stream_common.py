"""Shared TRAIN-fitted, standardised preprocessing for the two "standard"
AEMO input streams: standard_daily and standard_half_hourly.

Both streams follow the same three steps validated in
check_seasonal_residuals.py, now used for the actual detector runs
instead of only a QC check:

    1. Fit a seasonal profile on TRAIN only (fit_seasonal_profile) and
       apply it unchanged to TRAIN, Calibration and TEST
       (apply_seasonal_profile) -- Calibration and TEST never influence
       the fitted profile.
    2. (QC gate -- see check_seasonal_residuals.py; not repeated here.)
    3. Standardise every split using TRAIN residual mean/std only
       (never recalculated on Calibration or TEST).

The detector then runs continuously across TRAIN -> Calibration -> TEST,
exactly like run_aemo_detectors_raw_*.py -- one instance is warmed up
over TRAIN + Calibration before TEST-period alarms are scored, rather
than being created cold at the TEST boundary. Per the supervisor's
requested warm-up rule (see SHARED_DECISIONS.md Section 15).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import (
    aggregate_daily_demand,
    apply_seasonal_profile,
    fit_seasonal_profile,
)
from drift_lab.config import SPLIT

Cadence = Literal["daily", "half_hourly"]

TARGET_COLUMN = "TOTALDEMAND"
TEST_START = pd.Timestamp(SPLIT["test"][0])

_FREQUENCY_BY_CADENCE = {"daily": "daily", "half_hourly": "30min"}


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def _aggregate(series: pd.Series, cadence: Cadence) -> pd.Series:
    return aggregate_daily_demand(series) if cadence == "daily" else series


def build_standard_stream(region: str, cadence: Cadence) -> tuple[pd.Series, int]:
    """Return (standardised TRAIN+Calibration+TEST series, warm-up count).

    The warm-up count is how many leading observations fall before
    TEST_START -- everything from there is TRAIN/Calibration warm-up;
    only detections at or after it should be scored.
    """
    if cadence not in _FREQUENCY_BY_CADENCE:
        raise ValueError(f"cadence must be 'daily' or 'half_hourly', got {cadence!r}.")

    train, calibration, test = loader.load(region)

    train_agg = _aggregate(demand_series(train), cadence)
    calibration_agg = _aggregate(demand_series(calibration), cadence)
    test_agg = _aggregate(demand_series(test), cadence)

    profile = fit_seasonal_profile(train_agg, frequency=_FREQUENCY_BY_CADENCE[cadence])

    train_residual = apply_seasonal_profile(train_agg, profile)
    calibration_residual = apply_seasonal_profile(calibration_agg, profile)
    test_residual = apply_seasonal_profile(test_agg, profile)

    residual_mean = train_residual.mean()
    residual_std = train_residual.std(ddof=0)

    def standardise(residual: pd.Series) -> pd.Series:
        return (residual - residual_mean) / residual_std

    full = pd.concat(
        [
            standardise(train_residual),
            standardise(calibration_residual),
            standardise(test_residual),
        ]
    ).sort_index()
    full.name = TARGET_COLUMN

    warmup = int((full.index < TEST_START).sum())

    return full, warmup
