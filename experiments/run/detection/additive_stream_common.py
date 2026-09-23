"""Shared TRAIN-fitted, standardised preprocessing for the "additive"
AEMO input stream -- a second, independent deseasonalisation approach
kept fully separate from standard_stream_common.py.

Uses drift_lab.aemo.additive_deseasonalise.remove_daily_weekly_profile()
instead of deseasonalise.py's fit_seasonal_profile/apply_seasonal_profile.
The two modules differ in real ways, not just naming:

    - additive (this stream): expected = overall_mean + smoothed
      day-of-year effect + weekday/half-hour effect (offsets, summed).
    - standard_stream_common.py's stream: expected = base_demand *
      annual_factor * weekly_factor (ratios, multiplied).

remove_daily_weekly_profile() has no separate fit/apply step -- it
recomputes the profile from `reference` internally on every call. To
keep TRAIN, Calibration and TEST scored against the exact same
TRAIN-only profile (never refit on Calibration/TEST), it is called three
times here, always with reference=train_s -- the same reference series
holds the profile fixed across all three calls, at the cost of
recomputing it three times over instead of once.

half-hourly only: remove_daily_weekly_profile() hardcodes a half-hour
slot (hour*2 + minute//30) with no daily-aggregated mode, so there is no
additive counterpart to standard_stream_common.py's "daily" cadence.

Standardisation (TRAIN residual mean/std, ddof=0, applied unchanged to
Calibration/TEST) is the same convention standard_stream_common.py uses
-- plain arithmetic, not additive_deseasonalise.py's
ErrorZScorePreprocessor, which standardises a forecast-*error* stream
(a different signal) rather than a seasonal demand residual.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.additive_deseasonalise import remove_daily_weekly_profile
from drift_lab.config import SPLIT

TARGET_COLUMN = "TOTALDEMAND"
TEST_START = pd.Timestamp(SPLIT["test"][0])


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def build_additive_stream(region: str) -> tuple[pd.Series, int]:
    """Return (standardised TRAIN+Calibration+TEST series, warm-up count).

    The warm-up count is how many leading observations fall before
    TEST_START -- everything from there is TRAIN/Calibration warm-up;
    only detections at or after it should be scored. Per the
    supervisor's requested warm-up rule (see SHARED_DECISIONS.md
    Section 15).
    """
    train, calibration, test = loader.load(region)

    train_s = demand_series(train)
    calibration_s = demand_series(calibration)
    test_s = demand_series(test)

    train_residual = remove_daily_weekly_profile(train_s, reference=train_s)
    calibration_residual = remove_daily_weekly_profile(calibration_s, reference=train_s)
    test_residual = remove_daily_weekly_profile(test_s, reference=train_s)

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


def additive_half_hourly_demand_for_display(region: str) -> pd.Series:
    """Daily-mean resample of the additive (deseasonalised + standardised)
    series, TEST period only -- for display only, when the detector's own
    input is half-hourly and too dense to plot directly. Mirrors
    figure_f2_aemo_common.standard_half_hourly_demand_for_display() so it
    can be passed as figure_f2_aemo_common.build_all_figures()'s
    demand_for_display callable without editing that shared module."""
    full, warmup = build_additive_stream(region)
    return full.iloc[warmup:].resample("1D").mean().dropna()
