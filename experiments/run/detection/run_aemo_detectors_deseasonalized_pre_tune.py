"""AEMO detection on the deseasonalised test split, pre-tuning (class
default) configs.

Sibling of run_aemo_detectors_deseasonalized_post_tune.py: identical
deseasonalised half-hourly input (`remove_daily_weekly_profile(test,
reference=train+calibration)`) and both-tier (Tier 1 and Tier 2) event
matching, but plain `ADWINDetector()` / `KSWINDetector()` /
`PageHinkleyDetector()` defaults instead of the post-tuning, synthetic-tuned
configs -- completing the same pre/post-tuning comparison
run_aemo_detectors_daily_pre_tune.py gives the daily-aggregated input and
run_aemo_detectors_raw_pre_tune.py gives the raw input.

split_id "aemo_detect_deseasonalized_pre_tune_v1": its own id, distinct
from run_aemo_detectors_deseasonalized_post_tune.py's
"aemo_detect_deseasonalized_post_tune_v1" (same input, post-tuning configs
instead).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import remove_daily_weekly_profile
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_deseasonalized_pre_tune_v1"
TARGET_COLUMN = "TOTALDEMAND"

DETECTORS = (ADWINDetector(), KSWINDetector(), PageHinkleyDetector())


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def region_events(region: str) -> pd.DataFrame:
    """Every documented event (Tier 1 *and* Tier 2) for this region + NEM."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)

        # Historical data only estimates the seasonal profile; it is never
        # supplied to detector.detect() itself.
        seasonal_reference = pd.concat(
            [demand_series(train), demand_series(calibration)]
        ).sort_index()

        deseasonalized_test = remove_daily_weekly_profile(
            demand_series(test),
            reference=seasonal_reference,
        )

        events = region_events(region)

        for detector in DETECTORS:
            t0 = time.perf_counter()
            flagged = detector.detect(deseasonalized_test.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            detections = list(deseasonalized_test.index[flagged])

            record_run(
                method=detector.name,
                dataset="aemo",
                region=region,
                seed=None,
                config={
                    **config_of(detector),
                    "input_stream": "deseasonalized_half_hourly",
                    "parameter_selection": "class_defaults",
                },
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=0,
                detection=(detections, events, None),
            )
            print(
                f"{region} {detector.name}: {len(detections)} detections "
                f"on {len(deseasonalized_test)} deseasonalised test observations, "
                f"wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()
