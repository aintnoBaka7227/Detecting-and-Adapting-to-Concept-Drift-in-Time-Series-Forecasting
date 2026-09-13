"""AEMO detection runs on the daily-aggregated test split.

Companion to `run_aemo_detectors.py`, which feeds all three detectors the
full half-hourly series (warmed up on 2018-2023). This one feeds them
`daily_aggregate(test)` instead -- one point/day, test split only, no
warmup -- since raw half-hourly demand fires mostly on ordinary
seasonality rather than genuine drift. Matched against Tier 1 documented
events the same way, under its own `split_id` so the two never mix.
`train_samples=0`: no warmup here, unlike `run_aemo_detectors.py`.

Uses a widened 7-day tolerance for both point and period events (vs. the
frozen T2 defaults of 1/7 days), passed explicitly to `record_run` rather
than touching `evaluation.DOCUMENTED_POINT_TOLERANCE` -- this script is
exploratory, not the frozen Table T2 comparison, so it may loosen its own
matching window without affecting `run_aemo_detectors.py`'s rows. Lands
in `config`, so it gets its own `config_hash`.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import daily_aggregate
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_daily_test_v1"
TARGET_COLUMN = "TOTALDEMAND"
DETECTORS = (ADWINDetector(), KSWINDetector(), PageHinkleyDetector())
POINT_TOLERANCE = pd.Timedelta(days=7)
PERIOD_GRACE = pd.Timedelta(days=7)


def tier1_events(region: str) -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[
        (events["tier"] == 1) & events["region"].isin([region, "NEM"])
    ].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        _train, _calibration, test = loader.load(region)
        daily_test = daily_aggregate(test, column=TARGET_COLUMN)
        events = tier1_events(region)

        for detector in DETECTORS:
            t0 = time.perf_counter()
            flagged = detector.detect(daily_test.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            detections = list(daily_test.index[flagged])

            record_run(
                method=detector.name,
                dataset="aemo",
                region=region,
                seed=None,
                config={
                    **config_of(detector),
                    "input_stream": "daily_mean_demand",
                    "match_point_tolerance_days": POINT_TOLERANCE.days,
                    "match_period_grace_days": PERIOD_GRACE.days,
                },
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=0,
                detection=(detections, events, None),
                point_tolerance=POINT_TOLERANCE,
                period_grace=PERIOD_GRACE,
            )
            print(
                f"{region} {detector.name}: {len(detections)} detections "
                f"on {len(daily_test)} daily-aggregated test days, "
                f"wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()
