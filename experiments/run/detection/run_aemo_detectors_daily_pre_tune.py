"""AEMO detection on the daily-aggregated test split, pre-tuning (class
default) configs.

Sibling of run_aemo_detectors_daily_post_tune.py, same daily-aggregated
input (`aggregate_daily_demand(test)`) and the same both-tier (Tier 1 and
Tier 2) matching, but with plain `ADWINDetector()` / `KSWINDetector()` /
`PageHinkleyDetector()` defaults instead of the post-tuning, synthetic-tuned
configs -- so a produce script can compare pre- vs. post-tuning detection
quality on the same daily-aggregated input, holding everything else fixed.

An earlier `run_aemo_detectors_daily.py` covered this same pre-tuning,
daily-aggregated combination, but with Tier-1-only matching and its own
`split_id="aemo_detect_daily_test_v1"`; it was retired once
run_aemo_detectors_daily_post_tune.py existed to replace it as Table T2's
source (see DECISIONS.md). This script revives that comparison under a
new split_id, now matching both tiers like its post-tuning sibling so the
same tier1_unmatched/tier2_contextual breakdown is available on both sides
of the pre/post-tuning comparison.

split_id "aemo_detect_daily_pre_tune_v1": its own id, distinct from both
`run_aemo_detectors_daily_post_tune.py`'s "aemo_detect_daily_post_tune_v1"
(daily-aggregated, post-tuning configs) and
`run_aemo_detectors_raw_pre_tune.py`'s "aemo_detect_raw_pre_tune_v1" (raw
half-hourly, pre-tuning configs).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import aggregate_daily_demand
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_daily_pre_tune_v1"
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
        _train, _calibration, test = loader.load(region)
        daily_test = aggregate_daily_demand(demand_series(test))
        events = region_events(region)

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
                    "parameter_selection": "class_defaults",
                },
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=0,
                detection=(detections, events, None),
            )
            print(
                f"{region} {detector.name}: {len(detections)} detections "
                f"on {len(daily_test)} daily-aggregated test days, "
                f"wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()
