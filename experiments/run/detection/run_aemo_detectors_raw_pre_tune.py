"""AEMO detection runs for Figure F2.

Runs the three drift detectors on each region's full standardised demand
series, keeps the detections that fall in the frozen test window, and
matches those against the *full* documented event catalogue -- Tier 1
and Tier 2, not Tier 1 only -- via `evaluation.match_unmatch` /
`calculate_event_metrics`, inside `record_run`. `runs.csv` gets every
scalar `calculate_event_metrics` produces (including the Tier 1/Tier 2
breakdown and `event_recall`) plus one `delay_<event_id>` row per match;
the raw detection timestamps are dumped alongside for F2.

Feeds Figure F2, and (as a second, read-only consumer of the same rows)
Table T2's pre-tuning + raw-30-minute sibling
(produce_table_t2_raw_pre_tune.py) -- not the canonical Table T2, which
reads `run_aemo_detectors_daily_post_tune.py`'s daily-aggregated,
post-tuning rows instead.

split_id "aemo_detect_raw_pre_tune_v1": detectors see the whole 2018-2023
series so they are warmed up by the time the test window starts, but only
test-window detections are scored -- a different data scope from the
baselines' fit-then-predict "aemo_frozen_v1".

Deterministic given a fixed detector config, so `seed=None` (no 5-seed
sweep). The matching tolerance is frozen before the run (see
`evaluation.POINT_WINDOW` / `INTERVAL_GRACE`) and recorded in
the config, so re-tiering or re-tolerancing yields a distinct config_hash.

Per the supervisor: fix the tiering (events.csv `tier` column) and the
matching rule *before* running - choosing which events count after seeing
where detectors fired would invalidate the table. Coordinate the rule with
the other team.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS, SPLIT
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import INTERVAL_GRACE, POINT_WINDOW
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_raw_pre_tune_v1"
TEST_START = pd.Timestamp(SPLIT["test"][0])
DETECTORS = (ADWINDetector(), KSWINDetector(), PageHinkleyDetector())


def demand_series(region: str) -> pd.Series:
    """Full standardised 30-minute demand for `region`, gaps dropped so the
    river detectors never see NaN."""
    frame = loader.load_processed(region)
    return (
        pd.Series(
            frame["TOTALDEMAND"].to_numpy(),
            index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
            name="TOTALDEMAND",
        )
        .dropna()
        .sort_index()
    )


def region_events(region: str) -> pd.DataFrame:
    """Every documented event (Tier 1 *and* Tier 2) for this region + NEM."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        series = demand_series(region)
        events = region_events(region)
        warmup = int((series.index < TEST_START).sum())

        for detector in DETECTORS:
            t0 = time.perf_counter()
            flagged = detector.detect(series.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            timestamps = series.index[flagged]
            test_detections = list(timestamps[timestamps >= TEST_START])

            record_run(
                method=detector.name,
                dataset="aemo",
                region=region,
                seed=None,
                config={
                    **config_of(detector),
                    "match_point_tolerance_days": POINT_WINDOW.days,
                    "match_period_grace_days": INTERVAL_GRACE.days,
                },
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=warmup,
                detection=(test_detections, events, None),
            )
            print(
                f"{region} {detector.name}: {len(test_detections)} test-window "
                f"detections ({len(flagged)} total), wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()
