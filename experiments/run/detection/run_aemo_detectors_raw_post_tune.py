"""AEMO detection on the full raw half-hourly stream, post-tuning (frozen)
configs.

Sibling of run_aemo_detectors_raw_pre_tune.py: identical raw half-hourly
input, warm-up-then-test-window scoring, and both-tier (Tier 1 and Tier 2)
matching, but post_tune_detector_configs.make_post_tune_detectors() instead
of plain class defaults.

Filled a real gap: an earlier `run_post_tune_input_comparison_on_aemo.py`
ran post-tuning detectors on raw demand, but matched against Tier 1
events only, so its rows couldn't report a genuine
tier2_contextual/unmatched-across-both-tiers breakdown the way
run_aemo_detectors_daily_post_tune.py / run_aemo_detectors_deseasonalized_post_tune.py
can (that script was retired -- see DECISIONS.md). This script gives raw
demand the same both-tier-matched treatment as those two, so all three
post-tuning input streams (raw, daily-aggregated, deseasonalized) are
comparable in one table (produce_table_t2_all_streams_post_tune.py).

split_id "aemo_detect_raw_post_tune_v1": its own id, distinct from
run_aemo_detectors_raw_pre_tune.py's "aemo_detect_raw_pre_tune_v1"
(same input, pre-tuning configs instead).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS, SPLIT
from drift_lab.evaluation.evaluation import INTERVAL_GRACE, POINT_WINDOW
from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_raw_post_tune_v1"
TEST_START = pd.Timestamp(SPLIT["test"][0])


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

        for detector in make_post_tune_detectors():
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
                    "input_stream": "raw_30min",
                    "parameter_selection": "synthetic_only_budget",
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
