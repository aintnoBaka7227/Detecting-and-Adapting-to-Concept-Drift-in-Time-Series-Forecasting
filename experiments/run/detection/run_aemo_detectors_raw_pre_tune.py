"""AEMO detection runs on the raw demand stream, pre-tuning (output retired).

Runs the three detectors on each region's full 2018-2023 series (so they are
warmed up by the test window), keeps test-window detections and matches them
to the full event catalogue (Tier 1 and 2) inside record_run: every
calculate_event_metrics scalar plus delay_<event_id> rows, and the detection
timestamps. split_id "aemo_detect_raw_pre_tune_v1"; deterministic, so
seed=None. The tiering and matching tolerance are fixed before running and
recorded in the config, never chosen after seeing detections.

No table or figure reads these rows any more (the report uses the
standard-half-hourly, post-tuning stream); kept for reproducibility.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS, SPLIT
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import INTERVAL_GRACE, POINT_WINDOW, REFRACTORY_PERIOD
from experiments import results_io
from experiments.run.detection.detection_artifacts import match_and_persist
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
        test_period_days = (series.index.max() - TEST_START) / pd.Timedelta(days=1)

        for detector in DETECTORS:
            t0 = time.perf_counter()
            flagged = detector.detect(series.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            timestamps = series.index[flagged]
            test_detections = list(timestamps[timestamps >= TEST_START])

            config = {
                **config_of(detector),
                "input_stream": "raw_30min",
                "parameter_selection": "class_defaults",
                "preprocessing": "none",
                "refractory_period_days": REFRACTORY_PERIOD.days,
                "match_point_tolerance_days": POINT_WINDOW.days,
                "match_period_grace_days": INTERVAL_GRACE.days,
            }
            config_hash = results_io.config_hash(config)
            match_results, artifacts = match_and_persist(config_hash, region, test_detections, events)
            n_accepted = int((match_results["label"] != "Ignored").sum())

            record_run(
                method=detector.name,
                dataset="aemo",
                region=region,
                seed=None,
                config=config,
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=warmup,
                detection=(test_detections, events, None),
                test_period_days=test_period_days,
            )
            print(
                f"{region} {detector.name}: {len(test_detections)} raw signals, "
                f"{n_accepted} accepted after refractory ({len(flagged)} total incl. warm-up), "
                f"wall_clock={wall_clock_s:.1f}s -- {artifacts['accepted']}"
            )


if __name__ == "__main__":
    main()
