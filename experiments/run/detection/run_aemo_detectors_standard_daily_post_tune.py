"""AEMO detection on the standard-daily stream, post-tuning (frozen)
configs.

"standard-daily" = daily-aggregated demand, deseasonalised (TRAIN-fitted
day-of-year x day-of-week profile removed) and standardised (z-scored on
TRAIN residual mean/std only) -- see standard_stream_common.py for the
shared preprocessing. The detector runs continuously across TRAIN then
Calibration then TEST (warm-up), and only TEST-period alarms are scored.

Uses post_tune_detector_configs.make_post_tune_detectors(cadence="daily")
-- the daily-cadence winners chosen by run_fine_tune_on_synthetic.py's
synthetic-only sweep. AEMO data and AEMO events were not used to choose
these values.

Matched against the *full* event catalogue for the region -- Tier 1 and
Tier 2, not Tier 1 only.

Each detector is run and logged completely independently: three
detectors, three separate `record_run` calls per region, six rows total.

split_id "aemo_detect_standard_daily_post_tune_v1": its own id, distinct
from every other (stream, tuning-stage) combination.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.evaluation.evaluation import REFRACTORY_PERIOD
from experiments import results_io
from experiments.run.detection.detection_artifacts import match_and_persist
from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors
from experiments.run.detection.standard_stream_common import TEST_START, build_standard_stream
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_standard_daily_post_tune_v1"
CADENCE = "daily"


def region_events(region: str) -> pd.DataFrame:
    """Every documented event (Tier 1 *and* Tier 2) for this region + NEM."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        series, warmup = build_standard_stream(region, CADENCE)
        events = region_events(region)
        test_period_days = (series.index.max() - TEST_START) / pd.Timedelta(days=1)

        for detector in make_post_tune_detectors(CADENCE):
            t0 = time.perf_counter()
            flagged = detector.detect(series.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            timestamps = series.index[flagged]
            test_detections = list(timestamps[timestamps >= TEST_START])

            config = {
                **config_of(detector),
                "input_stream": "standard_daily",
                "parameter_selection": "synthetic_only_budget",
                "cadence": CADENCE,
                "preprocessing": "seasonal_profile_daily_v1",
                "refractory_period_days": REFRACTORY_PERIOD.days,
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
