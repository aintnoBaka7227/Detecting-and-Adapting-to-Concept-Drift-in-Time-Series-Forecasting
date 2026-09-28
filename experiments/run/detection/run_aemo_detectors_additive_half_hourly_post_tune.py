"""AEMO detection on the "additive" half-hourly stream, post-tuning
(frozen) configs.

"additive half-hourly" = raw half-hourly demand, deseasonalised via
drift_lab.aemo.additive_deseasonalise.remove_daily_weekly_profile()
(TRAIN-fitted overall-mean + smoothed day-of-year + weekday/half-hour
offsets, summed and subtracted) and standardised (z-scored on TRAIN
residual mean/std only) -- see additive_stream_common.py for the shared
preprocessing. This is a second, independent deseasonalisation approach
kept fully separate from run_aemo_detectors_standard_half_hourly_post_tune.py
and standard_stream_common.py (which fit a multiplicative profile
instead) -- neither file is read or modified by this script.

Uses post_tune_detector_configs.make_post_tune_detectors() -- the same
frozen winners run_aemo_detectors_standard_*_post_tune.py uses, chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep. Detector tuning is
independent of which deseasonalisation feeds it, so no separate tuning
pass was run for this stream.

Matched against the *full* event catalogue for the region -- Tier 1 and
Tier 2, not Tier 1 only.

Each detector is run and logged completely independently: three
detectors, three separate `record_run` calls per region, six rows total.

split_id "aemo_detect_additive_half_hourly_post_tune_v1": its own id,
never collides with any existing split_id.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.evaluation.evaluation import REFRACTORY_PERIOD
from experiments import results_io
from experiments.run.detection.additive_stream_common import TEST_START, build_additive_stream
from experiments.run.detection.detection_artifacts import match_and_persist
from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_additive_half_hourly_post_tune_v1"


def region_events(region: str) -> pd.DataFrame:
    """Every documented event (Tier 1 *and* Tier 2) for this region + NEM."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        series, warmup = build_additive_stream(region)
        events = region_events(region)
        test_period_days = (series.index.max() - TEST_START) / pd.Timedelta(days=1)

        for detector in make_post_tune_detectors():
            t0 = time.perf_counter()
            flagged = detector.detect(series.to_numpy())
            wall_clock_s = time.perf_counter() - t0

            timestamps = series.index[flagged]
            test_detections = list(timestamps[timestamps >= TEST_START])

            config = {
                **config_of(detector),
                "input_stream": "additive_half_hourly",
                "parameter_selection": "synthetic_only_budget",
                "preprocessing": "additive_profile_half_hourly_v1",
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
