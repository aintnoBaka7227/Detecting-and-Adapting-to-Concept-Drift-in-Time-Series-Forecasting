"""ADWIN change points for the adaptation arms -- detect once, reuse everywhere.

Runs the frozen post-tuning ADWIN a single time per region on the
standard-half-hourly stream (TRAIN-fitted seasonal residual, z-scored on
TRAIN), warmed up over TRAIN + Calibration exactly like
run_aemo_detectors_standard_half_hourly_post_tune.py, keeps TEST-period
alarms only, and applies the 14-day refractory filter through the shared
`evaluation.match_unmatch` (SHARED_DECISIONS.md 6-7: only accepted alarms
may trigger retraining). The accepted change points are written to

    results/runs/changepoints/adwin_standard_half_hourly_post_tune_<region>.csv
        timestamp   accepted alarm timestamp
        day         its calendar day (the retraining boundary is day 23:30)
        label       Match / Unmatch against the documented events (info only;
                    retraining never depends on it)

plus `..._config.json` with the detector settings. Adaptation arms C and D
read these files; nothing downstream re-runs the detector.

Usage:
    python -m experiments.run.detection.run_adwin_changepoints
"""

from __future__ import annotations

import json

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.evaluation.evaluation import REFRACTORY_PERIOD, match_unmatch
from experiments.results_io import RUNS_DIR
from experiments.run.detection.post_tune_detector_configs import (
    make_post_tune_detectors,
)
from experiments.run.detection.standard_stream_common import (
    TEST_START,
    build_standard_stream,
)
from experiments.run_harness import config_of

DETECTOR = "adwin"
STREAM_CADENCE = "half_hourly"
CHANGEPOINTS_DIR = RUNS_DIR / "changepoints"


def changepoints_path(region: str):
    return CHANGEPOINTS_DIR / f"{DETECTOR}_standard_half_hourly_post_tune_{region}.csv"


def load_changepoints(region: str) -> pd.DataFrame:
    """The accepted ADWIN change points for `region`, as written by main()."""
    path = changepoints_path(region)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run run_adwin_changepoints.py first")
    return pd.read_csv(path, parse_dates=["timestamp", "day"])


def make_detector():
    from drift_lab.detection.adwin import ADWINDetector

    return ADWINDetector(
        delta=1e-8,
        clock=48,
        min_window_length=336,
    )


def region_events(region: str) -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def main() -> None:
    CHANGEPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        series, _ = build_standard_stream(region, STREAM_CADENCE)
        detector = make_detector()

        flagged = detector.detect(series.to_numpy())
        timestamps = series.index[flagged]
        test_alarms = list(timestamps[timestamps >= TEST_START])

        results = match_unmatch(test_alarms, region_events(region), region)
        accepted = (
            results[results["label"] != "Ignored"]
            .sort_values("timestamp")
            .loc[:, ["timestamp", "label"]]
            .reset_index(drop=True)
        )
        accepted.insert(1, "day", pd.DatetimeIndex(accepted["timestamp"]).normalize())
        accepted.to_csv(changepoints_path(region), index=False)

        config = {
            **config_of(detector),
            "detector": DETECTOR,
            "input_stream": "standard_half_hourly",
            "preprocessing": "seasonal_profile_half_hourly_v1",
            "refractory_period_days": REFRACTORY_PERIOD.days,
        }
        changepoints_path(region).with_name(
            f"{DETECTOR}_standard_half_hourly_post_tune_{region}_config.json"
        ).write_text(json.dumps(config, sort_keys=True, default=str, indent=2))

        print(
            f"{region}: {len(test_alarms)} raw TEST alarms -> {len(accepted)} accepted "
            f"change points -- {changepoints_path(region)}"
        )


if __name__ == "__main__":
    main()
