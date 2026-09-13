"""AEMO detection on the *prediction-error stream*, not raw demand.

Companion to run_aemo_detectors.py. That one feeds the detectors raw
half-hourly demand, which is dominated by daily/weekly/annual seasonality
they read as drift -- hence thousands of unmatched detections. This one
feeds them each frozen baseline's 7-day rolling MAE over the test window
(the curve run_aemo_baselines.py already computed and dumped): a signal
that's flat while the model tracks and rises when concept drift bites, so
detections should be sparser and more meaningful. The briefing allows
exactly this -- "monitor the incoming stream, OR the model's own errors".

The rolling-MAE curve is read straight from the baseline's curve dump; no
forecasting is redone here. Detections are matched against the Tier 1
documented events (via evaluation.match_detections_to_events, inside
record_run), same as the raw-demand experiment.

split_id: "aemo_errstream_<baseline>_v1" -- one per baseline error stream,
deliberately distinct from the raw-demand experiment's "aemo_detect_full_v1"
so the two input streams are never conflated in runs.csv. `method` stays
the plain detector name; the baseline whose errors were detected on is in
split_id (and in the run config).

Deterministic given a fixed config, so seed=None.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import (
    DOCUMENTED_PERIOD_GRACE,
    DOCUMENTED_POINT_TOLERANCE,
)
from experiments.results_io import RUNS_CSV, curve_path
from experiments.run_harness import config_of, record_run

BASELINE_SPLIT_ID = "aemo_frozen_v1"  # where run_aemo_baselines.py logged the curves
BASELINES = ("seasonal_naive", "xgboost", "dhr_arima")
DETECTORS = (ADWINDetector(), KSWINDetector(), PageHinkleyDetector())


def split_id_for(baseline: str) -> str:
    return f"aemo_errstream_{baseline}_v1"


def tier1_events(region: str) -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[
        (events["tier"] == 1) & events["region"].isin([region, "NEM"])
    ].reset_index(drop=True)


def error_stream(runs: pd.DataFrame, baseline: str, region: str) -> pd.Series:
    """The frozen baseline's 7-day rolling-MAE curve over the test window,
    warm-up NaNs dropped."""
    rows = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["method"] == baseline)
        & (runs["region"] == region)
        & (runs["split_id"] == BASELINE_SPLIT_ID)
    ]
    if rows.empty:
        raise SystemExit(
            f"no baseline run for {baseline}/{region} -- run experiments.run_aemo_baselines first"
        )
    config_hash = rows.sort_values("timestamp")["config_hash"].iloc[-1]
    path = curve_path(config_hash, "aemo", region, None)
    curve = pd.read_csv(path, parse_dates=["index"]).set_index("index")[
        "rolling_mae_7d"
    ]
    return curve.dropna()


def main() -> None:
    if not RUNS_CSV.exists():
        raise SystemExit(
            f"{RUNS_CSV} not found -- run experiments.run_aemo_baselines first"
        )
    runs = pd.read_csv(RUNS_CSV)

    for region in REGIONS:
        events = tier1_events(region)
        for baseline in BASELINES:
            stream = error_stream(runs, baseline, region)

            for detector in DETECTORS:
                t0 = time.perf_counter()
                flagged = detector.detect(stream.to_numpy())
                wall_clock_s = time.perf_counter() - t0

                detections = list(stream.index[flagged])
                record_run(
                    method=detector.name,
                    dataset="aemo",
                    region=region,
                    seed=None,
                    config={
                        **config_of(detector),
                        "error_stream_source": baseline,
                        "match_point_tolerance_days": DOCUMENTED_POINT_TOLERANCE.days,
                        "match_period_grace_days": DOCUMENTED_PERIOD_GRACE.days,
                    },
                    wall_clock_s=wall_clock_s,
                    split_id=split_id_for(baseline),
                    train_samples=len(stream),
                    detection=(detections, events, None),
                )
                print(
                    f"{region} {detector.name} on {baseline} error: "
                    f"{len(detections)} detections, wall_clock={wall_clock_s:.1f}s"
                )


if __name__ == "__main__":
    main()
