"""AEMO detection on the daily-aggregated test split, frozen (tuned) configs.

Uses the frozen configurations chosen by `run_detector_budget_tuning.py`'s
synthetic-only sweep -- the same `ADWINDetector(delta=0.00075)` /
`KSWINDetector(alpha=0.005, window_size=300, stat_size=48, seed=42)` /
`PageHinkleyDetector(min_instances=30, delta=0.005, threshold=400.0)`
already used by `run_synthetic_detector_experiments.py` and
`run_detector_input_experiments.py`. AEMO data and AEMO events were not
used to choose these values.

Matched against the *full* event catalogue for the region -- Tier 1 and
Tier 2, not Tier 1 only -- so a produce script can report both a
Tier-1-specific unmatched count and an overall (both-tier) one from the
same run, instead of needing a second, differently-scoped run. (An
earlier `run_aemo_detectors_daily.py`, using plain class defaults and
Tier-1-only matching, served as the one-off sanity check for how much
daily aggregation cuts detection volume -- see DECISIONS.md -- and was
retired once this script existed to replace it.)

Each detector is run and logged completely independently: three
detectors, three separate `record_run` calls per region, six rows total.
Detections are never pooled or unioned across detectors before matching
-- each detector's output is matched against the event catalogue on its
own, exactly as if the other two detectors did not exist.

split_id "aemo_detect_daily_frozen_v1": its own id, distinct from
`run_aemo_detectors.py`'s "aemo_detect_full_v1" (daily-aggregated vs. the
full raw half-hourly stream).
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

SPLIT_ID = "aemo_detect_daily_frozen_v1"
TARGET_COLUMN = "TOTALDEMAND"


def make_frozen_detectors():
    """Final detector settings selected using synthetic data only (see
    run_detector_budget_tuning.py)."""
    return (
        ADWINDetector(delta=0.00075),
        KSWINDetector(alpha=0.005, window_size=300, stat_size=48, seed=42),
        PageHinkleyDetector(min_instances=30, delta=0.005, threshold=400.0),
    )


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

        for detector in make_frozen_detectors():
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
                    "parameter_selection": "synthetic_only_budget",
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
