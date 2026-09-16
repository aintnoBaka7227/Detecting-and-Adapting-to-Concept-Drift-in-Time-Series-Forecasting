"""AEMO detection on the daily-aggregated test split, post-tuning (frozen)
configs.

Uses post_tune_detector_configs.make_post_tune_detectors() -- the configs
chosen by `run_fine_tune_on_synthetic.py`'s synthetic-only sweep -- also
used by `run_post_tune_on_synthetic.py`,
`run_aemo_detectors_deseasonalized_post_tune.py`, and
`run_aemo_detectors_raw_post_tune.py`. AEMO data and AEMO events
were not used to choose these values.

Matched against the *full* event catalogue for the region -- Tier 1 and
Tier 2, not Tier 1 only -- so a produce script can report both a
Tier-1-specific unmatched count and an overall (both-tier) one from the
same run, instead of needing a second, differently-scoped run. (An
earlier `run_aemo_detectors_daily.py`, using plain class defaults and
Tier-1-only matching, served as the one-off sanity check for how much
daily aggregation cuts detection volume -- see DECISIONS.md -- and was
retired once this script existed to replace it as Table T2's source.
Its pre-tuning comparison role has since been revived, both-tier this
time, as `run_aemo_detectors_daily_pre_tune.py`.)

Each detector is run and logged completely independently: three
detectors, three separate `record_run` calls per region, six rows total.
Detections are never pooled or unioned across detectors before matching
-- each detector's output is matched against the event catalogue on its
own, exactly as if the other two detectors did not exist.

split_id "aemo_detect_daily_post_tune_v1": its own id, distinct from
`run_aemo_detectors_raw_pre_tune.py`'s "aemo_detect_raw_pre_tune_v1"
(daily-aggregated vs. the full raw half-hourly stream).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import aggregate_daily_demand
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_daily_post_tune_v1"
TARGET_COLUMN = "TOTALDEMAND"


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

        for detector in make_post_tune_detectors():
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
