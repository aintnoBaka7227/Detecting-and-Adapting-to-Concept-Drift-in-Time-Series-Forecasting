"""AEMO detection on the deseasonalised test split, post-tuning (frozen)
configs.

Sibling of run_aemo_detectors_daily_post_tune.py: same post-tuning detector
configs (post_tune_detector_configs.make_post_tune_detectors(), chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep) and the same
both-tier (Tier 1 and Tier 2) event matching, but fed
`remove_daily_weekly_profile(test, reference=train+calibration)` instead
of `aggregate_daily_demand(test)` -- half-hourly resolution with the
expected weekday/half-hour demand profile subtracted out, rather than
collapsed to one point per day. The seasonal profile is learned only from
`train`+`calibration`, never from the test stream itself, so nothing about
the test period leaks into what counts as "expected" demand.

An earlier `run_post_tune_input_comparison_on_aemo.py` ran this exact
detector-on-deseasonalised-input combination as one of three input
variants sharing a single split_id (distinguished only by a
`config.input_processing` field), which doesn't fit table_t2_common.py's
per-(method, region) "latest run" pivot -- pinning a Table T2 variant to
that split_id would have silently mixed all three inputs' rows together
the same way produce_table_t1_pre_tune.py used to mix pre/post-tuning runs
(see DECISIONS.md). It also only matched Tier 1 events. This script is a
dedicated, single-input, both-tier-matched source instead, exactly like
run_aemo_detectors_daily_post_tune.py / run_aemo_detectors_daily_pre_tune.py,
so a Table T2 variant can be pinned to one clean, fully comparable
split_id -- see DECISIONS.md for why the input-comparison script was
retired in favour of this one plus its raw-input sibling.

split_id "aemo_detect_deseasonalized_post_tune_v1": its own id, distinct
from every other AEMO detection split (raw / daily-aggregated x
pre-tuning / post-tuning).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import (
    remove_daily_weekly_profile,
    standardise_from_reference,
)
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.run.detection.post_tune_detector_configs import (
    make_post_tune_detectors,
)
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detect_deseasonalized_post_tune_v1"
TARGET_COLUMN = "TOTALDEMAND"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def region_events(region: str) -> pd.DataFrame:
    """Every documented event (Tier 1 and Tier 2) for this region + NEM."""
    events = pd.read_csv(
        DOCUMENTED_EVENTS_CSV,
        parse_dates=["start_date", "end_date"],
    )
    return events[
        events["region"].isin([region, "NEM"])
    ].reset_index(drop=True)


def main() -> None:
    for region in REGIONS:
        train, _calibration, test = loader.load(region)

        # Fit all preprocessing on the training window only.
        train_series = demand_series(train)
        test_series = demand_series(test)

        # Learn the seasonal profile from training and remove it from
        # both the training and test demand streams.
        deseasonalized_train = remove_daily_weekly_profile(
            train_series,
            reference=train_series,
        )

        deseasonalized_test = remove_daily_weekly_profile(
            test_series,
            reference=train_series,
        )

        # Learn z-score statistics from the deseasonalised training
        # stream and apply them unchanged to the test stream.
        standardized_test = standardise_from_reference(
            deseasonalized_test,
            reference=deseasonalized_train,
        )

        events = region_events(region)

        for detector in make_post_tune_detectors():
            t0 = time.perf_counter()

            flagged = detector.detect(
                standardized_test.to_numpy()
            )

            wall_clock_s = time.perf_counter() - t0

            detections = list(
                standardized_test.index[flagged]
            )

            record_run(
                method=detector.name,
                dataset="aemo",
                region=region,
                seed=None,
                config={
                    **config_of(detector),
                    "input_stream": (
                        "deseasonalized_standardized_half_hourly"
                    ),
                    "preprocessing_fit": "training_only",
                    "parameter_selection": "synthetic_only_budget",
                },
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=0,
                detection=(detections, events, None),
            )

            print(
                f"{region} {detector.name}: "
                f"{len(detections)} detections on "
                f"{len(standardized_test)} deseasonalised + "
                f"standardised test observations, "
                f"wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()
