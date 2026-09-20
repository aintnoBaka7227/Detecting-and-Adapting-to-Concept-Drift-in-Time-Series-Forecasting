"""Shared build logic for Table T2 (AEMO detection vs. documented events).

Six single-(stream, stage) producers read this module, one per (tuning
stage, input stream) combination, plus two combined producers that call
build_table() once per stream and concatenate:

- produce_table_t2_raw_pre_tune.py / _post_tune.py
    raw 30-minute demand              (run_aemo_detectors_raw_pre_tune.py / _post_tune.py)
- produce_table_t2_standard_daily_pre_tune.py / _post_tune.py
    standard-daily (deseasonalised + standardised, daily-aggregated)
                                       (run_aemo_detectors_standard_daily_pre_tune.py / _post_tune.py)
- produce_table_t2_standard_half_hourly_pre_tune.py / _post_tune.py
    standard-half-hourly (deseasonalised + standardised, native resolution)
                                       (run_aemo_detectors_standard_half_hourly_pre_tune.py / _post_tune.py)
- produce_table_t2_all_streams_post_tune.py / _all_tuning.py
    combined views across the above

All share the same columns, Tier 1 event ordering, and disclaimer; they
differ only in which run's `split_id` they pin.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.config import AEMO_5MIN_END, DOCUMENTED_EVENTS_CSV, SPLIT
from experiments.produce.chance_baseline import expected_matches_closed_form
from experiments.results_io import RUNS_CSV

TEST_START = pd.Timestamp(SPLIT["test"][0])
TEST_END = pd.Timestamp(AEMO_5MIN_END)  # last date AEMO data is available through

# Short column labels for the five Tier 1 events, keyed by event_id so the
# mapping survives a reorder -- full names are still printed alongside the
# table for reference.
AEMO_TIER1_COLUMNS = {
    "EVT-2020-02": "COVID delay",
    "EVT-2020-07": "solar/mild-weather delay",
    "EVT-2021-10": "5-minute settlement delay",
    "EVT-2022-03": "2022 suspension delay",
    "EVT-2023-14": "security-directions delay",
}

DISCLAIMER = (
    "The changepoints in T2 are historically documented events, not ground "
    "truth. A detection that matches none of them is reported as unmatched "
    "rather than as a false positive."
)


def tier1_events() -> pd.DataFrame:
    """Every Tier 1 event, in date order -- the master column order for
    every region (all five are tagged region='NEM', so the same columns
    apply everywhere)."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date"])
    return events.loc[events["tier"] == 1].sort_values("start_date")


def latest_detection_metrics(split_id: str) -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run the matching detection experiment first")
    runs = pd.read_csv(RUNS_CSV)
    det = runs[
        (runs["group"] == "detection")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == split_id)
    ].copy()
    if det.empty:
        raise SystemExit(
            f"no rows for split_id={split_id!r} in runs.csv -- "
            "run the matching detection experiment first"
        )
    # runs.csv is append-only; a re-run adds a fresh batch of rows. Every row
    # from one record_run call shares a timestamp, so keep only the rows from
    # the most recent call per (method, region) -- as a whole, not per
    # metric_name, so a metric an older run emitted but the latest one didn't
    # (e.g. an event it no longer matches) can't leak a stale value through.
    latest_timestamp = det.groupby(["method", "region"])["timestamp"].transform("max")
    return det[det["timestamp"] == latest_timestamp]


def build_table(split_id: str, include_chance_baseline: bool = True) -> pd.DataFrame:
    """Corrected Table T2. Tier 1 and Tier 2 precision are reported
    separately (never combined into one value -- see
    drift_lab.evaluation.calculate_event_metrics), alongside raw vs.
    accepted (post-refractory) detection counts and, unless disabled, the
    chance-matching baseline: `K * (1 - (1 - w/T)^N)` for each row's own
    accepted-detection count. Any result not clearly above that line is
    not a result."""
    metrics = latest_detection_metrics(split_id)
    tier1 = tier1_events()

    # One row per (method, region), one column per metric_name -- the only
    # reshape step, and it's a pure groupby: no manual per-(method, region)
    # filtering loop.
    wide = (
        metrics.groupby(["method", "region", "metric_name"])["metric_value"]
        .first()
        .unstack("metric_name")
    )

    rows = []
    for (method, region), metric in wide.iterrows():
        row = {"detector": method, "region": region}

        tier1_unmatched = 0
        for event in tier1.itertuples(index=False):
            column = AEMO_TIER1_COLUMNS[event.event_id]
            delay = metric.get(f"delay_{event.event_id}")
            if pd.isna(delay):
                row[column] = "not detected"
                tier1_unmatched += 1
            else:
                row[column] = round(delay)

        n_matched_t1 = int(metric.get("n_matched_t1", 0))
        n_matched_t2 = int(metric.get("n_matched_t2", 0))
        precision_t1 = metric.get("precision_t1")
        precision_t2 = metric.get("precision_t2")
        raw_count = int(metric.get("n_detections", 0))
        accepted_count = int(metric.get("n_effective_detections", 0))
        accepted_per_year = metric.get("accepted_detections_per_year")

        row["tier1_matched"] = n_matched_t1
        row["tier1_unmatched"] = tier1_unmatched
        row["precision_t1"] = None if pd.isna(precision_t1) else round(precision_t1, 2)
        row["tier2_matched"] = n_matched_t2
        row["precision_t2"] = None if pd.isna(precision_t2) else round(precision_t2, 2)
        row["unmatched_accepted"] = int(metric.get("n_unmatched_detections", 0))
        row["raw_signal_count"] = raw_count
        row["accepted_detection_count"] = accepted_count
        row["accepted_detections_per_year"] = (
            None if pd.isna(accepted_per_year) else round(accepted_per_year, 2)
        )

        if include_chance_baseline:
            k_tier1 = len(tier1)
            test_days = (TEST_END - TEST_START) / pd.Timedelta(days=1)
            row["chance_expected_matches"] = round(
                expected_matches_closed_form(k_tier1, test_days, accepted_count), 2
            )

        rows.append(row)

    columns = (
        ["detector", "region"]
        + [AEMO_TIER1_COLUMNS[event_id] for event_id in tier1["event_id"]]
        + [
            "tier1_matched",
            "tier1_unmatched",
            "precision_t1",
            "tier2_matched",
            "precision_t2",
            "unmatched_accepted",
            "raw_signal_count",
            "accepted_detection_count",
            "accepted_detections_per_year",
        ]
        + (["chance_expected_matches"] if include_chance_baseline else [])
    )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["region", "detector"])
        .reset_index(drop=True)
    )
