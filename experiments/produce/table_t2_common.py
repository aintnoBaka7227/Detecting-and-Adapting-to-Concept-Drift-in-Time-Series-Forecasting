"""Shared build logic for Table T2 (AEMO detection vs. documented events).

Four sibling producers read this module, one per (tuning stage, input
processing) combination:

- produce_table_t2_daily_post_tune.py     post-tuning, daily-aggregated
                                           (run_aemo_detectors_daily_post_tune.py)
- produce_table_t2_daily_pre_tune.py      pre-tuning,  daily-aggregated
                                           (run_aemo_detectors_daily_pre_tune.py)
- produce_table_t2_raw_pre_tune.py        pre-tuning,  raw 30-minute
                                           (run_aemo_detectors_raw_pre_tune.py)
- produce_table_t2_deseasonalized_post_tune.py   post-tuning, deseasonalized
                                           (run_aemo_detectors_deseasonalized_post_tune.py)

All four share the same columns, Tier 1 event ordering, and disclaimer;
they differ only in which run's `split_id` they pin.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV
from experiments.results_io import RUNS_CSV

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


def build_table(split_id: str) -> pd.DataFrame:
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

        row["tier1_unmatched"] = tier1_unmatched
        row["tier2_contextual"] = int(metric.get("n_matched_t2", 0))
        row["unmatched"] = int(metric.get("n_unmatched_events", 0))
        precision = metric.get("precision")
        row["precision"] = None if pd.isna(precision) else round(precision, 2)
        rows.append(row)

    columns = (
        ["detector", "region"]
        + [AEMO_TIER1_COLUMNS[event_id] for event_id in tier1["event_id"]]
        + ["tier1_unmatched", "tier2_contextual", "unmatched", "precision"]
    )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["region", "detector"])
        .reset_index(drop=True)
    )
