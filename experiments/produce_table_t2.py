"""Table T2 -- AEMO detection vs documented events.

Pure pivot of `results/runs.csv`. Reads the detection rows
`run_aemo_detectors.py` logged (`delay_<event_id>`, `precision`, unmatched
counts) and lays them out wide: one row per (detector, region), one column
per Tier 1 event.

Per the supervisor, T2 shows Tier 1 only, and an unmatched detection is
reported as `unmatched`, never as a false positive. Print the disclaimer
beside the table and keep it in the report verbatim.

T1 makes T2 believable -- produce it first, and never present T2 alone.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.results_io import FIGURES_DIR, RUNS_CSV

# Must match run_aemo_detectors.py::SPLIT_ID. Pinning it means running some
# other detection experiment (a different protocol / split) never silently
# changes this table -- the producer targets one specific experiment and
# selects the latest run *of that*.
SPLIT_ID = "aemo_detect_full_v1"

DISCLAIMER = (
    "The changepoints in T2 are historically documented events, not ground "
    "truth. A detection that matches none of them is reported as unmatched "
    "rather than as a false positive."
)


def tier1_events() -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV)
    return events.loc[events["tier"] == 1, ["event_id", "event_name", "region"]]


def latest_detection_metrics() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(
            f"{RUNS_CSV} not found -- run experiments.run_aemo_detectors first"
        )
    runs = pd.read_csv(RUNS_CSV)
    det = runs[
        (runs["group"] == "detection")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == SPLIT_ID)
    ].copy()
    if det.empty:
        raise SystemExit(
            f"no rows for split_id={SPLIT_ID!r} in runs.csv -- run experiments.run_aemo_detectors first"
        )
    # runs.csv is append-only; a re-run of run_aemo_detectors adds a fresh
    # batch of rows. Every row from one record_run call shares a timestamp,
    # so keep only the rows from the most recent call per (method, region) --
    # as a whole, not per metric_name, so a metric an older run emitted but
    # the latest one didn't (e.g. an event it no longer matches) can't leak
    # a stale value through.
    latest = det.groupby(["method", "region"])["timestamp"].transform("max")
    return det[det["timestamp"] == latest]


def build_table() -> pd.DataFrame:
    metrics = latest_detection_metrics()
    events = tier1_events()

    rows = []
    for region in REGIONS:
        region_events = events[events["region"].isin([region, "NEM"])]
        for method in sorted(metrics["method"].unique()):
            cell = metrics[
                (metrics["method"] == method) & (metrics["region"] == region)
            ]
            value = dict(zip(cell["metric_name"], cell["metric_value"]))

            row = {"detector": method, "region": region}
            for event in region_events.itertuples(index=False):
                delay = value.get(f"delay_{event.event_id}")
                row[event.event_id] = "not detected" if delay is None else round(delay)
            row["unmatched"] = int(value.get("n_unmatched_detections", 0))
            row["missed_events"] = int(value.get("n_unmatched_events", 0))
            precision = value.get("precision")
            row["precision"] = (
                None if precision is None or pd.isna(precision) else round(precision, 2)
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    table = build_table()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "table_t2_aemo_events.csv"
    table.to_csv(out, index=False)

    print(table.to_string(index=False))
    print()
    for event in tier1_events().drop_duplicates("event_id").itertuples(index=False):
        print(f"  {event.event_id}  {event.event_name}")
    print(f"\n{DISCLAIMER}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
