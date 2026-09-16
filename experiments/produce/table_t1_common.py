"""Shared build logic for Table T1 (synthetic drift detection).

Pre-tuning (produce_table_t1_pre_tune.py, reading run_pre_tune_on_synthetic.py's
rows) and post-tuning (produce_table_t1_post_tune.py, reading
run_post_tune_on_synthetic.py's rows) share every column definition and
aggregation rule; the only thing that differs between them is which
split_ids they read, via the `split_id_filter` callable passed to
build_table(). (The post-tuning table was briefly retired, then recreated
once a systematic pre/post-tuning x table-1/figure-2 matrix for the
synthetic benchmark was wanted -- see DECISIONS.md.)

`build_table()` takes an explicit `split_id_filter` rather than
hardcoding one, since the un-filtered version of this exact function
(back when there was only ever one T1) is what silently averaged
pre-tuning and post-tuning runs of the same detector/drift_type together
into one row once both existed in runs.csv -- kept explicit so that
mixing can't reappear by omission if a second T1 variant is added again.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from experiments.results_io import RUNS_CSV, RUNS_DIR

THRESHOLD_LABEL_BY_METHOD = {
    "adwin": lambda c: f"delta = {c['delta']}",
    "kswin": lambda c: f"alpha = {c['alpha']}",
    "page_hinkley": lambda c: f"threshold = {c['threshold']}",
}

TABLE_COLUMNS = [
    "detector",
    "drift type",
    "delay (mean ± sd)",
    "false alarms / 10k",
    "missed",
    "threshold",
    "seeds",
]


def threshold_label_for(config_hash: str, method: str) -> str:
    config = json.loads((RUNS_DIR / config_hash / "config.json").read_text())
    label = THRESHOLD_LABEL_BY_METHOD.get(method)
    return label(config) if label else json.dumps(config)


def format_delay_summary(delay: pd.Series) -> str:
    valid = delay.dropna()
    if valid.empty:
        return "n/a"
    if len(valid) == 1:
        return f"{valid.mean():.0f} ± n/a"
    return f"{valid.mean():.0f} ± {valid.std():.0f}"


def latest_synthetic_detection_rows(
    split_id_filter: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Detection rows for synthetic_* datasets matching `split_id_filter`,
    keeping only the most recent record_run() call per (method, dataset,
    seed).

    Both run_pre_tune_on_synthetic.py and run_post_tune_on_synthetic.py call
    record_run() once per (detector, drift_type, seed) -- each seed is its
    own call with its own timestamp, not one call covering all seeds. So
    the dedup key must include seed: grouping by (method, dataset) alone
    would keep only whichever single seed happened to run last and drop
    the other four. runs.csv is append-only, so a re-run of one seed adds
    a fresh row instead of replacing the old one; without this dedup, a
    partial re-run would double-count that seed in a sum() like `missed`.
    """
    df = pd.read_csv(RUNS_CSV)
    det = df[
        (df["group"] == "detection") & df["dataset"].str.startswith("synthetic_")
    ].copy()
    det = det[split_id_filter(det["split_id"])]
    det["drift_type"] = det["dataset"].str.removeprefix("synthetic_")

    latest_timestamp = det.groupby(["method", "dataset", "seed"])["timestamp"].transform(
        "max"
    )
    return det[det["timestamp"] == latest_timestamp]


def build_table(split_id_filter: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    det = latest_synthetic_detection_rows(split_id_filter)

    rows = []
    for (method, drift_type), g in det.groupby(["method", "drift_type"]):
        delay = g.loc[g["metric_name"] == "detection_delay", "metric_value"]
        alarms = g.loc[g["metric_name"] == "false_alarms_per_10000", "metric_value"]
        missed = g.loc[g["metric_name"] == "missed_detections", "metric_value"]
        rows.append(
            {
                "detector": method,
                "drift type": drift_type,
                "delay (mean ± sd)": format_delay_summary(delay),
                "false alarms / 10k": round(alarms.mean(), 1),
                "missed": int(missed.sum()),
                "threshold": threshold_label_for(g["config_hash"].iloc[0], method),
                "seeds": g["seed"].nunique(),
            }
        )

    return (
        pd.DataFrame(rows, columns=TABLE_COLUMNS)
        .sort_values(["detector", "drift type"])
        .reset_index(drop=True)
    )
