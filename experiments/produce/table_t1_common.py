"""Shared build logic for Table T1 (synthetic drift detection).

Read by produce_table_t1_post_tune.py (run_post_tune_on_synthetic.py's
rows), which picks its rows via the `split_id_filter` callable passed to
build_table(). (A pre-tuning T1 once shared this module; it was
retired.)

`build_table()` takes an explicit `split_id_filter` rather than
hardcoding one, since the un-filtered version of this exact function
(back when there was only ever one T1) is what silently averaged
pre-tuning and post-tuning runs of the same detector/drift_type together
into one row once both existed in runs.csv -- kept explicit so that
mixing can't reappear by omission if a second T1 variant is added again.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable

import pandas as pd

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.results_io import RUNS_CSV, RUNS_DIR

DETECTOR_CLASS_BY_METHOD = {
    "adwin": ADWINDetector,
    "kswin": KSWINDetector,
    "page_hinkley": PageHinkleyDetector,
}

TABLE_COLUMNS = [
    "detector",
    "drift type",
    "delay (mean ± sd)",
    "false alarms / 10k",
    "missed",
    "config",
    "seeds",
]


def _format_value(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def config_label_for(config_hash: str, method: str) -> str:
    """Every constructor hyperparameter of the detector, as the run
    recorded it -- not just the one that sets its threshold. Run
    bookkeeping in config.json (samples_per_year, parameter_selection, ...)
    is left out. Raises if the run didn't record a hyperparameter, rather
    than guessing a default: re-run the producing script instead."""
    config = json.loads((RUNS_DIR / config_hash / "config.json").read_text())
    detector_class = DETECTOR_CLASS_BY_METHOD[method]
    names = list(inspect.signature(detector_class.__init__).parameters)[1:]

    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(
            f"{method} run {config_hash} did not record {missing} in config.json -- "
            "re-run its producing script so every hyperparameter is logged"
        )
    return ", ".join(f"{name}={_format_value(config[name])}" for name in sorted(names))


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
                "config": config_label_for(g["config_hash"].iloc[0], method),
                "seeds": g["seed"].nunique(),
            }
        )

    return (
        pd.DataFrame(rows, columns=TABLE_COLUMNS)
        .sort_values(["detector", "drift type"])
        .reset_index(drop=True)
    )
