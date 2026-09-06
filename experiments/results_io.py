"""Filesystem side of the experiment ledger: results/runs.csv + curve dumps."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"

RUN_COLUMNS = [
    "group",
    "method",
    "dataset",
    "region",
    "seed",
    "config_hash",
    "split_id",
    "metric_name",
    "metric_value",
    "regime",
    "n_retrains",
    "train_samples",
    "wall_clock_s",
    "timestamp",
]


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def append_runs(rows: list[dict]) -> None:
    """Append rows to runs.csv under an exclusive lock (parallel-run safe)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    with locked_file(RUNS_CSV):
        frame.to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)


def dump_config(config_hash_: str, config: dict) -> Path:
    """Write the raw config a run used, so config_hash is traceable back to
    actual values (delta, alpha, ...) instead of just an opaque fingerprint.
    Idempotent: identical config -> identical file, written at most once.
    """
    out = RUNS_DIR / config_hash_
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if not path.exists():
        path.write_text(json.dumps(config, sort_keys=True, default=str, indent=2))
    return path


def curve_path(config_hash_: str, dataset: str, region: str | None, seed: int | None) -> Path:
    """The path dump_curve() writes to / a figure script reads from.

    `seed=None` -> the literal token "none", not Python's str(None) ("None")
    or a re-read NaN's str() ("nan") — both of those would drift apart after
    a runs.csv round-trip, making the file unfindable from the seed column.
    """
    seed_token = "none" if seed is None else str(seed)
    return RUNS_DIR / config_hash_ / f"curve_{dataset}_{region or '-'}_{seed_token}.csv"


def dump_curve(
    config_hash_: str,
    dataset: str,
    region: str | None,
    seed: int | None,
    curve: pd.Series,
) -> Path:
    """Write one run's full rolling-MAE curve for the figure scripts to read."""
    path = curve_path(config_hash_, dataset, region, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve.rename("rolling_mae_7d").to_csv(path, header=True)
    return path


@contextlib.contextmanager
def locked_file(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
