"""Shared raw/accepted-detection + event-assignment persistence for the
AEMO post-tuning detector runs.

Reuses `drift_lab.evaluation.match_unmatch` for both the refractory
(accepted-vs-raw) split and event assignment -- there is no second
matching or refractory implementation here. "Accepted" means every row
match_unmatch did not label "Ignored" (i.e. Match + Unmatch); "raw" means
every row it was given, unfiltered.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drift_lab.evaluation.evaluation import match_unmatch
from experiments.results_io import RUNS_DIR


def match_and_persist(
    config_hash: str,
    region: str,
    detected_timestamps,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run the shared matcher once and persist three artifacts under
    results/runs/<config_hash>/:

        raw_detections_<region>.csv        every raw signal timestamp
        accepted_detections_<region>.csv   raw minus refractory-ignored
                                            rows, with their match label,
                                            event_id, tier and delay
        event_assignments_<region>.csv     the full match_results frame,
                                            including Ignored rows

    Returns (match_results, {artifact_name: path}).
    """
    match_results = match_unmatch(detected_timestamps, events, region)

    base = RUNS_DIR / config_hash
    base.mkdir(parents=True, exist_ok=True)

    raw_path = base / f"raw_detections_{region}.csv"
    accepted_path = base / f"accepted_detections_{region}.csv"
    assignments_path = base / f"event_assignments_{region}.csv"

    match_results[["timestamp"]].sort_values("timestamp").to_csv(raw_path, index=False)

    accepted = match_results[match_results["label"] != "Ignored"].sort_values("timestamp")
    accepted.to_csv(accepted_path, index=False)

    match_results.sort_values("timestamp").to_csv(assignments_path, index=False)

    return match_results, {
        "raw": raw_path,
        "accepted": accepted_path,
        "assignments": assignments_path,
    }


def accepted_detections_path(config_hash: str, region: str) -> Path:
    return RUNS_DIR / config_hash / f"accepted_detections_{region}.csv"


def raw_detections_path(config_hash: str, region: str) -> Path:
    return RUNS_DIR / config_hash / f"raw_detections_{region}.csv"


def event_assignments_path(config_hash: str, region: str) -> Path:
    return RUNS_DIR / config_hash / f"event_assignments_{region}.csv"
