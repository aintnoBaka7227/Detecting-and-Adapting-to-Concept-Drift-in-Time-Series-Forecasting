"""Which drift detector triggers the adaptation arms, and how its results are named.

Arms C and D retrain on a detector's accepted alarms, so every trigger
detector has its own adaptation, uncertainty and escalation results, logged
under split_ids that name it. Arms A (never retrain) and B (every 30 days)
use no detector: they were run once, in the ADWIN batch (CONTROL_DETECTOR),
and every detector's tables and figures share those runs.
"""

from __future__ import annotations

import pandas as pd

TRIGGER_DETECTORS = ("adwin", "page_hinkley")
DETECTOR_LABELS = {"adwin": "ADWIN", "page_hinkley": "Page-Hinkley"}
CONTROL_DETECTOR = "adwin"
# run_aemo_adaptation_arms.ARM_A / ARM_B: the arms whose schedule ignores
# the detector.
CONTROL_ARMS = ("A_never_retrain", "B_every_30_days")


def adaptation_split_id(detector: str) -> str:
    return f"aemo_adapt_{detector}_std_hh_v1"


def uq_split_id(detector: str) -> str:
    return f"aemo_uq_{detector}_half_hourly_v1"


def escalation_split_id(detector: str, signal: str = "width") -> str:
    """Escalation rows of one detector and escalation signal
    (run_aemo_escalation_sweep.SIGNALS); width keeps its original id."""
    if signal == "width":
        return f"aemo_escalation_{detector}_half_hourly_v1"
    return f"aemo_escalation_{detector}_{signal}_half_hourly_v1"


# ADWIN's UQ and escalation runs were logged before the detector was part of
# every split_id. runs.csv is append-only, so those rows keep their original
# ids: new runs are written under the ids above, and readers accept both, so
# a rerun supersedes the old rows through the usual latest-timestamp pick.
LEGACY_SPLIT_IDS = {
    "aemo_uq_adwin_half_hourly_v1": ("aemo_uq_half_hourly_v1",),
    "aemo_escalation_adwin_half_hourly_v1": ("aemo_escalation_half_hourly_v1",),
}


def readable_split_ids(split_id: str) -> tuple[str, ...]:
    """`split_id` plus any legacy id its earlier runs were logged under."""
    return (split_id, *LEGACY_SPLIT_IDS.get(split_id, ()))


def detector_rows(runs: pd.DataFrame, group: str, split_id_of, detector: str) -> pd.DataFrame:
    """runs.csv rows of `group` for one trigger detector: its own drift-
    triggered arms plus the shared control arms, under the detector's
    split_id or a legacy one. `split_id_of` is one of the *_split_id
    functions above."""
    if detector not in TRIGGER_DETECTORS:
        raise ValueError(f"unknown trigger detector {detector!r}; expected one of {TRIGGER_DETECTORS}")
    rows = runs[runs["group"] == group]
    control = rows["method"].str.split("/").str[1].isin(CONTROL_ARMS)
    own = rows[rows["split_id"].isin(readable_split_ids(split_id_of(detector))) & ~control]
    shared = rows[rows["split_id"].isin(readable_split_ids(split_id_of(CONTROL_DETECTOR))) & control]
    return pd.concat([shared, own])
