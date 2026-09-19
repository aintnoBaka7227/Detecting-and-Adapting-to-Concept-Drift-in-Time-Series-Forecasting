"""Post-tuning (frozen) detector configs — the single source of truth for
every consumer that needs the "final" configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep: run_post_tune_on_synthetic.py,
run_aemo_detectors_standard_daily_post_tune.py,
run_aemo_detectors_standard_half_hourly_post_tune.py, and
run_aemo_detectors_raw_post_tune.py.

Loaded directly from results/tables/fine_tune_on_synthetic_winners.csv --
not hand-copied literals -- so a config here can never silently drift out
of sync with what the sweep actually selected. Tuned and frozen
SEPARATELY per cadence: a config's false-alarms-per-year rate depends on
how many samples make up one simulated year, so "daily" (365
samples/year) and "half_hourly" (48*365 samples/year) get independent
winners rather than sharing one. Eligibility per (config, cadence),
checked per row rather than as a mean: every no-drift seed has exactly
zero false alarms, every row's (no-drift and every drift scenario)
false-alarms/year is <= 2, and zero missed drifts across
sudden/gradual/recurring. AEMO data and AEMO events are never used to
choose these.

Fails loudly (RuntimeError) rather than silently falling back to a River
default if the winners file is missing, a (detector, cadence) row is
missing, or that row's status is not "selected" (i.e. the sweep recorded
`no_eligible_configuration` -- see run_fine_tune_on_synthetic.py).
"""

from __future__ import annotations

import json
from typing import Literal

import pandas as pd

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.results_io import TABLES_DIR

Cadence = Literal["daily", "half_hourly"]

CADENCES: tuple[Cadence, ...] = ("daily", "half_hourly")

WINNERS_CSV = TABLES_DIR / "fine_tune_on_synthetic_winners.csv"

_DETECTOR_CLASSES = {
    "adwin": ADWINDetector,
    "kswin": KSWINDetector,
    "page_hinkley": PageHinkleyDetector,
}


def load_winner_row(detector: str, cadence: Cadence) -> dict:
    """The winners-file row for one (detector, cadence), as a dict.

    Raises RuntimeError if the winners file doesn't exist, the
    (detector, cadence) row is missing, or its status isn't "selected"
    -- callers must not catch this and substitute a fallback.
    """
    if not WINNERS_CSV.exists():
        raise RuntimeError(
            f"{WINNERS_CSV} not found -- run run_fine_tune_on_synthetic.py "
            "before requesting a post-tuning configuration. Refusing to "
            "fall back to a River default."
        )

    winners = pd.read_csv(WINNERS_CSV)
    match = winners[(winners["detector"] == detector) & (winners["cadence"] == cadence)]

    if match.empty:
        raise RuntimeError(
            f"No fine-tuning row for detector={detector!r}, cadence={cadence!r} "
            f"in {WINNERS_CSV}. Re-run run_fine_tune_on_synthetic.py -- refusing "
            "to fall back to a River default or another cadence's configuration."
        )

    row = match.iloc[0].to_dict()

    if row["status"] != "selected":
        raise RuntimeError(
            f"detector={detector!r}, cadence={cadence!r} has status="
            f"{row['status']!r} in {WINNERS_CSV} (no configuration passed the "
            "synthetic acceptance criteria for this cadence). Refusing to "
            "silently use a River default or another cadence's configuration -- "
            "this (detector, cadence) has no valid frozen configuration to run."
        )

    return row


def make_post_tune_detectors(cadence: Cadence = "half_hourly"):
    """Return the three detectors at their post-tuning (frozen) settings
    for the given cadence, loaded live from fine_tune_on_synthetic_winners.csv."""

    if cadence not in CADENCES:
        raise ValueError(f"cadence must be 'daily' or 'half_hourly', got {cadence!r}.")

    detectors = []
    for name, detector_class in _DETECTOR_CLASSES.items():
        row = load_winner_row(name, cadence)
        kwargs = json.loads(row["detector_parameters"])
        detectors.append(detector_class(**kwargs))

    return tuple(detectors)
