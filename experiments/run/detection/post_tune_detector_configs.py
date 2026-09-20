"""Post-tuning (frozen) detector configs — the single source of truth for
every consumer that needs the "final" configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep: run_post_tune_on_synthetic.py,
run_aemo_detectors_standard_daily_post_tune.py,
run_aemo_detectors_standard_half_hourly_post_tune.py, and
run_aemo_detectors_raw_post_tune.py.

Loaded directly from results/tables/fine_tune_on_synthetic_winners.csv --
not hand-copied literals -- so a config here can never silently drift out
of sync with what the sweep actually selected. One winner per detector,
shared by every consumer regardless of which AEMO input stream (raw,
standard-daily, standard-half-hourly) it runs against -- the sweep itself
is scored against a single fixed samples-per-year assumption (see
run_fine_tune_on_synthetic.py), not one per deployment cadence.

Fails loudly (RuntimeError) rather than silently falling back to a River
default if the winners file is missing, a detector's row is missing, or
that row's status is not "selected" (i.e. the sweep recorded
`no_eligible_configuration` -- see run_fine_tune_on_synthetic.py).
"""

from __future__ import annotations

import json

import pandas as pd

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.results_io import TABLES_DIR

WINNERS_CSV = TABLES_DIR / "fine_tune_on_synthetic_winners.csv"

_DETECTOR_CLASSES = {
    "adwin": ADWINDetector,
    "kswin": KSWINDetector,
    "page_hinkley": PageHinkleyDetector,
}


def load_winner_row(detector: str) -> dict:
    """The winners-file row for one detector, as a dict.

    Raises RuntimeError if the winners file doesn't exist, the
    detector's row is missing, or its status isn't "selected" -- callers
    must not catch this and substitute a fallback.
    """
    if not WINNERS_CSV.exists():
        raise RuntimeError(
            f"{WINNERS_CSV} not found -- run run_fine_tune_on_synthetic.py "
            "before requesting a post-tuning configuration. Refusing to "
            "fall back to a River default."
        )

    winners = pd.read_csv(WINNERS_CSV)
    match = winners[winners["detector"] == detector]

    if match.empty:
        raise RuntimeError(
            f"No fine-tuning row for detector={detector!r} in {WINNERS_CSV}. "
            "Re-run run_fine_tune_on_synthetic.py -- refusing to fall back "
            "to a River default."
        )

    row = match.iloc[0].to_dict()

    if row["status"] != "selected":
        raise RuntimeError(
            f"detector={detector!r} has status={row['status']!r} in "
            f"{WINNERS_CSV} (no configuration passed the synthetic "
            "acceptance criteria). Refusing to silently use a River "
            "default -- this detector has no valid frozen configuration to run."
        )

    return row


def make_post_tune_detectors():
    """Return the three detectors at their post-tuning (frozen) settings,
    loaded live from fine_tune_on_synthetic_winners.csv."""

    detectors = []
    for name, detector_class in _DETECTOR_CLASSES.items():
        row = load_winner_row(name)
        kwargs = json.loads(row["detector_parameters"])
        detectors.append(detector_class(**kwargs))

    return tuple(detectors)
