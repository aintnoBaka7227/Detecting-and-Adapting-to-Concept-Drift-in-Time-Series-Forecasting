"""Post-tuning (frozen) detector configs — the single source of truth for
every consumer that needs the "final" configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep: run_post_tune_on_synthetic.py,
run_aemo_detectors_daily_post_tune.py, run_aemo_detectors_deseasonalized_post_tune.py,
and run_aemo_detectors_raw_post_tune.py.

Values below are run_fine_tune_on_synthetic.py's automatically selected
winners (results/tables/fine_tune_on_synthetic_winners.csv, see its
select_winners()), tuned and frozen SEPARATELY per cadence -- a config's
false-alarms-per-year rate depends on how many samples make up one
simulated year, so "daily" (365 samples/year) and "half_hourly"
(48*365 samples/year) get independent winners rather than sharing one.
Eligibility per (config, cadence), checked per row rather than as a
mean: every no-drift seed has exactly zero false alarms, every row's
(no-drift and every drift scenario) false-alarms/year is <= 2, and zero
missed drifts across sudden/gradual/recurring. AEMO data and AEMO events
are never used to choose these.

Not regenerated automatically: re-run run_fine_tune_on_synthetic.py and
update these values by hand if its grid or eligibility rule changes.
"""

from __future__ import annotations

from typing import Literal

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector

Cadence = Literal["daily", "half_hourly"]

CADENCES: tuple[Cadence, ...] = ("daily", "half_hourly")


def make_post_tune_detectors(cadence: Cadence = "half_hourly"):
    """Return the three detectors at their post-tuning (frozen) settings
    for the given cadence."""

    if cadence == "daily":
        return (
            ADWINDetector(delta=0.00075),
            KSWINDetector(alpha=0.001, window_size=450, stat_size=55, seed=42),
            PageHinkleyDetector(min_instances=20, delta=0.05, threshold=200.0),
        )

    if cadence == "half_hourly":
        return (
            ADWINDetector(delta=0.00075),
            KSWINDetector(alpha=0.001, window_size=450, stat_size=55, seed=42),
            PageHinkleyDetector(min_instances=20, delta=0.0005, threshold=400.0),
        )

    raise ValueError(f"cadence must be 'daily' or 'half_hourly', got {cadence!r}.")
