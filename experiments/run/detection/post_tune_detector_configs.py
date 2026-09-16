"""Post-tuning (frozen) detector configs — the single source of truth for
every consumer that needs the "final" configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep: run_post_tune_on_synthetic.py,
run_aemo_detectors_daily_post_tune.py, run_aemo_detectors_deseasonalized_post_tune.py,
and run_aemo_detectors_raw_post_tune.py.

Values below are run_fine_tune_on_synthetic.py's automatically selected
winners (results/tables/fine_tune_on_synthetic_winners.csv, see its
select_winners()) -- lowest mean detection delay among configs that passed
every criterion (<=2 false alarms per simulated year, zero missed drift
events, zero false alarms on the "none" no-drift stream) on every
(drift_type, seed) run. AEMO data and AEMO events are never used to choose
these.

Not regenerated automatically: re-run run_fine_tune_on_synthetic.py and
update these values by hand if its grid or acceptance rule changes.
"""

from __future__ import annotations

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector


def make_post_tune_detectors():
    """Return the three detectors at their post-tuning (frozen) settings."""
    return (
        ADWINDetector(delta=0.00075),
        KSWINDetector(alpha=0.001, window_size=450, stat_size=55, seed=42),
        PageHinkleyDetector(min_instances=20, delta=0.0005, threshold=400.0),
    )
