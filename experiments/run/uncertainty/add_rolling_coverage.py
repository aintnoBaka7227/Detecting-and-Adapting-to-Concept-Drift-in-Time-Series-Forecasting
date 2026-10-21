"""Back-fill F4's 7-day rolling coverage into saved interval files.

No interval is re-issued. Every results/runs/<hash>/intervals_aemo_*.csv
written before run_aemo_uq_replay.py saved rolling coverage gets the
rolling_coverage_7d_<uq method> columns that the replay now writes
(run_aemo_uq_replay.add_rolling_coverage), computed from the saved bounds
and rewritten in place. runs.csv is not touched. Safe to rerun: existing
columns are recomputed identically.

Usage:
    python -m experiments.run.uncertainty.add_rolling_coverage
"""

from __future__ import annotations

import pandas as pd

from experiments import results_io
from experiments.run.uncertainty.run_aemo_uq_replay import add_rolling_coverage


def main() -> None:
    paths = sorted(results_io.RUNS_DIR.glob("*/intervals_aemo_*.csv"))
    for path in paths:
        # round_trip: read the saved floats exactly, so rewriting the file
        # cannot move the bounds by a last binary digit.
        intervals = pd.read_csv(path, float_precision="round_trip")
        add_rolling_coverage(intervals).to_csv(path, index=False)
    print(f"added 7-day rolling coverage to {len(paths)} interval files")


if __name__ == "__main__":
    main()
