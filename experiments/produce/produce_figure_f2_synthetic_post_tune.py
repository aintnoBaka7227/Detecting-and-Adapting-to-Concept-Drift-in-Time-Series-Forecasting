"""Figure F2 (post-tuning) -- synthetic drift-detection visual companions.

Regenerates the deterministic synthetic series for visualisation only and
reads back persisted detector alarm indices from
run_post_tune_on_synthetic.py's rows (post-tuning, frozen detector
configs) -- it never reruns a detector. One figure per drift type with an
actual changepoint (sudden, gradual, recurring), one panel per seed. See
figure_f2_synthetic_common.py for the shared panel/figure logic and
produce_figure_f2_synthetic_pre_tune.py for the pre-tuning sibling.
"""

from __future__ import annotations

from experiments.produce.figure_f2_synthetic_common import KINDS, build_figure, load_runs
from experiments.results_io import FIGURES_DIR

# Must match run_post_tune_on_synthetic.py::SPLIT_ID.
SPLIT_ID = "synthetic_full_series_20000_observations"


def main() -> None:
    runs = load_runs()

    for kind in KINDS:
        output = FIGURES_DIR / f"f2_synthetic_post_tune_{kind}.png"
        build_figure(
            kind=kind,
            runs=runs,
            split_id_filter=lambda split_id: split_id == SPLIT_ID,
            title=f"F2 (post-tuning) — Synthetic {kind.capitalize()}-Drift Detection, all seeds",
            output=output,
        )
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
