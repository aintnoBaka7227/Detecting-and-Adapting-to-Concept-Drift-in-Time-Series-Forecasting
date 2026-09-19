"""Figure F2 (post-tuning) -- synthetic drift-detection visual companions.

Regenerates the deterministic synthetic series for visualisation only and
reads back persisted detector alarm indices from
run_post_tune_on_synthetic.py's rows (post-tuning, frozen detector
configs) -- it never reruns a detector. One figure per (cadence, drift
type with an actual changepoint), one panel per seed, since the two
cadences were tuned and frozen separately and can show different
detections. See figure_f2_synthetic_common.py for the shared panel/figure
logic and produce_figure_f2_synthetic_pre_tune.py for the pre-tuning
sibling.
"""

from __future__ import annotations

from experiments.produce.figure_f2_synthetic_common import KINDS, build_figure, load_runs
from experiments.results_io import FIGURES_DIR
from experiments.run.detection.post_tune_detector_configs import CADENCES
from experiments.run.detection.run_post_tune_on_synthetic import split_id_for


def main() -> None:
    runs = load_runs()

    for cadence in CADENCES:
        split_id = split_id_for(cadence)

        for kind in KINDS:
            output = FIGURES_DIR / f"f2_synthetic_post_tune_{cadence}_{kind}.png"
            build_figure(
                kind=kind,
                runs=runs,
                split_id_filter=lambda s, split_id=split_id: s == split_id,
                title=(
                    f"F2 (post-tuning, {cadence}) — "
                    f"Synthetic {kind.capitalize()}-Drift Detection, all seeds"
                ),
                output=output,
            )
            print(f"wrote {output}")


if __name__ == "__main__":
    main()
