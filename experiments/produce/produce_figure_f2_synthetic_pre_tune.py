"""Figure F2 (pre-tuning) -- synthetic drift-detection visual companions.

Sibling of produce_figure_f2_synthetic_post_tune.py: identical panel/figure
logic (see figure_f2_synthetic_common.py), reading
run_pre_tune_on_synthetic.py's rows instead -- class-default (untuned)
detector configs. run_pre_tune_on_synthetic.py logs one split_id per drift
type (it bakes the true changepoint positions into the split_id itself,
via its own split_id_for()), not one fixed split_id like the post-tuning
run, so the filter here matches the same "synth_n20000_cp*" prefix family
produce_table_t1_pre_tune.py pins to.
"""

from __future__ import annotations

from experiments.produce.figure_f2_synthetic_common import KINDS, build_figure, load_runs
from experiments.results_io import FIGURES_DIR

# Must match run_pre_tune_on_synthetic.py::N (its split_id_for() bakes N
# into the prefix).
PRE_TUNE_SPLIT_PREFIX = "synth_n20000_cp"


def main() -> None:
    runs = load_runs()

    for kind in KINDS:
        output = FIGURES_DIR / f"f2_synthetic_pre_tune_{kind}.png"
        build_figure(
            kind=kind,
            runs=runs,
            split_id_filter=lambda split_id: split_id.str.startswith(PRE_TUNE_SPLIT_PREFIX),
            title=f"F2 (pre-tuning) — Synthetic {kind.capitalize()}-Drift Detection, all seeds",
            output=output,
        )
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
