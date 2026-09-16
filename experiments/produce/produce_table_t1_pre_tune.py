"""T1 (pre-tuning) — detection on the synthetic benchmark, aggregated
across seeds.

Reads results/runs.csv (the one sanctioned top-level read) and group-bys it;
no metric is computed here that evaluation.py didn't already produce.

Pinned to run_pre_tune_on_synthetic.py's rows -- default (class-default,
untuned) detector hyperparameters, split_id "synth_n20000_cp*" (one
changepoint-tagged split_id per drift type; see
run_pre_tune_on_synthetic.py::split_id_for). This used to be the only T1 and
had no split_id filter at all, so once run_post_tune_on_synthetic.py's
"synthetic_full_series_20000_observations" rows existed in runs.csv, a
groupby(["method", "drift_type"]) here silently pooled both experiments'
seeds into the same row. See produce_table_t1_post_tune.py for the frozen
(post-tuning) sibling table this was split into.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t1_common import build_table
from experiments.results_io import TABLES_DIR

# Must match run_pre_tune_on_synthetic.py::N (its split_id_for() bakes N into
# the prefix). A pre-tuning run always uses this prefix family, never the
# single fixed post-tuning split_id.
PRE_TUNE_SPLIT_PREFIX = "synth_n20000_cp"


def main() -> None:
    table = build_table(lambda split_id: split_id.str.startswith(PRE_TUNE_SPLIT_PREFIX))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "table_t1_pre_tune_synthetic_detection.csv"
    table.to_csv(out_path, index=False)
    image_path = save_table_image(
        table,
        TABLES_DIR / "table_t1_pre_tune_synthetic_detection.png",
        title="T1 (pre-tuning) — Synthetic drift detection",
        subtitle="Detector hyperparameters are class defaults, not yet tuned against the false-alarm budget.",
    )
    print(table.to_string(index=False))
    print(f"\nwrote {out_path}")
    print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
