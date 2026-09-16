"""T1 (post-tuning) — detection on the synthetic benchmark, aggregated
across seeds, using the frozen detector configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep.

Sibling of produce_table_t1_pre_tune.py: same columns and aggregation
rules (see table_t1_common.py), pinned instead to
run_post_tune_on_synthetic.py's single split_id, so a pre-tuning re-run
can never leak into this table (or vice versa) through an un-filtered
groupby. Completes the pre/post-tuning x table-1/figure-2 matrix for the
synthetic benchmark alongside produce_figure_f2_synthetic_post_tune.py.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t1_common import build_table
from experiments.results_io import TABLES_DIR

# Must match run_post_tune_on_synthetic.py::SPLIT_ID.
POST_TUNE_SPLIT_ID = "synthetic_full_series_20000_observations"


def main() -> None:
    table = build_table(lambda split_id: split_id == POST_TUNE_SPLIT_ID)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "table_t1_post_tune_synthetic_detection.csv"
    table.to_csv(out_path, index=False)
    image_path = save_table_image(
        table,
        TABLES_DIR / "table_t1_post_tune_synthetic_detection.png",
        title="T1 (post-tuning) — Synthetic drift detection",
        subtitle="Detector hyperparameters are frozen from a synthetic-only false-alarm-budget sweep.",
    )
    print(table.to_string(index=False))
    print(f"\nwrote {out_path}")
    print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
