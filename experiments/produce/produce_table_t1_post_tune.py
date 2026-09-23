"""T1 (post-tuning) — detection on the synthetic benchmark, aggregated
across seeds, using the frozen detector configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep.

Sibling of produce_table_t1_pre_tune.py: same columns and aggregation
rules (see table_t1_common.py), pinned to run_post_tune_on_synthetic.py's
own split_id, so a pre-tuning re-run can never leak into this table
through an un-filtered groupby. Completes the pre/post-tuning x
table-1/figure-2 matrix for the synthetic benchmark alongside
produce_figure_f2_synthetic_post_tune.py.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t1_common import build_table, latest_synthetic_detection_rows
from experiments.results_io import TABLE1_DIR
from experiments.run.detection.run_post_tune_on_synthetic import SAMPLES_PER_YEAR, SPLIT_ID

OUTPUT_CSV = TABLE1_DIR / "table_t1_post_tune_synthetic_detection.csv"
OUTPUT_PNG = TABLE1_DIR / "table_t1_post_tune_synthetic_detection.png"


def add_false_alarms_per_year_column(table, samples_per_year: int):
    """Merge in the annualised false-alarm rate the eligibility rule was
    actually checked against -- table_t1_common's own columns only ever
    show the cadence-agnostic per-10k rate, which isn't comparable to the
    <=2/year budget without doing this conversion by hand (see the
    false_alarms_per_year rows run_post_tune_on_synthetic.py logs via
    record_run's samples_per_year= argument)."""

    det = latest_synthetic_detection_rows(lambda s: s == SPLIT_ID)
    column = f"false alarms / year ({samples_per_year})"

    fa_year = (
        det[det["metric_name"] == "false_alarms_per_year"]
        .groupby(["method", "drift_type"])["metric_value"]
        .mean()
        .round(2)
        .rename(column)
        .reset_index()
        .rename(columns={"method": "detector", "drift_type": "drift type"})
    )

    table = table.merge(fa_year, on=["detector", "drift type"], how="left")

    # Keep the new column next to the existing per-10k one instead of
    # trailing after "seeds".
    columns = list(table.columns)
    columns.remove(column)
    insert_at = columns.index("false alarms / 10k") + 1
    columns.insert(insert_at, column)
    return table[columns]


def main() -> None:
    TABLE1_DIR.mkdir(parents=True, exist_ok=True)

    table = build_table(lambda s: s == SPLIT_ID)
    table = add_false_alarms_per_year_column(table, SAMPLES_PER_YEAR)

    table.to_csv(OUTPUT_CSV, index=False)
    image_path = save_table_image(
        table,
        OUTPUT_PNG,
        title="T1 (post-tuning) — Synthetic drift detection",
        subtitle="Detector hyperparameters are frozen from a synthetic-only false-alarm-budget sweep.",
    )
    print(table.to_string(index=False))
    print(f"\nwrote {OUTPUT_CSV}")
    print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
