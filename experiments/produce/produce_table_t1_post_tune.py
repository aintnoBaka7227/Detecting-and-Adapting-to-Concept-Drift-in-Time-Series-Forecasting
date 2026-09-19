"""T1 (post-tuning) — detection on the synthetic benchmark, aggregated
across seeds, using the frozen detector configs chosen by
run_fine_tune_on_synthetic.py's synthetic-only sweep -- run once per
cadence, since the winners (and therefore their delay / false-alarm /
missed numbers) were tuned and frozen separately per cadence (see
post_tune_detector_configs.py) rather than shared between them.

Sibling of produce_table_t1_pre_tune.py: same columns and aggregation
rules (see table_t1_common.py), pinned to each cadence's own
run_post_tune_on_synthetic.py split_id (via split_id_for()), so a
pre-tuning re-run -- or the other cadence's post-tuning run -- can never
leak into this table through an un-filtered groupby. Completes the
pre/post-tuning x table-1/figure-2 matrix for the synthetic benchmark
alongside produce_figure_f2_synthetic_post_tune.py.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t1_common import build_table, latest_synthetic_detection_rows
from experiments.results_io import TABLES_DIR
from experiments.run.detection.post_tune_detector_configs import CADENCES
from experiments.run.detection.run_post_tune_on_synthetic import SAMPLES_PER_YEAR, split_id_for


def add_false_alarms_per_year_column(
    table,
    split_id: str,
    samples_per_year: int,
):
    """Merge in the annualised false-alarm rate this cadence's eligibility
    rule was actually checked against -- table_t1_common's own columns
    only ever show the cadence-agnostic per-10k rate, which isn't
    comparable to the <=2/year budget without doing this conversion by
    hand (see the false_alarms_per_year rows run_post_tune_on_synthetic.py
    logs via record_run's samples_per_year= argument)."""

    det = latest_synthetic_detection_rows(lambda s, split_id=split_id: s == split_id)
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
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for cadence in CADENCES:
        split_id = split_id_for(cadence)
        samples_per_year = SAMPLES_PER_YEAR[cadence]

        table = build_table(lambda s, split_id=split_id: s == split_id)
        table = add_false_alarms_per_year_column(table, split_id, samples_per_year)

        out_path = TABLES_DIR / f"table_t1_post_tune_synthetic_detection_{cadence}.csv"
        table.to_csv(out_path, index=False)
        image_path = save_table_image(
            table,
            TABLES_DIR / f"table_t1_post_tune_synthetic_detection_{cadence}.png",
            title=f"T1 (post-tuning, {cadence}) — Synthetic drift detection",
            subtitle=(
                "Detector hyperparameters are frozen from a synthetic-only "
                f"false-alarm-budget sweep tuned for the {cadence} cadence."
            ),
        )
        print(table.to_string(index=False))
        print(f"\nwrote {out_path}")
        print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
