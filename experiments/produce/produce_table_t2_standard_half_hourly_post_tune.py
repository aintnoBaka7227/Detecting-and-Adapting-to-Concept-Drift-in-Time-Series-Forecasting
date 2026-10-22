"""Table T2 (post-tuning, standard-half-hourly) -- AEMO detection vs
documented events.

Columns and aggregation rules live in table_t2_common.py. Pinned to
run_aemo_detectors_standard_half_hourly_post_tune.py's rows -- half-hourly
AEMO demand, deseasonalised (TRAIN-fitted profile removed) and
standardised (z-scored on TRAIN residual stats), using the frozen winners
from run_fine_tune_on_synthetic.py. This is the only T2: the project
reports the half-hourly stream, the one the adaptation arms use.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match run_aemo_detectors_standard_half_hourly_post_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_standard_half_hourly_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_standard_half_hourly_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_standard_half_hourly_post_tune.png",
        title="T2 (post-tuning, standard-half-hourly) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on the standard-half-hourly stream (deseasonalised and "
            "standardised half-hourly demand), using detector hyperparameters frozen "
            "after synthetic-only tuning."
        ),
    )

    print(table.to_string(index=False))
    print()
    for event in tier1_events().itertuples(index=False):
        print(f"  {event.event_id}  {event.event_name}")
    print(f"\n{DISCLAIMER}")
    print(f"\nwrote {out}")
    print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
