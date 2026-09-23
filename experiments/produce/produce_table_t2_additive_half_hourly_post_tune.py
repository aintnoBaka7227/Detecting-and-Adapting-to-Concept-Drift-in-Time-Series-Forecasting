"""Table T2 (post-tuning, additive-half-hourly) -- AEMO detection vs
documented events.

Second, independent deseasonalisation approach -- pinned to
run_aemo_detectors_additive_half_hourly_post_tune.py's rows (additive
overall-mean + day-of-year + weekday/half-hour offsets, summed and
subtracted, then standardised), using the same frozen winners from
run_fine_tune_on_synthetic.py every other post-tuning stream uses.
Reuses table_t2_common.py unchanged -- same columns and aggregation
rules as every other T2 sibling (produce_table_t2_standard_half_hourly_post_tune.py
etc.), differing only in which run's split_id it pins.

Output filenames are prefixed distinctly ("..._additive_half_hourly_...")
so they never collide with the standard-stream T2 tables.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match run_aemo_detectors_additive_half_hourly_post_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_additive_half_hourly_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_additive_half_hourly_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_additive_half_hourly_post_tune.png",
        title="T2 (post-tuning, additive-half-hourly) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on the additive-half-hourly stream (deseasonalised via "
            "additive_deseasonalise.remove_daily_weekly_profile and standardised), "
            "using detector hyperparameters frozen after synthetic-only tuning."
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
