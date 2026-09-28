"""Table T2 (post-tuning, raw 30-minute) -- AEMO detection vs documented
events.

Sibling of produce_table_t2_daily_post_tune.py: identical columns and
aggregation rules (see table_t2_common.py), pinned instead to
run_aemo_detectors_raw_post_tune.py's rows -- raw half-hourly AEMO demand,
post-tuning (frozen) detector configs.

Comparing this table to produce_table_t2_raw_pre_tune.py's output isolates
what tuning bought on raw demand, holding the input processing fixed;
comparing it to produce_table_t2_daily_post_tune.py's output isolates the
combined effect of both daily aggregation and tuning.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match run_aemo_detectors_raw_post_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_raw_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_raw_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_raw_post_tune.png",
        title="T2 (post-tuning, raw 30-minute) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on raw half-hourly AEMO demand, using detector "
            "hyperparameters frozen after synthetic-only tuning."
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
