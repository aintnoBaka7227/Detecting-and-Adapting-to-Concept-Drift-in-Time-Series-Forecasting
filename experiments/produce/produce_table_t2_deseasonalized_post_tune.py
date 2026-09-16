"""Table T2 (post-tuning, deseasonalized) -- AEMO detection vs documented
events.

Sibling of produce_table_t2_daily_post_tune.py: identical columns and
aggregation rules (see table_t2_common.py), pinned instead to
run_aemo_detectors_deseasonalized_post_tune.py's rows -- half-hourly AEMO
demand with the expected weekday/half-hour profile removed
(remove_daily_weekly_profile), using the same post-tuning (frozen)
detector configs as the canonical daily-aggregated table.

Comparing this table to produce_table_t2_daily_post_tune.py's output
isolates deseasonalising vs. daily aggregation as competing ways to remove
the intraday/weekly seasonality that makes raw-demand detection fire so
often, holding the (post-tuning) detector configs fixed.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLES_DIR

# Must match run_aemo_detectors_deseasonalized_post_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_deseasonalized_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table_t2_aemo_events_deseasonalized_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLES_DIR / "table_t2_aemo_events_deseasonalized_post_tune.png",
        title="T2 (post-tuning, deseasonalized) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on deseasonalized half-hourly AEMO demand (weekday/half-hour "
            "profile removed), using detector hyperparameters frozen after synthetic-only tuning."
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
