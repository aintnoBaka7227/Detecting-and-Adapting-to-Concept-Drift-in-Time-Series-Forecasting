"""Table T2 (pre-tuning, raw 30-minute) -- AEMO detection vs documented
events.

Sibling of produce_table_t2_daily_post_tune.py: identical columns and
aggregation rules (see table_t2_common.py), pinned instead to
run_aemo_detectors_raw_pre_tune.py's rows -- raw half-hourly AEMO demand,
plain class-default detector hyperparameters.
run_aemo_detectors_raw_pre_tune.py already runs exactly this (both-tier
matching, test-window-only scoring) to feed Figure F2; this script is
simply a second reader of its `split_id="aemo_detect_raw_pre_tune_v1"`
rows, not a new experiment.

Comparing this table to produce_table_t2_daily_pre_tune.py's output
isolates what daily aggregation bought before any tuning; comparing it to
produce_table_t2_daily_post_tune.py's output isolates the combined effect
of both daily aggregation and tuning.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match run_aemo_detectors_raw_pre_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_raw_pre_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_raw_pre_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_raw_pre_tune.png",
        title="T2 (pre-tuning, raw 30-minute) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on raw half-hourly AEMO demand, using default "
            "(untuned) detector hyperparameters."
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
