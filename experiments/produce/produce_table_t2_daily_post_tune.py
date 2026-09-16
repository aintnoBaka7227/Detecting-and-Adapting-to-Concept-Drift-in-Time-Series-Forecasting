"""Table T2 (post-tuning, daily-aggregated) -- AEMO detection vs documented
events.

Pure pivot of `results/runs.csv`. Reads the detection rows
`run_aemo_detectors_daily_post_tune.py` logged (`delay_<event_id>`,
`precision`, `n_unmatched_events`) and lays them out wide: one row per
(detector, region), one column per Tier 1 event, in event date order.

Per the supervisor, T2 shows Tier 1 by name, and an unmatched detection
is reported as `unmatched`, never as a false positive. Print the
disclaimer beside the table and keep it in the report verbatim.

Because the underlying run matches against *both* Tier 1 and Tier 2
events (not Tier-1-only), this table can additionally show how many of
the Tier 1 events specifically were missed (`tier1_unmatched`), how many
Tier 2 contextual events were matched (`tier2_contextual`, from the
harness's `n_matched_t2`), alongside the overall unmatched-event count
across both tiers (`unmatched`).

This is the canonical, post-tuning, daily-aggregated T2. Siblings --
produce_table_t2_raw_pre_tune.py, produce_table_t2_daily_pre_tune.py, and
produce_table_t2_deseasonalized_post_tune.py -- share every column and
rule here (see table_t2_common.py) but read a different (tuning stage,
input processing) combination instead.

T1 makes T2 believable -- produce it first, and never present T2 alone.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLES_DIR

# Must match run_aemo_detectors_daily_post_tune.py::SPLIT_ID. Pinning it
# means running some other detection experiment (a different protocol /
# split) never silently changes this table -- the producer targets one
# specific experiment and selects the latest run *of that*.
SPLIT_ID = "aemo_detect_daily_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table_t2_aemo_events_daily_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLES_DIR / "table_t2_aemo_events_daily_post_tune.png",
        title="T2 (post-tuning, daily-aggregated) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on daily-aggregated AEMO demand, using detector "
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
