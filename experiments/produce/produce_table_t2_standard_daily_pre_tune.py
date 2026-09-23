"""Table T2 (pre-tuning, standard-daily) -- AEMO detection vs documented
events.

Sibling of produce_table_t2_standard_daily_post_tune.py: identical
columns and aggregation rules (see table_t2_common.py), pinned instead to
run_aemo_detectors_standard_daily_pre_tune.py's rows -- same
standard-daily input stream, but class-default detector configs instead
of the post-tuning (frozen, daily-cadence) ones.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match run_aemo_detectors_standard_daily_pre_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_standard_daily_pre_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_standard_daily_pre_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_standard_daily_pre_tune.png",
        title="T2 (pre-tuning, standard-daily) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on the standard-daily stream (deseasonalised and standardised "
            "daily-aggregated demand), using class-default detector hyperparameters."
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
