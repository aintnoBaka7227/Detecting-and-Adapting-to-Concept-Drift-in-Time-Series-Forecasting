"""Table T2 (post-tuning, standard-daily) -- AEMO detection vs documented
events.

Pure pivot of `results/runs.csv`. Reads the detection rows
`run_aemo_detectors_standard_daily_post_tune.py` logged (`delay_<event_id>`,
`precision`, `n_unmatched_events`) and lays them out wide: one row per
(detector, region), one column per Tier 1 event, in event date order.

"standard-daily" = daily-aggregated demand, deseasonalised (TRAIN-fitted
profile removed) and standardised (z-scored on TRAIN residual stats) --
see standard_stream_common.py. Detector hyperparameters are the
daily-cadence winners from run_fine_tune_on_synthetic.py.

Per the supervisor, T2 shows Tier 1 by name, and an unmatched detection
is reported as `unmatched`, never as a false positive.

This is the canonical, post-tuning, standard-daily T2. Siblings --
produce_table_t2_raw_pre_tune.py, produce_table_t2_standard_daily_pre_tune.py,
and produce_table_t2_standard_half_hourly_post_tune.py -- share every
column and rule here (see table_t2_common.py) but read a different
(tuning stage, input stream) combination instead.
"""

from __future__ import annotations

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLES_DIR

# Must match run_aemo_detectors_standard_daily_post_tune.py::SPLIT_ID.
SPLIT_ID = "aemo_detect_standard_daily_post_tune_v1"


def main() -> None:
    table = build_table(SPLIT_ID)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table_t2_aemo_events_standard_daily_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLES_DIR / "table_t2_aemo_events_standard_daily_post_tune.png",
        title="T2 (post-tuning, standard-daily) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Detection run on the standard-daily stream (deseasonalised and standardised "
            "daily-aggregated demand), using detector hyperparameters frozen after "
            "synthetic-only, daily-cadence tuning."
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
