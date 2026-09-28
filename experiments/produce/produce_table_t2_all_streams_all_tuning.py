"""Table T2 (every input stream, every tuning stage) -- AEMO detection vs
documented events, one table across the full (input stream x tuning
stage) matrix.

Same columns as produce_table_t2_standard_daily_post_tune.py (see
table_t2_common.py), plus two new columns: `input_stream_type` and
`fine_tuned` (boolean). Built by calling `table_t2_common.build_table()`
once per one of the six single-input, single-tuning-stage experiments and
concatenating -- a pure reshape, no new metric.

Supersedes produce_table_t2_all_streams_post_tune.py's scope (post-tuning
only, three streams) by also including the three pre-tuning rows, but
does not replace it, or any of the six single-(stream, stage) tables --
this is an additional, widest view.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match each script's own SPLIT_ID.
COMBINATIONS = (
    ("raw_30min", False, "aemo_detect_raw_pre_tune_v1"),
    ("raw_30min", True, "aemo_detect_raw_post_tune_v1"),
    ("standard_daily", False, "aemo_detect_standard_daily_pre_tune_v1"),
    ("standard_daily", True, "aemo_detect_standard_daily_post_tune_v1"),
    ("standard_half_hourly", False, "aemo_detect_standard_half_hourly_pre_tune_v1"),
    ("standard_half_hourly", True, "aemo_detect_standard_half_hourly_post_tune_v1"),
)


def build_combined_table() -> pd.DataFrame:
    tables = []

    for input_stream_type, fine_tuned, split_id in COMBINATIONS:
        table = build_table(split_id)
        table.insert(1, "input_stream_type", input_stream_type)
        table.insert(2, "fine_tuned", fine_tuned)
        tables.append(table)

    return (
        pd.concat(tables, ignore_index=True)
        .sort_values(["input_stream_type", "fine_tuned", "region", "detector"])
        .reset_index(drop=True)
    )


def main() -> None:
    table = build_combined_table()
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_all_streams_all_tuning.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_all_streams_all_tuning.png",
        title="T2 (every input stream, every tuning stage) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "All three input streams (raw half-hourly, standard-daily, standard-half-hourly) "
            "x both tuning stages (class-default and post-tuning detector hyperparameters)."
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
