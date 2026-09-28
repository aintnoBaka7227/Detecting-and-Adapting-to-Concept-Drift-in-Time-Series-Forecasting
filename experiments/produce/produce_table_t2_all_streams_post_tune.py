"""Table T2 (post-tuning, all input streams) -- AEMO detection vs
documented events, one combined table across every post-tuning input
stream.

Same columns as produce_table_t2_standard_daily_post_tune.py (see
table_t2_common.py), plus one new `input_stream_type` column, built by
calling `table_t2_common.build_table()` once per post-tuning single-input
experiment and concatenating the results -- not a new metric, purely a
reshape of three existing tables into one.

Requires all three post-tuning input streams to be matched against the
*same* (full Tier 1 + Tier 2) event catalogue for the rows to be
comparable. That's why run_aemo_detectors_raw_post_tune.py exists: an
earlier `run_post_tune_input_comparison_on_aemo.py` ran post-tuning
detectors on raw demand, but only against Tier 1 events, so its
tier2_contextual/unmatched figures weren't on the same footing as the
standard-daily and standard-half-hourly rows here -- it was retired once
this raw-input, both-tier-matched sibling existed (see DECISIONS.md).

Does not replace or overwrite produce_table_t2_standard_daily_post_tune.py
(the canonical single-input table) or either of its siblings -- this is
an additional, wider view.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_image import save_table_image
from experiments.produce.table_t2_common import DISCLAIMER, build_table, tier1_events
from experiments.results_io import TABLE2_DIR

# Must match each script's own SPLIT_ID.
INPUT_STREAM_SPLIT_IDS = {
    "raw_30min": "aemo_detect_raw_post_tune_v1",
    "standard_daily": "aemo_detect_standard_daily_post_tune_v1",
    "standard_half_hourly": "aemo_detect_standard_half_hourly_post_tune_v1",
}


def build_combined_table() -> pd.DataFrame:
    tables = []

    for input_stream_type, split_id in INPUT_STREAM_SPLIT_IDS.items():
        table = build_table(split_id)
        table.insert(1, "input_stream_type", input_stream_type)
        tables.append(table)

    return (
        pd.concat(tables, ignore_index=True)
        .sort_values(["input_stream_type", "region", "detector"])
        .reset_index(drop=True)
    )


def main() -> None:
    table = build_combined_table()
    TABLE2_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE2_DIR / "table_t2_aemo_events_all_streams_post_tune.csv"
    table.to_csv(out, index=False)
    image_path = save_table_image(
        table,
        TABLE2_DIR / "table_t2_aemo_events_all_streams_post_tune.png",
        title="T2 (post-tuning, all input streams) — AEMO detection vs. documented Tier 1 events",
        subtitle=(
            "Post-tuning (frozen) detector hyperparameters, shared across three input "
            "streams: raw half-hourly, standard-daily, and standard-half-hourly demand."
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
