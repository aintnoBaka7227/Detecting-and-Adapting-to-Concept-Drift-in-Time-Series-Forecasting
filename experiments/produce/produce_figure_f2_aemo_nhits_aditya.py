"""AEMO Figure F2 producer for the Aditya NHITS model.

This follows the same pattern as the raw AEMO detection F2 producers, but is
configured for the model's split_id and method name. It reuses the repo's
shared AEMO Figure F2 builder instead of duplicating plotting logic.

Important: the shared AEMO F2 helper expects the same event-matching inputs
used by detection runs (documented-event assignments, matched events, and
``delay_<event_id>`` metrics). If your model is only recording forecast rows,
this script will not produce a valid F2 because that data is not present.
"""

from __future__ import annotations

import experiments.produce.figure_f2_aemo_common as common

MODEL_METHOD = "nhits_aditya"
SPLIT_ID = "aemo_nhits_aditya_block7d_v1"

# Reuse the repo's shared AEMO Figure F2 builder, but switch the per-method list
# to the model under test so the script follows the same output conventions as
# the detector F2 producers.
common.DETECTORS = (MODEL_METHOD,)


def main() -> None:
    common.build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.forecasting.run_aemo_nhits_aditya",
        demand_for_display=common.daily_mean_of_raw_demand,
        y_label="Daily mean demand (MW)",
        title_note="Aditya NHITS model on raw half-hourly test demand",
        output_prefix="f2_nhits_aditya_raw",
    )


if __name__ == "__main__":
    main()
