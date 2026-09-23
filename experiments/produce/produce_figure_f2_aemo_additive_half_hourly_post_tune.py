"""Figure F2 (post-tuning, additive-half-hourly) -- AEMO drift detection,
one figure per detector and region.

Reads run_aemo_detectors_additive_half_hourly_post_tune.py's rows.
Displays a daily-mean overlay of the additive-half-hourly (deseasonalised
+ standardised) series -- too dense to plot directly at half-hourly
resolution. Reuses figure_f2_aemo_common.build_all_figures() unchanged
(the same shared plotting logic every other AEMO F2 producer uses),
passing additive_stream_common.additive_half_hourly_demand_for_display
as the demand_for_display callable instead of editing that shared module.

Output filenames are prefixed distinctly
("f2_detection_additive_half_hourly_post_tune_...") so they never
collide with the standard-stream F2 figures.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import build_all_figures
from experiments.run.detection.additive_stream_common import additive_half_hourly_demand_for_display

SPLIT_ID = "aemo_detect_additive_half_hourly_post_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_additive_half_hourly_post_tune",
        demand_for_display=additive_half_hourly_demand_for_display,
        y_label="Daily mean standardised residual (z)",
        title_note="Post-tuning (frozen) detector configuration on the additive-half-hourly test stream",
        output_prefix="f2_detection_additive_half_hourly_post_tune",
    )


if __name__ == "__main__":
    main()
