"""Figure F2 (post-tuning, deseasonalized) -- AEMO drift detection, one
figure per detector and region.

Sibling of produce_figure_f2_aemo_deseasonalized_pre_tune.py: same
deseasonalized half-hourly input and display series, but reads
run_aemo_detectors_deseasonalized_post_tune.py's rows -- post-tuning
(frozen) detector configs instead of class defaults. See
figure_f2_aemo_common.py for the shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import (
    build_all_figures,
    daily_mean_of_deseasonalized_demand,
)

SPLIT_ID = "aemo_detect_deseasonalized_post_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_deseasonalized_post_tune",
        demand_for_display=daily_mean_of_deseasonalized_demand,
        y_label="Daily mean deseasonalized demand (MW)",
        title_note="Post-tuning detector configuration on deseasonalized half-hourly test demand",
        output_prefix="f2_detection_deseasonalized_post_tune",
    )


if __name__ == "__main__":
    main()
