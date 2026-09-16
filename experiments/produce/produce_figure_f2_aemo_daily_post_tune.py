"""Figure F2 (post-tuning, daily-aggregated) -- AEMO drift detection, one
figure per detector and region. The canonical AEMO F2 pair.

Reads run_aemo_detectors_daily_post_tune.py's rows. Displays
`aggregate_daily_demand` verbatim -- the exact series that detector run
saw, not a separately-computed resample. See figure_f2_aemo_common.py for
the shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import build_all_figures, daily_aggregated_demand

SPLIT_ID = "aemo_detect_daily_post_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_daily_post_tune",
        demand_for_display=daily_aggregated_demand,
        y_label="Daily mean demand (MW)",
        title_note="Post-tuning detector configuration on daily-aggregated test demand",
        output_prefix="f2_detection_daily_post_tune",
    )


if __name__ == "__main__":
    main()
