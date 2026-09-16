"""Figure F2 (pre-tuning, daily-aggregated) -- AEMO drift detection, one
figure per detector and region.

Reads run_aemo_detectors_daily_pre_tune.py's rows. Displays
`aggregate_daily_demand` verbatim -- the exact series that detector run
saw, not a separately-computed resample. See figure_f2_aemo_common.py for
the shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import build_all_figures, daily_aggregated_demand

SPLIT_ID = "aemo_detect_daily_pre_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_daily_pre_tune",
        demand_for_display=daily_aggregated_demand,
        y_label="Daily mean demand (MW)",
        title_note="Default detector configuration on daily-aggregated test demand",
        output_prefix="f2_detection_daily_pre_tune",
    )


if __name__ == "__main__":
    main()
