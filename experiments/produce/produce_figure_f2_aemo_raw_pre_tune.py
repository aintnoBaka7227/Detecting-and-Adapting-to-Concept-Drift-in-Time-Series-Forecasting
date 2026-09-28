"""Figure F2 (pre-tuning, raw 30-minute) -- AEMO drift detection, one figure
per detector and region.

Reads run_aemo_detectors_raw_pre_tune.py's rows. Displays a generic
daily-mean resample of raw demand for readability (the detector itself saw
the full raw half-hourly stream, too dense to plot directly); detections
and documented events are plotted at their real timestamps regardless.
See figure_f2_aemo_common.py for the shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import build_all_figures, daily_mean_of_raw_demand

SPLIT_ID = "aemo_detect_raw_pre_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_raw_pre_tune",
        demand_for_display=daily_mean_of_raw_demand,
        y_label="Daily mean demand (MW)",
        title_note="Default detector configuration on raw half-hourly test demand",
        output_prefix="f2_detection_raw_pre_tune",
    )


if __name__ == "__main__":
    main()
