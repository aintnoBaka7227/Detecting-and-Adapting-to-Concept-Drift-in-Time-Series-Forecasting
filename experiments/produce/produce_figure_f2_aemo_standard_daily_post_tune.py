"""Figure F2 (post-tuning, standard-daily) -- AEMO drift detection, one
figure per detector and region. The canonical AEMO F2, daily-cadence half.

Reads run_aemo_detectors_standard_daily_post_tune.py's rows. Displays the
standard-daily (deseasonalised + standardised) series verbatim -- the
exact series that detector run saw. See figure_f2_aemo_common.py for the
shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import build_all_figures, standard_daily_demand_for_display

SPLIT_ID = "aemo_detect_standard_daily_post_tune_v1"


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_standard_daily_post_tune",
        demand_for_display=standard_daily_demand_for_display,
        y_label="Standardised daily residual (z)",
        title_note="Post-tuning (daily-cadence) detector configuration on the standard-daily test stream",
        output_prefix="f2_detection_standard_daily_post_tune",
    )


if __name__ == "__main__":
    main()
