"""Figure F2 (post-tuning, standard-half-hourly) -- AEMO drift detection,
one figure per detector and region. The canonical AEMO F2, half-hourly
cadence half.

Reads run_aemo_detectors_standard_half_hourly_post_tune.py's rows.
Displays a daily-mean overlay of the standard-half-hourly (deseasonalised
+ standardised) series -- too dense to plot directly at half-hourly
resolution. The KSWIN and Page-Hinkley figures add a bottom panel with
each detector's replayed test value at every half-hourly step against its
firing threshold (KS -log10(p) vs alpha; PH cumulative deviation vs
lambda). ADWIN has none: river exposes no threshold statistic for it, and
its window width is just time since the last reset.
See figure_f2_aemo_common.py for the shared plotting logic.
"""

from __future__ import annotations

from experiments.produce.figure_f2_aemo_common import (
    StatisticTrace,
    build_all_figures,
    kswin_p_value_trace,
    page_hinkley_statistic_trace,
    standard_half_hourly_demand_for_display,
)

SPLIT_ID = "aemo_detect_standard_half_hourly_post_tune_v1"


_STATISTIC_TRACES = {
    "kswin": kswin_p_value_trace,
    "page_hinkley": page_hinkley_statistic_trace,
}


def statistic_trace(method: str, region: str) -> StatisticTrace | None:
    trace = _STATISTIC_TRACES.get(method)
    return trace(region, "half_hourly") if trace else None


def main() -> None:
    build_all_figures(
        split_id=SPLIT_ID,
        run_script_hint="experiments.run.detection.run_aemo_detectors_standard_half_hourly_post_tune",
        demand_for_display=standard_half_hourly_demand_for_display,
        y_label="Daily mean standardised residual (z)",
        title_note="Post-tuning (frozen) detector configuration on the standard-half-hourly test stream",
        output_prefix="f2_detection_standard_half_hourly_post_tune",
        statistic_trace=statistic_trace,
    )


if __name__ == "__main__":
    main()
