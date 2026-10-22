"""T7 -- the headline trigger comparison: Group A (us) vs Group B (Team 42).

Group A is Team 41's distribution-shift trigger, one configuration fixed at
the top of this file (TRIGGER, MODEL, ARM, UQ_METHOD, SIGNALS); every value
is read from runs.csv, mean ± sd across seeds per region:
    MAE during drift              adaptation rows, regime drift (as T3)
    Worst 24h coverage            UQ rows, regime drift (as T5)
    Review rate for 0.90 coverage escalation rows, regime drift (as T6)
    Detections raised             accepted test-period detections (as T2)
    Retrains triggered            n_retrains of the arm (as T4)
Group B is filled from GROUP_B_CSV when Team 42 provides it (columns:
metric, region, value -- metric as in the first column here); until then it
reads "awaiting Team 42". T7 is only a fair comparison if both groups share
the forecaster, data, arm, UQ method and escalation signal.

Usage:
    python -m experiments.produce.produce_table_t7
"""

from __future__ import annotations

import pandas as pd

from drift_lab.config import REGIONS
from experiments.produce.produce_figure_f5 import latest_escalation_rows
from experiments.produce.produce_table_t5 import latest_uq_rows
from experiments.produce.table_adaptation_common import (
    latest_adaptation_rows,
    mae_by_run,
)
from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLE7_DIR, output_path
from experiments.trigger_detectors import DETECTOR_LABELS

TRIGGER = "page_hinkley"
MODEL = "nhits"
ARM = "C_drift_full_history"
UQ_METHOD = "agaci"
SIGNALS = {"width": "interval width", "nonconformity_30min": "nonconformity, 30 min"}
DETECTION_SPLIT_ID = "aemo_detect_standard_half_hourly_post_tune_v1"
GROUP_B_CSV = TABLE7_DIR / "group_b_team42.csv"
PENDING = "awaiting Team 42"


def mean_sd(values: pd.Series, fmt: str) -> str:
    values = values.dropna()
    if values.empty:
        return "n/a"
    if len(values) == 1 or values.std() == 0:
        return format(values.mean(), fmt)
    return f"{format(values.mean(), fmt)} ± {format(values.std(), fmt)}"


def group_a(region: str) -> dict[str, str]:
    adaptation = latest_adaptation_rows(TRIGGER)
    per_run = mae_by_run(adaptation)
    arm = per_run[
        (per_run["model"] == MODEL)
        & (per_run["arm"] == ARM)
        & (per_run["region"] == region)
    ]

    uq = latest_uq_rows(TRIGGER)
    worst = uq[
        (uq["model"] == MODEL)
        & (uq["arm"] == ARM)
        & (uq["region"] == region)
        & (uq["uq"] == UQ_METHOD)
        & (uq["regime"] == "drift")
        & (uq["metric_name"] == "worst_24h_coverage")
    ]["metric_value"]

    runs = pd.read_csv(RUNS_CSV)
    detections = runs[
        (runs["split_id"] == DETECTION_SPLIT_ID)
        & (runs["method"] == TRIGGER)
        & (runs["region"] == region)
        & (runs["metric_name"] == "n_effective_detections")
    ].sort_values("timestamp")["metric_value"]

    values = {
        "MAE during drift (MW)": mean_sd(arm["drift"], ".0f"),
        "Worst 24h coverage": mean_sd(worst, ".2f"),
    }
    for signal, label in SIGNALS.items():
        esc = latest_escalation_rows(TRIGGER, signal)
        rate = esc[
            (esc["model"] == MODEL)
            & (esc["arm"] == ARM)
            & (esc["region"] == region)
            & (esc["uq"] == UQ_METHOD)
            & (esc["regime"] == "drift")
            & (esc["metric_name"] == "review_rate_for_coverage_0.90")
        ]["metric_value"]
        values[f"Review rate for 0.90 coverage ({label})"] = (
            "unreachable" if rate.isna().all() else mean_sd(rate.dropna(), ".1%")
        )
    values["Detections raised"] = (
        f"{detections.iloc[-1]:.0f}" if len(detections) else "n/a"
    )
    values["Retrains triggered"] = mean_sd(arm["n_retrains"], ".0f")
    return values


def group_b(region: str, metrics: list[str]) -> dict[str, str]:
    if not GROUP_B_CSV.exists():
        return dict.fromkeys(metrics, PENDING)
    team42 = pd.read_csv(GROUP_B_CSV)
    team42 = team42[team42["region"] == region].set_index("metric")["value"]
    return {m: str(team42.get(m, PENDING)) for m in metrics}


def build_table() -> pd.DataFrame:
    a = {region: group_a(region) for region in REGIONS}
    metrics = list(next(iter(a.values())))
    b = {region: group_b(region, metrics) for region in REGIONS}
    label = DETECTOR_LABELS[TRIGGER]
    table = pd.DataFrame({"metric": metrics})
    for region in REGIONS:
        table[f"Group A ({label}) {region}"] = [
            a[region][m] for m in metrics
        ]
    for region in REGIONS:
        table[f"Group B (Team 42) {region}"] = [
            b[region][m] for m in metrics
        ]
    return table


def main() -> None:
    table = build_table()
    out = output_path(TABLE7_DIR, TRIGGER, name=f"table_t7_trigger_comparison_{TRIGGER}.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")
    image = save_table_image(
        table,
        out.with_suffix(".png"),
        title="T7 — Trigger comparison: which trigger gives better-calibrated forecasts with less human work?",
        subtitle=(
            f"Group A: distribution-shift trigger, {DETECTOR_LABELS[TRIGGER]} on the standard-half-hourly seasonal residual; forecaster "
            f"{MODEL.upper()} (7 days -> 1 day ahead), arm C (retrain on all history at each accepted alarm), "
            f"{'AgACI' if UQ_METHOD == 'agaci' else 'fixed split conformal'} 90% intervals; drift = global "
            "drift regime; mean ± sd across 5 seeds; detections = accepted test-period alarms (14-day "
            "refractory). Group B: Team 42's error-based trigger. Same forecaster, data and conformal layer for both groups; only the trigger differs."
        ),
    )
    print(f"wrote {image}")


if __name__ == "__main__":
    main()
