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
reads "awaiting Team 42". SETTINGS (each team's trigger, forecaster, arm,
UQ method, escalation signals, regimes) is drawn above the metrics in the
image and saved next to it as a CSV, so differences between the two teams'
setups are visible rather than buried in a caption.

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
from experiments.produce.table_image import save_tables_image
from experiments.results_io import RUNS_CSV, TABLE7_DIR, output_path
from experiments.trigger_detectors import DETECTOR_LABELS

TRIGGER = "page_hinkley"
MODEL = "xgboost"
MODEL_LABELS = {"nhits": "NHITS", "xgboost": "XGBoost (gradient boosting)", "dhr_arima": "DHR + ARIMA"}
ARM = "D_drift_recent_180d"
UQ_METHOD = "agaci"
SIGNALS = {"width": "interval width", "nonconformity_30min": "nonconformity, 30 min"}
DETECTION_SPLIT_ID = "aemo_detect_standard_half_hourly_post_tune_v1"
GROUP_B_CSV = TABLE7_DIR / "group_b_team42.csv"
# Agreed setups of the two teams; the Team 41 column must match the constants above
# (its forecaster row follows MODEL).
SETTINGS = pd.DataFrame(
    [
        ("Trigger approach", "Distribution shift", "Forecast error"),
        ("Detector", "Page–Hinkley", "Page–Hinkley available"),
        ("Forecaster", MODEL_LABELS[MODEL], "Gradient Boosting (current Team 42)"),
        ("Adaptation", "Arm D · 180 days", "Arm D · 180 days"),
        ("Dataset / regions", "SA1 and NSW1", "SA1 and NSW1"),
        ("Seeds", "1–5", "1–5"),
        ("UQ method", "AgACI", "AgACI available"),
        (
            "Escalation signals",
            "Interval width, Lagged 30-minute nonconformity",
            "Interval width, Lagged 30-minute nonconformity",
        ),
        (
            "Regime definition",
            "Documented event + 7-day post-event buffer",
            "Documented event + 7-day post-event buffer",
        ),
    ],
    columns=["Setting", "Team 41 (Group A)", "Team 42 (Group B)"],
)
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
    settings_out = out.with_name(f"{out.stem}_settings.csv")
    SETTINGS.to_csv(settings_out, index=False)
    print(SETTINGS.to_string(index=False), "\n")
    print(table.to_string(index=False))
    print(f"wrote {out}\nwrote {settings_out}")
    image = save_tables_image(
        [("Settings", SETTINGS), ("Results", table)],
        out.with_suffix(".png"),
        title="T7 — Trigger comparison: which trigger gives better-calibrated forecasts with less human work?",
        subtitle=(
            "Results: mean ± sd across seeds per region; drift = global drift regime; review rate = the "
            "smallest share of drift-regime forecasts sent to review for the kept forecasts to reach 90% "
            "coverage; detections = accepted TEST-period alarms (14-day refractory)."
        ),
    )
    print(f"wrote {image}")


if __name__ == "__main__":
    main()
