"""Forecaster selection -- stage 4's Calibration-window evidence, and the pick.

Reads run_aemo_calibration_forecasts.py's rows (latest run per model x region
x seed) and reports per model and region: 7-day rolling MAE (mean over the
Calibration window), MAE, fit time on TRAIN, predict time on Calibration and
their total, as mean ± sd across seeds (dhr_arima: one deterministic run).

Selection rule, fixed before the run:
    1. per region, the lowest mean 7-day rolling MAE wins;
    2. if another model's mean is within one seed sd of the leader's (the
       larger of the two sds), the lower mean train + predict time wins;
    3. the selected forecaster is the model picked in every region; if the
       regions disagree, no single model is selected and the table says so.
Seasonal naive is not a candidate.

Writes results/tables/forecaster_selection/forecaster_selection.{csv,png}.

Usage:
    python -m experiments.produce.produce_table_forecaster_selection
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLES_DIR, output_path
from experiments.run.forecasting.run_aemo_calibration_forecasts import SPLIT_ID

METRICS = ("rolling_mae_7d_mean", "mae", "fit_seconds", "predict_seconds")
RUN_KEYS = ["method", "region", "seed_key"]


def latest_runs() -> pd.DataFrame:
    """One row per (model, region, seed): its metrics and total wall clock."""
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run run_aemo_calibration_forecasts.py first")
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[runs["split_id"] == SPLIT_ID].copy()
    if rows.empty:
        raise SystemExit(f"no rows for split_id={SPLIT_ID!r} -- run run_aemo_calibration_forecasts.py first")
    rows["seed_key"] = rows["seed"].fillna(-1)
    rows = rows[rows["timestamp"] == rows.groupby(RUN_KEYS)["timestamp"].transform("max")]
    per_run = rows[rows["metric_name"].isin(METRICS)].pivot_table(
        index=RUN_KEYS, columns="metric_name", values="metric_value"
    )
    per_run["total_seconds"] = rows.groupby(RUN_KEYS)["wall_clock_s"].first()
    return per_run.reset_index()


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    stats = per_run.groupby(["region", "method"]).agg(
        rolling_mean=("rolling_mae_7d_mean", "mean"),
        rolling_sd=("rolling_mae_7d_mean", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        fit_mean=("fit_seconds", "mean"),
        fit_sd=("fit_seconds", "std"),
        predict_mean=("predict_seconds", "mean"),
        predict_sd=("predict_seconds", "std"),
        total_mean=("total_seconds", "mean"),
        total_sd=("total_seconds", "std"),
        seeds=("seed_key", "nunique"),
    )
    return stats.reset_index()


def pick(region_stats: pd.DataFrame) -> tuple[str, str]:
    """(selected model, reason) for one region, by the rule in the docstring."""
    ranked = region_stats.sort_values("rolling_mean").reset_index(drop=True)
    leader = ranked.iloc[0]
    for row in ranked.iloc[1:].itertuples():
        tolerance = max(pd.Series([leader.rolling_sd, row.rolling_sd]).fillna(0))
        if row.rolling_mean - leader.rolling_mean <= tolerance and row.total_mean < leader.total_mean:
            return row.method, f"within one sd of {leader.method} on rolling MAE, and faster"
    return leader.method, "lowest 7-day rolling MAE"


def mean_sd(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f}" if pd.isna(sd) else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def build_table() -> tuple[pd.DataFrame, str]:
    stats = summarise(latest_runs())
    picks = {region: pick(g) for region, g in stats.groupby("region")}
    chosen = {model for model, _ in picks.values()}
    verdict = (
        f"Selected forecaster: {chosen.pop()} (picked in every region)"
        if len(chosen) == 1
        else "No single forecaster: the regions pick different models"
    )

    rows = []
    for r in stats.sort_values(["region", "rolling_mean"]).itertuples():
        model, reason = picks[r.region]
        rows.append(
            {
                "region": r.region,
                "model": r.method,
                "7-day rolling MAE (MW)": mean_sd(r.rolling_mean, r.rolling_sd, 1),
                "MAE (MW)": mean_sd(r.mae_mean, r.mae_sd, 1),
                "fit on TRAIN (s)": mean_sd(r.fit_mean, r.fit_sd, 1),
                "predict on Calibration (s)": mean_sd(r.predict_mean, r.predict_sd, 2),
                "train + predict (s)": mean_sd(r.total_mean, r.total_sd, 1),
                "seeds": int(r.seeds),
                "pick": f"✓ ({reason})" if r.method == model else "",
            }
        )
    return pd.DataFrame(rows), verdict


def main() -> None:
    table, verdict = build_table()
    out = output_path(TABLES_DIR, "forecaster_selection", name="forecaster_selection.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\n{verdict}\nwrote {out}")
    image = save_table_image(
        table,
        out.with_suffix(".png"),
        title="Forecaster selection on the Calibration window (Jan–Feb 2020)",
        subtitle=(
            "Each model fitted on TRAIN, then forecasting the Calibration window with its TEST protocol "
            "(NHITS: 48-step blocks with real context; xgboost / dhr_arima: blind). Rule, fixed before "
            "the run: lowest mean 7-day rolling MAE per region; within one seed sd, the lower "
            "train + predict wall clock wins; the forecaster is the model picked in every region. "
            f"Mean ± sd across seeds (dhr_arima is deterministic: 1 run). {verdict}."
        ),
    )
    print(f"wrote {image}")


if __name__ == "__main__":
    main()
