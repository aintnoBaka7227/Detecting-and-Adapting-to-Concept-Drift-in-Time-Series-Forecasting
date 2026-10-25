"""Forecaster selection -- one table per selection experiment, and the pick.

SELECTIONS lists the experiments, each read from its own runs.csv rows
(latest run per model x region x seed):
    holdout       run_aemo_holdout_selection.py: fitted on TRAIN to Oct 2019,
                  validated on Nov-Dec 2019 (the supervisor's protocol);
    calibration   run_aemo_calibration_forecasts.py (stage 4): fitted on
                  TRAIN, validated on the Calibration window (Jan-Feb 2020).
Per model and region: 7-day rolling MAE (mean over the validation window),
MAE, fit time, predict time and their total, as mean ± sd across seeds
(deterministic models: one run). Seasonal naive is the baseline, not a
candidate: CANDIDATES drops its rows (logged by an earlier holdout run).

Selection rule, fixed before the run:
    1. per region, the lowest mean 7-day rolling MAE wins;
    2. if another model's mean is within one seed sd of the leader's (the
       larger of the two sds), the lower mean train + predict time wins;
    3. the selected forecaster is the model picked in every region; if the
       regions disagree, no single model is selected and the table says so.

Writes results/tables/forecaster_selection/<stem>.{csv,png} per experiment.

Usage:
    python -m experiments.produce.produce_table_forecaster_selection
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLES_DIR, output_path
from experiments.run.forecasting import (
    run_aemo_calibration_forecasts,
    run_aemo_holdout_selection,
)

METRICS = ("rolling_mae_7d_mean", "mae", "fit_seconds", "predict_seconds")
RUN_KEYS = ["method", "region", "seed_key"]
CANDIDATES = ("nhits", "xgboost", "dhr_arima")
SELECTIONS = {
    "holdout": {
        "split_id": run_aemo_holdout_selection.SPLIT_ID,
        "runner": "run_aemo_holdout_selection.py",
        "stem": "forecaster_selection_holdout_nov_dec_2019",
        "title": "Forecaster selection on a validation window cut from TRAIN (Nov–Dec 2019)",
        "setup": (
            "Each model fitted on TRAIN up to 31 Oct 2019, then forecasting the held-out Nov–Dec 2019 "
            "with its TEST protocol (NHITS: 48-step blocks with real context; xgboost / dhr_arima: blind). Neither the Calibration window nor TEST "
            "is used."
        ),
    },
    "calibration": {
        "split_id": run_aemo_calibration_forecasts.SPLIT_ID,
        "runner": "run_aemo_calibration_forecasts.py",
        "stem": "forecaster_selection",
        "title": "Forecaster selection on the Calibration window (Jan–Feb 2020)",
        "setup": (
            "Each model fitted on TRAIN, then forecasting the Calibration window with its TEST protocol "
            "(NHITS: 48-step blocks with real context; xgboost / dhr_arima: blind)."
        ),
    },
}


def latest_runs(split_id: str, runner: str) -> pd.DataFrame:
    """One row per (model, region, seed): its metrics and total wall clock."""
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run {runner} first")
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[(runs["split_id"] == split_id) & runs["method"].isin(CANDIDATES)].copy()
    if rows.empty:
        raise SystemExit(f"no rows for split_id={split_id!r} -- run {runner} first")
    rows["seed_key"] = rows["seed"].fillna(-1)
    rows = rows[
        rows["timestamp"] == rows.groupby(RUN_KEYS)["timestamp"].transform("max")
    ]
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
        if (
            row.rolling_mean - leader.rolling_mean <= tolerance
            and row.total_mean < leader.total_mean
        ):
            return (
                row.method,
                f"within one sd of {leader.method} on rolling MAE, and faster",
            )
    return leader.method, "lowest 7-day rolling MAE"


def mean_sd(mean: float, sd: float, digits: int) -> str:
    return (
        f"{mean:.{digits}f}" if pd.isna(sd) else f"{mean:.{digits}f} ± {sd:.{digits}f}"
    )


def build_table(selection: dict) -> tuple[pd.DataFrame, str]:
    stats = summarise(latest_runs(selection["split_id"], selection["runner"]))
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
                "predict on validation (s)": mean_sd(r.predict_mean, r.predict_sd, 2),
                "train + predict (s)": mean_sd(r.total_mean, r.total_sd, 1),
                "seeds": int(r.seeds),
                "pick": f"✓ ({reason})" if r.method == model else "",
            }
        )
    return pd.DataFrame(rows), verdict


def main() -> None:
    for selection in SELECTIONS.values():
        table, verdict = build_table(selection)
        out = output_path(
            TABLES_DIR, "forecaster_selection", name=f"{selection['stem']}.csv"
        )
        table.to_csv(out, index=False)
        print(f"\n{selection['title']}\n{table.to_string(index=False)}")
        print(f"{verdict}\nwrote {out}")
        image = save_table_image(
            table,
            out.with_suffix(".png"),
            title=selection["title"],
            subtitle=(
                f"{selection['setup']} Rule, fixed before the run: lowest mean 7-day rolling MAE per "
                "region; within one seed sd, the lower train + predict wall clock wins; the forecaster "
                "is the model picked in every region. Mean ± sd across seeds (deterministic models: "
                f"1 run). {verdict}."
            ),
        )
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
