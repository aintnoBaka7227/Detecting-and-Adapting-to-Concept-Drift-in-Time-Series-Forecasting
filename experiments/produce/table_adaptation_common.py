"""Shared loading for Tables T3 and T4 (adaptation arms).

Reads run_aemo_adaptation_arms.py's rows from runs.csv (method is
"<model>/<arm>"), keeps the latest record_run batch per (model, arm,
region, seed), and marks the official arm D window per (model, region):
the recent-window run with the lowest seed-mean drift-regime MAE
(TEAM_DECISIONS.md 5.1). Every D window stays in the tables.
"""

from __future__ import annotations

import pandas as pd

from experiments.results_io import RUNS_CSV
from experiments.run.forecasting.run_aemo_adaptation_arms import ARM_A, SPLIT_ID

REGIMES = ("pre-drift", "drift", "post-drift")
D_PREFIX = "D_drift_recent_"


def latest_adaptation_rows() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run run_aemo_adaptation_arms.py first")
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[(runs["split_id"] == SPLIT_ID) & (runs["group"] == "adaptation")].copy()
    if rows.empty:
        raise SystemExit(f"no rows for split_id={SPLIT_ID!r} -- run run_aemo_adaptation_arms.py first")
    parts = rows["method"].str.split("/", n=1, expand=True)
    rows["model"], rows["arm"] = parts[0], parts[1]
    rows["seed_key"] = rows["seed"].fillna(-1)
    latest = rows.groupby(["model", "arm", "region", "seed_key"])["timestamp"].transform("max")
    return rows[rows["timestamp"] == latest]


def mae_by_run(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, arm, region, seed) with a column per regime MAE
    plus the run's cost fields."""
    mae = rows[(rows["metric_name"] == "mae") & rows["regime"].isin(REGIMES)]
    wide = mae.pivot_table(
        index=["model", "arm", "region", "seed_key"], columns="regime", values="metric_value"
    )
    cost = rows.groupby(["model", "arm", "region", "seed_key"])[
        ["n_retrains", "train_samples", "wall_clock_s"]
    ].first()
    return wide.join(cost).reset_index()


def official_d_arms(per_run: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(model, region) -> the D window with the lowest seed-mean drift MAE."""
    d = per_run[per_run["arm"].str.startswith(D_PREFIX)]
    means = d.groupby(["model", "region", "arm"])["drift"].mean().reset_index()
    best = means.loc[means.groupby(["model", "region"])["drift"].idxmin()]
    return {(r.model, r.region): r.arm for r in best.itertuples()}


def arm_label(arm: str, model: str, region: str, official: dict) -> str:
    if arm.startswith(D_PREFIX):
        tag = " (selected)" if official.get((model, region)) == arm else ""
        return f"{arm}{tag}"
    return arm


def mean_sd(values: pd.Series, digits: int = 1) -> str:
    valid = values.dropna()
    if valid.empty:
        return "n/a"
    if len(valid) == 1:
        return f"{valid.iloc[0]:.{digits}f} ± n/a"
    return f"{valid.mean():.{digits}f} ± {valid.std():.{digits}f}"


ARM_ORDER = {ARM_A: 0, "B_every_30_days": 1, "C_drift_full_history": 2}


def arm_sort_key(arm: str) -> tuple:
    if arm in ARM_ORDER:
        return (ARM_ORDER[arm], 0)
    return (3, int(arm.removeprefix(D_PREFIX).removesuffix("d")))
