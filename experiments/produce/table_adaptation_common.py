"""Shared loading for Tables T3 and T4 (adaptation arms).

Reads run_aemo_adaptation_arms.py's rows from runs.csv (method is
"<model>/<arm>") and keeps the latest record_run batch per (model, arm,
region, seed). Arm D is reported at the 180-day window only
(REPORTED_D_ARM) until the full sweep is run.

Two regime rules (TEAM_DECISIONS.md 5.3):
    global       every TEST timestamp, logged in runs.csv by the run /
                 rescore (`mae_by_run`) -- the official tables;
    event-local  only timestamps near an event, pooled here from each
                 run's saved curve (`mae_by_run_event_local`) -- the
                 comparison tables. runs.csv is not touched.
"""

from __future__ import annotations

import pandas as pd

from drift_lab import evaluation
from experiments import results_io
from experiments.results_io import RUNS_CSV
from experiments.run.adaptation.run_aemo_adaptation_arms import (
    ARM_A,
    arm_d,
    event_local_regime_labels,
)
from experiments.trigger_detectors import (
    CONTROL_DETECTOR,
    adaptation_split_id,
    detector_rows,
)

REGIMES = ("pre-drift", "drift", "post-drift")
D_PREFIX = "D_drift_recent_"
REPORTED_D_ARM = arm_d(180)
RUN_KEYS = ["model", "arm", "region", "seed_key"]


def latest_adaptation_rows(detector: str = CONTROL_DETECTOR) -> pd.DataFrame:
    """runs.csv rows of the latest run of every (model, arm, region, seed)
    for one trigger detector: its arms C and D plus the shared controls A
    and B (experiments.trigger_detectors.detector_rows)."""
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run run_aemo_adaptation_arms.py first")
    rows = detector_rows(pd.read_csv(RUNS_CSV), "adaptation", adaptation_split_id, detector).copy()
    if rows.empty:
        raise SystemExit(f"no adaptation rows for detector {detector!r} -- run run_aemo_adaptation_arms.py first")
    parts = rows["method"].str.split("/", n=1, expand=True)
    rows["model"], rows["arm"] = parts[0], parts[1]
    rows["seed_key"] = rows["seed"].fillna(-1)
    latest = rows.groupby(["model", "arm", "region", "seed_key"])["timestamp"].transform("max")
    return rows[rows["timestamp"] == latest]


def drop_unreported_d_arms(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep arms A-C and only the reported arm D window."""
    other = frame["arm"].str.startswith(D_PREFIX) & (frame["arm"] != REPORTED_D_ARM)
    return frame[~other].reset_index(drop=True)


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


def mae_by_run_event_local(rows: pd.DataFrame) -> pd.DataFrame:
    """`mae_by_run`'s shape, but each regime MAE pooled over the run's
    event-local timestamps only (evaluation.assign_event_local_regime),
    read from its saved curve. Timestamps outside every event window are
    left out. Cost fields come from runs.csv as in `mae_by_run`."""
    labels_by_region: dict[str, pd.Series] = {}
    records = []
    for (model, arm, region, seed_key, chash), _ in rows.groupby([*RUN_KEYS, "config_hash"]):
        seed = None if seed_key == -1 else int(seed_key)
        curve = pd.read_csv(
            results_io.curve_path(chash, "aemo", region, seed), parse_dates=["timestamp"]
        )
        index = pd.DatetimeIndex(curve["timestamp"])
        if region not in labels_by_region:
            labels_by_region[region] = event_local_regime_labels(index, region)
        labels = labels_by_region[region].reindex(index)
        inside = labels.notna().to_numpy()
        metrics = evaluation.calculate_forecast_metrics_by_regime(
            curve["actual"][inside], curve["forecast"][inside], labels[inside]
        )
        record = {"model": model, "arm": arm, "region": region, "seed_key": seed_key}
        for regime, values in metrics.items():
            record[regime.replace("_", "-")] = values["mae"]
        records.append(record)

    cost = rows.groupby(RUN_KEYS)[["n_retrains", "train_samples", "wall_clock_s"]].first()
    return pd.DataFrame(records).set_index(RUN_KEYS).join(cost).reset_index()


def pinball_by_run(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, arm, region, seed) with the whole-TEST interval
    pinball loss; empty when no run has intervals yet."""
    pinball = rows[(rows["metric_name"] == "pinball_loss") & (rows["regime"] == "full")]
    return pinball[["model", "arm", "region", "seed_key", "metric_value"]].rename(
        columns={"metric_value": "pinball_loss"}
    )


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
