"""T4 — adaptation cost comparison across the four required arms.

Reads adaptation forecast rows already written by the shared run harness.
No forecast metric is recalculated here.

Required columns:
    arm
    retrains
    cumulative train samples
    wall clock (s)
    MAE gain vs arm A
    gain per retrain
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_image import save_table_image
from experiments.results_io import FIGURES_DIR, RUNS_CSV, RUNS_DIR

ARM_ORDER = ("A", "B", "C", "D")

ARM_LABELS = {
    "A": "A — never retrain",
    "B": "B — retrain every 30 days",
    "C": "C — on drift, full history",
    "D": "D — on drift, recent window",
}

TABLE_COLUMNS = [
    "arm",
    "retrains",
    "cumulative train samples",
    "wall clock (s)",
    "MAE gain vs arm A",
    "gain per retrain",
]


def _arm_from_config(config_hash: str) -> str | None:
    """Read the arm identifier from the run's saved config."""
    path = RUNS_DIR / config_hash / "config.json"
    if not path.exists():
        return None

    config = pd.read_json(path, typ="series").to_dict()

    for key in ("arm", "adaptation_arm"):
        value = config.get(key)
        if value is not None:
            arm = str(value).strip().upper()
            if arm in ARM_ORDER:
                return arm

    return None


def _latest_run_batches(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest complete run batch for each arm/region/seed.

    Every metric emitted by one record_run() call shares its timestamp.
    Keeping the batch together prevents stale metrics from an older rerun
    leaking into the comparison.
    """
    keys = ["arm", "region", "seed"]

    latest = runs.groupby(keys, dropna=False)["timestamp"].transform("max")
    return runs[runs["timestamp"] == latest].copy()


def _mae_per_run(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the full-stream comparison MAE for each adaptation run.

    T4 compares overall forecast accuracy against Arm A. Regime MAEs must not
    be averaged because the regimes can contain different numbers of samples.
    The reviewed adaptation runner must therefore emit an explicit
    regime="full" MAE for T4.
    """
    mae = rows[
        (rows["metric_name"] == "mae")
        & (rows["regime"] == "full")
    ].copy()

    if mae.empty:
        raise SystemExit(
            "T4 requires explicit regime='full' MAE rows; "
            "do not average pre-drift/drift/post-drift MAEs."
        )

    return (
        mae.groupby(
            ["arm", "region", "seed"],
            dropna=False,
            as_index=False,
        )["metric_value"]
        .first()
        .rename(columns={"metric_value": "mae"})
    )


def build_table(split_id: str) -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(
            f"{RUNS_CSV} not found -- run the adaptation experiments first"
        )

    runs = pd.read_csv(RUNS_CSV)

    runs = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == split_id)
    ].copy()

    if runs.empty:
        raise SystemExit(
            f"no AEMO forecast rows for split_id={split_id!r}"
        )

    runs["arm"] = runs["config_hash"].map(_arm_from_config)
    runs = runs[runs["arm"].isin(ARM_ORDER)].copy()

    if runs.empty:
        raise SystemExit(
            "no adaptation runs with config arm=A/B/C/D were found"
        )

    runs = _latest_run_batches(runs)

    mae = _mae_per_run(runs)

    # Bookkeeping fields are repeated on every metric row from the same
    # record_run() call, so collapse them to one row per run.
    cost = (
        runs.groupby(
            ["arm", "region", "seed"],
            dropna=False,
            as_index=False,
        )
        .agg(
            retrains=("n_retrains", "first"),
            train_samples=("train_samples", "first"),
            wall_clock_s=("wall_clock_s", "first"),
        )
    )

    per_run = cost.merge(
        mae,
        on=["arm", "region", "seed"],
        how="inner",
        validate="one_to_one",
    )

    # Arm A is the control for each region/seed.
    arm_a = (
        per_run[per_run["arm"] == "A"][
            ["region", "seed", "mae"]
        ]
        .rename(columns={"mae": "arm_a_mae"})
    )

    per_run = per_run.merge(
        arm_a,
        on=["region", "seed"],
        how="left",
        validate="many_to_one",
    )

    if per_run["arm_a_mae"].isna().any():
        raise SystemExit(
            "T4 requires a matching Arm A control for every region/seed"
        )

    # Positive gain means lower MAE than never-retrain Arm A.
    per_run["mae_gain"] = per_run["arm_a_mae"] - per_run["mae"]

    per_run["gain_per_retrain"] = (
        per_run["mae_gain"]
        / per_run["retrains"].replace(0, pd.NA)
    )

    rows = []

    for arm in ARM_ORDER:
        group = per_run[per_run["arm"] == arm]
        if group.empty:
            continue

        rows.append(
            {
                "arm": ARM_LABELS[arm],
                "retrains": round(group["retrains"].mean(), 1),
                "cumulative train samples": round(
                    group["train_samples"].mean(),
                    1,
                ),
                "wall clock (s)": round(
                    group["wall_clock_s"].mean(),
                    2,
                ),
                "MAE gain vs arm A": round(
                    group["mae_gain"].mean(),
                    2,
                ),
                "gain per retrain": (
                    None
                    if group["gain_per_retrain"].dropna().empty
                    else round(
                        group["gain_per_retrain"].mean(),
                        3,
                    )
                ),
            }
        )

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-id",
        required=True,
        help="Adaptation experiment split_id to read from runs.csv.",
    )
    args = parser.parse_args()

    table = build_table(args.split_id)

    out_csv = FIGURES_DIR / "table_t4_adaptation_cost.csv"
    out_png = FIGURES_DIR / "table_t4_adaptation_cost.png"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    save_table_image(
        table,
        out_png,
        title="T4: Adaptation cost",
        subtitle=(
            "Positive MAE gain means lower forecast error than Arm A; "
            "gain per retrain is undefined for Arm A."
        ),
    )

    print(table.to_string(index=False))
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
