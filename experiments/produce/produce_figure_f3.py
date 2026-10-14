"""F3 — four-arm adaptation comparison.

Reads rolling-MAE curves and retraining timestamps already persisted by the
reviewed adaptation experiment. No forecast metric is recalculated here.

Required presentation:
- all four arms on one axis;
- 7-day rolling MAE in MW;
- retraining events shown as a rug beneath the curves;
- retrain count included in the legend.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from experiments import results_io

ARM_ORDER = ("A", "B", "C", "D")

ARM_LABELS = {
    "A": "A — never retrain",
    "B": "B — every 30 days",
    "C": "C — drift/full history",
    "D": "D — drift/recent window",
}

CONTROL_ARMS = {"A", "B"}


def _load_config(config_hash: str) -> dict:
    path = results_io.RUNS_DIR / config_hash / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(config_hash: str) -> str | None:
    config = _load_config(config_hash)

    for key in ("arm", "adaptation_arm"):
        value = config.get(key)
        if value is not None:
            candidate = str(value).strip().upper()
            if candidate in ARM_ORDER:
                return candidate

    return None


def _seed_value(value):
    return None if pd.isna(value) else int(value)


def _latest_runs(split_id: str, region: str) -> pd.DataFrame:
    if not results_io.RUNS_CSV.exists():
        raise SystemExit("results/runs.csv not found")

    runs = pd.read_csv(results_io.RUNS_CSV)

    runs = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["region"] == region)
        & (runs["split_id"] == split_id)
    ].copy()

    if runs.empty:
        raise SystemExit(
            f"no adaptation rows for region={region}, split_id={split_id}"
        )

    runs["arm"] = runs["config_hash"].map(_arm)
    runs = runs[runs["arm"].isin(ARM_ORDER)].copy()

    if runs.empty:
        raise SystemExit("no A/B/C/D adaptation runs found")

    # One record_run() call emits multiple metric rows sharing one timestamp.
    latest = runs.groupby(
        ["arm", "seed"],
        dropna=False,
    )["timestamp"].transform("max")

    return runs[runs["timestamp"] == latest].copy()


def _run_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["arm", "seed"],
            dropna=False,
            as_index=False,
        )
        .agg(
            config_hash=("config_hash", "first"),
            n_retrains=("n_retrains", "first"),
        )
    )


def _load_curve(row, region: str) -> pd.Series:
    seed = _seed_value(row.seed)

    path = results_io.curve_path(
        row.config_hash,
        "aemo",
        region,
        seed,
    )

    if not path.exists():
        raise SystemExit(f"missing rolling-MAE curve: {path}")

    frame = pd.read_csv(path, parse_dates=["index"])

    return pd.Series(
        frame["rolling_mae_7d"].to_numpy(),
        index=pd.DatetimeIndex(frame["index"]),
        name=row.arm,
    )


def _load_retrains(row, region: str) -> pd.DatetimeIndex:
    seed = _seed_value(row.seed)

    path = results_io.retrain_events_path(
        row.config_hash,
        "aemo",
        region,
        seed,
    )

    if not path.exists():
        if int(row.n_retrains) == 0:
            return pd.DatetimeIndex([])

        raise SystemExit(
            f"run reports {int(row.n_retrains)} retrains "
            f"but retrain artifact is missing: {path}"
        )

    frame = pd.read_csv(path, parse_dates=["timestamp"])
    stamps = pd.DatetimeIndex(frame["timestamp"])

    if len(stamps) != int(row.n_retrains):
        raise SystemExit(
            f"retrain artifact/count mismatch for Arm {row.arm}: "
            f"{len(stamps)} timestamps vs {int(row.n_retrains)} logged"
        )

    return stamps


def produce(split_id: str, region: str) -> None:
    rows = _run_metadata(_latest_runs(split_id, region))

    missing = [
        arm
        for arm in ARM_ORDER
        if arm not in set(rows["arm"])
    ]
    if missing:
        raise SystemExit(
            "F3 requires all four adaptation arms; missing: "
            + ", ".join(missing)
        )

    # F3 is a four-arm comparison. If stochastic experiments eventually use
    # multiple seeds, a reviewed aggregation policy must be chosen rather than
    # silently selecting one seed for the figure.
    counts = rows.groupby("arm", dropna=False).size()
    if (counts > 1).any():
        raise SystemExit(
            "F3 found multiple seeds per arm. "
            "Define the reviewed seed-aggregation policy before plotting."
        )

    rows["order"] = rows["arm"].map(
        {arm: i for i, arm in enumerate(ARM_ORDER)}
    )
    rows = rows.sort_values("order")

    fig, ax = plt.subplots(figsize=(12, 6))

    retrain_sets = {}

    for row in rows.itertuples():
        curve = _load_curve(row, region)
        retrains = _load_retrains(row, region)
        retrain_sets[row.arm] = retrains

        label = (
            f"{ARM_LABELS[row.arm]} "
            f"(retrains={int(row.n_retrains)})"
        )

        if row.arm in CONTROL_ARMS:
            ax.plot(
                curve.index,
                curve.to_numpy(),
                label=label,
                linewidth=1.4,
                alpha=0.55,
                color="0.55",
                linestyle="--" if row.arm == "B" else "-",
            )
        else:
            ax.plot(
                curve.index,
                curve.to_numpy(),
                label=label,
                linewidth=1.7,
            )

    # Separate rug rows below the x-axis prevent overlapping retrain marks
    # from becoming unreadable.
    rug_y = {
        "A": -0.035,
        "B": -0.055,
        "C": -0.075,
        "D": -0.095,
    }

    transform = ax.get_xaxis_transform()

    for arm in ARM_ORDER:
        stamps = retrain_sets[arm]
        if len(stamps) == 0:
            continue

        ax.vlines(
            stamps,
            rug_y[arm] - 0.008,
            rug_y[arm] + 0.008,
            transform=transform,
            linewidth=0.8,
            clip_on=False,
        )

    ax.set_title(
        f"F3: Four-arm adaptation comparison — {region}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("7-day rolling MAE (MW)")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.02,
        "Rug marks beneath the axis show actual model retraining events. "
        "Rolling window: 7 days.",
        ha="center",
        fontsize=9,
    )

    results_io.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        results_io.FIGURES_DIR
        / f"f3_four_arm_comparison_{region.lower()}.png"
    )

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-id",
        required=True,
        help="Reviewed four-arm adaptation experiment split_id.",
    )
    parser.add_argument(
        "--region",
        required=True,
        choices=("SA1", "NSW1"),
    )
    args = parser.parse_args()

    produce(args.split_id, args.region)


if __name__ == "__main__":
    main()
