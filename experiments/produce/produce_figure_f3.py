from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.produce.table_adaptation_common import (
    ARM_A,
    latest_adaptation_rows,
    mae_by_run,
    official_d_arms,
)
from experiments.results_io import FIGURES_DIR, RUNS_DIR


ROLLING_WINDOW = 336

ARM_B = "B_every_30_days"
ARM_C = "C_drift_full_history"


def latest_run_hashes() -> dict[tuple[str, str, str, int], str]:
    rows = latest_adaptation_rows()

    parts = rows["method"].str.split("/", n=1, expand=True)
    rows = rows.copy()
    rows["model"] = parts[0]
    rows["arm"] = parts[1]
    rows["seed_key"] = rows["seed"].fillna(-1).astype(int)

    latest = (
        rows.sort_values("timestamp")
        .groupby(["model", "arm", "region", "seed_key"], as_index=False)
        .tail(1)
    )

    return {
        (r.model, r.arm, r.region, int(r.seed_key)): r.config_hash
        for r in latest.itertuples()
    }


def load_curve(
    config_hash: str,
    region: str,
    seed_key: int,
) -> pd.DataFrame:
    seed_token = "none" if seed_key == -1 else str(seed_key)
    path = RUNS_DIR / config_hash / f"curve_aemo_{region}_{seed_token}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_retrains(
    config_hash: str,
    region: str,
    seed_key: int,
) -> pd.DatetimeIndex:
    seed_token = "none" if seed_key == -1 else str(seed_key)
    path = RUNS_DIR / config_hash / f"retrains_aemo_{region}_{seed_token}.csv"

    if not path.exists():
        return pd.DatetimeIndex([])

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DatetimeIndex(df["timestamp"])


def build_region_figure(region: str, model_name: str) -> Path:
    rows = latest_adaptation_rows()
    per_run = mae_by_run(rows)
    official_d = official_d_arms(per_run)

    hashes = latest_run_hashes()

    model_region = per_run[per_run["region"] == region].copy()
    if model_region.empty:
        raise RuntimeError(f"No adaptation runs found for region {region}")



    d_arm = official_d[(model_name, region)]
    arms = [ARM_A, ARM_B, ARM_C, d_arm]

    seed_values = sorted(
        model_region[
            (model_region["model"] == model_name)
            & (model_region["arm"].isin(arms))
        ]["seed_key"].unique()
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    arm_colors = {
        ARM_A: "0.72",
        ARM_B: "0.32",
        ARM_C: "tab:green",
        d_arm: "tab:red",
    }

    for arm in arms:
        curves = []

        for seed_key in seed_values:
            key = (model_name, arm, region, int(seed_key))
            config_hash = hashes.get(key)
            if config_hash is None:
                continue

            curve = load_curve(config_hash, region, int(seed_key))

            if (
                "rolling_mae_7d" not in curve.columns
                or curve["rolling_mae_7d"].isna().all()
            ):
                curve["rolling_mae_7d"] = (
                    curve["absolute_error"]
                    .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
                    .mean()
                )

            curves.append(
                curve.set_index("timestamp")["rolling_mae_7d"].rename(
                    str(seed_key)
                )
            )

        if not curves:
            continue

        combined = pd.concat(curves, axis=1)
        mean_curve = combined.mean(axis=1)

        retrain_count = 0

        for seed_key in seed_values:
            key = (model_name, arm, region, int(seed_key))
            config_hash = hashes.get(key)

            if config_hash is None:
                continue

            retrain_count += len(
                load_retrains(config_hash, region, int(seed_key))
            )

        if arm == d_arm:
            label = f"{arm} ({retrain_count} retrains, selected)"
        else:
            label = f"{arm} ({retrain_count} retrains)"

        ax.plot(
            mean_curve.index,
            mean_curve.values,
            label=label,
            color=arm_colors[arm],
        )

    # documented-event reference lines from regime transitions
    reference_curve = None

    for arm in arms:
        for seed_key in seed_values:
            key = (model_name, arm, region, int(seed_key))
            config_hash = hashes.get(key)

            if config_hash is not None:
                reference_curve = load_curve(
                    config_hash,
                    region,
                    int(seed_key),
                )
                break

        if reference_curve is not None:
            break

    if reference_curve is not None and "regime" in reference_curve.columns:
        changed = reference_curve["regime"].ne(reference_curve["regime"].shift())
        transition_times = reference_curve.loc[changed, "timestamp"].iloc[1:]
        for ts in transition_times:
            ax.axvline(ts, linestyle="--", linewidth=0.8, alpha=0.35)
    arm_a_pre = per_run[
        (per_run["model"] == model_name)
        & (per_run["arm"] == ARM_A)
        & (per_run["region"] == region)
    ]["pre-drift"].mean()

    if pd.notna(arm_a_pre):
        ax.axhline(
            arm_a_pre,
            linestyle=":",
            linewidth=1.5,
            color="0.25",
            label=f"Arm A pre-drift MAE ({arm_a_pre:.1f} MW)",
        )

    ax.set_title(f"F3 – Four-arm adaptation comparison – {region}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("7-day rolling MAE (MW)")
    ax.legend()
    ax.grid(alpha=0.2)

    # retraining rugs
    ymin, ymax = ax.get_ylim()
    rug_levels = [
        ymin + (ymax - ymin) * 0.015,
        ymin + (ymax - ymin) * 0.035,
        ymin + (ymax - ymin) * 0.055,
        ymin + (ymax - ymin) * 0.075,
    ]

    for arm, rug_y in zip(arms, rug_levels):
        retrain_times = []

        for seed_key in seed_values:
            key = (model_name, arm, region, int(seed_key))
            config_hash = hashes.get(key)
            if config_hash is None:
                continue
            retrain_times.extend(load_retrains(config_hash, region, int(seed_key)))

        if retrain_times:
            ax.plot(
                retrain_times,
                [rug_y] * len(retrain_times),
                "|",
                markersize=6,
                alpha=0.5,
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"f3_four_arm_comparison_{region}.png"

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    model_name = "nhits"

    for region in ("SA1","NSW1"):
        out = build_region_figure(region, model_name)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()