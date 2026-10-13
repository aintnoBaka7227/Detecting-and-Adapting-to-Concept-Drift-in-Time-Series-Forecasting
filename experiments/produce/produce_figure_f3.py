"""F3 -- four-arm comparison: 7-day rolling MAE through the TEST period.

One figure per model x region (the MW scales differ by region and by
model). Each shows exactly four lines -- arms A, B, C and the official
arm D window -- as the mean 7-day (336-observation) rolling MAE across
seeds, with a ±1 standard-deviation band when the model has more than one
seed. Controls A and B are grey; the drift-triggered arms C and D are in
colour. The dashed horizontal line is arm A's pre-drift MAE. Below the
curves, one rug row per arm marks every retrain.

Reads the saved curve and retrain files of the latest adaptation runs; no
metric is recalculated here.

Usage:
    python -m experiments.produce.produce_figure_f3
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import results_io
from experiments.produce.table_adaptation_common import (
    D_PREFIX,
    latest_adaptation_rows,
    mae_by_run,
    official_d_arms,
)
from experiments.run.adaptation.run_aemo_adaptation_arms import ARM_A, ARM_B, ARM_C

MODEL_NAMES = {"nhits": "NHITS", "xgboost": "XGBoost", "dhr_arima": "DHR + ARIMA"}

# (label, colour, line style, line width). Controls grey, drift arms in colour.
ARM_STYLE = {
    "A": ("A  never retrain", "#52514e", "-", 1.1),
    "B": ("B  every 30 days", "#a3a29c", (0, (5, 2)), 1.1),
    "C": ("C  drift alarm, full history", "#2a78d6", "-", 1.4),
    "D": ("D  drift alarm, recent {days} days", "#eb6834", "-", 1.4),
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def load_arm(rows: pd.DataFrame, model: str, region: str, arm: str):
    """(index, seeds x time rolling-MAE matrix, retrain timestamps) for one arm."""
    runs = rows[(rows["model"] == model) & (rows["region"] == region) & (rows["arm"] == arm)]
    runs = runs.drop_duplicates(["seed_key", "config_hash"]).sort_values("seed_key")
    curves, index, retrains = [], None, pd.DatetimeIndex([])
    for run in runs.itertuples():
        seed = None if run.seed_key == -1 else int(run.seed_key)
        curve = pd.read_csv(
            results_io.curve_path(run.config_hash, "aemo", region, seed), parse_dates=["timestamp"]
        )
        index = pd.DatetimeIndex(curve["timestamp"])
        curves.append(curve["rolling_mae_7d"].to_numpy())
        if len(retrains) == 0:  # the retrain schedule does not depend on the seed
            stamps = pd.read_csv(
                results_io.retrains_path(run.config_hash, "aemo", region, seed),
                parse_dates=["timestamp"],
            )
            retrains = pd.DatetimeIndex(stamps["timestamp"])
    return index, np.vstack(curves), retrains


def plot_figure(rows: pd.DataFrame, per_run: pd.DataFrame, official: dict, model: str, region: str):
    d_arm = official[(model, region)]
    days = d_arm.removeprefix(D_PREFIX).removesuffix("d")
    arms = {"A": ARM_A, "B": ARM_B, "C": ARM_C, "D": d_arm}

    fig, (ax, rug) = plt.subplots(
        2, 1, figsize=(14, 6.6), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06},
    )
    n_seeds, top = 1, 0.0
    for row, (key, arm) in enumerate(arms.items()):
        label, colour, style, width = ARM_STYLE[key]
        index, curves, retrains = load_arm(rows, model, region, arm)
        n_seeds = max(n_seeds, len(curves))
        with warnings.catch_warnings():  # the first 335 rolling values are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(curves, axis=0)
            sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        if len(curves) > 1:
            ax.fill_between(index, mean - sd, mean + sd, color=colour, alpha=0.16, linewidth=0)
        top = max(top, float(np.nanmax(mean + sd)))
        ax.plot(
            index, mean, color=colour, linestyle=style, linewidth=width,
            label=f"{label.format(days=days)}  ({len(retrains)} retrains)",
        )
        rug.vlines(retrains, row - 0.36, row + 0.36, color=colour, linewidth=1.3)

    reference = per_run[
        (per_run["model"] == model) & (per_run["region"] == region) & (per_run["arm"] == ARM_A)
    ]["pre-drift"].mean()
    ax.axhline(
        reference, color=TEXT, linestyle=(0, (1, 2)), linewidth=1.2,
        label=f"arm A pre-drift MAE  ({reference:.0f} MW)",
    )

    spread = " Lines are the mean across seeds; bands are ±1 standard deviation." if n_seeds > 1 else ""
    seeds = f"{n_seeds} seeds" if n_seeds > 1 else "deterministic, one run"
    fig.suptitle(
        f"F3 — Four adaptation arms, {MODEL_NAMES.get(model, model)}, {region}",
        x=0.125, y=0.97, ha="left", fontsize=14, fontweight="bold", color=TEXT,
    )
    ax.set_title(
        f"7-day rolling MAE (336 half-hourly observations) over the TEST period; {seeds}.{spread}",
        loc="left", fontsize=10, color=TEXT_MUTED,
    )
    ax.set_ylabel("7-day rolling MAE (MW)", color=TEXT_MUTED)
    ax.set_ylim(0, top * 1.22)  # headroom so the legend never sits on a curve
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9.5, labelcolor=TEXT)

    rug.set_ylim(len(arms) - 0.5, -0.5)
    rug.set_yticks(range(len(arms)), list(arms))
    rug.set_ylabel("retrains", color=TEXT_MUTED)
    rug.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    rug.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for axis in (ax, rug):
        axis.set_facecolor(SURFACE)
        axis.tick_params(colors=TEXT_MUTED, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.margins(x=0.005)

    out = results_io.FIGURE3_DIR / f"f3_four_arm_comparison_{model}_{region}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    results_io.FIGURE3_DIR.mkdir(parents=True, exist_ok=True)
    rows = latest_adaptation_rows()
    per_run = mae_by_run(rows)
    official = official_d_arms(per_run)
    for model, region in sorted(official):
        print(f"wrote {plot_figure(rows, per_run, official, model, region)}")


if __name__ == "__main__":
    main()
