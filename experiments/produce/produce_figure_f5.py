"""F5 -- escalation curve: review rate against retained-set coverage.

One figure per model x region, AgACI intervals only: the required-outputs
spec asks for one curve per regime, not per UQ method, and fixed split
conformal has a constant width, so its only operating points are 0% and
100% review (T6 reports it). Columns are the adaptation arms (A, B, C and
arm D at the 180-day window, as in T4/T5). Each panel has one curve per regime -- pre-drift
(stable), drift, post-drift -- so the stable and drift curves stay
separate. The x axis is the fraction of forecasts escalated for review
(width > tau), the y axis the empirical coverage of the forecasts kept.

Seeds: each seed's curve is read as a step function of review rate (the
coverage at the largest attainable review rate not above x, the T6 budget
convention) and the line is the mean across seeds with a ±1 sd band.

The 0.90 crossing is the logged review_rate_for_coverage_0.90 (mean across
seeds), marked on the target line and stated in each panel. A constant-
width method has only the 0% and 100% review points, so it is drawn as its
0% point only, not as a curve.

Reads the escalation runs.csv rows and, through their config_hash, the
curve files written by run_aemo_escalation_sweep.py; no metric is
recalculated here.

Every output is written once per trigger detector (experiments.trigger_detectors:
ADWIN, Page-Hinkley) with the detector in the file name; arms A and B are the
shared controls, arms C and D are that detector's runs.

Usage:
    python -m experiments.produce.produce_figure_f5
"""

from __future__ import annotations

import warnings
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiments import results_io
from experiments.produce.figure_text import axes_right_in, line_height_in, wrap_to_width
from experiments.produce.table_adaptation_common import arm_sort_key
from experiments.run.uncertainty.run_aemo_escalation_sweep import SIGNAL_RULES, SIGNALS
from experiments.trigger_detectors import (
    CONTROL_DETECTOR,
    DETECTOR_LABELS,
    TRIGGER_DETECTORS,
    detector_rows,
    escalation_split_id,
)

MODEL_NAMES = {"nhits": "NHITS", "xgboost": "XGBoost", "dhr_arima": "DHR + ARIMA"}
UQ_ORDER = {"agaci": "AgACI", "fixed_split_conformal": "Fixed split conformal"}
# Width gives fixed split conformal only its 0% and 100% points, so its width
# figure shows AgACI alone; every nonconformity signal shows both methods.
WIDTH_UQS = ("agaci",)
ARM_LABELS = {
    "A_never_retrain": "A  never retrain",
    "B_every_30_days": "B  every 30 days",
    "C_drift_full_history": "C  drift alarm, full history",
}
# (label, colour). Stable in blue, drift in orange, post-drift grey.
REGIME_STYLE = {
    "pre-drift": ("pre-drift (stable)", "#2a78d6"),
    "drift": ("drift", "#eb6834"),
    "post-drift": ("post-drift", "#7a7974"),
}
TARGET = 0.90
HEADER_IN = 0.8  # title, subtitle and legend band above the panels, inches
HEADER_LEFT = 0.06  # figure fraction where the title, subtitle and legend start
SUBTITLE_SIZE = 9.5
REVIEW_GRID = np.linspace(0, 1, 1001)[:-1]  # 100% review keeps nothing: no coverage
# Display only: a curve stops once fewer forecasts than this (one day) are kept,
# where coverage is a few-sample artefact. The logged metrics are unaffected.
DISPLAY_MIN_RETAINED = 48
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
RUN_KEYS = ["model", "arm", "region", "uq", "seed_key"]


def latest_escalation_rows(detector: str = CONTROL_DETECTOR, signal: str = "width") -> pd.DataFrame:
    """runs.csv rows of the latest sweep of every (model, arm, region, UQ
    method, seed) for one trigger detector (its arms C and D plus the
    shared controls A and B); method is "<model>/<arm>/<uq method>"."""
    hint = "run run_aemo_escalation_sweep.py first"
    if not results_io.RUNS_CSV.exists():
        raise SystemExit(f"{results_io.RUNS_CSV} not found -- {hint}")
    rows = detector_rows(
        pd.read_csv(results_io.RUNS_CSV), "escalation", partial(escalation_split_id, signal=signal), detector
    ).copy()
    if rows.empty:
        raise SystemExit(f"no escalation rows for detector {detector!r} -- {hint}")
    parts = rows["method"].str.split("/", n=2, expand=True)
    rows["model"], rows["arm"], rows["uq"] = parts[0], parts[1], parts[2]
    rows["seed_key"] = rows["seed"].fillna(-1)
    latest = rows.groupby(RUN_KEYS)["timestamp"].transform("max")
    return rows[rows["timestamp"] == latest]


def arm_label(arm: str) -> str:
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    return f"D  drift alarm, recent {arm.rsplit('_', 1)[-1].removesuffix('d')} days"


def coverage_on_grid(curve: pd.DataFrame) -> np.ndarray:
    """Retained coverage at the largest attainable review rate <= each grid
    point (the no-review point always qualifies); NaN where fewer than
    DISPLAY_MIN_RETAINED forecasts are kept."""
    curve = curve.sort_values("review_rate")
    rates = curve["review_rate"].to_numpy()
    index = np.searchsorted(rates, REVIEW_GRID, side="right") - 1
    coverage = curve["retained_coverage"].to_numpy()[index]
    return np.where(curve["n_retained"].to_numpy()[index] >= DISPLAY_MIN_RETAINED, coverage, np.nan)


def load_panel(runs: pd.DataFrame, region: str) -> dict:
    """{regime: (seeds x grid coverage, no-review coverages, constant width)}."""
    curves = {regime: [] for regime in REGIME_STYLE}
    base = {regime: [] for regime in REGIME_STYLE}
    constant = True
    for run in runs.drop_duplicates(["seed_key", "config_hash"]).itertuples():
        seed = None if run.seed_key == -1 else int(run.seed_key)
        frame = pd.read_csv(results_io.escalation_curve_path(run.config_hash, "aemo", region, seed))
        frame["regime"] = frame["regime"].str.replace("_", "-", regex=False)
        for regime in REGIME_STYLE:
            curve = frame[frame["regime"] == regime]
            if curve.empty:
                continue
            constant &= curve["review_rate"].nunique() <= 2
            curves[regime].append(coverage_on_grid(curve))
            base[regime].append(float(curve.loc[curve["review_rate"] == 0, "retained_coverage"].iloc[0]))
    return {
        regime: (np.vstack(curves[regime]), np.array(base[regime]), constant)
        for regime in REGIME_STYLE
        if curves[regime]
    }


def crossing_note(points: pd.DataFrame, regime: str) -> tuple[str, float | None]:
    """Panel text for the 0.90 review rate, and the mean rate to mark."""
    values = points.loc[
        (points["regime"] == regime) & (points["metric_name"] == f"review_rate_for_coverage_{TARGET:.2f}"),
        "metric_value",
    ]
    reached = values.dropna()
    if reached.empty:
        return "unreachable", None
    mean = reached.mean()
    spread = reached.std() if len(reached) > 1 else 0.0
    text = f"{mean:.0%}" if round(spread, 2) == 0 else f"{mean:.0%} ± {spread:.0%}"
    if len(reached) < len(values):
        text += f" (unreachable in {len(values) - len(reached)}/{len(values)} seeds)"
    return text, mean


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1)


def plot_figure(rows: pd.DataFrame, model: str, region: str, detector: str, signal: str):
    subset = rows[(rows["model"] == model) & (rows["region"] == region)]
    arms = sorted(subset["arm"].unique(), key=arm_sort_key)
    shown = WIDTH_UQS if signal == "width" else tuple(UQ_ORDER)
    uqs = [uq for uq in UQ_ORDER if uq in shown and uq in set(subset["uq"])]

    fig, axes = plt.subplots(
        len(uqs), len(arms), figsize=(4.1 * len(arms), 3.5 * len(uqs) + HEADER_IN),
        sharex=True, sharey=True, facecolor=SURFACE, squeeze=False,
    )
    height = fig.get_figheight()

    def from_top(inches: float) -> float:
        """Figure-fraction y that sits `inches` below the top edge."""
        return 1 - inches / height

    bottom, n_seeds = 1.0, 1
    for i, uq in enumerate(uqs):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            runs = subset[(subset["uq"] == uq) & (subset["arm"] == arm)]
            n_seeds = max(n_seeds, runs["seed_key"].nunique())
            panel = load_panel(runs, region)
            notes = []
            for regime, (curves, base, constant) in panel.items():
                label, colour = REGIME_STYLE[regime]
                text, crossing = crossing_note(runs, regime)
                notes.append((f"{label}: 0.90 {'unreachable' if crossing is None else 'at ' + text}", colour))
                if constant:
                    ax.errorbar(
                        0, base.mean(), yerr=base.std(ddof=1) if len(base) > 1 else None,
                        fmt="o", color=colour, markersize=5, capsize=3, label=label,
                    )
                    bottom = min(bottom, float(base.min()))
                    continue
                with warnings.catch_warnings():  # grid points past every seed's cut-off
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(curves, axis=0)
                    sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else None
                if sd is not None:
                    ax.fill_between(REVIEW_GRID, mean - sd, mean + sd, color=colour, alpha=0.16, linewidth=0)
                ax.plot(REVIEW_GRID, mean, color=colour, linewidth=1.4, label=label)
                bottom = min(bottom, float(np.nanmin(mean)))
                if crossing is not None:
                    ax.plot(crossing, TARGET, marker="o", markersize=6, markerfacecolor=SURFACE,
                            markeredgecolor=colour, markeredgewidth=1.6, zorder=5)
            if panel and next(iter(panel.values()))[2]:
                notes.append(("constant width: only 0% and 100% review attainable", TEXT_MUTED))
            ax.axhline(TARGET, color=TEXT, linestyle=(0, (4, 3)), linewidth=1.0)
            for k, (note, colour) in enumerate(notes):
                ax.text(0.98, 0.04 + 0.075 * (len(notes) - 1 - k), note, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color=colour)
            style_axis(ax)
            if i == 0:
                ax.set_title(arm_label(arm), loc="left", fontsize=10, color=TEXT)
            if j == 0:
                ax.set_ylabel(f"{UQ_ORDER[uq]}\nretained coverage", color=TEXT_MUTED, fontsize=9)
            if i == len(uqs) - 1:
                ax.set_xlabel("review rate (forecasts escalated)", color=TEXT_MUTED, fontsize=9)

    for ax in axes.flat:
        ax.set_ylim(max(0.0, bottom - 0.12), 1.005)
    seeds = f"mean ± sd across {n_seeds} seeds" if n_seeds > 1 else "deterministic, one run"
    subtitle = (
        f"{SIGNAL_RULES[signal]}; τ swept over every observed value. "
        f"Coverage of the forecasts kept, per global regime; {seeds}. "
        "Review rates for 0.90 coverage are stated in each panel. "
        f"Curves stop once fewer than {DISPLAY_MIN_RETAINED} forecasts (one day) are kept."
    )
    # Lay the panels out once to learn their width, wrap the subtitle to it,
    # then make room for any extra subtitle lines above the panels.
    fig.tight_layout(rect=(0.04, 0, 1, from_top(HEADER_IN)))
    left_in = HEADER_LEFT * fig.get_figwidth()
    subtitle = wrap_to_width(fig, subtitle, axes_right_in(fig, axes.flat) - left_in, SUBTITLE_SIZE)
    extra_in = subtitle.count("\n") * line_height_in(SUBTITLE_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=TEXT, linestyle=(0, (4, 3)), linewidth=1.0))
    labels.append("0.90 target")
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
                              markeredgecolor=TEXT_MUTED, markeredgewidth=1.6))
    labels.append("0.90 crossing" + (" (mean across seeds)" if n_seeds > 1 else ""))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(HEADER_LEFT, from_top(0.62 + extra_in)),
               ncol=5, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(
        f"F5 — Escalation curve, {MODEL_NAMES.get(model, model)}, {region}, "
        f"{DETECTOR_LABELS[detector]} trigger, {signal.replace('_', ' ')}",
        x=HEADER_LEFT, y=from_top(0.08), ha="left", fontsize=14, fontweight="bold", color=TEXT,
    )
    fig.text(HEADER_LEFT, from_top(0.30), subtitle, ha="left", va="top", fontsize=SUBTITLE_SIZE, color=TEXT_MUTED)
    fig.tight_layout(rect=(0.04, 0, 1, from_top(HEADER_IN + extra_in)))

    out = results_io.output_path(
        results_io.FIGURE5_DIR,
        detector,
        model,
        signal,
        name=f"f5_escalation_curve_{detector}_{signal}_{model}_{region}.png",
    )
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    results_io.FIGURE5_DIR.mkdir(parents=True, exist_ok=True)
    for detector, signal in product(TRIGGER_DETECTORS, SIGNALS):
        rows = latest_escalation_rows(detector, signal)
        for (model, region), _ in rows.groupby(["model", "region"]):
            print(f"wrote {plot_figure(rows, model, region, detector, signal)}")


if __name__ == "__main__":
    main()
