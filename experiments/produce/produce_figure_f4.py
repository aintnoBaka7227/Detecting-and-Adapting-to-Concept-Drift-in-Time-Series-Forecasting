"""F4 -- coverage trajectory: 7-day rolling coverage through the TEST period.

One figure per model x region; rows are arms A, B, C and D (180 days), each
with an AgACI and a fixed split conformal line (mean ± sd across seeds) of
the 90% intervals' coverage over the last 336 half-hours (F1's window).
Below the dashed 0.90 target the gap is filled, showing when calibration
failed; the drift regime is shaded and Tier 1 events are drawn. The
rolling coverage is read from the interval files (computed at the run stage
by run_aemo_uq_replay.add_rolling_coverage); the first 335 values are blank.

Outputs go to <table|figure>/<detector>/[<model>/[<signal>/]] (results_io.output_path);
arms A and B are the shared controls, C and D that detector's runs.

Usage:
    python -m experiments.produce.produce_figure_f4
"""

from __future__ import annotations

import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiments import results_io
from experiments.produce.figure_text import axes_right_in, line_height_in, wrap_to_width
from experiments.produce.produce_figure_f1 import events_in_window, load_events
from experiments.produce.produce_table_t5 import latest_uq_rows
from experiments.produce.table_adaptation_common import arm_sort_key
from experiments.run.uncertainty.run_aemo_uq_replay import (
    AGACI,
    FIXED,
    ROLLING_COVERAGE_WINDOW,
)
from experiments.trigger_detectors import DETECTOR_LABELS, TRIGGER_DETECTORS

MODEL_NAMES = {"nhits": "NHITS", "xgboost": "XGBoost", "dhr_arima": "DHR + ARIMA"}
ARM_LABELS = {
    "A_never_retrain": "A  never retrain",
    "B_every_30_days": "B  every 30 days",
    "C_drift_full_history": "C  drift alarm, full history",
}
OFFICIAL_D = "D_drift_recent_180d"  # arm D at the 180-day window, as in T4/T5
# (label, colour), drawn in this order.
UQ_STYLE = {
    FIXED: ("fixed split conformal", "#eb6834"),
    AGACI: ("AgACI", "#2a78d6"),
}
TARGET = 0.90
DRIFT_SHADE = ("#c9b8a6", 0.22)  # (colour, alpha) of the global drift regime
EVENT_COLOUR = "#7a7974"
HEADER_IN = 0.75  # title, subtitle and legend band above the panels, inches
HEADER_LEFT = 0.06  # figure fraction where the title, subtitle and legend start
SUBTITLE_SIZE = 9.5
HALF_HOUR = pd.Timedelta(minutes=30)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def arm_label(arm: str) -> str:
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    return f"D  drift alarm, recent {arm.rsplit('_', 1)[-1].removesuffix('d')} days"


def load_arm(runs: pd.DataFrame, region: str):
    """(index, regime labels, {uq: seeds x time rolling coverage}) for one arm.
    Every UQ run of one seed points at the same interval file."""
    coverage = {uq: [] for uq in UQ_STYLE}
    index, regimes = None, None
    for run in runs.drop_duplicates("seed_key").sort_values("seed_key").itertuples():
        seed = None if run.seed_key == -1 else int(run.seed_key)
        uq_config = json.loads((results_io.RUNS_DIR / run.config_hash / "config.json").read_text())
        intervals = pd.read_csv(
            results_io.intervals_path(uq_config["source_config_hash"], "aemo", region, seed),
            parse_dates=["timestamp"],
            float_precision="round_trip",
        )
        index, regimes = pd.DatetimeIndex(intervals["timestamp"]), intervals["regime"]
        for uq in UQ_STYLE:
            column = f"rolling_coverage_7d_{uq}"
            if column not in intervals:
                raise SystemExit(
                    f"{column} missing from {run.config_hash}'s interval file -- "
                    "run experiments.run.uncertainty.add_rolling_coverage first"
                )
            coverage[uq].append(intervals[column].to_numpy())
    return index, regimes, {uq: np.vstack(curves) for uq, curves in coverage.items()}


def drift_spans(index: pd.DatetimeIndex, regimes: pd.Series) -> list[tuple]:
    """(start, end) of every contiguous run of the drift regime; end is the
    half-hour after its last stamp."""
    run_id = (regimes != regimes.shift()).cumsum()
    spans = []
    for _, run in regimes.groupby(run_id):
        if run.iloc[0] == "drift":
            spans.append((index[run.index[0]], index[run.index[-1]] + HALF_HOUR))
    return spans


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_figure(rows: pd.DataFrame, events: pd.DataFrame, model: str, region: str, detector: str):
    subset = rows[(rows["model"] == model) & (rows["region"] == region)]
    arms = sorted(
        (arm for arm in subset["arm"].unique() if not arm.startswith("D_") or arm == OFFICIAL_D),
        key=arm_sort_key,
    )

    fig, axes = plt.subplots(
        len(arms), 1, figsize=(14, 2.35 * len(arms) + HEADER_IN),
        sharex=True, sharey=True, facecolor=SURFACE, squeeze=False,
    )
    axes = axes[:, 0]
    height = fig.get_figheight()

    def from_top(inches: float) -> float:
        """Figure-fraction y that sits `inches` below the top edge."""
        return 1 - inches / height

    bottom, n_seeds, index, regimes = 1.0, 1, None, None
    for ax, arm in zip(axes, arms):
        index, regimes, coverage = load_arm(subset[subset["arm"] == arm], region)
        for uq, curves in coverage.items():
            label, colour = UQ_STYLE[uq]
            n_seeds = max(n_seeds, len(curves))
            with warnings.catch_warnings():  # the first 335 rolling values are all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(curves, axis=0)
                sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else None
            if sd is not None:  # coverage is a fraction: clip the band to [0, 1]
                low, high = np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1)
                ax.fill_between(index, low, high, color=colour, alpha=0.14, linewidth=0)
                bottom = min(bottom, float(np.nanmin(low)))
            # Below-target periods of the mean line: where calibration failed.
            ax.fill_between(index, mean, TARGET, where=mean < TARGET, interpolate=True,
                            color=colour, alpha=0.30, linewidth=0)
            ax.plot(index, mean, color=colour, linewidth=1.1, label=label)
            bottom = min(bottom, float(np.nanmin(mean)))
        for start, end in drift_spans(index, regimes):
            ax.axvspan(start, end, color=DRIFT_SHADE[0], alpha=DRIFT_SHADE[1], linewidth=0, zorder=0)
        ax.axhline(TARGET, color=TEXT, linestyle=(0, (4, 3)), linewidth=1.0, zorder=4)
        style_axis(ax)
        ax.set_title(arm_label(arm), loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("7-day rolling\ncoverage", color=TEXT_MUTED, fontsize=9)

    # Tier 1 documented events: a thin line in every row, numbered at the
    # foot of the top row and named in the subtitle.
    tier1 = events_in_window(events[events["tier"] == 1], region, index[0], index[-1])
    for number, event in enumerate(tier1.itertuples(), start=1):
        for ax in axes:
            ax.axvline(event.start_date, color=EVENT_COLOUR, linewidth=0.8, zorder=1)
        axes[0].annotate(
            str(number), (event.start_date, 0), xycoords=("data", "axes fraction"),
            xytext=(3, 3), textcoords="offset points", ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=EVENT_COLOUR,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
        )
    event_names = "; ".join(
        f"{number} {str(event.event_name).split(' - ')[0].removeprefix('NEM ')} ({event.start_date:%d %b %Y})"
        for number, event in enumerate(tier1.itertuples(), start=1)
    )

    # Fixed limits: the 0.90 target is never clipped, whatever the data do.
    axes[0].set_ylim(max(min(bottom, TARGET) - 0.03, -0.02), 1.01)
    axes[0].set_xlim(index[0], index[-1])
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("date (TEST period, half-hourly observations)", color=TEXT_MUTED, fontsize=9)

    seeds = f"mean ± sd across {n_seeds} seeds" if n_seeds > 1 else "deterministic, one run"
    subtitle = (
        f"Empirical coverage of the 90% intervals over a 7-day rolling window ({ROLLING_COVERAGE_WINDOW} half-hourly "
        f"observations) through the TEST period; {seeds}. Dashed line = 0.90 target; filled dips "
        "are periods below target. Global drift regime shaded. The first 7 days are blank (window "
        f"warm-up). Tier 1 documented events (thin lines): {event_names}."
    )
    # Lay the panels out once to learn their width, wrap the subtitle to it,
    # then make room for any extra subtitle lines above the panels.
    fig.tight_layout(rect=(0.04, 0, 1, from_top(HEADER_IN)))
    left_in = HEADER_LEFT * fig.get_figwidth()
    subtitle = wrap_to_width(fig, subtitle, axes_right_in(fig, axes) - left_in, SUBTITLE_SIZE)
    extra_in = subtitle.count("\n") * line_height_in(SUBTITLE_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=TEXT, linestyle=(0, (4, 3)), linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, color=DRIFT_SHADE[0], alpha=DRIFT_SHADE[1] * 2, linewidth=0),
        plt.Line2D([], [], color=EVENT_COLOUR, linewidth=0.8),
    ]
    labels += ["0.90 target", "drift regime", "Tier 1 event"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(HEADER_LEFT, from_top(0.62 + extra_in)),
               ncol=5, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(
        f"F4 — Coverage trajectory, {MODEL_NAMES.get(model, model)}, {region}, "
        f"{DETECTOR_LABELS[detector]} trigger",
        x=HEADER_LEFT, y=from_top(0.08), ha="left", fontsize=14, fontweight="bold", color=TEXT,
    )
    fig.text(HEADER_LEFT, from_top(0.30), subtitle, ha="left", va="top", fontsize=SUBTITLE_SIZE, color=TEXT_MUTED)
    fig.tight_layout(rect=(0.04, 0, 1, from_top(HEADER_IN + extra_in)))

    out = results_io.output_path(
        results_io.FIGURE4_DIR, detector, model, name=f"f4_coverage_trajectory_{detector}_{model}_{region}.png"
    )
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    results_io.FIGURE4_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    for detector in TRIGGER_DETECTORS:
        rows = latest_uq_rows(detector)
        for (model, region), _ in rows.groupby(["model", "region"]):
            print(f"wrote {plot_figure(rows, events, model, region, detector)}")


if __name__ == "__main__":
    main()
