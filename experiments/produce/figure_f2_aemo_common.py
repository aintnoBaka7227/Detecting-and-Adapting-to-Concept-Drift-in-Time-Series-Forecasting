"""Shared build logic for the AEMO Figure F2 companions.

Read by produce_figure_f2_aemo_standard_half_hourly_post_tune.py -- the
post-tuning, standard-half-hourly input stream. (The raw, standard-daily
and pre-tuning F2 figures were retired; their detector runs still feed
Tables T1/T2.)

It draws one figure per (detector, region) -- test-period demand with
Tier 1 documented events marked and stored matched/unmatched detections
from the detector run, plus an optional test-statistic panel. Detections
are read from their dumps and are never recomputed here. Demand is shown
as a daily-mean overlay, since the half-hourly stream is too dense to
plot directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.results_io import FIGURE2_DIR, RUNS_CSV
from experiments.run.detection.detection_artifacts import event_assignments_path
from experiments.run.detection.post_tune_detector_configs import load_winner_row
from experiments.run.detection.standard_stream_common import (
    TEST_START,
    Cadence,
    build_standard_stream,
)

DETECTORS = ("adwin", "kswin", "page_hinkley")
DISPLAY_SMOOTH_WINDOW = 15
STATISTIC_COLOR = "#2B7A99"


@dataclass(frozen=True)
class StatisticTrace:
    """A detector's per-step test statistic over TEST, drawn as a panel
    under the demand plot. `values` is NaN wherever no test ran; the
    detector fires when a value crosses `threshold`."""

    values: pd.Series
    threshold: float
    y_label: str
    threshold_label: str

RAW_COLOR = "#A9D6E5"
SMOOTH_COLOR = "#1464A5"
EVENT_LINE_COLOR = "#303030"
PERIOD_SHADE_COLOR = "#F4A261"
MATCH_COLOR = "#198754"
FALSE_ALARM_COLOR = "#C62828"
SUPPRESSED_COLOR = "#B0B6BD"
GRID_COLOR = "#D9DEE5"


def tier1_events(region: str) -> pd.DataFrame:
    """Return the Tier 1 documented events for a region."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    events = events[(events["tier"] == 1) & events["region"].isin([region, "NEM"])].copy()
    events["label"] = events["event_name"].str.split(" - ").str[0].str.slice(0, 30)
    return events.sort_values("start_date").reset_index(drop=True)


def standard_half_hourly_demand_for_display(region: str) -> pd.Series:
    """Daily-mean resample of the standard-half-hourly (deseasonalised +
    standardised) series, TEST period only -- for display only, when the
    detector's own input is half-hourly and too dense to plot directly."""
    full, warmup = build_standard_stream(region, "half_hourly")
    return full.iloc[warmup:].resample("1D").mean().dropna()


def kswin_p_value_trace(region: str, cadence: Cadence) -> StatisticTrace:
    """Replay the frozen post-tune KSWIN over the same TRAIN -> Calibration
    -> TEST standard stream its detector run used, and return -log10(p)
    for the TEST period. Display only -- detections still come from the
    persisted dumps; this just exposes the p-value behind them."""
    kwargs = json.loads(load_winner_row("kswin")["detector_parameters"])
    detector = KSWINDetector(**kwargs)
    series, _ = build_standard_stream(region, cadence)
    _, p_values = detector.p_value_trace(series.to_numpy())

    # ks_2samp can underflow to 0 for extreme splits; clip before the log.
    neg_log_p = -np.log10(np.clip(p_values, 1e-300, None))
    trace = pd.Series(neg_log_p, index=series.index)
    return StatisticTrace(
        values=trace[trace.index >= TEST_START],
        threshold=-np.log10(detector.alpha),
        y_label=r"KS test $-\log_{10}(p)$",
        threshold_label=rf"Threshold $\alpha$ = {detector.alpha:g}",
    )


def page_hinkley_statistic_trace(region: str, cadence: Cadence) -> StatisticTrace:
    """Replay the frozen post-tune Page-Hinkley over the same standard
    stream its detector run used, and return its test value (cumulative
    deviation from the running mean, in z units) for the TEST period.
    Display only, like kswin_p_value_trace."""
    kwargs = json.loads(load_winner_row("page_hinkley")["detector_parameters"])
    detector = PageHinkleyDetector(**kwargs)
    series, _ = build_standard_stream(region, cadence)
    _, statistic = detector.statistic_trace(series.to_numpy())

    trace = pd.Series(statistic, index=series.index)
    return StatisticTrace(
        values=trace[trace.index >= TEST_START],
        threshold=detector.threshold,
        y_label="Page-Hinkley test value",
        threshold_label=rf"Threshold $\lambda$ = {detector.threshold:g}",
    )


def smoothed(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average used only as a display overlay."""
    pad = window // 2
    padded = np.concatenate([np.full(pad, values[0]), values, np.full(pad, values[-1])])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def latest_metrics(split_id: str, run_script_hint: str) -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run {run_script_hint} first")

    runs = pd.read_csv(RUNS_CSV)
    detections = runs[
        (runs["group"] == "detection") & (runs["dataset"] == "aemo") & (runs["split_id"] == split_id)
    ].copy()

    if detections.empty:
        raise SystemExit(f"no rows for split_id={split_id!r} in runs.csv -- run {run_script_hint} first")

    detections["timestamp"] = pd.to_datetime(detections["timestamp"], errors="coerce", utc=True)
    latest = detections.groupby(["method", "region"])["timestamp"].transform("max")
    return detections[detections["timestamp"] == latest]


def detector_slice(
    metrics: pd.DataFrame,
    method: str,
    region: str,
    split_id: str,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict[str, float], int]:
    """Returns (accepted_detections, raw_suppressed_detections, delays,
    recorded_unmatched). "Accepted" = Match + Unmatch rows from the
    persisted event_assignments dump (detection_artifacts.py) --
    refractory-ignored rows are returned separately as
    `raw_suppressed_detections`, not folded into the plotted signal."""
    cell = metrics[(metrics["method"] == method) & (metrics["region"] == region)]
    if cell.empty:
        raise SystemExit(f"no metrics found for method={method!r}, region={region!r}, split_id={split_id!r}")

    values = dict(zip(cell["metric_name"], cell["metric_value"]))
    config_hash = cell["config_hash"].iloc[0]
    dump = event_assignments_path(config_hash, region)

    if not dump.exists():
        raise SystemExit(f"event-assignment dump not found for {method}/{region}: {dump}")

    assignments = pd.read_csv(dump, parse_dates=["timestamp"])
    accepted = pd.DatetimeIndex(
        assignments.loc[assignments["label"] != "Ignored", "timestamp"]
    ).sort_values()
    suppressed = pd.DatetimeIndex(
        assignments.loc[assignments["label"] == "Ignored", "timestamp"]
    ).sort_values()

    delays = {
        name.removeprefix("delay_"): float(value)
        for name, value in values.items()
        if name.startswith("delay_") and pd.notna(value)
    }
    return accepted, suppressed, delays, int(values.get("n_unmatched_detections", 0))


def matched_detection_times(
    detections: pd.DatetimeIndex,
    delays: dict[str, float],
    events: pd.DataFrame,
) -> dict[str, pd.Timestamp]:
    """Map matched event IDs to timestamps that exist in the dump."""
    event_starts = events.set_index("event_id")["start_date"].to_dict()
    normalised_detections = {
        pd.Timestamp(timestamp).normalize(): pd.Timestamp(timestamp) for timestamp in detections
    }

    matched: dict[str, pd.Timestamp] = {}
    for event_id, delay in delays.items():
        if event_id not in event_starts:
            continue
        expected_day = (pd.Timestamp(event_starts[event_id]) + pd.Timedelta(days=delay)).normalize()
        if expected_day in normalised_detections:
            matched[event_id] = normalised_detections[expected_day]
    return matched


def plot_one(
    *,
    method: str,
    region: str,
    demand: pd.Series,
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    split_id: str,
    y_label: str,
    title_note: str,
    output_prefix: str,
    trace: StatisticTrace | None = None,
) -> None:
    detections, suppressed, delays, recorded_unmatched = detector_slice(metrics, method, region, split_id)
    matched_times = matched_detection_times(detections, delays, events)
    matched_days = {pd.Timestamp(timestamp).normalize() for timestamp in matched_times.values()}
    unmatched = pd.DatetimeIndex(
        [ts for ts in detections if pd.Timestamp(ts).normalize() not in matched_days]
    )

    if len(unmatched) != recorded_unmatched:
        print(
            f"warning: {method}/{region} reconstructed {len(unmatched)} "
            f"unmatched detections; runs.csv records {recorded_unmatched}"
        )

    label_by_id = dict(zip(events["event_id"], events["label"]))

    if trace is None:
        fig, ax = plt.subplots(figsize=(15.5, 6.8))
        stat_ax = None
    else:
        fig, (ax, stat_ax) = plt.subplots(
            2,
            1,
            figsize=(15.5, 9.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.06},
        )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFC")

    ax.plot(demand.index, demand.to_numpy(), color=RAW_COLOR, lw=0.8, alpha=0.9, zorder=1)
    ax.plot(
        demand.index,
        smoothed(demand.to_numpy(), DISPLAY_SMOOTH_WINDOW),
        color=SMOOTH_COLOR,
        lw=2.0,
        zorder=3,
    )

    demand_range = float(demand.max() - demand.min())
    ax.set_ylim(
        float(demand.min()) - 0.08 * demand_range,
        float(demand.max()) + 0.42 * demand_range,
    )
    label_transform = blended_transform_factory(ax.transData, ax.transAxes)

    for event_number, event in enumerate(events.itertuples(index=False)):
        if event.end_date != event.start_date:
            ax.axvspan(
                event.start_date,
                event.end_date + pd.Timedelta(days=1),
                color=PERIOD_SHADE_COLOR,
                alpha=0.14,
                lw=0,
                zorder=0,
            )
        ax.axvline(event.start_date, color=EVENT_LINE_COLOR, lw=1.0, alpha=0.65, zorder=2)
        ax.text(
            event.start_date,
            0.975 - 0.075 * (event_number % 2),
            event.label,
            transform=label_transform,
            rotation=0,
            ha="center",
            va="top",
            fontsize=6.5,
            color=EVENT_LINE_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            clip_on=True,
            zorder=7,
        )

    for detected in suppressed:
        ax.axvline(detected, color=SUPPRESSED_COLOR, ls=":", lw=0.6, alpha=0.5, zorder=3)

    for detected in unmatched:
        ax.axvline(detected, color=FALSE_ALARM_COLOR, ls="--", lw=0.9, alpha=0.65, zorder=4)

    for match_number, (event_id, detected) in enumerate(matched_times.items()):
        ax.axvline(detected, color=MATCH_COLOR, ls="--", lw=1.6, alpha=0.95, zorder=5)
        ax.text(
            detected,
            0.84 - 0.07 * (match_number % 2),
            f"MATCHED\n{label_by_id[event_id]}\n{delays[event_id]:.0f} d delay",
            transform=label_transform,
            fontsize=6.5,
            color=MATCH_COLOR,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": MATCH_COLOR,
                "alpha": 0.88,
                "boxstyle": "round,pad=0.25",
                "linewidth": 0.5,
            },
            clip_on=True,
            zorder=9,
        )

    ax.set_title(
        f"F2: {method.replace('_', ' ').title()} detections — {region}",
        loc="right",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0,
        1.015,
        title_note,
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax.set_ylabel(y_label)
    bottom_ax = ax if stat_ax is None else stat_ax
    bottom_ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.8)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BD")

    legend_handles = [
        Line2D([], [], color=RAW_COLOR, lw=1.2, label="Daily mean"),
        Line2D([], [], color=SMOOTH_COLOR, lw=2.0, label=f"{DISPLAY_SMOOTH_WINDOW}-day mean"),
        Line2D([], [], color=EVENT_LINE_COLOR, lw=1.0, label="Documented event"),
        Line2D([], [], color=MATCH_COLOR, lw=1.6, ls="--", label="Matched (accepted)"),
        Line2D([], [], color=FALSE_ALARM_COLOR, lw=0.9, ls="--", label="Unmatched (accepted)"),
        Line2D([], [], color=SUPPRESSED_COLOR, lw=0.6, ls=":", label="Raw signal, refractory-suppressed"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=3,
        fontsize=8,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#C7CDD4",
    )

    if stat_ax is not None:
        plot_statistic_panel(stat_ax, trace, events, matched_times.values(), unmatched)
        stat_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        stat_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    bottom_ax.text(
        1.0,
        -0.16 if stat_ax is None else -0.38,
        f"Accepted: {len(detections)}  |  Matched: {len(matched_days)}  |  No match: {len(unmatched)}"
        f"  |  Suppressed (refractory): {len(suppressed)}",
        transform=bottom_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    if stat_ax is None:
        fig.tight_layout()
    out = FIGURE2_DIR / f"{output_prefix}_{method}_{region}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_statistic_panel(
    ax,
    trace: StatisticTrace,
    events: pd.DataFrame,
    matched: Iterable[pd.Timestamp],
    unmatched: pd.DatetimeIndex,
) -> None:
    """The detector's test statistic under the demand plot: gaps where no
    test ran, the firing threshold dashed, and a marker on each accepted
    detection (green matched, red unmatched)."""
    ax.set_facecolor("#FAFBFC")
    for event in events.itertuples(index=False):
        if event.end_date != event.start_date:
            ax.axvspan(
                event.start_date,
                event.end_date + pd.Timedelta(days=1),
                color=PERIOD_SHADE_COLOR,
                alpha=0.14,
                lw=0,
                zorder=0,
            )

    values = trace.values
    ax.plot(values.index, values.to_numpy(), color=STATISTIC_COLOR, lw=0.9, zorder=2)
    ax.axhline(trace.threshold, color=FALSE_ALARM_COLOR, ls="--", lw=1.1, zorder=3)

    top = max(float(np.nanmax(values.to_numpy())), trace.threshold) * 1.2
    for timestamps, color in ((pd.DatetimeIndex(list(matched)), MATCH_COLOR), (unmatched, FALSE_ALARM_COLOR)):
        if len(timestamps) == 0:
            continue
        # The stored timestamp is the exact step the test fired on.
        heights = values.reindex(timestamps).fillna(trace.threshold).to_numpy()
        ax.scatter(timestamps, heights, marker="v", s=34, color=color, edgecolor="white", lw=0.5, zorder=4)

    ax.set_ylim(0, top)
    ax.set_ylabel(trace.y_label)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.8)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BD")
    ax.legend(
        handles=[
            Line2D([], [], color=STATISTIC_COLOR, lw=1.0, label="Test statistic (gaps: no test run)"),
            Line2D([], [], color=FALSE_ALARM_COLOR, lw=1.1, ls="--", label=trace.threshold_label),
        ],
        loc="upper left",
        ncol=2,
        fontsize=8,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#C7CDD4",
    )


def build_all_figures(
    *,
    split_id: str,
    run_script_hint: str,
    demand_for_display: Callable[[str], pd.Series],
    y_label: str,
    title_note: str,
    output_prefix: str,
    statistic_trace: Callable[[str, str], StatisticTrace | None] | None = None,
) -> None:
    """`statistic_trace(method, region)` optionally supplies a detector's
    test statistic for a bottom panel; None (or a None return) keeps the
    single-panel figure."""
    metrics = latest_metrics(split_id, run_script_hint)
    FIGURE2_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        demand = demand_for_display(region)
        events = tier1_events(region)
        for method in DETECTORS:
            plot_one(
                method=method,
                region=region,
                demand=demand,
                events=events,
                metrics=metrics,
                split_id=split_id,
                y_label=y_label,
                title_note=title_note,
                output_prefix=output_prefix,
                trace=statistic_trace(method, region) if statistic_trace else None,
            )
