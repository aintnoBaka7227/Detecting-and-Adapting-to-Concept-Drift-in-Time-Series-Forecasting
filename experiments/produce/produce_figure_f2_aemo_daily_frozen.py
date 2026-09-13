"""Figure F2 for daily-aggregated demand and frozen detector configs.

Creates one figure per detector and region from stored experiment outputs.
Detections are read from their dumps and are never recomputed here.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import aggregate_daily_demand
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.results_io import FIGURES_DIR, RUNS_CSV, detections_path

SPLIT_ID = "aemo_detect_daily_frozen_v1"
DETECTORS = ("adwin", "kswin", "page_hinkley")
DAILY_MEAN_WINDOW = 15

RAW_COLOR = "#A9D6E5"
SMOOTH_COLOR = "#1464A5"
EVENT_LINE_COLOR = "#303030"
PERIOD_SHADE_COLOR = "#F4A261"
MATCH_COLOR = "#198754"
FALSE_ALARM_COLOR = "#C62828"
GRID_COLOR = "#D9DEE5"


def tier1_events(region: str) -> pd.DataFrame:
    """Return the Tier 1 documented events for a region."""
    events = pd.read_csv(
        DOCUMENTED_EVENTS_CSV,
        parse_dates=["start_date", "end_date"],
    )
    events = events[
        (events["tier"] == 1)
        & events["region"].isin([region, "NEM"])
    ].copy()
    events["label"] = (
        events["event_name"]
        .str.split(" - ")
        .str[0]
        .str.slice(0, 30)
    )
    return events.sort_values("start_date").reset_index(drop=True)


def latest_metrics() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(
            f"{RUNS_CSV} not found -- run "
            "experiments.run.detection.run_aemo_detectors_daily_frozen first"
        )

    runs = pd.read_csv(RUNS_CSV)
    detections = runs[
        (runs["group"] == "detection")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == SPLIT_ID)
    ].copy()

    if detections.empty:
        raise SystemExit(
            f"no rows for split_id={SPLIT_ID!r} in runs.csv -- run "
            "experiments.run.detection.run_aemo_detectors_daily_frozen first"
        )

    detections["timestamp"] = pd.to_datetime(
        detections["timestamp"],
        errors="coerce",
        utc=True,
    )
    latest = detections.groupby(["method", "region"])["timestamp"].transform("max")
    return detections[detections["timestamp"] == latest]


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    ).sort_index()


def daily_demand(region: str) -> pd.Series:
    """Return the exact daily input used by the frozen detector run."""
    test = loader.load(region)[2]
    return aggregate_daily_demand(demand_series(test))


def smoothed(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average used only as a display overlay."""
    pad = window // 2
    padded = np.concatenate(
        [
            np.full(pad, values[0]),
            values,
            np.full(pad, values[-1]),
        ]
    )
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def detector_slice(
    metrics: pd.DataFrame,
    method: str,
    region: str,
) -> tuple[pd.DatetimeIndex, dict[str, float], int]:
    cell = metrics[
        (metrics["method"] == method)
        & (metrics["region"] == region)
    ]
    if cell.empty:
        raise SystemExit(
            f"no metrics found for method={method!r}, region={region!r}, "
            f"split_id={SPLIT_ID!r}"
        )

    values = dict(zip(cell["metric_name"], cell["metric_value"]))
    config_hash = cell["config_hash"].iloc[0]
    dump = detections_path(config_hash, "aemo", region, None)

    if not dump.exists():
        raise SystemExit(
            f"detection dump not found for {method}/{region}: {dump}"
        )

    detection_frame = pd.read_csv(dump, parse_dates=["timestamp"])
    detections = pd.DatetimeIndex(detection_frame["timestamp"]).sort_values()
    delays = {
        name.removeprefix("delay_"): float(value)
        for name, value in values.items()
        if name.startswith("delay_") and pd.notna(value)
    }
    return (
        detections,
        delays,
        int(values.get("n_unmatched_detections", 0)),
    )


def matched_detection_times(
    detections: pd.DatetimeIndex,
    delays: dict[str, float],
    events: pd.DataFrame,
) -> dict[str, pd.Timestamp]:
    """Map matched event IDs to timestamps that exist in the dump."""
    event_starts = events.set_index("event_id")["start_date"].to_dict()
    normalised_detections = {
        pd.Timestamp(timestamp).normalize(): pd.Timestamp(timestamp)
        for timestamp in detections
    }

    matched: dict[str, pd.Timestamp] = {}
    for event_id, delay in delays.items():
        if event_id not in event_starts:
            continue
        expected_day = (
            pd.Timestamp(event_starts[event_id])
            + pd.Timedelta(days=delay)
        ).normalize()
        if expected_day in normalised_detections:
            matched[event_id] = normalised_detections[expected_day]
    return matched


def plot(
    method: str,
    region: str,
    demand: pd.Series,
    events: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    detections, delays, recorded_unmatched = detector_slice(
        metrics,
        method,
        region,
    )
    matched_times = matched_detection_times(detections, delays, events)
    matched_days = {
        pd.Timestamp(timestamp).normalize()
        for timestamp in matched_times.values()
    }
    unmatched = pd.DatetimeIndex(
        [
            timestamp
            for timestamp in detections
            if pd.Timestamp(timestamp).normalize() not in matched_days
        ]
    )

    if len(unmatched) != recorded_unmatched:
        print(
            f"warning: {method}/{region} reconstructed {len(unmatched)} "
            f"unmatched detections; runs.csv records {recorded_unmatched}"
        )

    label_by_id = dict(zip(events["event_id"], events["label"]))

    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFC")

    ax.plot(
        demand.index,
        demand.to_numpy(),
        color=RAW_COLOR,
        lw=0.8,
        alpha=0.9,
        zorder=1,
    )
    ax.plot(
        demand.index,
        smoothed(demand.to_numpy(), DAILY_MEAN_WINDOW),
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
        ax.axvline(
            event.start_date,
            color=EVENT_LINE_COLOR,
            lw=1.0,
            alpha=0.65,
            zorder=2,
        )
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
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.2,
            },
            clip_on=True,
            zorder=7,
        )

    # Draw only genuinely unmatched detections in red.
    for detection_number, detected in enumerate(unmatched):
        ax.axvline(
            detected,
            color=FALSE_ALARM_COLOR,
            ls="--",
            lw=0.9,
            alpha=0.65,
            zorder=4,
        )
        ax.text(
            detected,
            0.70 - 0.055 * (detection_number % 3),
            "NO MATCH",
            transform=label_transform,
            rotation=90,
            ha="right",
            va="top",
            fontsize=6.2,
            color=FALSE_ALARM_COLOR,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "pad": 0.8,
            },
            clip_on=True,
            zorder=8,
        )

    # Matched detections are green and display the linked event and delay.
    for match_number, (event_id, detected) in enumerate(matched_times.items()):
        ax.axvline(
            detected,
            color=MATCH_COLOR,
            ls="--",
            lw=1.6,
            alpha=0.95,
            zorder=5,
        )
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
        "Frozen configuration on daily-aggregated test demand",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax.set_ylabel("Daily mean demand (MW)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.8)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BD")

    legend_handles = [
        Line2D([], [], color=RAW_COLOR, lw=1.2, label="Daily mean demand"),
        Line2D(
            [],
            [],
            color=SMOOTH_COLOR,
            lw=2.0,
            label=f"{DAILY_MEAN_WINDOW}-day mean",
        ),
        Line2D(
            [],
            [],
            color=EVENT_LINE_COLOR,
            lw=1.0,
            label="Documented event",
        ),
        Line2D(
            [],
            [],
            color=MATCH_COLOR,
            lw=1.6,
            ls="--",
            label="Matched detection",
        ),
        Line2D(
            [],
            [],
            color=FALSE_ALARM_COLOR,
            lw=0.9,
            ls="--",
            label="Unmatched detection",
        ),
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

    ax.text(
        1.0,
        -0.16,
        f"Detections: {len(detections)}  |  "
        f"Matched: {len(matched_days)}  |  "
        f"No match: {len(unmatched)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    out = FIGURES_DIR / (
        f"f2_detection_daily_frozen_{method}_{region}.png"
    )
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    metrics = latest_metrics()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        demand = daily_demand(region)
        events = tier1_events(region)
        for method in DETECTORS:
            plot(method, region, demand, events, metrics)


if __name__ == "__main__":
    main()
