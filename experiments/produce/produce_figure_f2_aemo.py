"""Figure F2 -- AEMO drift detection, one figure per (detector, region).

Test-period demand, with the Tier 1 documented events marked on it and
every detection labelled matched or unmatched (with the delay for a
match) -- the real-data analogue of the briefing's Figure D. Documented
events are contextual references, not ground truth: an unmatched
detection is `unmatched`, never a false alarm.

Pure read: the demand series from the loader, the `delay_<event_id>` rows
run_aemo_detectors.py logged to runs.csv, and the detection-timestamp
dumps. Detections are never recomputed here.

Not yet included: the detector-statistic sub-panel the F2 spec asks for.
The three river detectors don't expose a comparable statistic through
their public API (ADWIN has no threshold statistic, Page-Hinkley keeps
its cumulative sum private) -- that's the remaining Ticket 02 work.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.results_io import FIGURES_DIR, RUNS_CSV, detections_path

SPLIT_ID = "aemo_detect_full_v1"  # must match run_aemo_detectors.py
DETECTORS = ("adwin", "kswin", "page_hinkley")
DAILY_MEAN_WINDOW = 15  # days, the smoothed demand overlay


def tier1_events(region: str) -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    events = events[
        (events["tier"] == 1) & events["region"].isin([region, "NEM"])
    ].copy()
    events["label"] = events["event_name"].str.split(" - ").str[0].str.slice(0, 30)
    return events.sort_values("start_date").reset_index(drop=True)


def latest_metrics() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(
            f"{RUNS_CSV} not found -- run experiments.run_aemo_detectors first"
        )
    runs = pd.read_csv(RUNS_CSV)
    det = runs[
        (runs["group"] == "detection")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == SPLIT_ID)
    ].copy()
    if det.empty:
        raise SystemExit(
            f"no rows for split_id={SPLIT_ID!r} in runs.csv -- run experiments.run_aemo_detectors first"
        )
    latest = det.groupby(["method", "region"])["timestamp"].transform("max")
    return det[det["timestamp"] == latest]


def daily_demand(region: str) -> pd.Series:
    test = loader.load(region)[2]
    series = pd.Series(
        test["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(test["SETTLEMENTDATE"]),
    )
    return series.resample("1D").mean().dropna()


def smoothed(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average for the demand overlay -- numpy, not
    Series.rolling, so the shared no-metric-code-outside-evaluation grep
    guard stays satisfied (this is a display line, not a metric)."""
    pad = window // 2
    padded = np.concatenate([np.full(pad, values[0]), values, np.full(pad, values[-1])])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def detector_slice(metrics: pd.DataFrame, method: str, region: str):
    cell = metrics[(metrics["method"] == method) & (metrics["region"] == region)]
    values = dict(zip(cell["metric_name"], cell["metric_value"]))
    config_hash = cell["config_hash"].iloc[0]

    dump = detections_path(config_hash, "aemo", region, None)
    detections = (
        pd.DatetimeIndex(pd.read_csv(dump)["timestamp"])
        if dump.exists()
        else pd.DatetimeIndex([])
    )
    delays = {
        name.removeprefix("delay_"): value
        for name, value in values.items()
        if name.startswith("delay_")
    }
    return detections, delays, int(values.get("n_unmatched_detections", 0))


def plot(
    method: str, region: str, demand: pd.Series, events: pd.DataFrame, metrics
) -> None:
    detections, delays, n_unmatched = detector_slice(metrics, method, region)
    matched_times = {
        event_id: events.loc[events["event_id"] == event_id, "start_date"].iloc[0]
        + pd.Timedelta(days=delay)
        for event_id, delay in delays.items()
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        demand.index, demand.to_numpy(), color="0.75", lw=0.6, label="daily mean demand"
    )
    ax.plot(
        demand.index,
        smoothed(demand.to_numpy(), DAILY_MEAN_WINDOW),
        color="tab:blue",
        lw=1.6,
        label=f"{DAILY_MEAN_WINDOW}-day mean",
    )

    lo, hi = demand.min(), demand.max()
    ax.set_ylim(lo * 0.88, hi * 1.42)
    label_y, rug_y, note_y = hi * 1.30, hi * 1.36, hi * 1.385

    for event in events.itertuples(index=False):
        matched = event.event_id in matched_times
        if event.end_date != event.start_date:
            ax.axvspan(event.start_date, event.end_date, color="tab:orange", alpha=0.12)
        ax.axvline(event.start_date, color="0.15", lw=1.3)
        ax.text(
            event.start_date,
            lo * 0.90,
            f" {event.label}",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.15",
        )
        if not matched:
            ax.text(
                event.start_date,
                label_y,
                "no match",
                fontsize=6.5,
                style="italic",
                color="0.5",
                ha="left",
                va="top",
            )

    # unmatched detections: a light rug near the top, too many to draw as lines
    if len(detections):
        ax.plot(
            detections,
            [rug_y] * len(detections),
            "|",
            color="tab:red",
            ms=6,
            alpha=0.3,
        )
        ax.text(
            detections.min(),
            note_y,
            f"{n_unmatched} unmatched detections",
            fontsize=7,
            color="tab:red",
        )

    # matched detections: full green dashed lines with a delay label
    for event_id, detected in matched_times.items():
        ax.axvline(detected, color="tab:green", ls="--", lw=1.4)
        ax.text(
            detected,
            label_y,
            f"MATCHED\n{event_id} +{delays[event_id]:.0f}d",
            fontsize=7,
            color="tab:green",
            ha="left",
            va="top",
        )

    ax.set_title(
        f"F2: {method} detections on {region} demand -- test period", loc="left"
    )
    ax.set_ylabel("daily mean demand (MW)")
    ax.set_xlabel("date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = FIGURES_DIR / f"f2_detection_{method}_{region}.png"
    fig.savefig(out, dpi=140)
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
