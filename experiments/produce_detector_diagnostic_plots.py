"""Produce diagnostic plots for the final frozen AEMO detectors.

The detector wrappers remain unchanged. This experiment reproduces the
River detector state sequentially in order to visualise detector statistics
and configured decision parameters.

Plots are generated from the final preferred AEMO representation:
complete-day daily mean demand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import drift

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.data.deseasonalise import aggregate_daily_demand


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures" / "diagnostics"
CHANGEPOINT_DIR = RESULTS_DIR / "changepoints"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    """Return TOTALDEMAND indexed by settlement timestamp."""

    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def trace_adwin(stream: pd.Series) -> pd.DataFrame:
    """Trace public ADWIN diagnostics."""

    detector = drift.ADWIN(delta=0.00075)

    rows = []

    for timestamp, value in stream.items():
        detector.update(float(value))

        rows.append(
            {
                "timestamp": timestamp,
                "value": float(value),
                "estimation": float(detector.estimation),
                "width": float(detector.width),
                "variance": float(detector.variance),
                "drift": bool(detector.drift_detected),
            }
        )

    return pd.DataFrame(rows)


def trace_kswin(stream: pd.Series) -> pd.DataFrame:
    """Trace KSWIN p-values and decision threshold."""

    detector = drift.KSWIN(
        alpha=0.005,
        window_size=300,
        stat_size=48,
        seed=42,
    )

    rows = []

    for timestamp, value in stream.items():
        detector.update(float(value))

        rows.append(
            {
                "timestamp": timestamp,
                "value": float(value),
                "p_value": float(detector.p_value),
                "threshold": float(detector.alpha),
                "drift": bool(detector.drift_detected),
            }
        )

    return pd.DataFrame(rows)


def trace_page_hinkley(stream: pd.Series) -> pd.DataFrame:
    """Trace the Page-Hinkley cumulative decision statistic."""

    detector = drift.PageHinkley(
        min_instances=30,
        delta=0.005,
        threshold=400.0,
    )

    rows = []

    for timestamp, value in stream.items():
        detector.update(float(value))

        increase = float(
            detector._sum_increase - detector._min_increase
        )

        decrease = float(
            detector._max_decrease - detector._sum_decrease
        )

        statistic = max(increase, decrease)

        rows.append(
            {
                "timestamp": timestamp,
                "value": float(value),
                "statistic": statistic,
                "threshold": float(detector.threshold),
                "drift": bool(detector.drift_detected),
            }
        )

    return pd.DataFrame(rows)


def add_drift_lines(ax, frame: pd.DataFrame) -> None:
    """Add vertical lines at detected changepoints."""

    detections = frame.loc[
        frame["drift"],
        "timestamp",
    ]

    for timestamp in detections:
        ax.axvline(
            timestamp,
            linewidth=0.7,
            alpha=0.35,
        )


def plot_adwin(region: str, frame: pd.DataFrame) -> Path:
    """Plot ADWIN demand and public adaptive-window diagnostics."""

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
    )

    axes[0].plot(
        frame["timestamp"],
        frame["value"],
        linewidth=0.8,
    )
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_title(
        f"{region} — ADWIN diagnostics "
        "(delta=0.00075)"
    )

    axes[1].plot(
        frame["timestamp"],
        frame["estimation"],
        linewidth=0.9,
        label="Adaptive-window mean",
    )
    axes[1].set_ylabel("Estimation")
    axes[1].legend()

    axes[2].plot(
        frame["timestamp"],
        frame["width"],
        linewidth=0.9,
        label="Adaptive window width",
    )
    axes[2].set_ylabel("Window width")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    for ax in axes:
        add_drift_lines(ax, frame)
        ax.grid(alpha=0.2)

    fig.text(
        0.5,
        0.01,
        (
            "ADWIN uses a dynamic internal cut threshold. "
            "River does not expose that numeric threshold through "
            "the public API; delta=0.00075 is the configured "
            "significance parameter."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))

    output = FIGURE_DIR / (
        f"{region}_adwin_daily_diagnostics.png"
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output


def plot_kswin(region: str, frame: pd.DataFrame) -> Path:
    """Plot KSWIN demand, p-value and alpha threshold."""

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
    )

    axes[0].plot(
        frame["timestamp"],
        frame["value"],
        linewidth=0.8,
    )
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_title(
        f"{region} — KSWIN diagnostics"
    )

    axes[1].plot(
        frame["timestamp"],
        frame["p_value"],
        linewidth=0.8,
        label="KSWIN p-value",
    )

    axes[1].axhline(
        y=0.005,
        linestyle="--",
        label="alpha = 0.005",
    )

    axes[1].set_ylabel("p-value")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    for ax in axes:
        add_drift_lines(ax, frame)
        ax.grid(alpha=0.2)

    fig.tight_layout()

    output = FIGURE_DIR / (
        f"{region}_kswin_daily_diagnostics.png"
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output


def plot_page_hinkley(
    region: str,
    frame: pd.DataFrame,
) -> Path:
    """Plot Page-Hinkley demand and decision statistic."""

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
    )

    axes[0].plot(
        frame["timestamp"],
        frame["value"],
        linewidth=0.8,
    )
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_title(
        f"{region} — Page-Hinkley diagnostics"
    )

    axes[1].plot(
        frame["timestamp"],
        frame["statistic"],
        linewidth=0.8,
        label="Page-Hinkley statistic",
    )

    axes[1].axhline(
        y=400.0,
        linestyle="--",
        label="threshold = 400",
    )

    axes[1].set_ylabel("Test statistic")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    for ax in axes:
        add_drift_lines(ax, frame)
        ax.grid(alpha=0.2)

    fig.tight_layout()

    output = FIGURE_DIR / (
        f"{region}_page_hinkley_daily_diagnostics.png"
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output


def main() -> None:
    """Produce final daily diagnostic figures."""

    for region in REGIONS:
        _, _, test = loader.load(region)

        raw = demand_series(test)

        raw = raw.loc[
            "2020-03-01":"2023-12-31 23:59:59"
        ]

        daily = aggregate_daily_demand(raw)

        adwin_trace = trace_adwin(daily)
        kswin_trace = trace_kswin(daily)
        ph_trace = trace_page_hinkley(daily)

        outputs = (
            plot_adwin(region, adwin_trace),
            plot_kswin(region, kswin_trace),
            plot_page_hinkley(region, ph_trace),
        )

        print(f"\n{region}")

        for output in outputs:
            print(f"  saved: {output}")


if __name__ == "__main__":
    main()