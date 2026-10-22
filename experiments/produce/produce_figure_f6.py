"""F6 -- where each trigger type fires over the TEST period, and where they disagree.

One panel per region, one row per trigger:
    Group A   Team 41's distribution-shift trigger (TRIGGER): its accepted
              change points from run_detector_changepoints.py, i.e. the days
              arm C retrains on;
    Group B   Team 42's error-based trigger, read from GROUP_B_CSV (columns:
              timestamp, region) once they provide it; an empty row until then.
An alarm is a disagreement when the other trigger has no alarm within
AGREE_DAYS of it (only shown once both rows exist). Tier 1 documented events
are shaded for reference.

Usage:
    python -m experiments.produce.produce_figure_f6
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from drift_lab.config import REGIONS
from experiments.produce.figure_text import axes_right_in, line_height_in, wrap_to_width
from experiments.produce.produce_figure_f1 import load_events
from experiments.results_io import FIGURE6_DIR, output_path
from experiments.run.detection.run_detector_changepoints import load_changepoints
from experiments.run.detection.standard_stream_common import TEST_START
from experiments.trigger_detectors import DETECTOR_LABELS

TRIGGER = "page_hinkley"
GROUP_B_CSV = FIGURE6_DIR / "group_b_team42_alarms.csv"
AGREE_DAYS = 7
TEST_END = pd.Timestamp("2024-01-01")
GROUP_A_COLOR = "#0D2C54"
GROUP_B_COLOR = "#C1666B"
DISAGREE_COLOR = "#E0A800"
EVENT_COLOR = "#D9DEE5"


def group_b_alarms(region: str) -> pd.DatetimeIndex | None:
    if not GROUP_B_CSV.exists():
        return None
    alarms = pd.read_csv(GROUP_B_CSV, parse_dates=["timestamp"])
    return pd.DatetimeIndex(
        alarms.loc[alarms["region"] == region, "timestamp"]
    ).sort_values()


def unmatched(alarms: pd.DatetimeIndex, other: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Alarms with no alarm in `other` within AGREE_DAYS."""
    tolerance = pd.Timedelta(days=AGREE_DAYS)
    return pd.DatetimeIndex(
        [t for t in alarms if not ((other - t).to_series().abs() <= tolerance).any()]
    )


def draw_region(ax: plt.Axes, region: str, events: pd.DataFrame) -> int | None:
    tier1 = events[(events["tier"] == 1) & events["region"].isin([region, "NEM"])]
    for e in tier1.itertuples():
        ax.axvspan(
            e.start_date, e.end_date + pd.Timedelta(days=1), color=EVENT_COLOR, zorder=0
        )

    a = pd.DatetimeIndex(load_changepoints(TRIGGER, region)["timestamp"]).sort_values()
    b = group_b_alarms(region)
    ax.vlines(a, 1.6, 2.4, color=GROUP_A_COLOR, linewidth=1.6)
    n_disagree = None
    if b is None:
        ax.text(
            TEST_START + pd.Timedelta(days=20),
            1,
            "awaiting Team 42 alarm file",
            va="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )
    else:
        ax.vlines(b, 0.6, 1.4, color=GROUP_B_COLOR, linewidth=1.6)
        only_a, only_b = unmatched(a, b), unmatched(b, a)
        ax.vlines(only_a, 1.6, 2.4, color=DISAGREE_COLOR, linewidth=2.4)
        ax.vlines(only_b, 0.6, 1.4, color=DISAGREE_COLOR, linewidth=2.4)
        n_disagree = len(only_a) + len(only_b)

    ax.set_ylim(0.3, 2.7)
    ax.set_yticks(
        [2, 1],
        [
            f"Group A\n{DETECTOR_LABELS[TRIGGER]} ({len(a)})",
            "Group B\nTeam 42" + ("" if b is None else f" ({len(b)})"),
        ],
    )
    ax.set_xlim(TEST_START, TEST_END)
    ax.set_title(region, loc="left", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color=EVENT_COLOR, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return n_disagree


def main() -> None:
    events = load_events()
    fig, axes = plt.subplots(
        len(REGIONS), 1, figsize=(12, 2.4 * len(REGIONS) + 1.6), sharex=True
    )
    disagreements = {
        region: draw_region(ax, region, events) for ax, region in zip(axes, REGIONS)
    }

    handles = [
        Line2D(
            [],
            [],
            color=GROUP_A_COLOR,
            linewidth=2,
            label="Group A alarm (distribution shift)",
        ),
        Line2D(
            [],
            [],
            color=GROUP_B_COLOR,
            linewidth=2,
            label="Group B alarm (error-based)",
        ),
        Line2D(
            [],
            [],
            color=DISAGREE_COLOR,
            linewidth=2.4,
            label=f"disagreement (no other alarm within ±{AGREE_DAYS} d)",
        ),
        Patch(color=EVENT_COLOR, label="Tier 1 documented event"),
    ]
    axes[-1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    counts = (
        "Disagreements: awaiting Team 42."
        if any(v is None for v in disagreements.values())
        else "Disagreements: "
        + ", ".join(f"{r} {n}" for r, n in disagreements.items())
        + "."
    )
    title = "F6 — Where each trigger type fires, and where the two disagree"
    subtitle = (
        f"TEST period (Mar 2020 – Dec 2023). Group A: accepted {DETECTOR_LABELS[TRIGGER]} alarms on the "
        "standard-half-hourly seasonal residual (14-day refractory), the days arm C retrains on. Group B: "
        f"Team 42's error-based trigger. An alarm is a disagreement if the other trigger has no alarm within "
        f"±{AGREE_DAYS} days. Count per row in brackets. {counts}"
    )
    plot_in = fig.get_figheight()
    fig.subplots_adjust(
        left=0.13, right=0.98, bottom=1.0 / plot_in, top=1 - 0.3 / plot_in, hspace=0.45
    )
    left_in = 0.13 * fig.get_figwidth()
    width_in = axes_right_in(fig, axes) - left_in
    title = wrap_to_width(fig, title, width_in, 13, fontweight="bold")
    subtitle = wrap_to_width(fig, subtitle, width_in, 9.5)
    header_in = (
        0.2
        + line_height_in(13) * (title.count("\n") + 1)
        + line_height_in(9.5) * (subtitle.count("\n") + 1)
    )
    fig.set_figheight(plot_in + header_in)
    height = fig.get_figheight()
    fig.subplots_adjust(bottom=1.0 / height, top=1 - (header_in + 0.3) / height)
    fig.text(0.13, 1 - 0.1 / height, title, fontsize=13, fontweight="bold", va="top")
    fig.text(
        0.13,
        1 - (0.15 + line_height_in(13) * (title.count("\n") + 1)) / height,
        subtitle,
        fontsize=9.5,
        va="top",
        color="#333333",
    )

    out = output_path(
        FIGURE6_DIR, TRIGGER, name=f"figure_f6_trigger_timeline_{TRIGGER}.png"
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
