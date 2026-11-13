
"""
Create Figure 2 using the AEMO detection results.
Plot daily residuals, documented events and detector alarms.
Save individual figures by region and one combined figure.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set the folder paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"
figure_dir = output_dir / "figure"
src_dir = project_dir / "src"

sys.path.insert(0, str(src_dir))

from aemo_preprocessing import load_aemo_data, split_aemo_data
from deseasonalise import fit_seasonal_profile, transform_data


# Detectors and regions used in the experiment
detectors = ["ADWIN", "KSWIN", "Page-Hinkley"]
regions = ["NSW1", "SA1"]

# Positive Tier 1 events
event_names = {
    "EVT-2020-02": "COVID-19 demand change",
    "EVT-2020-07": "December solar and mild weather",
    "EVT-2022-03": "Market suspension",
    "EVT-2023-14": "System-security directions"
}

# Colours used in the figures
colors = {
    "daily": "#a8d5e8",
    "average": "#176a9a",
    "tier1": "#16804a",
    "tier2": "#8b5fbf",
    "unmatched": "#d84b4b",
    "ignored": "#f2a3a3",
    "event": "#4b5563",
    "window": "#f3c58b"
}


def load_region_data(region):

    # Load the demand data
    filename = "NSW.csv" if region == "NSW1" else "SA.csv"

    data = load_aemo_data(filename, region)
    train, calibration, test = split_aemo_data(data)

    # Fit seasonal patterns using training data only
    profile = fit_seasonal_profile(train)

    # Apply the seasonal profile to the test data
    test = transform_data(test, profile)

    test = test.copy()
    test["SETTLEMENTDATE"] = pd.to_datetime(
        test["SETTLEMENTDATE"]
    )

    # Calculate daily mean residuals
    daily = (
        test.set_index("SETTLEMENTDATE")["standardised_residual"]
        .resample("D")
        .mean()
    )

    # Calculate the 15-day moving average
    average = daily.rolling(
        window=15,
        min_periods=1
    ).mean()

    return daily, average


def load_matches(region, detector):

    # Load the saved evaluation results
    detector_name = detector.lower().replace("-", "_")

    filename = (
        f"{region.lower()}_{detector_name}_matches.csv"
    )

    matches = pd.read_csv(
        output_dir / filename
    )

    matches["timestamp"] = pd.to_datetime(
        matches["timestamp"]
    )

    matches["tier"] = pd.to_numeric(
        matches["tier"],
        errors="coerce"
    )

    return matches


def plot_panel(ax, region, detector, daily, average, events):

    # Plot the daily mean residual
    ax.plot(
        daily.index,
        daily.values,
        color=colors["daily"],
        linewidth=0.7,
        alpha=0.7,
        zorder=2
    )

    # Plot the 15-day moving average
    ax.plot(
        average.index,
        average.values,
        color=colors["average"],
        linewidth=1.8,
        zorder=3
    )

    # Load detection results
    matches = load_matches(region, detector)

    tier1_matches = matches[
        (matches["label"] == "Match")
        & (matches["tier"] == 1)
    ]

    tier2_matches = matches[
        (matches["label"] == "Match")
        & (matches["tier"] == 2)
    ]

    unmatched = matches[
        matches["label"] == "Unmatch"
    ]

    ignored = matches[
        matches["label"] == "Ignored"
    ]

    # Plot ignored alarms
    for timestamp in ignored["timestamp"]:

        ax.axvline(
            timestamp,
            color=colors["ignored"],
            linestyle=":",
            linewidth=0.4,
            alpha=0.2,
            zorder=1
        )

    # Plot unmatched accepted alarms
    for timestamp in unmatched["timestamp"]:

        ax.axvline(
            timestamp,
            color=colors["unmatched"],
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
            zorder=1
        )

    # Select documented events for this region
    region_events = events[
        events["region"].isin(["NEM", region])
    ].sort_values("start_date")

    # Plot the event periods and start dates
    for _, event in region_events.iterrows():

        start = pd.Timestamp(event["start_date"])

        end = (
            pd.Timestamp(event["end_date"])
            + pd.Timedelta(days=1)
        )

        ax.axvspan(
            start,
            end,
            color=colors["window"],
            alpha=0.16,
            zorder=0
        )

        ax.axvline(
            start,
            color=colors["event"],
            linewidth=1.0,
            alpha=0.75,
            zorder=4
        )

    # Plot Tier 2 matched detections
    for timestamp in tier2_matches["timestamp"]:

        ax.axvline(
            timestamp,
            color=colors["tier2"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            zorder=5
        )

    # Plot Tier 1 matched detections
    for timestamp in tier1_matches["timestamp"]:

        ax.axvline(
            timestamp,
            color=colors["tier1"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            zorder=6
        )

    # Add labels for matched Tier 1 events
    for _, match in tier1_matches.iterrows():

        event_id = match["event_id"]

        if event_id not in event_names:
            continue

        event_name = event_names[event_id]
        delay = float(match["delay_days"])
        timestamp = match["timestamp"]

        # Move labels left when they are near the right edge
        if timestamp >= pd.Timestamp("2023-06-01"):
            x_offset = -5
            horizontal = "right"
        else:
            x_offset = 5
            horizontal = "left"

        ax.annotate(
            f"{event_name}\n{delay:.1f} d delay",
            xy=(timestamp, 0.96),
            xycoords=("data", "axes fraction"),
            xytext=(x_offset, -3),
            textcoords="offset points",
            fontsize=8,
            color=colors["tier1"],
            ha=horizontal,
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": colors["tier1"],
                "alpha": 0.95,
                "linewidth": 0.7
            },
            zorder=10
        )

    # Count accepted and ignored alarms
    accepted_count = (
        len(tier1_matches)
        + len(tier2_matches)
        + len(unmatched)
    )

    matched_count = (
        len(tier1_matches)
        + len(tier2_matches)
    )

    raw_count = accepted_count + len(ignored)

    # Add the detector name
    ax.set_title(
        f"{region} - {detector}",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel("Daily mean residual (z)")
    ax.grid(alpha=0.2)

    # Add the detection counts
    count_text = (
        f"Raw: {raw_count} | "
        f"Accepted: {accepted_count} | "
        f"Tier 1: {len(tier1_matches)} | "
        f"Tier 2: {len(tier2_matches)} | "
        f"Unmatched: {len(unmatched)} | "
        f"Ignored: {len(ignored)}"
    )

    ax.text(
        0.99,
        0.02,
        count_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.95,
            "pad": 3
        }
    )

    # Keep the plot within the test period
    ax.set_xlim(
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2024-01-01")
    )


def create_legend():

    # Create one common legend for all figures
    legend_items = [
        Line2D(
            [0], [0],
            color=colors["daily"],
            label="Daily mean residual"
        ),
        Line2D(
            [0], [0],
            color=colors["average"],
            linewidth=2,
            label="15-day mean"
        ),
        Line2D(
            [0], [0],
            color=colors["tier1"],
            linestyle="--",
            label="Tier 1 matched"
        ),
        Line2D(
            [0], [0],
            color=colors["tier2"],
            linestyle="--",
            label="Tier 2 matched"
        ),
        Line2D(
            [0], [0],
            color=colors["unmatched"],
            linestyle="--",
            label="Unmatched accepted"
        ),
        Line2D(
            [0], [0],
            color=colors["ignored"],
            linestyle=":",
            label="Ignored alarm"
        ),
        Line2D(
            [0], [0],
            color=colors["event"],
            label="Documented event start"
        ),
        Patch(
            facecolor=colors["window"],
            alpha=0.3,
            label="Tier 1 event period"
        )
    ]

    return legend_items


def main():

    # Create separate folders for each region
    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for region in regions:

        region_dir = figure_dir / region.lower()

        region_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # Load the documented event catalogue
    events = pd.read_csv(
        data_dir / "events.csv"
    )

    # Select positive Tier 1 events
    events = events[
        (pd.to_numeric(events["tier"]) == 1)
        & (events["event_id"] != "EVT-2021-10")
    ].copy()

    legend_items = create_legend()

    # Create the combined detection panel
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(20, 15),
        sharex=True
    )

    for col, region in enumerate(regions):

        # Load the residuals once for each region
        daily, average = load_region_data(region)

        region_dir = figure_dir / region.lower()

        for row, detector in enumerate(detectors):

            # Plot this detector in the combined figure
            plot_panel(
                axes[row, col],
                region,
                detector,
                daily,
                average,
                events
            )

            # Create an individual figure
            single_fig, single_ax = plt.subplots(
                figsize=(17, 6)
            )

            plot_panel(
                single_ax,
                region,
                detector,
                daily,
                average,
                events
            )

            single_ax.set_xlabel("Date")

            single_fig.legend(
                handles=legend_items,
                loc="lower center",
                ncol=4,
                fontsize=8
            )

            single_fig.tight_layout(
                rect=[0, 0.12, 1, 1]
            )

            detector_name = (
                detector.lower().replace("-", "_")
            )

            filename = (
                f"figure2_{region.lower()}_{detector_name}.png"
            )

            # Save the figure in its region folder
            single_fig.savefig(
                region_dir / filename,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(single_fig)

            print("Saved:", region.lower(), "/", filename)

    # Add labels to the combined figure
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")

    fig.suptitle(
        "Figure 2 - AEMO Drift Detection",
        fontsize=16,
        fontweight="bold"
    )

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        fontsize=9
    )

    fig.tight_layout(
        rect=[0, 0.08, 1, 0.96]
    )

    # Save the combined figure
    fig.savefig(
        figure_dir / "figure2_detection_panel.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nFigure 2 completed.")


if __name__ == "__main__":
    main()
