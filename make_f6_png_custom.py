from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from experiments.run.detection.run_detector_changepoints import load_changepoints
from experiments.produce.produce_figure_f1 import load_events


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parent

OUT_DIR = (
    ROOT
    / "results"
    / "figures"
    / "figure6"
    / "bingqi"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEAM42_XLSX = (
    OUT_DIR
    / "Team42_PH_5571.xlsx"
)

OUT_PNG = (
    OUT_DIR
    / "F6_custom_final.png"
)

OUT_CSV = (
    OUT_DIR
    / "F6_disagreement_summary.csv"
)


# =========================================================
# Settings
# =========================================================
REGIONS = [
    "SA1",
    "NSW1"
]

AGREE_DAYS = 7

TEST_START = pd.Timestamp(
    "2020-03-01"
)

TEST_END = pd.Timestamp(
    "2024-01-01"
)


# =========================================================
# Group A
# =========================================================
def load_group_a():

    rows = []

    for region in REGIONS:

        df = load_changepoints(
            "page_hinkley",
            region
        ).copy()

        if "timestamp" not in df.columns:
            raise ValueError(
                f"No timestamp column found for {region}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        for timestamp in df["timestamp"]:

            rows.append(
                {
                    "region": region,
                    "team": "Team 41",
                    "timestamp": timestamp
                }
            )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "region",
            "timestamp"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\n=== GROUP A PAGE-HINKLEY ALARMS ==="
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nGroup A counts:"
    )

    print(
        result.groupby(
            "region"
        ).size()
    )

    return result


# =========================================================
# Excel helper
# =========================================================
def find_column(
    columns,
    keywords
):

    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    for keyword in keywords:

        for lower_name, original_name in normalized.items():

            if lower_name == keyword:
                return original_name

    for keyword in keywords:

        for lower_name, original_name in normalized.items():

            if keyword in lower_name:
                return original_name

    return None


# =========================================================
# Group B
# =========================================================
def load_group_b():

    if not TEAM42_XLSX.exists():

        raise FileNotFoundError(
            f"\nCannot find Team 42 file:\n{TEAM42_XLSX}"
        )

    excel = pd.ExcelFile(
        TEAM42_XLSX
    )

    preferred_sheet = None

    for sheet in excel.sheet_names:

        if (
            "accepted" in sheet.lower()
            and "detection" in sheet.lower()
        ):
            preferred_sheet = sheet
            break

    if preferred_sheet is None:
        preferred_sheet = excel.sheet_names[0]

    df = pd.read_excel(
        TEAM42_XLSX,
        sheet_name=preferred_sheet
    )

    print(
        f"\nUsing Team 42 sheet: {preferred_sheet}"
    )

    print(
        "Columns:",
        list(df.columns)
    )

    region_col = find_column(
        df.columns,
        [
            "region",
            "aemo region"
        ]
    )

    timestamp_col = find_column(
        df.columns,
        [
            "timestamp",
            "datetime",
            "date time",
            "alarm time",
            "detection time"
        ]
    )

    if region_col is None:
        region_col = df.columns[0]

    if timestamp_col is None:
        timestamp_col = df.columns[1]

    result = pd.DataFrame(
        {
            "region": (
                df[region_col]
                .astype(str)
                .str.strip()
            ),

            "timestamp": pd.to_datetime(
                df[timestamp_col],
                errors="coerce"
            )
        }
    )

    result = result[
        result["region"].isin(
            REGIONS
        )
    ].copy()

    result = result.dropna(
        subset=["timestamp"]
    )

    result["team"] = "Team 42"

    result = result.sort_values(
        [
            "region",
            "timestamp"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\n=== GROUP B PAGE-HINKLEY ALARMS ==="
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nGroup B counts:"
    )

    print(
        result.groupby(
            "region"
        ).size()
    )

    return result


# =========================================================
# Find unmatched alarms
# =========================================================
def unmatched(
    alarms,
    other
):

    tolerance = pd.Timedelta(
        days=AGREE_DAYS
    )

    unmatched_list = []

    for timestamp in alarms:

        if len(other) == 0:

            unmatched_list.append(
                timestamp
            )

            continue

        differences = (
            other - timestamp
        )

        within_window = (
            differences
            .to_series()
            .abs()
            <= tolerance
        ).any()

        if not within_window:

            unmatched_list.append(
                timestamp
            )

    return pd.DatetimeIndex(
        unmatched_list
    )


# =========================================================
# Summary table
# =========================================================
def build_summary(
    group_a,
    group_b
):

    rows = []

    for region in REGIONS:

        a = pd.DatetimeIndex(
            group_a.loc[
                group_a["region"] == region,
                "timestamp"
            ]
        ).sort_values()

        b = pd.DatetimeIndex(
            group_b.loc[
                group_b["region"] == region,
                "timestamp"
            ]
        ).sort_values()

        only_a = unmatched(
            a,
            b
        )

        only_b = unmatched(
            b,
            a
        )

        rows.append(
            {
                "region": region,
                "team41_alarms": len(a),
                "team42_alarms": len(b),
                "team41_disagreements": len(only_a),
                "team42_disagreements": len(only_b),
                "total_disagreements": (
                    len(only_a)
                    + len(only_b)
                )
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Draw one region
# =========================================================
def draw_region(
    ax,
    region,
    group_a,
    group_b,
    events
):

    # -----------------------------------------------------
    # Group A alarm timestamps
    # -----------------------------------------------------
    a = pd.DatetimeIndex(
        group_a.loc[
            group_a["region"] == region,
            "timestamp"
        ]
    ).sort_values()

    # -----------------------------------------------------
    # Group B alarm timestamps
    # -----------------------------------------------------
    b = pd.DatetimeIndex(
        group_b.loc[
            group_b["region"] == region,
            "timestamp"
        ]
    ).sort_values()

    # -----------------------------------------------------
    # Disagreements
    # -----------------------------------------------------
    only_a = unmatched(
        a,
        b
    )

    only_b = unmatched(
        b,
        a
    )

    matched_a = a.difference(
        only_a
    )

    matched_b = b.difference(
        only_b
    )

    # -----------------------------------------------------
    # Y positions
    # -----------------------------------------------------
    y_a = 1.0
    y_b = 0.0
    y_event = 1.65

    # -----------------------------------------------------
    # Main horizontal timelines
    # -----------------------------------------------------
    ax.hlines(
        y_a,
        TEST_START,
        TEST_END,
        linewidth=1.0,
        alpha=0.35
    )

    ax.hlines(
        y_b,
        TEST_START,
        TEST_END,
        linewidth=1.0,
        alpha=0.35
    )

    # -----------------------------------------------------
    # Tier 1 event band
    # Different visual style from Phong:
    # shown as a band at the top instead of full background
    # -----------------------------------------------------
    tier1 = events[
        (events["tier"] == 1)
        & (
            events["region"].isin(
                [
                    region,
                    "NEM"
                ]
            )
        )
    ].copy()

    for event in tier1.itertuples():

        start = pd.Timestamp(
            event.start_date
        )

        end = (
            pd.Timestamp(
                event.end_date
            )
            + pd.Timedelta(
                days=1
            )
        )

        ax.plot(
            [
                start,
                end
            ],
            [
                y_event,
                y_event
            ],
            linewidth=8,
            alpha=0.25,
            solid_capstyle="butt"
        )

    # -----------------------------------------------------
    # Matched Team 41 alarms
    # -----------------------------------------------------
    ax.scatter(
        matched_a,
        [y_a] * len(matched_a),
        marker="o",
        s=65,
        zorder=4
    )

    # -----------------------------------------------------
    # Matched Team 42 alarms
    # -----------------------------------------------------
    ax.scatter(
        matched_b,
        [y_b] * len(matched_b),
        marker="o",
        s=65,
        zorder=4
    )

    # -----------------------------------------------------
    # Team 41 disagreements
    # -----------------------------------------------------
    ax.scatter(
        only_a,
        [y_a] * len(only_a),
        marker="X",
        s=85,
        zorder=5
    )

    # -----------------------------------------------------
    # Team 42 disagreements
    # -----------------------------------------------------
    ax.scatter(
        only_b,
        [y_b] * len(only_b),
        marker="X",
        s=85,
        zorder=5
    )

    # -----------------------------------------------------
    # Connect matched pairs
    # -----------------------------------------------------
    for timestamp in matched_a:

        if len(b) == 0:
            continue

        diffs = (
            b - timestamp
        )

        nearest_index = (
            diffs
            .to_series()
            .abs()
            .argmin()
        )

        nearest = b[
            nearest_index
        ]

        if (
            abs(
                nearest - timestamp
            )
            <= pd.Timedelta(
                days=AGREE_DAYS
            )
        ):

            ax.plot(
                [
                    timestamp,
                    nearest
                ],
                [
                    y_a,
                    y_b
                ],
                linestyle="--",
                linewidth=0.9,
                alpha=0.45,
                zorder=2
            )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------
    ax.set_yticks(
        [
            y_event,
            y_a,
            y_b
        ]
    )

    ax.set_yticklabels(
        [
            "Tier 1 events",

            (
                "Team 41\n"
                f"{len(a)} alarms / "
                f"{len(only_a)} disagreements"
            ),

            (
                "Team 42\n"
                f"{len(b)} alarms / "
                f"{len(only_b)} disagreements"
            )
        ],
        fontsize=10
    )

    # -----------------------------------------------------
    # Region title
    # -----------------------------------------------------
    ax.set_title(
        region,
        loc="left",
        fontsize=15,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # Axes
    # -----------------------------------------------------
    ax.set_ylim(
        -0.5,
        2.0
    )

    ax.set_xlim(
        TEST_START,
        TEST_END
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            bymonth=[
                1,
                7
            ]
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b %Y"
        )
    )

    ax.grid(
        axis="x",
        alpha=0.15
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    return {
        "team41_alarms": len(a),
        "team42_alarms": len(b),
        "team41_disagreements": len(only_a),
        "team42_disagreements": len(only_b),
        "total_disagreements": (
            len(only_a)
            + len(only_b)
        )
    }


# =========================================================
# Figure
# =========================================================
def make_figure(
    group_a,
    group_b,
    events,
    summary
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            16,
            8.5
        ),
        sharex=True
    )

    region_stats = {}

    for ax, region in zip(
        axes,
        REGIONS
    ):

        region_stats[
            region
        ] = draw_region(
            ax,
            region,
            group_a,
            group_b,
            events
        )

    # =====================================================
    # Main title
    # =====================================================
    fig.suptitle(
        (
            "F6 - Trigger agreement "
            "and disagreement timeline"
        ),
        fontsize=21,
        fontweight="bold",
        y=0.985
    )

    # =====================================================
    # Subtitle
    # =====================================================
    subtitle = (
        "Team 41 uses a distribution-shift Page-Hinkley trigger; "
        "Team 42 uses a forecast-error Page-Hinkley trigger. "
        "Circles indicate alarms with a corresponding alarm within ±7 days; "
        "X markers indicate disagreements. "
        "Tier 1 documented events are shown in the upper event band."
    )

    fig.text(
        0.08,
        0.925,
        subtitle,
        fontsize=10.5,
        ha="left",
        va="top"
    )

    # =====================================================
    # Summary text
    # =====================================================
    summary_text = (
        f"SA1: {region_stats['SA1']['total_disagreements']} total disagreements   |   "
        f"NSW1: {region_stats['NSW1']['total_disagreements']} total disagreements"
    )

    fig.text(
        0.08,
        0.885,
        summary_text,
        fontsize=10.5,
        fontweight="bold",
        ha="left"
    )

    # =====================================================
    # Legend
    # =====================================================
    handles = [

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Matched alarm"
        ),

        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markersize=9,
            label="Disagreement"
        ),

        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1,
            label="Matched pair within ±7 days"
        ),

        Line2D(
            [0],
            [0],
            linewidth=8,
            alpha=0.25,
            label="Tier 1 documented event"
        )
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.015
        ),
        fontsize=9.5
    )

    # =====================================================
    # Layout
    # =====================================================
    plt.subplots_adjust(
        top=0.83,
        bottom=0.15,
        left=0.18,
        right=0.97,
        hspace=0.38
    )

    # =====================================================
    # Save
    # =====================================================
    fig.savefig(
        OUT_PNG,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# =========================================================
# Main
# =========================================================
def main():

    group_a = load_group_a()

    group_b = load_group_b()

    events = load_events()

    summary = build_summary(
        group_a,
        group_b
    )

    print(
        "\n=== F6 DISAGREEMENT SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        OUT_CSV,
        index=False
    )

    make_figure(
        group_a,
        group_b,
        events,
        summary
    )

    print(
        "\n======================================"
    )

    print(
        "F6 completed."
    )

    print(
        "\nPNG:"
    )

    print(
        OUT_PNG
    )

    print(
        "\nSummary CSV:"
    )

    print(
        OUT_CSV
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()