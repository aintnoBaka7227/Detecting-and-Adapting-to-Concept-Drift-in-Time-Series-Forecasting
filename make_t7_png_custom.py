from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from experiments.run.detection.run_detector_changepoints import load_changepoints


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parent

RUNS_CSV = ROOT / "results" / "runs.csv"

T5_CSV = (
    ROOT
    / "results"
    / "tables"
    / "table5"
    / "page_hinkley"
    / "table_t5_coverage_and_width_page_hinkley.csv"
)

T6_WIDTH_CSV = (
    ROOT
    / "results"
    / "tables"
    / "table6"
    / "page_hinkley"
    / "table_t6_escalation_operating_points_page_hinkley_width.csv"
)

T6_NONCONF_CSV = (
    ROOT
    / "results"
    / "tables"
    / "table6"
    / "page_hinkley"
    / "table_t6_escalation_operating_points_page_hinkley_nonconformity_30min.csv"
)

GROUP_B_CSV = (
    ROOT
    / "results"
    / "tables"
    / "table7"
    / "group_b_team42.csv"
)

OUT_DIR = ROOT / "results" / "tables" / "table7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "T7_custom.csv"
OUT_PNG = OUT_DIR / "T7_custom_clean.png"


# =========================================================
# Helper functions
# =========================================================
def fmt_mean_sd(mean, sd, decimals=1, suffix=""):
    if pd.isna(sd):
        sd = 0.0

    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}{suffix}"


def wrap_text(value, width):
    if pd.isna(value):
        return ""

    return textwrap.fill(
        str(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False
    )


def normalize_percent(value):
    text = str(value).strip()

    if text in {"0%", "0.0%", "0.00%"}:
        return "0.0%"

    return text


# =========================================================
# Group A: MAE + retrains from runs.csv
# =========================================================
def load_group_a_adaptation():

    df = pd.read_csv(RUNS_CSV)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    filtered = df[
        (df["group"] == "adaptation")
        & (df["method"] == "xgboost/D_drift_recent_180d")
        & (df["metric_name"] == "mae")
        & (df["regime"] == "drift")
        & (df["region"].isin(["SA1", "NSW1"]))
        & (df["seed"].isin([1, 2, 3, 4, 5]))
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No Group A XGBoost Arm D 180-day drift MAE rows were found."
        )

    filtered = (
        filtered
        .sort_values("timestamp")
        .groupby(["region", "seed"], as_index=False)
        .tail(1)
        .sort_values(["region", "seed"])
    )

    print("\n=== Latest Group A adaptation rows ===")

    print(
        filtered[
            [
                "region",
                "seed",
                "metric_value",
                "n_retrains",
                "config_hash",
                "timestamp"
            ]
        ].to_string(index=False)
    )

    mae_stats = (
        filtered
        .groupby("region")["metric_value"]
        .agg(["count", "mean", "std"])
    )

    mae = {}
    retrains = {}

    for region in ["SA1", "NSW1"]:

        row = mae_stats.loc[region]

        mae[region] = fmt_mean_sd(
            row["mean"],
            row["std"],
            decimals=1,
            suffix=" MW"
        )

        region_rows = filtered[
            filtered["region"] == region
        ]

        retrain_values = (
            region_rows["n_retrains"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(retrain_values) == 1:
            retrains[region] = str(
                retrain_values[0]
            )
        else:
            retrains[region] = str(
                sorted(
                    retrain_values.tolist()
                )
            )

    return mae, retrains


# =========================================================
# Group A: Page-Hinkley detections
# =========================================================
def load_group_a_detections():

    detections = {}

    for region in ["SA1", "NSW1"]:

        cps = load_changepoints(
            "page_hinkley",
            region
        )

        detections[region] = str(
            len(cps)
        )

    return detections


# =========================================================
# Group A: worst 24h coverage from T5
# =========================================================
def load_group_a_worst24():

    t5 = pd.read_csv(T5_CSV)

    filtered = t5[
        (t5["model"] == "xgboost")
        & (t5["arm"] == "D_drift_recent_180d")
        & (t5["UQ method"] == "AgACI")
        & (t5["regime"] == "drift")
        & (t5["region"].isin(["SA1", "NSW1"]))
    ].copy()

    print("\n=== Group A T5 rows used ===")

    print(
        filtered[
            [
                "region",
                "worst 24h coverage",
                "seeds"
            ]
        ].to_string(index=False)
    )

    result = {}

    for region in ["SA1", "NSW1"]:

        rows = filtered[
            filtered["region"] == region
        ]

        if rows.empty:
            result[region] = "n/a"
        else:
            result[region] = str(
                rows.iloc[-1][
                    "worst 24h coverage"
                ]
            )

    return result


# =========================================================
# Group A: T6 review rate
# =========================================================
def load_review_rate(csv_path):

    df = pd.read_csv(csv_path)

    filtered = df[
        (df["model"] == "xgboost")
        & (df["arm"] == "D_drift_recent_180d")
        & (df["UQ method"] == "AgACI")
        & (df["regime"] == "drift")
        & (df["region"].isin(["SA1", "NSW1"]))
    ].copy()

    print(
        f"\n=== T6 rows used: {csv_path.name} ==="
    )

    print(
        filtered[
            [
                "region",
                "review rate for 0.90"
            ]
        ].to_string(index=False)
    )

    result = {}

    for region in ["SA1", "NSW1"]:

        rows = filtered[
            filtered["region"] == region
        ]

        if rows.empty:
            result[region] = "n/a"
        else:
            result[region] = normalize_percent(
                rows.iloc[-1][
                    "review rate for 0.90"
                ]
            )

    return result


# =========================================================
# Group B results
# =========================================================
def load_group_b():

    df = pd.read_csv(
        GROUP_B_CSV
    )

    required_metrics = [
        "MAE during drift (MW)",
        "Worst 24h coverage",
        "Review rate for 0.90 coverage (interval width)",
        "Review rate for 0.90 coverage (nonconformity, 30 min)",
        "Detections raised",
        "Retrains triggered",
    ]

    result = {}

    for metric in required_metrics:

        result[metric] = {}

        for region in ["SA1", "NSW1"]:

            rows = df[
                (df["metric"] == metric)
                & (df["region"] == region)
            ]

            if rows.empty:

                result[metric][region] = "n/a"

            else:

                value = str(
                    rows.iloc[-1]["value"]
                )

                if (
                    metric == "MAE during drift (MW)"
                    and "MW" not in value
                ):
                    value = value + " MW"

                if "Review rate" in metric:
                    value = normalize_percent(
                        value
                    )

                result[metric][region] = value

    return result


# =========================================================
# Configuration table
# =========================================================
def build_settings_df():

    rows = [
        [
            "Drift trigger",
            "Distribution shift",
            "Forecast error",
            "Primary variable: distribution shift vs forecast error",
        ],
        [
            "Detector",
            "Page-Hinkley",
            "Page-Hinkley",
            "Consistent Page-Hinkley detection algorithm",
        ],
        [
            "Forecasting model",
            "XGBoost",
            "HistGradientBoostingRegressor",
            "Same gradient-boosting family; different model implementations",
        ],
        [
            "Adaptation policy",
            "Arm D · recent 180 days",
            "Arm D · recent 180 days",
            "Common Arm D, 180-day retraining window",
        ],
        [
            "Dataset / Regions",
            "SA1 and NSW1",
            "SA1 and NSW1",
            "Identical AEMO datasets and evaluation periods",
        ],
        [
            "Experiment seeds",
            "1-5",
            "1-5",
            "Five-seed experimental protocol",
        ],
        [
            "UQ method",
            "AgACI",
            "AgACI",
            "AgACI-based methods; implementation equivalence unverified",
        ],
        [
            "Escalation signals",
            "Interval width; lagged nonconformity",
            "Interval width; lagged 30-minute error",
            "Comparable review criteria; methodological differences remain",
        ],
        [
            "Regime definition",
            "Documented events + 7-day post-event buffer",
            "Documented events + 7-day post-event buffer",
            "Consistent event-based definition with a seven-day buffer",
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Setting",
            "Team 41",
            "Team 42",
            "Comparison status",
        ]
    )


# =========================================================
# T7 outcome table
# =========================================================
def build_t7():

    mae, retrains = load_group_a_adaptation()

    detections = load_group_a_detections()

    worst24 = load_group_a_worst24()

    review_width = load_review_rate(
        T6_WIDTH_CSV
    )

    review_nonconf = load_review_rate(
        T6_NONCONF_CSV
    )

    group_b = load_group_b()

    rows = [
        [
            "MAE during drift",
            mae["SA1"],
            mae["NSW1"],
            group_b[
                "MAE during drift (MW)"
            ]["SA1"],
            group_b[
                "MAE during drift (MW)"
            ]["NSW1"],
        ],

        [
            "Worst 24h coverage",
            worst24["SA1"],
            worst24["NSW1"],
            group_b[
                "Worst 24h coverage"
            ]["SA1"],
            group_b[
                "Worst 24h coverage"
            ]["NSW1"],
        ],

        [
            "Review rate for 0.90 coverage (interval width)",
            review_width["SA1"],
            review_width["NSW1"],
            group_b[
                "Review rate for 0.90 coverage (interval width)"
            ]["SA1"],
            group_b[
                "Review rate for 0.90 coverage (interval width)"
            ]["NSW1"],
        ],

        [
            "Review rate for 0.90 coverage (nonconformity, 30 min)",
            review_nonconf["SA1"],
            review_nonconf["NSW1"],
            group_b[
                "Review rate for 0.90 coverage (nonconformity, 30 min)"
            ]["SA1"],
            group_b[
                "Review rate for 0.90 coverage (nonconformity, 30 min)"
            ]["NSW1"],
        ],

        [
            "Detections raised",
            detections["SA1"],
            detections["NSW1"],
            group_b[
                "Detections raised"
            ]["SA1"],
            group_b[
                "Detections raised"
            ]["NSW1"],
        ],

        [
            "Retrains triggered",
            retrains["SA1"],
            retrains["NSW1"],
            group_b[
                "Retrains triggered"
            ]["SA1"],
            group_b[
                "Retrains triggered"
            ]["NSW1"],
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Outcome",
            "Team 41 · SA1",
            "Team 41 · NSW1",
            "Team 42 · SA1",
            "Team 42 · NSW1",
        ]
    )


# =========================================================
# Text wrapping
# =========================================================
def wrap_settings_df(df):

    df = df.copy()

    df["Setting"] = df[
        "Setting"
    ].apply(
        lambda x: wrap_text(
            x,
            18
        )
    )

    df["Team 41"] = df[
        "Team 41"
    ].apply(
        lambda x: wrap_text(
            x,
            27
        )
    )

    df["Team 42"] = df[
        "Team 42"
    ].apply(
        lambda x: wrap_text(
            x,
            27
        )
    )

    df["Comparison status"] = df[
        "Comparison status"
    ].apply(
        lambda x: wrap_text(
            x,
            40
        )
    )

    return df


def wrap_outcomes_df(df):

    df = df.copy()

    df["Outcome"] = df[
        "Outcome"
    ].apply(
        lambda x: wrap_text(
            x,
            38
        )
    )

    return df


# =========================================================
# Draw table
# =========================================================
def draw_table(
    ax,
    df,
    title,
    col_widths,
    font_size,
    row_height_scale,
):

    ax.axis(
        "off"
    )

    ax.text(
        0.0,
        1.00,
        title,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        font_size
    )

    table.scale(
        1,
        row_height_scale
    )

    for (row, col), cell in table.get_celld().items():

        cell.set_linewidth(
            0.7
        )

        text = cell.get_text()

        text.set_wrap(
            True
        )

        if row == 0:

            text.set_weight(
                "bold"
            )

            text.set_ha(
                "center"
            )

            cell.set_height(
                cell.get_height()
                * 1.15
            )

        else:

            if col == 0:

                text.set_ha(
                    "left"
                )

            else:

                text.set_ha(
                    "center"
                )

    return table


# =========================================================
# Main
# =========================================================
def main():

    settings_df = build_settings_df()

    t7_df = build_t7()

    # Save raw result table
    t7_df.to_csv(
        OUT_CSV,
        index=False
    )

    # Display versions with wrapped text
    settings_show = wrap_settings_df(
        settings_df
    )

    t7_show = wrap_outcomes_df(
        t7_df
    )

    print(
        "\n=== FINAL CUSTOM T7 ==="
    )

    print(
        t7_df.to_string(
            index=False
        )
    )

    # =====================================================
    # Figure
    # =====================================================
    fig = plt.figure(
        figsize=(22, 13),
        dpi=220
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            1.18,
            1.0
        ],
        hspace=0.20
    )

    ax1 = fig.add_subplot(
        gs[0]
    )

    ax2 = fig.add_subplot(
        gs[1]
    )

    # =====================================================
    # Main title
    # =====================================================
    fig.suptitle(
        "T7 - Trigger comparison outcomes (SA1 and NSW1)",
        fontsize=25,
        fontweight="bold",
        y=0.985
    )

    # =====================================================
    # Subtitle
    # =====================================================
    subtitle = (
        "Team 41 values are generated from our own project data "
        "(runs.csv / T5 / T6 / Page-Hinkley changepoints). "
        "Team 42 values are taken from the results they shared."
    )

    fig.text(
        0.05,
        0.935,
        subtitle,
        fontsize=10.5,
        ha="left",
        va="top"
    )

    # =====================================================
    # Configuration table
    # =====================================================
    draw_table(
        ax1,
        settings_show,
        title=(
            "Experimental configuration "
            "and comparability"
        ),
        col_widths=[
            0.14,
            0.22,
            0.22,
            0.42,
        ],
        font_size=10,
        row_height_scale=2.15,
    )

    # =====================================================
    # T7 results table
    # =====================================================
    draw_table(
        ax2,
        t7_show,
        title=(
            "T7 · Comparison outcomes"
        ),
        col_widths=[
            0.37,
            0.1575,
            0.1575,
            0.1575,
            0.1575,
        ],
        font_size=11,
        row_height_scale=2.25,
    )

    # =====================================================
    # Layout
    # =====================================================
    plt.subplots_adjust(
        top=0.865,
        bottom=0.04,
        left=0.04,
        right=0.96
    )

    # =====================================================
    # Save PNG
    # =====================================================
    fig.savefig(
        OUT_PNG,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        "\nSaved:"
    )

    print(
        OUT_CSV
    )

    print(
        OUT_PNG
    )


if __name__ == "__main__":
    main()