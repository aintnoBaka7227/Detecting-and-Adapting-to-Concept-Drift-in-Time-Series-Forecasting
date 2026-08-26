"""Visualise the processed AEMO electricity demand and price time series.

Plotting is kept separate from loading, cleaning and splitting so that
visualisation never modifies the underlying dataset. These plots are used
for data-quality inspection, exploratory analysis and report figures.

The full-series plots show TOTALDEMAND / RRP against SETTLEMENTDATE in
chronological order, with monthly ticks to make long-term patterns and
changes across years easier to inspect.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _plot_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path | None,
) -> None:
    """Plot a single value column against `date_col` and show or save it."""

    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col]
    )

    data = data.sort_values(date_col)

    fig, ax = plt.subplots(
        figsize=(20, 7)
    )

    ax.plot(
        data[date_col],
        data[value_col],
        linewidth=0.5,
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=1)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(alpha=0.3)

    plt.xticks(rotation=90)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved figure: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_demand_series(
    df: pd.DataFrame,
    region: str,
    date_col: str = "SETTLEMENTDATE",
    demand_col: str = "TOTALDEMAND",
    output_path: Path | None = None,
) -> None:
    """Plot the complete electricity-demand time series for one region.

    The input DataFrame is copied and sorted chronologically before
    plotting. No values in the original DataFrame are modified.

    The x-axis shows one tick per month and the y-axis shows electricity
    demand in MW.

    If `output_path` is given, the figure is saved there instead of shown
    interactively.
    """

    _plot_series(
        df,
        date_col=date_col,
        value_col=demand_col,
        ylabel="Total Demand (MW)",
        title=f"{region} Total Electricity Demand",
        output_path=output_path,
    )


def plot_price_series(
    df: pd.DataFrame,
    region: str,
    date_col: str = "SETTLEMENTDATE",
    price_col: str = "RRP",
    output_path: Path | None = None,
) -> None:
    """Plot the complete electricity-price (RRP) time series for one region.

    The input DataFrame is copied and sorted chronologically before
    plotting. No values in the original DataFrame are modified.

    The x-axis shows one tick per month and the y-axis shows the Regional
    Reference Price in $/MWh.

    If `output_path` is given, the figure is saved there instead of shown
    interactively.
    """

    _plot_series(
        df,
        date_col=date_col,
        value_col=price_col,
        ylabel="RRP ($/MWh)",
        title=f"{region} Regional Reference Price",
        output_path=output_path,
    )

def plot_week_comparison(
    df: pd.DataFrame,
    region: str,
    month: int,
    day: int,
    year_a: int = 2019,
    year_b: int = 2021,
    date_col: str = "SETTLEMENTDATE",
    value_col: str = "TOTALDEMAND",
    ylabel: str = "Total Demand (MW)",
    output_path: Path | None = None,
) -> None:
    """Overlay one calendar week from `year_a` against a week in `year_b`.

    `month`/`day` is an anchor date; each year's window independently
    starts on the Monday on or after that date in that year, since the
    same "MM-DD" is not the same weekday across years two years apart
    (e.g. Aug 5 is a Monday in 2019 but a Thursday in 2021 - 2020 is a
    leap year). Snapping per-year keeps the "Mon..Sun" x-axis correct for
    both lines even though the two windows may start a few days apart.

    Both weeks are plotted against a shared "hours since Monday 00:00"
    x-axis so the two years overlap directly, rather than against their
    (different) calendar dates.

    If `output_path` is given, the figure is saved there instead of shown
    interactively.
    """

    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col]
    )

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    for year, color in ((year_a, "tab:blue"), (year_b, "tab:red")):
        anchor = pd.Timestamp(year=year, month=month, day=day)
        week_start_ts = anchor + pd.Timedelta(
            days=(7 - anchor.dayofweek) % 7
        )
        week_end_ts = week_start_ts + pd.Timedelta(days=7)

        window = data[
            (data[date_col] >= week_start_ts)
            & (data[date_col] < week_end_ts)
        ].copy()

        if window.empty:
            raise ValueError(
                f"No rows found for {week_start_ts.date()} "
                f"through {week_end_ts.date()}."
            )

        elapsed_hours = (
            (window[date_col] - week_start_ts)
            / pd.Timedelta(hours=1)
        )

        ax.plot(
            elapsed_hours,
            window[value_col],
            linewidth=1,
            label=f"{year} (from {week_start_ts.date()})",
            color=color,
        )

    ax.set_xticks(range(0, 24 * 7 + 1, 24))
    ax.set_xticklabels(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]
    )
    ax.set_xlim(0, 24 * 7)

    ax.set_xlabel("Day of week")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{region} week of {month:02d}-{day:02d} — {year_a} vs {year_b}"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved figure: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_boxplot_by_region(
    data_by_region: dict[str, pd.DataFrame],
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path | None = None,
) -> None:
    """Boxplot of `value_col`, one box per region, to compare outliers.

    `data_by_region` maps region name to its DataFrame, for example
    `{"SA1": sa1_df, "NSW1": nsw1_df}`. Boxes are drawn side by side in
    that order so outlier spread (whiskers, fliers) can be compared
    directly across regions.

    If `output_path` is given, the figure is saved there instead of shown
    interactively.
    """

    regions = list(data_by_region.keys())
    values = [
        data_by_region[region][value_col].dropna()
        for region in regions
    ]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.boxplot(
        values,
        tick_labels=regions,
        showmeans=True,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved figure: {output_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    import random

    from drift_forecasting.config import PROCESSED_DATA_DIR, REGIONS

    data_by_region = {}

    for region in REGIONS:
        input_file = (
            PROCESSED_DATA_DIR
            / f"{region}_201801_202312_cleaned_30min.csv"
        )

        df = pd.read_csv(input_file)
        data_by_region[region] = df

        demand_output_path = (
            PROCESSED_DATA_DIR
            / f"{region}_demand_series.png"
        )

        price_output_path = (
            PROCESSED_DATA_DIR
            / f"{region}_price_series.png"
        )

        plot_demand_series(df, region=region, output_path=demand_output_path)
        plot_price_series(df, region=region, output_path=price_output_path)

        # Picked randomly for now
        random_anchor = pd.Timestamp("2019-01-01") + pd.Timedelta(
            days=random.randint(0, 358)
        )

        week_output_path = (
            PROCESSED_DATA_DIR
            / f"{region}_week_comparison_{random_anchor.strftime('%m%d')}.png"
        )

        plot_week_comparison(
            df,
            region=region,
            month=random_anchor.month,
            day=random_anchor.day,
            output_path=week_output_path,
        )

    demand_boxplot_path = (
        PROCESSED_DATA_DIR
        / "demand_boxplot_by_region.png"
    )

    price_boxplot_path = (
        PROCESSED_DATA_DIR
        / "price_boxplot_by_region.png"
    )

    plot_boxplot_by_region(
        data_by_region,
        value_col="TOTALDEMAND",
        ylabel="Total Demand (MW)",
        title="Total Electricity Demand by Region",
        output_path=demand_boxplot_path,
    )

    plot_boxplot_by_region(
        data_by_region,
        value_col="RRP",
        ylabel="RRP ($/MWh)",
        title="Regional Reference Price by Region",
        output_path=price_boxplot_path,
    )