"""F1 — 7-day rolling-MAE degradation curves for frozen baselines.

Reads saved curves and all Tier 1 and Tier 2 documented events. No metric is
recalculated.
Writes one yearly figure with SA1 and NSW1 shown as stacked panels, plus a
joined figure covering every test year on one continuous time axis.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.results_io import FIGURE1_DIR, RUNS_CSV, curve_path

MODEL_STYLE = {
    "seasonal_naive": ("Seasonal naive (lag 48)", "#198754"),
    "xgboost": ("XGBoost (lag + calendar)", "#1464A5"),
    "dhr_arima": ("DHR + ARIMA", "#E67E22"),
    "nhits": ("NHITS (7-day blocks)", "#7D3C98"),
}

EVENT_STYLE = {
    1: ("#C62828", "#F4A261", 0.13),
    2: ("#6F42C1", "#C9B6E4", 0.08),
}
GRID_COLOR = "#D9DEE5"
EVENT_LABEL_LEVELS = {
    1: (0.99, 0.955),
    2: (0.91, 0.865, 0.82),
}


def load_model_curves(region: str) -> dict[str, pd.Series]:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run the baseline experiments first")

    runs = pd.read_csv(RUNS_CSV)
    baseline = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["region"] == region)
    ].copy()

    if "timestamp" in baseline:
        baseline["timestamp"] = pd.to_datetime(
            baseline["timestamp"], errors="coerce", utc=True
        )
        baseline = baseline.sort_values("timestamp")

    curves: dict[str, pd.Series] = {}
    for method in MODEL_STYLE:
        rows = baseline[baseline["method"] == method]
        if rows.empty:
            continue

        row = rows.iloc[-1]
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        path = curve_path(row["config_hash"], "aemo", region, seed)
        if not path.exists():
            print(f"warning: curve dump not found for {method}/{region}: {path}")
            continue

        curve = pd.read_csv(path, parse_dates=["index"])
        curves[method] = pd.Series(
            curve["rolling_mae_7d"].to_numpy(),
            index=pd.DatetimeIndex(curve["index"]),
            name=method,
        ).sort_index()

    if not curves:
        raise SystemExit(f"no baseline curves found for {region}")
    return curves


def load_events() -> pd.DataFrame:
    """Load the full Tier 1 and Tier 2 documented-event catalogue."""
    events = pd.read_csv(
        DOCUMENTED_EVENTS_CSV,
        parse_dates=["start_date", "end_date"],
    )
    return events[events["tier"].isin([1, 2])].copy()


def test_years(
    region_curves: dict[str, dict[str, pd.Series]],
) -> list[int]:
    """Return calendar years with at least two valid rows in every curve."""
    all_curves = [
        curve
        for curves in region_curves.values()
        for curve in curves.values()
    ]
    years = sorted(
        {
            timestamp.year
            for curve in all_curves
            for timestamp in curve.dropna().index
        }
    )
    return [
        year
        for year in years
        if all(
            (curve.dropna().index.year == year).sum() >= 2
            for curve in all_curves
        )
    ]


def year_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(year=year, month=1, day=1),
        pd.Timestamp(year=year + 1, month=1, day=1),
    )


def events_in_window(
    events: pd.DataFrame,
    region: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    return (
        events[
            events["region"].isin([region, "NEM"])
            & (events["start_date"] < window_end)
            & (events["end_date"] >= window_start)
        ]
        .sort_values("start_date")
        .reset_index(drop=True)
    )


def annotate_events_with_names(
    ax: plt.Axes,
    events: pd.DataFrame,
    region: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> None:
    """Draw tier-coloured markers and compact vertical event names."""
    year_start, year_end = window_start, window_end
    in_scope = events_in_window(events, region, window_start, window_end)

    tier_counts = {1: 0, 2: 0}
    for _, event in in_scope.iterrows():
        tier = int(event["tier"])
        colour, shade_colour, shade_alpha = EVENT_STYLE[tier]
        is_period = event["end_date"] != event["start_date"]
        levels = EVENT_LABEL_LEVELS[tier]
        level = levels[tier_counts[tier] % len(levels)]
        tier_counts[tier] += 1
        short_name = str(event["event_name"]).split(" - ")[0][:38]
        label = f"[T{tier}] {short_name}"

        if is_period:
            span_start = max(event["start_date"], year_start)
            span_end = min(event["end_date"] + pd.Timedelta(days=1), year_end)
            ax.axvspan(
                span_start,
                span_end,
                color=shade_colour,
                alpha=shade_alpha,
                lw=0,
                zorder=0,
            )
            for edge in (event["start_date"], event["end_date"]):
                if year_start <= edge < year_end:
                    ax.axvline(
                        edge,
                        color=colour,
                        ls="--",
                        lw=0.8,
                        alpha=0.55,
                        zorder=2,
                    )
            anchor = span_start
        else:
            anchor = event["start_date"]
            ax.axvline(
                anchor,
                color=colour,
                ls="--",
                lw=0.9,
                alpha=0.65,
                zorder=2,
            )

        ax.text(
            anchor,
            level,
            label,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=6.2 if tier == 1 else 5.8,
            color=colour,
            fontweight="bold" if tier == 1 else "normal",
            linespacing=1.1,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
                "pad": 0.8,
            },
            clip_on=True,
            zorder=6,
        )


def format_year_axis(ax: plt.Axes, year: int) -> None:
    ax.set_xlim(*year_window(year))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_axes(ax)


def format_multi_year_axis(ax: plt.Axes, years: list[int]) -> None:
    ax.set_xlim(year_window(years[0])[0], year_window(years[-1])[1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    style_axes(ax)
    for year in years[1:]:
        ax.axvline(
            year_window(year)[0],
            color="#7A8594",
            lw=1.0,
            alpha=0.7,
            zorder=1,
        )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.85)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.45, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BD")


def plot_curves(ax: plt.Axes, curves: dict[str, pd.Series]) -> float:
    """Draw each saved curve and return the highest plotted value."""
    peak = 0.0
    for method, curve in curves.items():
        label, colour = MODEL_STYLE[method]
        ax.plot(
            curve.index,
            curve.to_numpy(),
            label=label,
            color=colour,
            lw=1.8,
            alpha=0.95,
            zorder=3,
        )
        if curve.notna().any():
            peak = max(peak, float(curve.max()))
    return peak


def plot_year(
    year: int,
    region_curves: dict[str, dict[str, pd.Series]],
    events: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        len(REGIONS),
        1,
        figsize=(15.5, 5.2 * len(REGIONS)),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for ax, region in zip(axes, REGIONS):
        ax.set_facecolor("#FAFBFC")
        year_curves = {
            method: curve[curve.index.year == year]
            for method, curve in region_curves[region].items()
        }
        peak = plot_curves(ax, year_curves)
        if peak > 0:
            # Reserve the upper section for event names so model lines do not
            # cover the labels. The highest curve occupies about 61% of the
            # plotting height, while event labels begin above 82%.
            ax.set_ylim(0, peak * 1.65)

        annotate_events_with_names(ax, events, region, *year_window(year))
        ax.set_title(region, loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("7-day rolling MAE (MW)")
        format_year_axis(ax, year)
        ax.legend(
            loc="lower right",
            ncol=2,
            fontsize=8,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#C7CDD4",
        )

    axes[-1].set_xlabel(f"Date ({year})")
    fig.suptitle(
        f"F1: 7-day rolling MAE for frozen baselines — {year}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.969,
        "Higher values indicate larger forecasting errors; vertical labels show all documented Tier 1 and Tier 2 events.",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )

    add_event_legend(fig)

    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=2.0)
    out_path = FIGURE1_DIR / f"f1_degradation_{year}.png"
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {out_path}")


def add_event_legend(fig: plt.Figure) -> None:
    event_legend = [
        Line2D(
            [], [], color=EVENT_STYLE[1][0], lw=1.1, ls="--",
            label="Tier 1 documented event",
        ),
        Line2D(
            [], [], color=EVENT_STYLE[2][0], lw=0.9, ls="--",
            label="Tier 2 documented event",
        ),
    ]
    fig.legend(
        handles=event_legend,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8,
    )


def plot_all_years(
    years: list[int],
    region_curves: dict[str, dict[str, pd.Series]],
    events: pd.DataFrame,
) -> None:
    """Draw every test year on one continuous axis, one panel per region."""
    window_start = year_window(years[0])[0]
    window_end = year_window(years[-1])[1]
    fig, axes = plt.subplots(
        len(REGIONS),
        1,
        figsize=(8.0 * len(years), 5.6 * len(REGIONS)),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for ax, region in zip(axes, REGIONS):
        ax.set_facecolor("#FAFBFC")
        window_curves = {
            method: curve[(curve.index >= window_start) & (curve.index < window_end)]
            for method, curve in region_curves[region].items()
        }
        peak = plot_curves(ax, window_curves)
        if peak > 0:
            ax.set_ylim(0, peak * 1.65)

        annotate_events_with_names(ax, events, region, window_start, window_end)
        ax.set_title(region, loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("7-day rolling MAE (MW)")
        format_multi_year_axis(ax, years)
        ax.legend(
            loc="lower right",
            ncol=4,
            fontsize=8,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#C7CDD4",
        )

    axes[-1].set_xlabel(f"Date ({years[0]}–{years[-1]})")
    fig.suptitle(
        f"F1: 7-day rolling MAE for frozen baselines — {years[0]}–{years[-1]}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.969,
        "All test years joined on one axis; solid grey lines mark year boundaries. "
        "Vertical labels show all documented Tier 1 and Tier 2 events.",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    add_event_legend(fig)

    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=2.0)
    out_path = FIGURE1_DIR / f"f1_degradation_{years[0]}_{years[-1]}.png"
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    FIGURE1_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    region_curves = {
        region: load_model_curves(region)
        for region in REGIONS
    }

    years = test_years(region_curves)
    for year in years:
        plot_year(year, region_curves, events)
    if len(years) > 1:
        plot_all_years(years, region_curves, events)


if __name__ == "__main__":
    main()