"""F1 — 7-day rolling-MAE degradation curve for the frozen baselines.

Reads results/runs.csv for each baseline's config_hash, the curve dump that
hash points to, and the documented events catalogue. No metric is computed
here — every line is a curve `run_aemo_baselines.py` (or, for nhits,
`run_aemo_nhits.py`) already produced; this script doesn't care which
split_id a method's rows came from, just group/dataset/region/method.

Writes:
  f1_degradation_<region>.png   one per region, the full test period, markers only
  f1_degradation_<year>.png     one per year, both regions stacked, markers + names
"""

from __future__ import annotations

import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from experiments.results_io import FIGURES_DIR, RUNS_CSV, curve_path

MODEL_STYLE = {
    "seasonal_naive": ("Seasonal naive (lag 48)", "tab:green"),
    "xgboost": ("XGBoost (lag + calendar)", "tab:blue"),
    "dhr_arima": ("DHR + ARIMA", "tab:orange"),
    "nhits": ("NHITS (pilot, weekly re-grounded)", "tab:purple"),
}

EVENT_COLOR = "tab:red"
EVENT_LABEL_WIDTH = 22  # chars/line — wraps a long event name into a compact block
EVENT_LABEL_LEVELS = (0.97, 0.83, 0.69, 0.55, 0.41, 0.27, 0.13)  # stagger down the axis
EVENT_LABEL_GAP = pd.Timedelta(days=3)


def load_model_curves(region: str) -> dict[str, pd.Series]:
    runs = pd.read_csv(RUNS_CSV)
    baseline = runs[
        (runs["group"] == "baseline") & (runs["dataset"] == "aemo") & (runs["region"] == region)
    ]

    curves: dict[str, pd.Series] = {}
    for method in MODEL_STYLE:
        rows = baseline[baseline["method"] == method]
        if rows.empty:
            continue
        row = rows.iloc[-1]  # latest run for this (method, region)
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        path = curve_path(row["config_hash"], "aemo", region, seed)
        curve = pd.read_csv(path, parse_dates=["index"])
        curves[method] = pd.Series(curve["rolling_mae_7d"].to_numpy(), index=curve["index"])
    return curves


def load_events() -> pd.DataFrame:
    return pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])


def test_years(region_curves: dict[str, dict[str, pd.Series]]) -> list[int]:
    """Calendar years with real coverage. AEMO stamps an interval by its end,
    so the very last test row lands on 2024-01-01 — a single point, not a
    real year of data — so a year needs >=2 rows to count."""
    all_curves = [c for curves in region_curves.values() for c in curves.values()]
    all_years = sorted({ts.year for c in all_curves for ts in c.dropna().index})
    return [y for y in all_years if all((c.dropna().index.year == y).sum() >= 2 for c in all_curves)]


def annotate_events(ax: plt.Axes, events: pd.DataFrame, region: str) -> None:
    """Markers only — for the full multi-year overview, where names would just clutter."""
    in_scope = events[events["region"].isin([region, "NEM"])]
    for _, event in in_scope.iterrows():
        ax.axvline(event["start_date"], color=EVENT_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
        if event["end_date"] != event["start_date"]:
            ax.axvspan(event["start_date"], event["end_date"], color=EVENT_COLOR, alpha=0.06)


def annotate_events_with_names(ax: plt.Axes, events: pd.DataFrame, region: str, year: int) -> None:
    """Markers + the event name, wrapped onto several short horizontal lines
    (never one long trailing line) and staggered vertically so consecutive
    events don't overlap."""
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)

    in_scope = (
        events[
            events["region"].isin([region, "NEM"])
            & (events["start_date"] < year_end)
            & (events["end_date"] >= year_start)
        ]
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    for i, event in in_scope.iterrows():
        is_period = event["end_date"] != event["start_date"]
        level = EVENT_LABEL_LEVELS[i % len(EVENT_LABEL_LEVELS)]
        label = "\n".join(textwrap.wrap(event["event_name"], width=EVENT_LABEL_WIDTH))
        text_kw = {
            "transform": ax.get_xaxis_transform(),
            "va": "top",
            "fontsize": 6.5,
            "color": EVENT_COLOR,
            "linespacing": 1.15,
        }

        if is_period:
            span_start = max(event["start_date"], year_start)
            span_end = min(event["end_date"], year_end)
            for edge in (event["start_date"], event["end_date"]):
                if year_start <= edge < year_end:
                    ax.axvline(edge, color=EVENT_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvspan(span_start, span_end, color=EVENT_COLOR, alpha=0.07)
            ax.text(span_end + EVENT_LABEL_GAP, level, label, ha="left", **text_kw)
        else:
            edge = event["start_date"]
            ax.axvline(edge, color=EVENT_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
            near_right = edge > year_end - pd.Timedelta(days=45)
            ax.text(
                edge - EVENT_LABEL_GAP if near_right else edge + EVENT_LABEL_GAP,
                level,
                label,
                ha="right" if near_right else "left",
                **text_kw,
            )


def format_full_period_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.grid(alpha=0.25)


def format_year_axis(ax: plt.Axes, year: int) -> None:
    ax.set_xlim(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.grid(alpha=0.25)


def plot_curves(ax: plt.Axes, curves: dict[str, pd.Series]) -> float:
    """Draw every model's line; return the max y-value plotted (0 if none)."""
    peak = 0.0
    for method, curve in curves.items():
        label, color = MODEL_STYLE[method]
        ax.plot(curve.index, curve.to_numpy(), label=label, color=color, linewidth=1.2)
        if curve.notna().any():
            peak = max(peak, curve.max())
    return peak


def plot_full_period(region: str, curves: dict[str, pd.Series], events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_curves(ax, curves)
    annotate_events(ax, events, region)

    ax.set_title(f"F1: rolling 7-day MAE, frozen baselines — {region}", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("7-day rolling MAE (MW)")
    ax.set_ylim(bottom=0)
    format_full_period_axis(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False, borderaxespad=0.0)
    fig.tight_layout()

    out_path = FIGURES_DIR / f"f1_degradation_{region}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_year(year: int, region_curves: dict[str, dict[str, pd.Series]], events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(15, 5 * len(REGIONS)))

    for ax, region in zip(axes, REGIONS):
        year_curves = {
            method: curve[curve.index.year == year] for method, curve in region_curves[region].items()
        }
        peak = plot_curves(ax, year_curves)
        if peak > 0:
            ax.set_ylim(0, peak * 1.4)  # headroom for the staggered labels

        annotate_events_with_names(ax, events, region, year)

        ax.set_title(f"{region}", loc="left")
        ax.set_ylabel("7-day rolling MAE (MW)")
        format_year_axis(ax, year)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False, borderaxespad=0.0)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"F1: rolling 7-day MAE, frozen baselines — {year}", x=0.01, ha="left")
    fig.tight_layout()

    out_path = FIGURES_DIR / f"f1_degradation_{year}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    region_curves = {region: load_model_curves(region) for region in REGIONS}

    for region in REGIONS:
        plot_full_period(region, region_curves[region], events)

    for year in test_years(region_curves):
        plot_year(year, region_curves, events)


if __name__ == "__main__":
    main()
