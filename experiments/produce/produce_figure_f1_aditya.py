"""F1 degradation curves for Aditya's seasonal-naive AEMO run."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.config import REGIONS
from experiments.produce.produce_figure_f1 import (
    EVENT_STYLE,
    annotate_events_with_names,
    format_year_axis,
    load_events,
    test_years,
)
from experiments.results_io import FIGURE1_DIR, RUNS_CSV, curve_path

SPLIT_ID = "aemo_frozen_v1_aditya"
OUTPUT_DIR = FIGURE1_DIR / "seasonal_naive_aditya"
MODEL_STYLE = {
    "seasonal_naive": ("Seasonal naive (lag 48), Aditya run", "#198754"),
}


def load_model_curves(region: str) -> dict[str, pd.Series]:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run the seasonal-naive experiment first")

    runs = pd.read_csv(RUNS_CSV)
    matching_runs = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["region"] == region)
        & (runs["method"] == "seasonal_naive")
        & (runs["split_id"] == SPLIT_ID)
    ].copy()
    if matching_runs.empty:
        raise SystemExit(f"no {SPLIT_ID} seasonal-naive run found for {region}")

    if "timestamp" in matching_runs:
        matching_runs["timestamp"] = pd.to_datetime(
            matching_runs["timestamp"], errors="coerce", utc=True
        )
        matching_runs = matching_runs.sort_values("timestamp")

    row = matching_runs.iloc[-1]
    seed = None if pd.isna(row["seed"]) else int(row["seed"])
    path = curve_path(row["config_hash"], "aemo", region, seed)
    if not path.exists():
        raise SystemExit(f"curve dump not found for {region}: {path}")

    curve = pd.read_csv(path, parse_dates=["index"])
    return {
        "seasonal_naive": pd.Series(
            curve["rolling_mae_7d"].to_numpy(),
            index=pd.DatetimeIndex(curve["index"]),
            name="seasonal_naive",
        ).sort_index()
    }


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
        curve = region_curves[region]["seasonal_naive"]
        year_curve = curve[curve.index.year == year]
        label, colour = MODEL_STYLE["seasonal_naive"]
        ax.plot(
            year_curve.index,
            year_curve.to_numpy(),
            label=label,
            color=colour,
            lw=1.8,
            alpha=0.95,
            zorder=3,
        )
        if year_curve.notna().any():
            ax.set_ylim(0, float(year_curve.max()) * 1.65)

        annotate_events_with_names(ax, events, region, year)
        ax.set_title(region, loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("7-day rolling MAE (MW)")
        format_year_axis(ax, year)
        ax.legend(
            loc="lower right",
            fontsize=8,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#C7CDD4",
        )

    axes[-1].set_xlabel(f"Date ({year})")
    fig.suptitle(
        f"F1: Aditya's seasonal-naive 7-day rolling MAE — {year}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    event_legend = [
        plt.Line2D([], [], color=EVENT_STYLE[1][0], lw=1.1, ls="--", label="Tier 1 documented event"),
        plt.Line2D([], [], color=EVENT_STYLE[2][0], lw=0.9, ls="--", label="Tier 2 documented event"),
    ]
    fig.legend(
        handles=event_legend,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"f1_degradation_{year}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    region_curves = {
        region: load_model_curves(region)
        for region in REGIONS
    }
    events = load_events()

    for year in test_years(region_curves):
        plot_year(year, region_curves, events)


if __name__ == "__main__":
    main()