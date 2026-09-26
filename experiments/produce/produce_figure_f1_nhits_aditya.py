"""F1 — yearly rolling-MAE degradation curves for Aditya's NHITS model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from drift_lab.config import REGIONS
from experiments.produce.produce_figure_f1 import (
    EVENT_STYLE,
    annotate_events_with_names,
    format_year_axis,
    load_events,
)
from experiments.results_io import FIGURE1_DIR, RUNS_CSV, curve_path

MODEL_METHOD = "nhits_aditya"
MODEL_LABEL = "NHITS Aditya (7-day blocks)"
MODEL_COLOR = "#1464A5"


def load_model_curve(region: str) -> pd.Series:
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- run the NHITS Aditya experiment first")

    runs = pd.read_csv(RUNS_CSV)
    rows = runs[
        (runs["group"] == "baseline")
        & (runs["method"] == MODEL_METHOD)
        & (runs["dataset"] == "aemo")
        & (runs["region"] == region)
    ].copy()
    if rows.empty:
        raise SystemExit(
            f"no baseline rows for {MODEL_METHOD!r}, region={region!r} in runs.csv"
        )

    if "timestamp" in rows:
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
        rows = rows.sort_values("timestamp")

    row = rows.iloc[-1]
    seed = None if pd.isna(row["seed"]) else int(row["seed"])
    path = curve_path(row["config_hash"], "aemo", region, seed)
    if not path.exists():
        raise SystemExit(f"curve dump not found for {MODEL_METHOD}/{region}: {path}")

    curve = pd.read_csv(path, parse_dates=["index"])
    return pd.Series(
        curve["rolling_mae_7d"].to_numpy(),
        index=pd.DatetimeIndex(curve["index"]),
        name=MODEL_METHOD,
    ).sort_index()


def main() -> None:
    FIGURE1_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    region_curves = {region: load_model_curve(region) for region in REGIONS}
    years = sorted(
        {
            timestamp.year
            for curve in region_curves.values()
            for timestamp in curve.dropna().index
        }
    )
    years = [
        year
        for year in years
        if all(
            (curve.dropna().index.year == year).sum() >= 2
            for curve in region_curves.values()
        )
    ]

    for year in years:
        fig, axes = plt.subplots(
            len(REGIONS), 1, figsize=(15.5, 5.2 * len(REGIONS)), sharex=True
        )
        fig.patch.set_facecolor("white")

        for ax, region in zip(axes, REGIONS):
            ax.set_facecolor("#FAFBFC")
            curve = region_curves[region]
            year_curve = curve[curve.index.year == year]
            ax.plot(
                year_curve.index,
                year_curve.to_numpy(),
                label=MODEL_LABEL,
                color=MODEL_COLOR,
                lw=1.8,
                alpha=0.95,
                zorder=3,
            )
            peak = float(year_curve.max()) if year_curve.notna().any() else 0.0
            if peak > 0:
                ax.set_ylim(0, peak * 1.65)

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
            f"F1: 7-day rolling MAE for Aditya NHITS — {year}",
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
        event_legend = [
            Line2D([], [], color=EVENT_STYLE[1][0], lw=1.1, ls="--", label="Tier 1 documented event"),
            Line2D([], [], color=EVENT_STYLE[2][0], lw=0.9, ls="--", label="Tier 2 documented event"),
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
        out_path = FIGURE1_DIR / f"f1_nhits_aditya_{year}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()