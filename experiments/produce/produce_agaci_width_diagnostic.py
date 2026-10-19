"""QC -- AgACI interval width over the TEST period, per model (diagnostic).

Not a report figure. One figure per region, one row per model, for arm A
(never retrain, so only the UQ method adapts) and one seed (seed 1; the
deterministic dhr_arima has none), so spikes are not averaged away:

    daily median width       the typical AgACI interval;
    daily min-max band       how far it moves within each day;
    fixed split conformal    its constant width, for comparison;
    2 x M_t                  the width AgACI tends to when its weight sits on
                             candidates whose radius is infinite: each such
                             radius is replaced by M_t = max(|calibration
                             errors|, |errors| in the 7-day buffer) -- see
                             uncertainty/agaci.py.

The global drift regime is shaded. Reads the saved interval and calibration
files; nothing is re-issued.

Usage:
    python -m experiments.produce.produce_agaci_width_diagnostic
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from experiments import results_io
from experiments.produce.figure_text import axes_right_in, line_height_in, wrap_to_width
from experiments.produce.produce_table_t5 import latest_uq_rows
from experiments.run.uncertainty.run_aemo_uq_replay import AGACI, FIXED

MODEL_NAMES = {"dhr_arima": "DHR + ARIMA", "nhits": "NHITS", "xgboost": "XGBoost"}
ARM = "A_never_retrain"
SEED = 1
BUFFER = 336  # AgACI's provisional 7-day error buffer
AGACI_COLOUR = "#2a78d6"
FIXED_COLOUR = "#eb6834"
CAP_COLOUR = "#7a7974"
DRIFT_SHADE = "#e9e4de"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SUBTITLE_SIZE = 9.5


def load_run(run) -> pd.DataFrame:
    """Half-hourly widths, the replacement cap 2 x M_t and the regime."""
    seed = None if run.seed_key == -1 else int(run.seed_key)
    source = json.loads((results_io.RUNS_DIR / run.config_hash / "config.json").read_text())["source_config_hash"]
    d = pd.read_csv(results_io.intervals_path(source, "aemo", run.region, seed), parse_dates=["timestamp"])
    calibration_max = pd.read_csv(results_io.calibration_path(run.model, run.region, seed))["residual"].abs().max()
    # M_t uses only outcomes before t: the buffer holds the latest BUFFER errors.
    buffer_max = (d["actual"] - d["forecast"]).abs().rolling(BUFFER, min_periods=1).max().shift(1)
    return pd.DataFrame(
        {
            "agaci": d[f"upper_{AGACI}"] - d[f"lower_{AGACI}"],
            "fixed": d[f"upper_{FIXED}"] - d[f"lower_{FIXED}"],
            "cap": 2 * buffer_max.fillna(0).clip(lower=calibration_max),
            "drift": d["regime"] == "drift",
        }
    ).set_index(d["timestamp"])


def drift_spans(drift: pd.Series) -> list[tuple]:
    block = (drift != drift.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in drift.groupby(block) if g.iloc[0]]


def plot_region(runs: pd.DataFrame, region: str):
    models = sorted(runs["model"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 3.1 * len(models) + 1.2), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[:, 0]
    for ax, model in zip(axes, models):
        run = runs[runs["model"] == model].iloc[0]
        d = load_run(run)
        daily = d["agaci"].resample("D").agg(["median", "min", "max"])
        for start, end in drift_spans(d["drift"]):
            ax.axvspan(start, end, color=DRIFT_SHADE, linewidth=0, zorder=0)
        ax.fill_between(daily.index, daily["min"], daily["max"], color=AGACI_COLOUR, alpha=0.18,
                        linewidth=0, label="AgACI daily min–max")
        ax.plot(daily.index, daily["median"], color=AGACI_COLOUR, linewidth=1.1, label="AgACI daily median")
        ax.plot(d.index, d["fixed"], color=FIXED_COLOUR, linestyle=(0, (5, 2)), linewidth=1.1,
                label="fixed split conformal")
        ax.plot(d.index, d["cap"], color=CAP_COLOUR, linestyle=(0, (1, 2)), linewidth=1.2,
                label="2 × M (infinite-candidate cap)")
        share = (d["agaci"] > 0.9 * d["cap"]).mean()
        ax.set_title(
            f"{MODEL_NAMES.get(model, model)}   median {d['agaci'].median():,.0f} MW · "
            f"99th pct {d['agaci'].quantile(0.99):,.0f} MW · within 10% of the cap {share:.1%} of half-hours",
            loc="left", fontsize=10, color=TEXT,
        )
        ax.set_ylabel("interval width (MW)", color=TEXT_MUTED, fontsize=9)
        ax.set_ylim(0, d["cap"].max() * 1.08)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].margins(x=0.005)

    height = fig.get_figheight()
    header_in = 0.7
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / height))
    subtitle = wrap_to_width(
        fig,
        f"QC diagnostic, not a report figure. 90% AgACI intervals, arm A (never retrain), seed {SEED} "
        "(dhr_arima: deterministic, one run), issued one half-hour ahead over the TEST period. Blue: daily "
        "median and min–max of the half-hourly width. Orange: the constant fixed split conformal width. "
        "Grey dotted: 2 × M, the width AgACI tends to when its weight sits on candidates with an infinite "
        f"radius (M = largest |error| in the calibration window or the {BUFFER}-observation buffer). "
        "Global drift regime shaded.",
        axes_right_in(fig, axes) - 0.01 * fig.get_figwidth(),
        SUBTITLE_SIZE,
    )
    extra = subtitle.count("\n") * line_height_in(SUBTITLE_SIZE)
    fig.suptitle(f"QC — AgACI interval width over time, {region}", x=0.01, y=1 - 0.08 / height,
                 ha="left", va="top", fontsize=14, fontweight="bold", color=TEXT)
    fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=SUBTITLE_SIZE, color=TEXT_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1 - (0.5 + extra + 0.2) / height),
               ncol=4, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 1 - (header_in + extra + 0.15) / height))

    out = results_io.FIGURES_DIR / f"qc_agaci_width_{region}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    rows = latest_uq_rows().drop_duplicates(["method", "region", "seed_key"])
    rows = rows[(rows["uq"] == AGACI) & (rows["arm"] == ARM) & rows["seed_key"].isin([SEED, -1])]
    for region, runs in rows.groupby("region"):
        print(f"wrote {plot_region(runs, region)}")


if __name__ == "__main__":
    main()
