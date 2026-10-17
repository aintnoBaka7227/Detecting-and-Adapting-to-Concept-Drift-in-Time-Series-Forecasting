"""Streamlit page for the project's tables and figures, up to adaptation.

Run from the repo root:

    streamlit run app/streamlit_app.py

Read-only: it never runs a model or recomputes a metric. Detection pages
show the tables and figures already in results/. The Adaptation page is built
live from results/runs.csv: every row with the adaptation runner's split_id
(ADWIN-triggered arms), keeping the most recent run per model x arm x region x
seed, through the same loader that produces T3, T4 and F3
(experiments/produce/table_adaptation_common.py). Run the adaptation runner
again and refresh the page to see the new rows.

A missing file shows a hint about which script creates it instead of
crashing the page.

Uncertainty and escalation (T5/T6, F4/F5) have a placeholder tab; add their
files to UNCERTAINTY_ITEMS when they exist.
"""

from __future__ import annotations

from pathlib import Path

import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):  # so `experiments` and `drift_lab` import without installing
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

ADAPT_SPLIT_ID = "aemo_adapt_adwin_std_hh_v1"
MODELS = ("nhits", "xgboost", "dhr_arima")
REGIONS = ("SA1", "NSW1")

# (title, csv path, producing script) ---------------------------------------
T1 = ("Table 1: detection on the synthetic benchmark",
      TABLES / "table1" / "table_t1_post_tune_synthetic_detection.csv",
      "experiments.produce.produce_table_t1_post_tune")
T2 = ("Table 2: detection on AEMO vs documented events",
      TABLES / "table2" / "table_t2_aemo_events_all_streams_post_tune.csv",
      "experiments.produce.produce_table_t2_all_streams_post_tune")
T3 = ("Table 3: forecast accuracy by regime",
      TABLES / "table3" / "table_t3_forecast_accuracy.csv",
      "experiments.produce.produce_table_t3")
T4 = ("Table 4: what each adaptation arm cost",
      TABLES / "table4" / "table_t4_adaptation_cost.csv",
      "experiments.produce.produce_table_t4")

# Add finished uncertainty / escalation outputs here: (title, path, script).
UNCERTAINTY_ITEMS: list[tuple[str, Path, str]] = []


def _stretch(fn, *args, **kwargs):
    """Full-width widgets across Streamlit versions: newer releases take
    width="stretch", older ones use_container_width=True."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    """mtime is only part of the cache key, so a regenerated file is re-read."""
    return pd.read_csv(path)


def show_table(spec, filters: dict[str, list[str]] | None = None) -> None:
    title, path, script = spec
    st.subheader(title)
    if not path.exists():
        st.info(f"Not generated yet. Create it with `python -m {script}`.")
        return
    df = load_csv(str(path), path.stat().st_mtime)
    for column, wanted in (filters or {}).items():
        if column in df.columns and wanted:
            df = df[df[column].isin(wanted)]
    _stretch(st.dataframe, df, hide_index=True)
    st.download_button(
        "Download CSV",
        path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        key=f"dl-{path.name}",
    )


def show_image(path: Path, caption: str, script: str) -> None:
    if path.exists():
        _stretch(st.image, str(path), caption=caption)
    else:
        st.info(f"{path.name} not generated yet. Create it with `python -m {script}`.")


def tab_overview() -> None:
    st.markdown(
        """
**Detecting and adapting to concept drift in time series forecasting**

Applied to two AEMO electricity-demand regions (SA1, NSW1) and a synthetic
benchmark with known drift.

```
Data  ->  Detect  ->  Adapt  ->  Quantify (+ Escalate)
 done      done       done        coming
```

This page shows the results up to adaptation. Uncertainty and escalation
are added when those steps are finished.
"""
    )
    runs_path = RESULTS / "runs.csv"
    if not runs_path.exists():
        st.warning("results/runs.csv not found.")
        return
    runs = load_csv(str(runs_path), runs_path.stat().st_mtime)
    adapt = runs[runs["split_id"] == ADAPT_SPLIT_ID] if "split_id" in runs.columns else runs.iloc[0:0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in runs.csv", f"{len(runs):,}")
    c2.metric("Adaptation rows", f"{len(adapt):,}")
    c3.metric(
        "Adaptation runs (model/arm)",
        adapt["method"].nunique() if "method" in adapt.columns else 0,
    )


def tab_forecasting() -> None:
    st.header("Frozen baselines (Figure 1)")
    st.caption("7-day rolling MAE of the frozen baselines through the test period.")
    figs = sorted((FIGURES / "figure1").glob("f1_degradation_*.png")) if FIGURES.exists() else []
    if not figs:
        st.info("No figures yet. Create them with `python -m experiments.produce.produce_figure_f1`.")
        return
    labels = {p.stem.removeprefix("f1_degradation_"): p for p in figs}
    choice = st.selectbox("Period", list(labels), index=len(labels) - 1)
    show_image(labels[choice], f"Figure 1: {choice}", "experiments.produce.produce_figure_f1")


def tab_detection() -> None:
    st.header("Drift detection")
    show_table(T1)
    kind = st.radio("Synthetic drift type (Figure 2)", ["sudden", "gradual", "recurring"], horizontal=True)
    show_image(
        FIGURES / "figure2" / f"f2_synthetic_post_tune_{kind}.png",
        f"Figure 2, synthetic: {kind}",
        "experiments.produce.produce_figure_f2_synthetic_post_tune",
    )
    st.divider()
    t2 = T2
    st.subheader(t2[0])
    if t2[1].exists():
        df = load_csv(str(t2[1]), t2[1].stat().st_mtime)
        c1, c2 = st.columns(2)
        regions = c1.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
        dets = c2.multiselect("Detector", sorted(df["detector"].unique()), default=sorted(df["detector"].unique()))
        show_table(t2, {"region": regions, "detector": dets})
    else:
        show_table(t2)
    c1, c2 = st.columns(2)
    det = c1.selectbox("Detector (Figure 2, AEMO)", ["adwin", "kswin", "page_hinkley"])
    reg = c2.selectbox("Region (Figure 2, AEMO)", list(REGIONS))
    show_image(
        FIGURES / "figure2" / f"f2_detection_standard_half_hourly_post_tune_{det}_{reg}.png",
        f"Figure 2, AEMO: {det}, {reg}",
        "experiments.produce.produce_figure_f2_aemo_standard_half_hourly_post_tune",
    )


ARM_NAMES = {
    "A": "A  never retrain",
    "B": "B  every 30 days",
    "C": "C  drift alarm, full history",
    "D": "D  drift alarm, recent window",
}


def arm_family(arm: str) -> str:
    return arm[0]


@st.cache_data(show_spinner=False)
def adaptation_tables(runs_mtime: float):
    """T3, T4 and the per-run MAE table, built from the latest ADWIN rows in
    runs.csv with the project's own producers. Returns (t3, t4, per_run, split_id)
    or an error string."""
    try:
        from experiments.produce import produce_table_t3, produce_table_t4
        from experiments.produce.table_adaptation_common import (
            latest_adaptation_rows,
            mae_by_run,
            official_d_arms,
        )
        from experiments.run.adaptation.run_aemo_adaptation_arms import SPLIT_ID

        latest = latest_adaptation_rows()
        per_run = mae_by_run(latest)
        official = official_d_arms(per_run)
        return (
            produce_table_t3.build_table(),
            produce_table_t4.build_table(),
            per_run,
            official,
            SPLIT_ID,
            latest["timestamp"].max(),
        )
    except SystemExit as exc:  # the producers exit with a message when rows are missing
        return f"No adaptation rows yet: {exc}"
    except Exception as exc:  # noqa: BLE001 - show the reason on the page
        return f"Could not build the adaptation tables from runs.csv: {exc!r}"


def apply_filters(df, models, regions, arms):
    out = df[df["model"].isin(models) & df["region"].isin(regions)]
    if arms:
        out = out[out["arm"].map(arm_family).isin(arms)]
    return out


@st.cache_data(show_spinner=False)
def f3_curves(runs_mtime: float, model: str, region: str):
    """Mean 7-day rolling MAE across seeds for arms A, B, C and the official D,
    read from the saved curve files of the latest runs. None if the curve files
    are not on this machine (they are gitignored)."""
    try:
        from experiments.produce.produce_figure_f3 import load_arm
        from experiments.produce.table_adaptation_common import (
            latest_adaptation_rows,
            mae_by_run,
            official_d_arms,
        )
        from experiments.run.adaptation.run_aemo_adaptation_arms import ARM_A, ARM_B, ARM_C

        rows = latest_adaptation_rows()
        official = official_d_arms(mae_by_run(rows))
        d_arm = official[(model, region)]
        arms = {"A": ARM_A, "B": ARM_B, "C": ARM_C, f"D ({d_arm.removeprefix('D_drift_recent_')})": d_arm}
        frame = {}
        retrains = {}
        for label, arm in arms.items():
            index, curves, stamps = load_arm(rows, model, region, arm)
            frame[label] = pd.Series(curves.mean(axis=0), index=index)
            retrains[label] = len(stamps)
        return pd.DataFrame(frame), retrains
    except Exception:  # noqa: BLE001 - missing curve files, missing rows, etc.
        return None


def cross_check_table(runs: pd.DataFrame) -> pd.DataFrame | None:
    """Ishan's Arm A/B runner (experiments/run/forecasting/run_aemo_adaptation_arms.py,
    split_id aemo_adaptation_arms_v1) against the team runner, whole-TEST MAE."""
    mine_id = "aemo_adaptation_arms_v1"
    team_id = ADAPT_SPLIT_ID
    if "split_id" not in runs.columns or mine_id not in set(runs["split_id"]):
        return None
    mine = runs[(runs["split_id"] == mine_id) & (runs["metric_name"] == "mae")].copy()
    latest = mine.groupby(["method", "region"])["timestamp"].transform("max")
    mine = mine[mine["timestamp"] == latest]
    parts = mine["method"].str.split("+", n=1, expand=True)
    mine["model"] = parts[0]
    mine["arm"] = parts[1].map({"never_retrain": "A", "scheduled_retrain_30d": "B"})
    mine = mine.dropna(subset=["arm"]).groupby(["model", "arm", "region"], as_index=False).agg(
        ishan_mae=("metric_value", "first"), ishan_retrains=("n_retrains", "first")
    )

    team = runs[
        (runs["split_id"] == team_id) & (runs["metric_name"] == "mae") & (runs["regime"] == "full")
    ].copy()
    if team.empty:
        return None
    parts = team["method"].str.split("/", n=1, expand=True)
    team["model"], team["arm"] = parts[0], parts[1].str[0]
    team = team[team["arm"].isin(["A", "B"])]
    team["seed_key"] = team["seed"].fillna(-1)
    latest = team.groupby(["model", "arm", "region", "seed_key"])["timestamp"].transform("max")
    team = team[team["timestamp"] == latest]
    team = team.groupby(["model", "arm", "region"], as_index=False).agg(
        team_mae=("metric_value", "mean"), team_seeds=("seed_key", "nunique")
    )
    both = mine.merge(team, on=["model", "arm", "region"], how="outer").sort_values(
        ["model", "region", "arm"]
    )
    both["difference (%)"] = (both["ishan_mae"] / both["team_mae"] - 1) * 100
    return both.round(1).reset_index(drop=True)


def tab_adaptation() -> None:
    st.header("Adaptation arms (ADWIN-triggered)")
    st.markdown(
        "- **A** never retrain  \n"
        "- **B** retrain every 30 days  \n"
        "- **C** retrain on each accepted ADWIN alarm, on all history  \n"
        "- **D** retrain on each accepted ADWIN alarm, on a recent window "
        "(30 / 60 / 120 / 180 days; the best is marked *selected*)"
    )
    runs_path = RESULTS / "runs.csv"
    if not runs_path.exists():
        st.warning("results/runs.csv not found.")
        return
    mtime = runs_path.stat().st_mtime
    built = adaptation_tables(mtime)

    if isinstance(built, str):
        st.info(built)
        st.caption("Showing the committed tables instead.")
        show_table(T3)
        show_table(T4)
        return

    t3, t4, per_run, official, split_id, newest = built
    st.caption(
        f"Live from results/runs.csv, split_id `{split_id}`: the most recent run per "
        f"model x arm x region x seed (newest row {pd.to_datetime(newest):%Y-%m-%d %H:%M} UTC)."
    )

    c1, c2, c3 = st.columns(3)
    all_models = sorted(t3["model"].unique())
    models = c1.multiselect("Model", all_models, default=all_models)
    regions = c2.multiselect("Region", list(REGIONS), default=list(REGIONS))
    arms = c3.multiselect(
        "Arm", list(ARM_NAMES), default=list(ARM_NAMES), format_func=ARM_NAMES.get
    )

    st.subheader("Table 3: forecast accuracy by regime")
    _stretch(st.dataframe, apply_filters(t3, models, regions, arms), hide_index=True)
    st.caption(
        "MAE in MW, mean ± sd across seeds. Pinball loss reads 'pending UQ' until the "
        "uncertainty intervals are scored. Every Arm D window is listed."
    )

    st.subheader("Table 4: what each arm cost")
    _stretch(st.dataframe, apply_filters(t4, models, regions, arms), hide_index=True)
    st.caption("Negative gain = better than never retraining (arm A). Arm D is reported at 180 days.")

    with st.expander("Every seed, one row per run"):
        runs_view = per_run.rename(columns={"seed_key": "seed"})
        runs_view["seed"] = runs_view["seed"].where(runs_view["seed"] != -1, other=pd.NA)
        _stretch(st.dataframe, apply_filters(runs_view, models, regions, arms), hide_index=True)

    st.subheader("Figure 3: four-arm comparison")
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 3)", all_models)
    region = c2.selectbox("Region (Figure 3)", list(REGIONS))
    curves = f3_curves(mtime, model, region)
    if curves is not None:
        frame, retrains = curves
        st.line_chart(frame, y_label="7-day rolling MAE (MW)")
        st.caption(
            "Mean 7-day rolling MAE across seeds over the TEST period. Retrains: "
            + ", ".join(f"{k} {v}" for k, v in retrains.items())
        )
    else:
        st.caption("Curve files are not on this machine (they are gitignored); showing the saved figure.")
        show_image(
            FIGURES / "figure3" / f"f3_four_arm_comparison_{model}_{region}.png",
            f"Figure 3: {model}, {region}",
            "experiments.produce.produce_figure_f3",
        )

    runs = load_csv(str(runs_path), mtime)
    check = cross_check_table(runs)
    if check is not None:
        with st.expander("Cross-check: Ishan's Arm A/B runner vs the team runner"):
            st.caption(
                "Whole-TEST MAE (MW) of Arm A and B. Ishan's runner refits Arm B on all history; "
                "the team runner refits on the last 30 days only, and averages seeds."
            )
            _stretch(st.dataframe, check, hide_index=True)


def tab_uncertainty() -> None:
    st.header("Uncertainty and escalation")
    if not UNCERTAINTY_ITEMS:
        st.info(
            "Coming soon. Tables 5 and 6 and Figures 4 and 5 will appear here once "
            "the uncertainty and escalation steps are finished."
        )
        return
    for title, path, script in UNCERTAINTY_ITEMS:
        if path.suffix.lower() == ".csv":
            show_table((title, path, script))
        else:
            show_image(path, title, script)


def main() -> None:
    st.set_page_config(page_title="Concept drift in forecasting", layout="wide")
    st.title("Detecting and adapting to concept drift")
    st.caption("Team PG-S2-41 | AEMO electricity demand, SA1 and NSW1")
    tabs = st.tabs(["Overview", "Baselines", "Detection", "Adaptation", "Uncertainty & escalation"])
    for tab, fn in zip(tabs, (tab_overview, tab_forecasting, tab_detection, tab_adaptation, tab_uncertainty)):
        with tab:
            fn()


main()
