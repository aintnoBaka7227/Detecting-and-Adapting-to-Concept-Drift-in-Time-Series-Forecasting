"""Streamlit page for the project's tables and figures, from data to escalation.

Run from the repo root:

    python -m streamlit run app/streamlit_app.py

Read-only: it never runs a model or recomputes a metric. It shows the tables
and figures the produce_* scripts have already written to results/, and
reads results/runs.csv for the overview counts and the cross-check. Every
table and figure has a short "what this shows" note, and the Method and
settings tab lists the settings behind the results (detector parameters, the
four arms, the models, the data split, the uncertainty and escalation
methods).

Both trigger detectors are available wherever a table depends on one:
ADWIN (the control, which also runs arms A and B) and Page-Hinkley.

A missing file shows a hint about which script creates it instead of
crashing the page. Rerun the pipeline (python -m experiments.run_pipeline)
and refresh the page to see new results.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
DETECTORS = {"adwin": "ADWIN", "page_hinkley": "Page-Hinkley"}
SIGNALS = {
    "width": "Interval width",
    "nonconformity_30min": "Nonconformity, 30 minutes",
    "nonconformity_1d": "Nonconformity, 1 day",
    "nonconformity_7d": "Nonconformity, 7 days",
}
ARM_NAMES = {
    "A": "A  never retrain",
    "B": "B  every 30 days",
    "C": "C  drift alarm, full history",
    "D": "D  drift alarm, recent window",
}


# ---------------------------------------------------------------- helpers
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


def read(path: Path) -> pd.DataFrame | None:
    return load_csv(str(path), path.stat().st_mtime) if path.exists() else None


def num(value) -> float:
    """The leading number of a cell such as '659.0 ± 3.1' or '0.93 ± n/a'."""
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else float("nan")


def explain(what: str, how: str | None = None) -> None:
    with st.expander("What this shows and how to read it"):
        st.markdown(what)
        if how:
            st.markdown(how)


def show_df(df: pd.DataFrame, path: Path | None = None) -> None:
    _stretch(st.dataframe, df, hide_index=True)
    if path is not None:
        st.download_button(
            "Download CSV",
            path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=f"dl-{path.parent.name}-{path.name}",
        )


def show_table(title: str, path: Path, script: str, filters: dict | None = None, level: int = 3) -> None:
    (st.subheader if level == 3 else st.markdown)(title if level == 3 else f"**{title}**")
    df = read(path)
    if df is None:
        st.info(f"Not generated yet. Create it with `python -m {script}`.")
        return
    for column, wanted in (filters or {}).items():
        if column in df.columns and wanted:
            df = df[df[column].isin(wanted)]
    show_df(df, path)


def show_image(path: Path, caption: str, script: str) -> None:
    if path.exists():
        _stretch(st.image, str(path), caption=caption)
    else:
        st.info(f"{path.name} not generated yet. Create it with `python -m {script}`.")


def pick_detector(key: str) -> str:
    return st.radio(
        "Trigger detector",
        list(DETECTORS),
        format_func=DETECTORS.get,
        horizontal=True,
        key=f"det-{key}",
        help="The detector whose accepted alarms trigger arms C and D. Arms A and B use no detector.",
    )


def pick_version(key: str) -> bool:
    """True for the event-local version of T3 / T4 / the arm D sweep."""
    choice = st.radio(
        "Drift regime",
        ["Global regimes (official)", "Event-local"],
        horizontal=True,
        key=f"ver-{key}",
        help="Global: the drift regime is every documented event plus a 7-day buffer. "
        "Event-local: only the 7 days before, during and after an event.",
    )
    return choice == "Event-local"


# ------------------------------------------------- adaptation file paths
def t3_path(det: str, local: bool) -> Path:
    tag = "_event_local" if local else ""
    return TABLES / "table3" / det / f"table_t3_forecast_accuracy{tag}_{det}.csv"


def t4_path(det: str, local: bool) -> Path:
    tag = "_event_local" if local else ""
    return TABLES / "table4" / det / f"table_t4_adaptation_cost{tag}_{det}.csv"


def sweep_path(det: str, local: bool) -> Path:
    tag = "_event_local" if local else ""
    return TABLES / "arm_d_sweep" / det / f"table_arm_d_sweep{tag}_{det}.csv"


def t5_path(det: str) -> Path:
    return TABLES / "table5" / det / f"table_t5_coverage_and_width_{det}.csv"


def t6_path(det: str, signal: str) -> Path:
    return TABLES / "table6" / det / f"table_t6_escalation_operating_points_{det}_{signal}.csv"


T7 = TABLES / "table7" / "page_hinkley" / "table_t7_trigger_comparison_page_hinkley.csv"
F6 = FIGURES / "figure6" / "page_hinkley" / "figure_f6_trigger_timeline_page_hinkley.png"
SELECTION = TABLES / "forecaster_selection" / "forecaster_selection.csv"
WINNERS = TABLES / "fine_tune_on_synthetic_winners.csv"


# ------------------------------------------------------ headline findings
def arm_family(arm: str) -> str:
    return str(arm)[0]


def adaptation_findings(det: str) -> list[str]:
    """Plain-language findings computed from the committed Table 4 and Table 5,
    so they stay true when the tables are regenerated."""
    out: list[str] = []
    t4 = read(t4_path(det, False))
    if t4 is not None and "drift MAE gain vs arm A (MW)" in t4.columns:
        t4 = t4.assign(gain=t4["drift MAE gain vs arm A (MW)"].map(num), fam=t4["arm"].map(arm_family))
        cells = t4[["model", "region"]].drop_duplicates()
        n = len(cells)
        wins = {
            fam: int((t4[(t4["fam"] == fam)].set_index(["model", "region"])["gain"] < 0).sum())
            for fam in ("B", "C", "D")
        }
        names = {"B": "retraining every 30 days (B)", "C": "retraining on a drift alarm with full history (C)",
                 "D": "retraining on a drift alarm with a 180-day window (D)"}
        for fam in ("C", "B", "D"):
            out.append(f"{names[fam][0].upper() + names[fam][1:]} beat never retraining (A) on drift-period error "
                       f"in **{wins[fam]} of {n}** model and region combinations.")
        best = (
            t4.sort_values("gain").groupby(["model", "region"], as_index=False).first()
        )
        per_model = {}
        for model, grp in best.groupby("model"):
            per_model[model] = ", ".join(f"{r}: {ARM_NAMES[arm_family(a)].split('  ')[0]}"
                                         for r, a in zip(grp["region"], grp["arm"]))
        out.append("Best arm on drift-period error, by model: "
                   + "; ".join(f"**{m}** ({v})" for m, v in per_model.items()) + ".")
    t5 = read(t5_path(det))
    if t5 is not None and {"regime", "coverage", "UQ method"} <= set(t5.columns):
        drift = t5[t5["regime"] == "drift"].assign(cov=lambda d: d["coverage"].map(num))
        for method, grp in drift.groupby("UQ method"):
            out.append(f"During drift, **{method}** intervals covered {grp['cov'].min():.2f} to "
                       f"{grp['cov'].max():.2f} of actual demand (target 0.90).")
    return out


def selected_forecaster() -> str | None:
    sel = read(SELECTION)
    if sel is None or "pick" not in sel.columns:
        return None
    picked = sel[sel["pick"].astype(str).str.startswith("✓")]
    models = sorted(picked["model"].unique())
    if len(models) == 1 and len(picked) == sel["region"].nunique():
        return models[0]
    return None


# ------------------------------------------------------------------- tabs
def tab_overview() -> None:
    st.markdown(
        """
**Detecting and adapting to concept drift in time series forecasting**

Electricity demand changes over time (new rules, events, shocks). A forecaster
trained on old data gets worse when that happens. This project asks two
questions on two AEMO regions (SA1, NSW1) and a synthetic benchmark with known
drift: **can we detect the change, and does adapting the model help?** It then
checks how far the forecasts can be trusted, and when a person should step in.
"""
    )
    st.markdown(
        """
```
Data -> Detect -> Forecast -> Adapt -> Uncertainty -> Escalation
```
"""
    )
    st.markdown(
        "| Stage | The question it answers | Where to look |\n"
        "|---|---|---|\n"
        "| Data | Is the demand series clean? | Method and settings |\n"
        "| Detect | Which detector finds real changes without false alarms? | Detection (T1, T2, F2) |\n"
        "| Forecast | How fast do frozen forecasters degrade? Which one do we use? | Baselines (F1) |\n"
        "| Adapt | Does retraining on drift alarms beat never or scheduled retraining? | Adaptation (T3, T4, F3) |\n"
        "| Uncertainty | Do the 90% intervals keep their coverage through drift? | Uncertainty (T5, F4) |\n"
        "| Escalation | How much human review keeps coverage on target? | Escalation (T6, F5, T7, F6) |"
    )

    runs_path = RESULTS / "runs.csv"
    if runs_path.exists():
        runs = load_csv(str(runs_path), runs_path.stat().st_mtime)
        sid = runs["split_id"].astype(str) if "split_id" in runs.columns else pd.Series(dtype=str)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in runs.csv", f"{len(runs):,}")
        c2.metric("Adaptation rows", f"{int(sid.str.startswith('aemo_adapt_').sum()):,}")
        c3.metric("Uncertainty rows", f"{int(sid.str.startswith('aemo_uq_').sum()):,}")
        c4.metric("Escalation rows", f"{int(sid.str.startswith('aemo_escalation_').sum()):,}")
    else:
        st.warning("results/runs.csv not found.")

    st.subheader("Main findings so far")
    chosen = selected_forecaster()
    if chosen:
        st.markdown(f"- Forecaster selected on the Calibration window: **{chosen}**.")
    for det in DETECTORS:
        lines = adaptation_findings(det)
        if lines:
            with st.expander(f"Trigger detector: {DETECTORS[det]}", expanded=det == "adwin"):
                for line in lines:
                    st.markdown(f"- {line}")
    st.caption(
        "These lines are computed from the committed tables, so they change when the tables are "
        "regenerated. Arm D is read at its 180-day window. DHR+ARIMA is deterministic, so it has "
        "one run per region; NHITS and XGBoost have five seeds."
    )


def tab_method() -> None:
    st.header("Method and settings")
    st.caption("What was run and with which settings, so every table and figure can be read in context.")

    with st.expander("Data and splits", expanded=True):
        try:
            from drift_lab.config import REGIME_DRIFT_MARGIN, SEEDS, SPLIT

            split, margin, seeds = SPLIT, REGIME_DRIFT_MARGIN, SEEDS
        except Exception:  # noqa: BLE001 - the page still works without the package
            split = {"train": ("2018-01-01", "2019-12-31"), "calibration": ("2020-01-01", "2020-02-29"),
                     "test": ("2020-03-01", None)}
            margin, seeds = 336, (1, 2, 3, 4, 5)
        st.markdown(
            f"- **Data:** AEMO electricity demand (TOTALDEMAND), half-hourly (48 readings a day), "
            f"regions {', '.join(REGIONS)}.\n"
            f"- **Train:** {split['train'][0]} to {split['train'][1]}. Models are fitted here.\n"
            f"- **Calibration:** {split['calibration'][0]} to {split['calibration'][1]}. Used to "
            f"select the forecaster and to calibrate the uncertainty intervals.\n"
            f"- **Test:** from {split['test'][0]} to the end of the data (2023). Everything in the "
            f"results is scored here.\n"
            f"- **Seeds:** {', '.join(map(str, seeds))} for NHITS and XGBoost. DHR+ARIMA is "
            f"deterministic, so it runs once.\n"
            f"- **Regimes:** each test timestamp is labelled *pre-drift*, *drift* (a documented "
            f"event plus a {margin // 48}-day buffer) or *post-drift*."
        )

    with st.expander("Drift detectors"):
        st.markdown(
            "Three detectors were tuned on the synthetic benchmark (where the true change points "
            "are known) and then frozen. **ADWIN** and **Page-Hinkley** are the trigger detectors "
            "for the adaptation arms. KSWIN is evaluated in the detection tables only. An alarm is "
            "*accepted* if it survives a 14-day refractory filter."
        )
        winners = read(WINNERS)
        if winners is not None:
            keep = [c for c in ("detector", "detector_parameters", "mean_false_alarms_per_year",
                                "mean_detection_delay", "status") if c in winners.columns]
            show_df(winners[keep].rename(columns={
                "detector_parameters": "frozen parameters",
                "mean_false_alarms_per_year": "false alarms / year (synthetic, no drift)",
                "mean_detection_delay": "mean detection delay (observations)",
            }))

    with st.expander("Forecasting models"):
        st.markdown(
            "- **NHITS:** a neural network. 7 days of input (336 half-hours), forecasts 1 day "
            "(48 half-hours) ahead, 500 training steps.\n"
            "- **XGBoost:** gradient-boosted trees.\n"
            "- **DHR+ARIMA:** dynamic harmonic regression (Fourier terms for the daily and weekly "
            "cycles) plus ARIMA on the leftover errors.\n"
            "- **Seasonal naive:** the simple baseline. It is not a candidate for selection.\n\n"
            "**Forecaster selection** (fixed before the run): per region the lowest 7-day rolling "
            "MAE on the Calibration window wins. If another model is within one seed spread, the "
            "faster one wins. A forecaster is selected only if the same model wins in every region."
        )
        sel = read(SELECTION)
        if sel is not None:
            show_df(sel)

    with st.expander("Adaptation arms"):
        st.markdown(
            "| Arm | Policy |\n|---|---|\n"
            "| A | Never retrain (control) |\n"
            "| B | Retrain every 30 days on the last 30 days (scheduled control) |\n"
            "| C | Retrain on each accepted drift alarm, on all history |\n"
            "| D | Retrain on each accepted drift alarm, on the last 30, 60, 120 or 180 days "
            "(180 is reported in the main tables; the sweep shows all four) |\n\n"
            "Arms A and B use no detector, so they are the shared controls. ADWIN runs all four "
            "arms. Page-Hinkley runs C and D."
        )

    with st.expander("Uncertainty and escalation"):
        st.markdown(
            "- **Interval:** a 90% prediction interval on every forecast, issued one half-hour at "
            "a time.\n"
            "- **Fixed split conformal:** one interval radius, set once from the Calibration "
            "residuals and never updated.\n"
            "- **AgACI:** an adaptive method. It keeps several candidate intervals and re-weights "
            "them after every observed outcome.\n"
            "- **Escalation:** a forecast is sent for human review when a signal passes a "
            "threshold. Signals: the interval width, or a nonconformity score (the recent mean "
            "absolute error over 30 minutes, 1 day or 7 days, lagged so it only uses past errors).\n"
            "- **Operating points:** the review rate needed to reach 0.90 or 0.95 coverage, and the "
            "coverage reached when only 5%, 10% or 20% of forecasts are reviewed."
        )

    with st.expander("Metrics glossary"):
        st.markdown(
            "| Term | Meaning |\n|---|---|\n"
            "| MAE (MW) | Mean absolute error of the forecast. Lower is better. |\n"
            "| 7-day rolling MAE | MAE over a sliding 336-half-hour window, used for the curves. |\n"
            "| Drift MAE gain vs arm A | An arm's drift-period MAE minus arm A's. **Negative means "
            "better than never retraining.** |\n"
            "| Coverage | Share of actual values that fall inside the 90% interval. Target 0.90. |\n"
            "| Mean and normalised width | Interval size, in MW and as a share of mean TRAIN demand. |\n"
            "| Worst 24h coverage | The lowest coverage in any 24-hour window of that regime. |\n"
            "| Pinball loss | A score for the interval bounds. Lower is better. |\n"
            "| Review rate | Share of forecasts escalated to a person. |\n"
            "| mean ± sd | Mean and spread across seeds. 'n/a' means there is only one run. |"
        )

    st.caption("Source: README.md (pipeline and execution order), src/drift_lab/config.py and the tables in results/.")


def tab_forecasting() -> None:
    st.header("Frozen baselines (Figure 1)")
    explain(
        "Forecasters fitted once on the Train period and never updated, then run through the "
        "whole Test period. The line is the 7-day rolling MAE.",
        "A line that climbs means the model is getting worse as demand changes. This is the "
        "reason adaptation is needed. NHITS uses a more forgiving block protocol than XGBoost and "
        "DHR+ARIMA, so the lines are not an exact like-for-like comparison.",
    )
    figs = sorted((FIGURES / "figure1").glob("f1_degradation_*.png")) if FIGURES.exists() else []
    if not figs:
        st.info("No figures yet. Create them with `python -m experiments.produce.produce_figure_f1`.")
    else:
        labels = {p.stem.removeprefix("f1_degradation_"): p for p in figs}
        choice = st.selectbox("Period", list(labels), index=len(labels) - 1)
        show_image(labels[choice], f"Figure 1: {choice}", "experiments.produce.produce_figure_f1")

    st.subheader("Forecaster selection")
    explain(
        "Each model is fitted on Train and scored on the Calibration window. The ✓ marks the pick.",
        "The rule is lowest 7-day rolling MAE per region, with the faster model winning a near tie. "
        "A forecaster is selected only if the same model wins in both regions.",
    )
    sel = read(SELECTION)
    if sel is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_forecaster_selection`.")
    else:
        show_df(sel, SELECTION)


def tab_detection() -> None:
    st.header("Drift detection")

    st.subheader("Table 1: detection on the synthetic benchmark")
    explain(
        "Series with known, planted drift (sudden, gradual, recurring) and a no-drift control. "
        "Each row is a detector with its frozen parameters.",
        "Delay is how long after the real change the detector fires. False alarms are per 10,000 "
        "observations. Missed means a real change was never detected. A good detector has a short "
        "delay and no false alarms on the no-drift series.",
    )
    t1 = TABLES / "table1" / "table_t1_post_tune_synthetic_detection.csv"
    df = read(t1)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t1_post_tune`.")
    else:
        show_df(df, t1)

    kind = st.radio("Synthetic drift type (Figure 2)", ["sudden", "gradual", "recurring"], horizontal=True)
    show_image(
        FIGURES / "figure2" / f"f2_synthetic_post_tune_{kind}.png",
        f"Figure 2, synthetic: {kind}",
        "experiments.produce.produce_figure_f2_synthetic_post_tune",
    )
    explain(
        "One synthetic series with the true change point marked and each detector's alarms on it.",
        "Alarms close to the marked change are good. Alarms far from it, or many of them, are false alarms.",
    )

    st.divider()
    st.subheader("Table 2: detection on AEMO vs documented events")
    explain(
        "Real demand data, compared with a list of documented events (for example COVID-19 and "
        "market rule changes). Alarms are matched to events.",
        "The documented events are not complete ground truth, so an alarm outside every event "
        "window is labelled *unmatched*, not counted as a false alarm. Precision and recall are "
        "computed against the events only.",
    )
    t2 = TABLES / "table2" / "table_t2_aemo_events_standard_half_hourly_post_tune.csv"
    df = read(t2)
    if df is None:
        st.info("Not generated yet. Create it with "
                "`python -m experiments.produce.produce_table_t2_standard_half_hourly_post_tune`.")
    else:
        c1, c2 = st.columns(2)
        if "region" in df.columns and "detector" in df.columns:
            regions = c1.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
            dets = c2.multiselect("Detector", sorted(df["detector"].unique()), default=sorted(df["detector"].unique()))
            df = df[df["region"].isin(regions) & df["detector"].isin(dets)]
        show_df(df, t2)

    c1, c2 = st.columns(2)
    det = c1.selectbox("Detector (Figure 2, AEMO)", ["adwin", "kswin", "page_hinkley"])
    reg = c2.selectbox("Region (Figure 2, AEMO)", list(REGIONS))
    show_image(
        FIGURES / "figure2" / f"f2_detection_standard_half_hourly_post_tune_{det}_{reg}.png",
        f"Figure 2, AEMO: {det}, {reg}",
        "experiments.produce.produce_figure_f2_aemo_standard_half_hourly_post_tune",
    )


def adaptation_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)
    models = c1.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()), key=f"m-{key}")
    regions = c2.multiselect("Region", list(REGIONS), default=list(REGIONS), key=f"r-{key}")
    arms = c3.multiselect("Arm", list(ARM_NAMES), default=list(ARM_NAMES), format_func=ARM_NAMES.get, key=f"a-{key}")
    out = df[df["model"].isin(models) & df["region"].isin(regions)]
    if arms and "arm" in out.columns:
        out = out[out["arm"].map(arm_family).isin(arms)]
    return out


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
    st.header("Adaptation arms")
    st.markdown(
        "- **A** never retrain  \n"
        "- **B** retrain every 30 days, on the last 30 days  \n"
        "- **C** retrain on each accepted drift alarm, on all history  \n"
        "- **D** retrain on each accepted drift alarm, on a recent window "
        "(30 / 60 / 120 / 180 days; 180 is reported in the main tables)"
    )
    det = pick_detector("adapt")
    local = pick_version("adapt")
    st.caption(
        f"Arms C and D are triggered by **{DETECTORS[det]}** alarms. Arms A and B use no detector "
        "and are the same in both views."
    )

    st.subheader("Table 3: forecast accuracy by regime")
    explain(
        "Forecast error (MAE, in MW) of each arm, split into the pre-drift, drift and post-drift "
        "regimes, plus the pinball loss of the 90% interval over the whole test period.",
        "Compare arms within the same model and region. Lower MAE is better. The drift column "
        "matters most: it shows which arm copes best while demand is changing. 'mean ± sd' is the "
        "spread across seeds, and 'n/a' means a single run.",
    )
    df = read(t3_path(det, local))
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t3`.")
    else:
        show_df(adaptation_filters(df, "t3"), t3_path(det, local))

    st.subheader("Table 4: what each arm cost")
    explain(
        "How many retrains each arm did, how much training data and time that cost, and how much "
        "it changed the drift-period error compared with never retraining (arm A).",
        "**A negative gain is good** (better than arm A). Gain per retrain shows whether the extra "
        "retraining was worth its cost.",
    )
    df = read(t4_path(det, local))
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t4`.")
    else:
        show_df(adaptation_filters(df, "t4"), t4_path(det, local))

    st.subheader("Arm D sweep: which recent window is best?")
    explain(
        "Arm D's four window lengths side by side. The *winner* column marks the window with the "
        "lowest drift-period MAE for that model and region ('tie' if the runner-up is within one "
        "seed spread).",
        "The winner is chosen on test-period error, so treat it as a result to report, not as a "
        "tuned setting.",
    )
    df = read(sweep_path(det, local))
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_arm_d_sweep`.")
    else:
        show_df(df, sweep_path(det, local))

    st.subheader("Figure 3: four-arm comparison")
    explain(
        "The 7-day rolling MAE through the test period for arms A, B, C and the best arm-D window. "
        "The shaded band is ± 1 sd across seeds, the dashed line is arm A's pre-drift error, and "
        "the marks underneath show every retrain.",
        "A line that stays below arm A's after a drift period shows that adapting helped. Retrains "
        "that do not lower the line were wasted effort.",
    )
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 3)", list(MODELS), key="f3-model")
    region = c2.selectbox("Region (Figure 3)", list(REGIONS), key="f3-region")
    show_image(
        FIGURES / "figure3" / det / model / f"f3_four_arm_comparison_{det}_{model}_{region}.png",
        f"Figure 3: {model}, {region}, {DETECTORS[det]}",
        "experiments.produce.produce_figure_f3",
    )

    runs_path = RESULTS / "runs.csv"
    if runs_path.exists():
        check = cross_check_table(load_csv(str(runs_path), runs_path.stat().st_mtime))
        if check is not None:
            with st.expander("Cross-check: Ishan's Arm A/B runner vs the team runner"):
                st.caption(
                    "Whole-test MAE (MW) of Arm A and B from the ADWIN runs. Ishan's runner refits "
                    "Arm B on all history; the team runner refits on the last 30 days only, and "
                    "averages seeds."
                )
                _stretch(st.dataframe, check, hide_index=True)


def tab_uncertainty() -> None:
    st.header("Uncertainty")
    st.markdown(
        "Every forecast gets a 90% interval. Two methods are compared: **fixed split conformal** "
        "(one radius set once) and **AgACI** (adapts after every outcome)."
    )
    det = pick_detector("uq")

    st.subheader("Table 5: coverage and interval width by regime")
    explain(
        "For each arm and UQ method: how often the actual value fell inside the interval "
        "(coverage), how wide the interval was, and the worst 24-hour coverage in that regime.",
        "Coverage should sit near **0.90**. Below it, the intervals are too confident. Far above "
        "it, they are wider than needed. Compare coverage with width: a method that reaches 0.90 "
        "with a narrower interval is better. Pay most attention to the drift rows.",
    )
    path = t5_path(det)
    df = read(path)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t5`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        models = c1.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()), key="t5m")
        regions = c2.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()), key="t5r")
        methods = c3.multiselect("UQ method", sorted(df["UQ method"].unique()), default=sorted(df["UQ method"].unique()), key="t5u")
        regimes = c4.multiselect("Regime", ["pre-drift", "drift", "post-drift"], default=["drift"], key="t5g")
        view = df[df["model"].isin(models) & df["region"].isin(regions)
                  & df["UQ method"].isin(methods) & df["regime"].isin(regimes)]
        show_df(view, path)

    st.subheader("Figure 4: coverage through time")
    explain(
        "The 7-day rolling coverage of the 90% intervals through the test period, one row per arm, "
        "with a line for each UQ method.",
        "Where a line drops below the dashed 0.90 target the gap is filled, showing when "
        "calibration failed. The shaded bands mark the drift regime.",
    )
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 4)", list(MODELS), key="f4-model")
    region = c2.selectbox("Region (Figure 4)", list(REGIONS), key="f4-region")
    show_image(
        FIGURES / "figure4" / det / model / f"f4_coverage_trajectory_{det}_{model}_{region}.png",
        f"Figure 4: {model}, {region}, {DETECTORS[det]}",
        "experiments.produce.produce_figure_f4",
    )


def tab_escalation() -> None:
    st.header("Escalation")
    st.markdown(
        "When the intervals cannot be trusted, a person reviews the forecast. These pages show how "
        "much review it takes to keep coverage on target."
    )
    det = pick_detector("esc")
    signal = st.radio(
        "Escalation signal",
        list(SIGNALS),
        format_func=SIGNALS.get,
        horizontal=True,
        help="What decides that a forecast needs review: the interval width, or the recent error.",
    )

    st.subheader("Table 6: escalation operating points")
    explain(
        "For each arm, UQ method and regime: the smallest review rate that reaches 0.90 and 0.95 "
        "coverage on the forecasts that are kept, and the coverage reached if only 5%, 10% or 20% "
        "of forecasts are reviewed.",
        "A lower review rate is better. *Unreachable* means no threshold gets there. *Not "
        "attainable* means that review budget cannot be met (for example, a constant-width "
        "interval can only review 0% or 100% by width).",
    )
    path = t6_path(det, signal)
    df = read(path)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t6`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        models = c1.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()), key="t6m")
        regions = c2.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()), key="t6r")
        methods = c3.multiselect("UQ method", sorted(df["UQ method"].unique()), default=sorted(df["UQ method"].unique()), key="t6u")
        regimes = c4.multiselect("Regime", ["pre-drift", "drift", "post-drift"], default=["drift"], key="t6g")
        view = df[df["model"].isin(models) & df["region"].isin(regions)
                  & df["UQ method"].isin(methods) & df["regime"].isin(regimes)]
        show_df(view, path)

    st.subheader("Figure 5: escalation curve")
    explain(
        "Review rate (x axis) against the coverage of the forecasts that are kept (y axis), one "
        "panel per arm and one curve per regime.",
        "A curve that rises quickly means a little review buys a lot of coverage. The marked point "
        "is where the curve crosses 0.90. Width figures show AgACI only, because fixed split "
        "conformal has a constant width.",
    )
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 5)", list(MODELS), key="f5-model")
    region = c2.selectbox("Region (Figure 5)", list(REGIONS), key="f5-region")
    show_image(
        FIGURES / "figure5" / det / model / signal / f"f5_escalation_curve_{det}_{signal}_{model}_{region}.png",
        f"Figure 5: {model}, {region}, {DETECTORS[det]}, {SIGNALS[signal]}",
        "experiments.produce.produce_figure_f5",
    )

    st.divider()
    st.subheader("Table 7: our trigger vs Team 42's trigger")
    explain(
        "A headline comparison of Group A (this team's distribution-shift trigger, Page-Hinkley) "
        "and Group B (Team 42's error-based trigger) on the same forecaster, arm, UQ method and "
        "escalation signal.",
        "Group B reads 'awaiting Team 42' until their results are provided. The comparison is only "
        "fair once both groups share the same setup.",
    )
    df = read(T7)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.produce_table_t7`.")
    else:
        show_df(df, T7)
    show_image(F6, "Figure 6: where each trigger fires", "experiments.produce.produce_figure_f6")
    explain(
        "Where each trigger raises alarms through the test period, one row per group, with "
        "documented events shaded.",
        "Alarms that have no alarm from the other group nearby are disagreements. The Group B row "
        "stays empty until Team 42's alarms are provided.",
    )


def main() -> None:
    st.set_page_config(page_title="Concept drift in forecasting", layout="wide")
    st.title("Detecting and adapting to concept drift")
    st.caption("Team PG-S2-41 | AEMO electricity demand, SA1 and NSW1")
    tabs = st.tabs([
        "Overview", "Method and settings", "Baselines", "Detection",
        "Adaptation", "Uncertainty", "Escalation",
    ])
    for tab, fn in zip(
        tabs,
        (tab_overview, tab_method, tab_forecasting, tab_detection,
         tab_adaptation, tab_uncertainty, tab_escalation),
    ):
        with tab:
            fn()


main()
