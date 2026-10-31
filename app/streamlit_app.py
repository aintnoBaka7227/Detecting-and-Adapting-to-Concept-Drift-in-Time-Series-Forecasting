"""Streamlit page for the project's tables and figures, from data to escalation.

Run from the repo root:

    python -m streamlit run app/streamlit_app.py

Read-only: it never runs a model or recomputes a metric. It shows the tables
and figures the run/produce scripts have already written to results/. The
only thing drawn live is the synthetic benchmark preview on the Data tab,
which is instant. Every table and figure has a short "what this shows" note.

Tabs follow the pipeline, in the order the report presents it, so tables
T1-T7 and figures F1-F6 appear in sequence: Data -> Forecast -> Detection ->
Adaptation -> Uncertainty -> Escalation, then the shared comparison with
Team 42. From Table 3 on, every table and figure opens on the reported setup
(NHITS forecasts, Page-Hinkley trigger); the other forecasters and ADWIN stay
selectable. The shared comparison keeps its own setup.

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

REGIONS = ("SA1", "NSW1")
MODELS = ("nhits", "xgboost", "dhr_arima")
DETECTORS = {"adwin": "ADWIN", "page_hinkley": "Page-Hinkley"}
# From Table 3 on, every table and figure opens on the reported setup (NHITS
# forecasts, Page-Hinkley trigger); the other models and ADWIN stay selectable.
# The shared comparison (T7, F6) keeps its own setup.
DET = "page_hinkley"
DET_LABEL = DETECTORS[DET]
MODEL = "nhits"
MODEL_LABEL = "NHITS"
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


def method(text: str) -> None:
    """The stage's method and settings, at the top of its tab."""
    with st.expander("Method: how this stage works"):
        st.markdown(text)


def explain(what: str, how: str | None = None) -> None:
    with st.expander("What this shows and how to read it"):
        st.markdown(what)
        if how:
            st.markdown(how)


def _is_category(col: pd.Series) -> bool:
    """A text column with a short list of repeated values (model, region, arm,
    regime, metric...). Result cells such as '659.0 ± 3.1' are not categories."""
    if not pd.api.types.is_string_dtype(col):
        return False
    values = col.dropna().astype(str)
    if not 2 <= values.nunique() <= 30:
        return False
    return not values.str.contains("±").any() and values.str.match(r"\s*-?\d").mean() < 0.5


def filter_columns(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Let the viewer pick which columns to show and filter rows by any
    category column. Nothing selected means no filter, so the table starts
    unchanged. Used by every table, so each one is easy to filter live."""
    with st.expander("Filter and choose columns"):
        shown = st.multiselect(
            "Columns to show", list(df.columns), default=list(df.columns), key=f"cols-{key}"
        )
        cats = [c for c in df.columns if _is_category(df[c])]
        if cats:
            st.caption("Filter rows (leave empty to keep all)")
            for start in range(0, len(cats), 4):
                row = st.columns(4)
                for box, col in zip(row, cats[start : start + 4]):
                    wanted = box.multiselect(
                        col, sorted(df[col].dropna().astype(str).unique()), key=f"flt-{key}-{col}"
                    )
                    if wanted:
                        df = df[df[col].astype(str).isin(wanted)]
    return df[shown] if shown else df


def show_df(df: pd.DataFrame, path: Path | None = None) -> None:
    key = f"{path.parent.name}-{path.name}" if path is not None else "nopath-" + "|".join(map(str, df.columns))
    _stretch(st.dataframe, filter_columns(df, key), hide_index=True)
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
        index=list(DETECTORS).index(DET),
        format_func=DETECTORS.get,
        horizontal=True,
        key=f"det-{key}",
        help="The detector whose accepted alarms trigger arms C and D. Arms A and B use no detector.",
    )


def pick_models(df: pd.DataFrame, box, key: str) -> list[str]:
    """Model filter that opens on the reported forecaster only."""
    options = sorted(df["model"].unique())
    return box.multiselect("Model", options, default=[m for m in options if m == MODEL], key=key)


def pick_signal(key: str) -> str:
    """Escalation signal, chosen separately for T6 and F5."""
    return st.radio(
        "Escalation signal",
        list(SIGNALS),
        format_func=SIGNALS.get,
        horizontal=True,
        key=f"sig-{key}",
        help="What decides that a forecast needs review: the interval width, or the recent error. "
        "Width and the 30-minute nonconformity are the reported signals; 1 day and 7 days are "
        "backup, to show why the shortest look-back was kept.",
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
DATA_AUDIT = TABLES / "data" / "data_audit.csv"
DATA_SPLITS = TABLES / "data" / "data_splits.csv"
EVENTS = ROOT / "src" / "drift_lab" / "aemo" / "events.csv"
# The synthetic benchmark's four series (drift_lab/synthetic/generator.py, n = 20,000).
SYNTHETIC_KINDS = pd.DataFrame(
    [
        ("None", "No drift", "none", "False alarms: any alarm is one"),
        ("Sudden", "+8 at once at t = 5,000", "5,000", "Delay on an abrupt change"),
        ("Gradual", "Ramp from 0 to +8 over t = 5,000-6,000", "5,000-6,000 (one interval)",
         "Whether a slow drift is caught, and how late"),
        ("Recurring", "+8 from t = 6,666 to 13,333, then back", "6,666 and 13,333",
         "Whether the return to the old pattern is caught too"),
    ],
    columns=["kind", "drift", "true change points", "what it tests"],
)


# ------------------------------------------------------ headline findings
def arm_family(arm: str) -> str:
    return str(arm)[0]


def adaptation_findings() -> list[str]:
    """Plain-language findings for NHITS with the Page-Hinkley trigger, computed
    from the committed Table 4 and Table 5 so they stay true when the tables
    are regenerated."""
    out: list[str] = []
    t4 = read(t4_path(DET, False))
    if t4 is not None and "drift MAE gain vs arm A (MW)" in t4.columns:
        t4 = t4[t4["model"] == MODEL].assign(gain=t4["drift MAE gain vs arm A (MW)"].map(num))
        for region, grp in t4.groupby("region", sort=False):
            gains = ", ".join(
                f"{arm_family(arm)} {gain:+.0f} MW"
                for arm, gain in zip(grp["arm"], grp["gain"])
                if arm_family(arm) != "A"
            )
            out.append(f"{region}, drift-period MAE change vs never retraining (A): {gains} "
                       "(negative is better).")
    t5 = read(t5_path(DET))
    if t5 is not None and {"regime", "coverage", "UQ method"} <= set(t5.columns):
        drift = t5[(t5["model"] == MODEL) & (t5["regime"] == "drift")]
        drift = drift.assign(cov=drift["coverage"].map(num))
        for method, grp in drift.groupby("UQ method"):
            out.append(f"During drift, **{method}** intervals covered {grp['cov'].min():.2f} to "
                       f"{grp['cov'].max():.2f} of actual demand across the four arms and both "
                       "regions (target 0.90).")
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
Data -> Forecast -> Detect -> Adapt -> Uncertainty -> Escalation -> Comparison
```
"""
    )
    st.markdown(
        "| Stage | The question it answers | Tab |\n"
        "|---|---|---|\n"
        "| Data | How is the demand series audited, cleaned, split and deseasonalised? How is the synthetic benchmark built? | Data |\n"
        "| Forecast | How fast do frozen forecasters degrade? Which one do we use? | Forecast (F1) |\n"
        "| Detect | Which detector finds real changes without false alarms? | Detection (T1, F2, T2) |\n"
        "| Adapt | Does retraining on drift alarms beat never or scheduled retraining? | Adaptation (T3, T4, F3) |\n"
        "| Uncertainty | Do the 90% intervals keep their coverage through drift? | Uncertainty (T5, F4) |\n"
        "| Escalation | How much human review keeps coverage on target? | Escalation (T6, F5) |\n"
        "| Comparison | Our distribution-shift trigger vs Team 42's error-based trigger | Comparison (T7, F6) |"
    )

    st.subheader("Main findings so far")
    chosen = selected_forecaster()
    if chosen:
        st.markdown(f"- Forecaster selected on the Calibration window: **{chosen}**.")
    for line in adaptation_findings():
        st.markdown(f"- {line}")
    st.caption(
        f"{MODEL_LABEL} forecasts with the {DET_LABEL} trigger, mean of five seeds. These lines are "
        "computed from the committed tables, so they change when the tables are regenerated. Arm D "
        "is read at its 180-day window."
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


def synthetic_preview():
    """The four synthetic series (seed 1) with their true change points marked,
    drawn live from the generator."""
    import matplotlib.pyplot as plt

    from drift_lab.synthetic.generator import make_series

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, kind in zip(axes, ("none", "sudden", "gradual", "recurring")):
        y, cps = make_series(kind, seed=1)
        ax.plot(y, color="#0D2C54", linewidth=0.3)
        if kind == "gradual":
            ax.axvspan(cps[0], cps[1], color="#E0A800", alpha=0.3)
        for cp in cps:
            ax.axvline(cp, color="#C1666B", linestyle="--", linewidth=1.2)
        ax.set_title(kind, loc="left", fontsize=10, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("step (half-hour)")
    fig.tight_layout()
    return fig


def tab_data() -> None:
    st.header("Data")
    st.caption("Two datasets: real AEMO demand, and a synthetic benchmark where the true change points are known.")

    st.subheader("AEMO demand data")
    st.markdown(
        "Monthly AEMO CSVs for **SA1** and **NSW1**, January 2018 to December 2023, with five "
        "columns: `REGION`, `SETTLEMENTDATE`, `TOTALDEMAND` (MW), `RRP` (price, $/MWh) and "
        "`PERIODTYPE`.\n\n"
        "**Scope:** price is loaded, audited and kept, but the project studies **total demand "
        "only**. No detector, forecaster or interval method reads price."
    )
    st.markdown(
        """
```
Read -> Audit and clean -> 30-minute grid -> Split -> Deseasonalise
```
"""
    )

    st.markdown("**1. Read, audit and clean** (`src/drift_lab/aemo/loader.py`)")
    explain(
        "Every check loader.py runs, in its order, with the action taken and what it found in "
        "each region.",
        "Cleaning is **non-destructive**: it only drops rows that cannot be placed in time "
        "(invalid timestamps, duplicates) and never alters or clips a demand or price value. "
        "Extremes such as SA1's negative demand (rooftop solar) and price spikes are real market "
        "behaviour, so they are kept.",
    )
    audit = read(DATA_AUDIT)
    if audit is None:
        st.info("Not generated yet. Create it with `python -m experiments.run.detection.check_data_audit`.")
    else:
        show_df(audit, DATA_AUDIT)

    st.markdown("**2. One 30-minute grid**")
    st.markdown(
        "AEMO moved from 30-minute to 5-minute settlement on **1 October 2021**. Every interval is "
        "averaged onto one regular 30-minute grid, labelled by interval end, so the format change "
        "is not mistaken for drift. A gap would become an empty (NaN) half-hour, never filled in."
    )

    st.markdown("**3. Split by time**")
    explain(
        "The frozen Train / Calibration / Test split and how many half-hours each holds.",
        "Everything is fitted on Train only. Test starts in March 2020, so every Tier 1 event and "
        "all but one Tier 2 event fall in Test; the exception, EVT-2020-01 (NSW bushfire demand, "
        "4 Jan 2020), is in Calibration, where no alarm is scored.",
    )
    splits = read(DATA_SPLITS)
    if splits is None:
        st.info("Not generated yet. Create it with `python -m experiments.run.detection.check_data_audit`.")
    else:
        show_df(splits, DATA_SPLITS)

    st.markdown("**4. Deseasonalise** (`src/drift_lab/aemo/deseasonalise.py`): the detector's input")
    st.markdown(
        "Demand swings by season, weekday and time of day; a detector fed raw demand would alarm "
        "every summer and every Monday. So the expected seasonal shape is removed first:\n\n"
        "1. **Fit on Train only:** expected demand = base level × day-of-year factor × "
        "(weekday, half-hour) factor. The base level is mean Train demand; the day-of-year factor "
        "is smoothed with a 31-day rolling mean that wraps December into January (29 February is "
        "interpolated); the weekday × half-hour factor has 7 × 48 = 336 slots.\n"
        "2. **Apply it frozen** to Train, Calibration and Test: residual = actual − expected. Later "
        "data never changes the profile.\n"
        "3. **Standardise** the residual with Train's mean and standard deviation (z-score units).\n"
        "4. **Detect** on one continuous stream: Train + Calibration warm the detector up, and only "
        "Test alarms are scored.\n\n"
        "What is left is what seasonality does not explain; a lasting shift in it is drift. The "
        "residual feeds **detection only**. The forecasters use the 30-minute demand directly."
    )

    st.divider()
    st.subheader("Synthetic benchmark")
    st.markdown(
        "AEMO has no exact ground truth for when drift starts. Synthetic series have known change "
        "points, so detectors are tuned and scored exactly (false alarms, missed drifts, delay) "
        "before they touch real data. Built in NumPy (`src/drift_lab/synthetic/generator.py`); "
        "river provides the detectors that run on it.\n\n"
        "**y(t) = 20 + 5·sin(2πt / 48) + drift(t) + noise**, noise ~ N(0, 1)\n\n"
        "- A daily cycle of 48 steps, like half-hourly demand.\n"
        "- 20,000 steps (about 417 days), seeds 1-5.\n"
        "- Drift is a level shift of **+8**, eight times the noise standard deviation."
    )
    show_df(SYNTHETIC_KINDS)
    try:
        _stretch(st.pyplot, synthetic_preview())
        st.caption("Seed 1. Dashed lines are the true change points; the shaded band is the gradual ramp.")
    except Exception as exc:  # noqa: BLE001 - the page still works without the package
        st.info(f"Synthetic preview unavailable: {exc}")
    st.caption(
        "Limits: one daily cycle, Gaussian noise and level shifts only. No weekly or annual "
        "seasonality and no change in variance, which is why detectors are also validated on AEMO."
    )


def tab_forecasting() -> None:
    st.header("Forecast")
    method(
        "**Models** (all forecast half-hourly demand in MW, from the 30-minute grid):\n"
        "- **NHITS:** a neural network. 7 days of input (336 half-hours), forecasts 1 day "
        "(48 half-hours) ahead, 500 training steps, seeds 1-5.\n"
        "- **XGBoost:** gradient-boosted trees, seeds 1-5.\n"
        "- **DHR+ARIMA:** dynamic harmonic regression (Fourier terms for the daily and weekly "
        "cycles) plus ARIMA on the leftover errors. Deterministic, so one run.\n"
        "- **Seasonal naive:** the simple baseline. It is not a candidate for selection.\n\n"
        "**Frozen baselines (F1):** each model is fitted once on Train and never updated, then "
        "run through the whole Test period. The 7-day rolling MAE shows how fast it degrades.\n\n"
        "**Forecaster selection** (rule fixed before the run): each model is fitted on Train and "
        "scored on the Calibration window. Per region the lowest 7-day rolling MAE wins; if another "
        "model is within one seed spread, the faster one wins. A forecaster is selected only if the "
        "same model wins in every region. The selected forecaster is the one carried through "
        "Adaptation, Uncertainty and Escalation."
    )
    st.subheader("Documented events")
    explain(
        "The shared event catalogue both teams use: 33 events from January 2020 to December 2023. "
        "*Date type* is a single date or a date range. Every event also has a region, a public "
        "source (mostly AEMO Quarterly Energy Dynamics, plus AEMO market notices and AER reports) "
        "and the reason for its tier; those are in the downloaded CSV.",
        "**Tier 1** events are well documented with a clear, NEM-wide effect on demand (COVID-19, "
        "the December 2020 solar shift, the June 2022 market suspension, the late-2023 "
        "system-security directions). **Tier 2** events are real but smaller or more local, mostly "
        "minimum-demand records. NEM events count for both regions. These are reference events, "
        "not complete ground truth; F1 marks them, and T2 matches alarms against them.",
    )
    events = read(EVENTS)
    if events is None:
        st.info(f"{EVENTS.name} not found.")
    else:
        events = events.assign(tier="Tier " + events["tier"].astype(str))[
            ["event_name", "tier", "date_precision", "start_date", "end_date"]
        ].rename(columns={
            "event_name": "event", "date_precision": "date type",
            "start_date": "start", "end_date": "end",
        })
        show_df(events, EVENTS)

    st.subheader("Figure 1: frozen baselines")
    explain(
        "Forecasters fitted once on the Train period and never updated, then run through the "
        "whole Test period. The line is the 7-day rolling MAE.",
        "A line that climbs means the model is getting worse as demand changes. This is the "
        "reason adaptation is needed. NHITS uses a more forgiving block protocol than XGBoost and "
        "DHR+ARIMA, so the lines are not an exact like-for-like comparison.",
    )
    figs = sorted((FIGURES / "figure1").glob("f1_degradation_*.png")) if FIGURES.exists() else []
    if not figs:
        st.info("No figures yet. Create them with `python -m experiments.produce.forecasting.produce_figure_f1`.")
    else:
        labels = {p.stem.removeprefix("f1_degradation_"): p for p in figs}
        default = list(labels).index("2020_2023") if "2020_2023" in labels else len(labels) - 1
        choice = st.selectbox("Period (per-year figures are the backup)", list(labels), index=default)
        show_image(labels[choice], f"Figure 1: {choice}", "experiments.produce.forecasting.produce_figure_f1")

    st.subheader("Forecaster selection")
    explain(
        "Each model is fitted on Train and scored on the Calibration window. The ✓ marks the pick.",
        "The rule is lowest 7-day rolling MAE per region, with the faster model winning a near tie. "
        "A forecaster is selected only if the same model wins in both regions.",
    )
    sel = read(SELECTION)
    if sel is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.forecasting.produce_table_forecaster_selection`.")
    else:
        show_df(sel, SELECTION)


def tab_detection() -> None:
    st.header("Drift detection")
    method(
        "**Detectors:** ADWIN, KSWIN and Page-Hinkley, from the river library.\n\n"
        "**Tuning (synthetic only):** a grid of settings per detector is run on the synthetic "
        "benchmark, seeds 1-5. A setting is *eligible* only if, on every seed, it raises no false "
        "alarm on the no-drift series, stays at or below 2 false alarms per year on every series, "
        "and misses no drift. The winner has the fewest false alarms per year, then the shortest "
        "mean delay. Winners are frozen and used unchanged everywhere after.\n\n"
        "**On AEMO:** each detector reads the deseasonalised, standardised demand residual (Data "
        "tab) as one continuous stream. Train and Calibration warm it up; only Test alarms are "
        "scored. An alarm is *accepted* if it comes at least 14 days after the previous accepted "
        "one (refractory period); accepted alarms are what trigger retraining.\n\n"
        "**Matching to documented events (T2):** an event's window runs from its start to its end "
        "plus 7 days. An accepted alarm inside a window matches that event; alarms before an event "
        "starts never match it; one alarm matches at most one event, and each event is credited "
        "once. Events are not complete ground truth, so an alarm outside every window is "
        "*unmatched*, not a false alarm. Tier 1 and Tier 2 events are scored separately, and a "
        "chance baseline (the matches expected if the same number of alarms fell at random) shows "
        "what matching by luck looks like."
    )

    st.subheader("Detector decision: tuning on synthetic data")
    explain(
        "Each detector's parameters are tuned on the synthetic benchmark, where the true change "
        "points are known. The winners are frozen and used for every later result.",
        "Winners is the short list that is carried forward. The full tuning table and the "
        "calibration summary show everything that was tried.",
    )
    for label, name, module in (
        ("Winners (frozen settings)", "fine_tune_on_synthetic_winners.csv", "run.detection.run_fine_tune_on_synthetic"),
        ("Full tuning table", "fine_tune_on_synthetic.csv", "run.detection.run_fine_tune_on_synthetic"),
        ("Calibration summary", "fine_tune_calibration_summary.csv", "produce.produce_fine_tune_calibration_summary"),
    ):
        with st.expander(label, expanded=name.endswith("winners.csv")):
            tuning = read(TABLES / name)
            if tuning is None:
                st.info(f"{name} not generated yet. Create it with `python -m experiments.{module}`.")
            elif name == "fine_tune_calibration_summary.csv":
                # One tab per detector, without the other detectors' (empty)
                # parameter columns.
                st.caption(
                    "A blank cell means that value was not reached: the sweep stops a config at "
                    "its first failure, and the all-row averages are given for eligible configs only."
                )
                tuning = tuning.assign(fail_reason=tuning["fail_reason"].fillna(""))
                detectors = list(dict.fromkeys(tuning["detector"]))
                for tab, detector in zip(st.tabs(detectors), detectors):
                    with tab:
                        show_df(tuning[tuning["detector"] == detector].dropna(axis=1, how="all"))
                st.download_button(
                    "Download CSV", (TABLES / name).read_bytes(), file_name=name, mime="text/csv",
                    key=f"dl-{name}",
                )
            else:
                show_df(tuning, TABLES / name)

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
        st.info("Not generated yet. Create it with `python -m experiments.produce.detection.produce_table_t1_post_tune`.")
    else:
        show_df(df, t1)

    kind = st.radio("Synthetic drift type (Figure 2)", ["sudden", "gradual", "recurring"], horizontal=True)
    show_image(
        FIGURES / "figure2" / f"f2_synthetic_post_tune_{kind}.png",
        f"Figure 2, synthetic: {kind}",
        "experiments.produce.detection.produce_figure_f2_synthetic_post_tune",
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
                "`python -m experiments.produce.detection.produce_table_t2_standard_half_hourly_post_tune`.")
    else:
        c1, c2 = st.columns(2)
        if "region" in df.columns and "detector" in df.columns:
            regions = c1.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
            dets = c2.multiselect("Detector", sorted(df["detector"].unique()), default=sorted(df["detector"].unique()))
            df = df[df["region"].isin(regions) & df["detector"].isin(dets)]
        show_df(df, t2)

    c1, c2 = st.columns(2)
    det = c1.selectbox("Detector (Figure 2, AEMO)", ["adwin", "kswin", "page_hinkley"], index=2)
    reg = c2.selectbox("Region (Figure 2, AEMO)", list(REGIONS))
    show_image(
        FIGURES / "figure2" / f"f2_detection_standard_half_hourly_post_tune_{det}_{reg}.png",
        f"Figure 2, AEMO: {det}, {reg}",
        "experiments.produce.detection.produce_figure_f2_aemo_standard_half_hourly_post_tune",
    )


def adaptation_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)
    models = pick_models(df, c1, f"m-{key}")
    regions = c2.multiselect("Region", list(REGIONS), default=list(REGIONS), key=f"r-{key}")
    arms = c3.multiselect("Arm", list(ARM_NAMES), default=list(ARM_NAMES), format_func=ARM_NAMES.get, key=f"a-{key}")
    out = df[df["model"].isin(models) & df["region"].isin(regions)]
    if arms and "arm" in out.columns:
        out = out[out["arm"].map(arm_family).isin(arms)]
    return out


def tab_adaptation() -> None:
    st.header("Adaptation arms")
    method(
        "**Four arms:**\n"
        "- **A** never retrain (control)\n"
        "- **B** retrain every 30 days, on the last 30 days (scheduled control)\n"
        "- **C** retrain on each accepted drift alarm, on all history\n"
        "- **D** retrain on each accepted drift alarm, on a recent window only. The window is "
        "swept over 30, 60, 120 and 180 days; 180 is reported in the main tables.\n\n"
        "Arms A and B use no detector, so they are shared by both detectors. Arms C and D retrain "
        "on the detector's accepted alarms (Detection tab).\n\n"
        "**Retraining boundary:** for an accepted alarm on day *d*, the old model forecasts "
        "through the end of day *d*, retraining uses data up to day *d*, and the new model "
        "forecasts from day *d* + 1.\n\n"
        "**Regimes:** each Test timestamp is labelled pre-drift, drift or post-drift.\n"
        "- **Global (primary):** drift is every documented event plus a 7-day buffer after it; "
        "pre-drift is before the first event; post-drift runs from the end of one drift window "
        "to the next event.\n"
        "- **Event-local (extra comparison):** only the 7 days before an event, the event itself "
        "and the 7 days after it; everything else is left out.\n\n"
        "**Recorded per arm:** MAE by regime, number of retrains, training samples, wall-clock "
        "time, drift MAE gain vs arm A, and gain per retrain."
    )
    det = pick_detector("adapt")
    local = pick_version("adapt")
    st.caption(
        f"Opens on **{MODEL_LABEL}** forecasts. Arms C and D are triggered by **{DETECTORS[det]}** "
        "alarms. Arms A and B use no detector "
        "and are the same in both views."
    )

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
        st.info("Not generated yet. Create it with `python -m experiments.produce.adaptation.produce_table_arm_d_sweep`.")
    else:
        models = pick_models(df, st, "m-sweep")
        show_df(df[df["model"].isin(models)], sweep_path(det, local))

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
        st.info("Not generated yet. Create it with `python -m experiments.produce.adaptation.produce_table_t3`.")
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
        st.info("Not generated yet. Create it with `python -m experiments.produce.adaptation.produce_table_t4`.")
    else:
        show_df(adaptation_filters(df, "t4"), t4_path(det, local))

    st.subheader("Figure 3: four-arm comparison")
    explain(
        "The 7-day rolling MAE through the test period for arms A, B, C and the best arm-D window. "
        "The shaded band is ± 1 sd across seeds, the dashed line is arm A's pre-drift error, and "
        "the marks underneath show every retrain.",
        "A line that stays below arm A's after a drift period shows that adapting helped. Retrains "
        "that do not lower the line were wasted effort.",
    )
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 3)", list(MODELS), index=MODELS.index(MODEL), key="f3-model")
    region = c2.selectbox("Region (Figure 3)", list(REGIONS), key="f3-region")
    show_image(
        FIGURES / "figure3" / det / model / f"f3_four_arm_comparison_{det}_{model}_{region}.png",
        f"Figure 3: {model}, {region}, {DETECTORS[det]}",
        "experiments.produce.adaptation.produce_figure_f3",
    )


def tab_uncertainty() -> None:
    st.header("Uncertainty")
    st.markdown(
        "Every forecast gets a 90% interval. Two methods are compared: **fixed split conformal** "
        "(one radius set once) and **AgACI** (adapts after every outcome)."
    )
    method(
        "**Target:** a 90% prediction interval around every half-hourly forecast, issued one "
        "half-hour at a time, replayed over each adaptation arm's saved forecasts.\n\n"
        "**Calibration:** both methods start from the Calibration window's absolute errors "
        "|actual − forecast|.\n\n"
        "- **Fixed split conformal:** the radius is the conformal quantile of the Calibration "
        "errors, set once and never updated on Test. Every interval is forecast ± radius. Its "
        "coverage guarantee assumes the errors do not drift, so it is the non-adaptive baseline.\n"
        "- **AgACI:** several candidate intervals, each adjusting its own level after every miss "
        "or hit (adaptive conformal inference) from a rolling 7-day buffer of recent errors. Their "
        "bounds are combined by online aggregation, re-weighted after every outcome, so the "
        "interval can widen, narrow and be asymmetric. One instance per forecaster × region × "
        "arm × seed; it is not reset when the forecaster is retrained.\n\n"
        "**Metrics, by regime (T5):** coverage (target 0.90), mean and normalised width, and worst "
        "24-hour coverage. The interval's pinball loss is reported with forecast accuracy in T3."
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
        st.info("Not generated yet. Create it with `python -m experiments.produce.uncertainty.produce_table_t5`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        models = pick_models(df, c1, "t5m")
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
    model = c1.selectbox("Model (Figure 4)", list(MODELS), index=MODELS.index(MODEL), key="f4-model")
    region = c2.selectbox("Region (Figure 4)", list(REGIONS), key="f4-region")
    show_image(
        FIGURES / "figure4" / det / model / f"f4_coverage_trajectory_{det}_{model}_{region}.png",
        f"Figure 4: {model}, {region}, {DETECTORS[det]}",
        "experiments.produce.uncertainty.produce_figure_f4",
    )


def tab_escalation() -> None:
    st.header("Escalation")
    st.markdown(
        "When the intervals cannot be trusted, a person reviews the forecast. These pages show how "
        "much review it takes to keep coverage on target."
    )
    method(
        "**Rule:** a forecast is sent for human review when its signal is above a threshold τ. "
        "Reviewed forecasts are taken out; coverage is then measured on the forecasts that are "
        "kept, against the intervals as issued.\n\n"
        "**Signals:**\n"
        "- **Interval width:** upper − lower of the 90% interval.\n"
        "- **Nonconformity:** the mean |actual − forecast| over the previous 30 minutes (reported), "
        "1 day or 7 days (backup). Lagged, so a forecast only uses errors already observed.\n\n"
        "**Threshold sweep:** every distinct signal value is tried as τ, giving the escalation "
        "curve (F5): review rate against kept coverage. Under width, fixed split conformal has "
        "one constant width, so only 0% or 100% review is possible.\n\n"
        "**Operating points (T6):** the smallest review rate that keeps 0.90 or 0.95 coverage "
        "(*unreachable* if none does), and the coverage reached when 5%, 10% or 20% of forecasts "
        "are reviewed."
    )
    det = pick_detector("esc")
    st.caption(
        "Each operating point is found by sweeping the threshold on the same test data it is scored "
        "on, so it is the best trade-off available in hindsight, not a threshold fixed in advance."
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
    signal = pick_signal("t6")
    path = t6_path(det, signal)
    df = read(path)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.escalation.produce_table_t6`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        models = pick_models(df, c1, "t6m")
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
    signal = pick_signal("f5")
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model (Figure 5)", list(MODELS), index=MODELS.index(MODEL), key="f5-model")
    region = c2.selectbox("Region (Figure 5)", list(REGIONS), key="f5-region")
    show_image(
        FIGURES / "figure5" / det / model / signal / f"f5_escalation_curve_{det}_{signal}_{model}_{region}.png",
        f"Figure 5: {model}, {region}, {DETECTORS[det]}, {SIGNALS[signal]}",
        "experiments.produce.escalation.produce_figure_f5",
    )


def tab_comparison() -> None:
    st.header("Comparison with Team 42")
    st.markdown(
        "A shared experiment, not one of this team's own findings: Group A is this team's "
        "distribution-shift trigger (Page-Hinkley on the deseasonalised demand residual), Group B "
        "is Team 42's error-based trigger (Page-Hinkley on forecast error). Both use arm D (180 "
        "days), AgACI and the same regions, seeds and regimes. The forecasters differ: XGBoost "
        "for Group A, HistGradientBoostingRegressor for Group B."
    )

    st.subheader("Table 7: our trigger vs Team 42's trigger")
    t7_settings = T7.with_name("table_t7_trigger_comparison_page_hinkley_settings.csv")
    settings = read(t7_settings)
    if settings is not None:
        st.markdown("**Settings behind this comparison**")
        show_df(settings, t7_settings)
    explain(
        "Drift-period MAE, worst 24-hour coverage, review rates for both escalation signals, and "
        "how many alarms and retrains each trigger produced, per region.",
        "Read the settings first: the comparison is only fair where both groups share the setup, "
        "and the last column says where they do not.",
    )
    df = read(T7)
    if df is None:
        st.info("Not generated yet. Create it with `python -m experiments.produce.comparison.produce_table_t7`.")
    else:
        show_df(df, T7)

    st.subheader("Figure 6: where each trigger fires")
    explain(
        "Where each trigger raises alarms through the test period, one row per group, with "
        "Tier 1 documented events shaded.",
        "An alarm with no alarm from the other group within ±7 days is a disagreement.",
    )
    show_image(F6, "Figure 6: where each trigger fires", "experiments.produce.comparison.produce_figure_f6")
    alarms = read(FIGURES / "figure6" / "group_b_team42_alarms.csv")
    if alarms is not None:
        with st.expander("Team 42's accepted alarms"):
            show_df(alarms, FIGURES / "figure6" / "group_b_team42_alarms.csv")


def main() -> None:
    st.set_page_config(page_title="Concept drift in forecasting", layout="wide")
    st.title("Detecting and adapting to concept drift")
    st.caption("Team PG-S2-41 | AEMO electricity demand, SA1 and NSW1")
    tabs = st.tabs([
        "Overview", "Data", "Forecast", "Detection",
        "Adaptation", "Uncertainty", "Escalation", "Comparison",
    ])
    for tab, fn in zip(
        tabs,
        (tab_overview, tab_data, tab_forecasting, tab_detection,
         tab_adaptation, tab_uncertainty, tab_escalation, tab_comparison),
    ):
        with tab:
            fn()


main()
