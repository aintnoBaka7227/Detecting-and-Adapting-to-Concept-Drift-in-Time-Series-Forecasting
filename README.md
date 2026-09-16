# Detecting and Adapting to Concept Drift in Time Series Forecasting

This project builds an automated pipeline that detects concept drift in a
time series and adapts the forecasting model when drift occurs. It also
reports prediction confidence and flags results that may need human review.

The pipeline, conceptually:

```
Data -> Detect -> Adapt -> Quantify (+ Escalate)
```

Applied to two AEMO electricity-demand regions (**SA1**, **NSW1**) and a
synthetic benchmark with known ground-truth drift.

---

## Setup

```bash
pip install -e ".[dev]"
```

This installs the `drift_lab` package from `src/` in editable mode. You can
then import it from notebooks, tests or scripts without changing Python paths.
It also installs the `pytest` and `ruff` development tools.

The AEMO raw CSVs are already committed under `data/raw/{SA1,NSW1}/` —
there's no download step. `data/processed/`, `data/synthetic/` and
`results/runs/` are write targets for cached/generated output and are
gitignored (see [`.gitignore`](.gitignore)); `results/runs.csv` itself
*is* tracked, since it's the shared experiment ledger.

---

## Quickstart

```bash
# Fit the three frozen baselines on both AEMO regions, log to runs.csv
python -m experiments.run.forecasting.run_aemo_baselines

# Run the three detectors against the synthetic benchmark (5 seeds x 4 drift types)
python -m experiments.run.detection.run_synthetic_detectors

# Turn runs.csv into the deliverables
python -m experiments.produce.produce_table_t1     # results/tables/table_t1_synthetic_detection.csv
python -m experiments.produce.produce_figure_f1    # results/figures/f1_degradation_*.png

# Run the test suite
pytest
```

Everything under `results/figures/`, `results/tables/` and `results/runs/` is
generated output. It can be recreated from `results/runs.csv`. The ledger can
also be recreated by running the `run_*.py` scripts again.

---

## Repository layout

This tree is a direct listing of what's actually in the repo right now —
including a couple of pre-refactor leftovers, called out explicitly below
rather than quietly omitted, so nobody mistakes "not mentioned" for "not
there."

```
.
├── src/drift_lab/                  # the installable package — implementation only, nothing runs here
│   ├── config.py                    # paths, REGIONS, SEEDS, train/calibration/test SPLIT, shared constants
│   │
│   ├── aemo/                        # AEMO demand data
│   │   ├── loader.py                  load(region) -> (train, calibration, test); load_processed(region)
│   │   ├── deseasonalise.py           aggregate_daily_demand(series) / remove_daily_weekly_profile(series, reference) -- for detectors
│   │   └── events.csv                 documented AEMO/AER events, tiered (see docs/event_tiering_criteria.md)
│   │
│   ├── synthetic/                   # synthetic drift benchmark
│   │   └── generator.py               make_series(kind, n, noise, seed) -> (y, changepoints)
│   │
│   ├── forecasting/                 # Forecaster implementations (fit / predict)
│   │   ├── base.py                    the frozen contract
│   │   ├── seasonal_naive.py          lag-48 (one day ago) baseline
│   │   ├── xgboost_forecaster.py       recursive XGBoost on lag + calendar features
│   │   ├── dhr_arima.py                dynamic harmonic regression + ARIMA errors
│   │   ├── nhits_forecaster.py         NHITS (Nixtla neuralforecast) direct multi-step neural model
│   │   └── nixtla_common.py            shared Forecaster <-> neuralforecast adapter (NHITS today, PatchTST-shaped later)
│   │
│   ├── detection/                   # DriftDetector implementations (detect(stream) -> indices)
│   │   ├── base.py                    the frozen contract + detect_with_river() shared helper
│   │   └── adwin.py, kswin.py, page_hinkley.py    river-backed detectors
│   │
│   ├── adaptation/                  # Adapter implementations (adapt(changepoints, model, data) -> model)
│   │   ├── base.py                    the frozen contract
│   │   └── retrain_using_3_month_windows.py   retrain on drift, trailing 3 calendar months
│   │
│   ├── uncertainty/                 # UncertaintyQuantifier contract — scaffold only
│   │   └── base.py
│   │
│   └── evaluation/                  # the ONE place every metric is computed
│       ├── evaluation.py              calculate_mae, calculate_rolling_mae, evaluate_detections, match_unmatch, assign_regime, ...
│       └── __init__.py                 re-exports the public functions
│
├── experiments/                     # the only place that runs anything or writes to results/
│   ├── run_harness.py                 record_run() — turns one run's arrays into runs.csv rows
│   ├── results_io.py                   runs.csv / config.json / curve-dump filesystem plumbing
│   │                                    (both shared by everything below — not experiment scripts themselves)
│   │
│   ├── run/                          split further by what the script exercises:
│   │   ├── forecasting/               a Forecaster is the thing under test (record_run(forecast=...))
│   │   │   ├── run_aemo_baselines.py     fits seasonal_naive + xgboost + dhr_arima on AEMO
│   │   │   ├── run_aemo_nhits.py         NHITS baseline on AEMO
│   │   │   ├── run_aemo_nhits_retrain3mo_kswin_nsw.py   NHITS + kswin + RetrainUsing3MonthWindows, NSW1
│   │   │   └── run_aemo_nhits_retrain3mo_kswin_sa.py    same, SA1
│   │   └── detection/                 a DriftDetector is the thing under test (record_run(detection=...))
│   │       ├── run_aemo_detectors.py            adwin/kswin/page_hinkley on raw half-hourly AEMO demand -- feeds Figure F2
│   │       ├── run_aemo_detectors_daily_frozen.py  same three, daily-aggregated + frozen configs, both-tier matching -- feeds Table T2
│   │       ├── run_detector_budget_tuning.py    synthetic-only sweep that picks the frozen configs above
│   │       ├── run_detector_input_experiments.py   compares raw / daily-mean / deseasonalised AEMO demand as detector input
│   │       ├── run_synthetic_detectors.py       runs adwin/kswin/page_hinkley on the synthetic benchmark (Step 4 / T1 deliverable)
│   │       ├── run_synthetic_detector_experiments.py  the canonical all-three-detectors synthetic sweep (split_id="synth_n20000")
│   │       ├── run_synthetic_adwin.py, run_synthetic_kswin.py   single-detector synthetic runs, one script per detector
│   │       └── run_synthetic_generator.py       plots the synthetic benchmark itself (no runs.csv row)
│   │
│   └── produce/                      one file per table/figure, run with `python -m experiments.produce.<name>`
│       ├── produce_table_t1.py         synthetic detection table, grouped from runs.csv
│       ├── produce_table_t2.py         AEMO detector-vs-documented-event table, grouped from runs.csv
│       ├── produce_figure_f1.py        AEMO rolling-MAE degradation figures, grouped from runs.csv
│       ├── produce_figure_f2_aemo.py           detection figure per (detector, region), raw half-hourly run
│       ├── produce_figure_f2_aemo_daily_frozen.py   same figure, daily-aggregated + frozen-config run
│       ├── produce_adwin_results.py    ⚠ pre-refactor: synthetic-only ADWIN summary, not part of the T1/T2/F1/F2 pipeline above
│       └── produce_kswin_results.py    ⚠ pre-refactor: synthetic-only KSWIN summary, same caveat
│
├── results/
│   ├── runs.csv                       the experiment ledger — tracked in git
│   ├── runs/<config_hash>/             config.json + per-run curve dumps — gitignored
│   ├── tables/                        generated tables (produce_table_*.py) — gitignored
│   └── figures/                       generated figures (produce_figure_*.py) — gitignored
│
├── tests/                           # pytest, one file per package/module it covers (14 files)
├── data/
│   ├── raw/{SA1,NSW1}/                 committed source CSVs
│   ├── processed/, synthetic/          generated caches — gitignored
│   └── baseline/                       committed reference CSVs + figures that
│   │                                    tests/test_xgboost_forecaster.py and
│   │                                    tests/test_dhr_arima.py reproduce against
│   │                                    (RUN_BASELINE_REPRO=1, opt-in, slow)
├── docs/
│   ├── interfaces.md, refactor.md, DATA_QUALITY.md, event_tiering_criteria.md
│   └── methods/adwin.md, methods/kswin.md   per-detector method notes
├── notebooks/
│   ├── exploration/                    scratch EDA — not imported by anything, not production code
│   └── research/                       modelling scratch work (gradient boosting, NHITS) — same caveat
└── pyproject.toml                    # package metadata + dependencies
```

`⚠` marks the two files this tree lists for completeness but that the rest of
this README doesn't otherwise document: `produce_adwin_results.py` /
`produce_kswin_results.py` summarise synthetic-only ADWIN/KSWIN runs from
before the team settled on `produce_table_t1.py` as the one canonical
synthetic-detection table. They still run against today's `runs.csv` schema,
but nothing downstream depends on their output.

---

## The interfaces

Three interfaces were agreed early so team members could work in parallel
without breaking each other's code:

```python
detector(stream) -> changepoint indices
adapter(changepoints, model, data) -> updated model
uq(point_forecast, calibration_residuals) -> lower bound, upper bound, escalate flag
```

A fourth interface, `Forecaster` (`fit`/`predict`), is required by the
adapter signature. Giving every model the same interface allows an adapter to
retrain any of them. The full explanation is in
[`docs/interfaces.md`](docs/interfaces.md).

**There is no registry.** Import the required model or detector class and
create it directly:

```python
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster
from drift_lab.detection.adwin import ADWINDetector

model = XGBoostForecaster(max_depth=4)
detector = ADWINDetector(delta=0.002)
```

### `Forecaster` (`src/drift_lab/forecasting/base.py`)

```python
class Forecaster(ABC):
    name: str
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...     # point forecasts only, no metrics
    def observe(self, y: pd.Series) -> None: ...              # optional: extend history, no retrain
```

`fit` is called once, on the training window only. `observe` (default:
no-op) lets a model absorb the calibration window's real values so the
first test-set forecasts can resolve lags across that boundary, without
retraining. `predict` returns one point forecast per row of `X` — never
a metric; metrics only ever come from `evaluation/`.

| Model | File | Notes |
|---|---|---|
| `SeasonalNaive` | `seasonal_naive.py` | forecast = observed value one season (day) ago |
| `XGBoostForecaster` | `xgboost_forecaster.py` | recursive, lag + calendar features |
| `DHRArima` | `dhr_arima.py` | Fourier terms (day/week) + SARIMAX errors |

To add a model: new file next to these, subclass `Forecaster`, read
`forecasting/base.py`'s docstring first (it has a worked example).

### `DriftDetector` (`src/drift_lab/detection/base.py`)

```python
class DriftDetector(ABC):
    name: str
    def detect(self, stream: np.ndarray | pd.Series) -> list[int]: ...   # positional indices
    def reset(self) -> None: ...   # optional, default no-op
```

| Detector | File | Backing |
|---|---|---|
| `ADWINDetector` | `adwin.py` | `river.drift.ADWIN` |
| `KSWINDetector` | `kswin.py` | `river.drift.KSWIN` |
| `PageHinkleyDetector` | `page_hinkley.py` | `river.drift.PageHinkley` |

All three wrap a `river` online detector via the shared
`detect_with_river()` helper in `detection/base.py` — feeds the stream
through one value at a time, collects the indices where
`drift_detected` fires. New detectors should reuse that helper rather
than reimplementing the loop.

### `Adapter` (`src/drift_lab/adaptation/base.py`)

```python
class Adapter(ABC):
    def adapt(self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame) -> Forecaster: ...
```

| Arm | File | Policy |
|---|---|---|
| `RetrainUsing3MonthWindows` | `retrain_using_3_month_windows.py` | On any non-empty `changepoints`, refit `model` from scratch on the 3 calendar months ending at the most recent flagged date. No cooldown — every call with detections retrains. |

`data` must carry a `DatetimeIndex` with the target as its only column
(or pass `target_column=`) — the same shape every `Forecaster.fit(X, y)`
in this codebase already expects (`pd.DataFrame(index=y.index)` alongside
`y`), which is what makes this arm callable against any model
unchanged: `SeasonalNaive`, `XGBoostForecaster`, `DHRArima`, or
`NHITSForecaster`. The retrain window is free to reach back across split
boundaries (into calibration, even training) — that data was already
observable by "now", so it isn't leakage; the arm just never uses
anything *after* the changepoint it's reacting to. Full rationale,
including why there's no throttle here, in `DECISIONS.md`.

### `UncertaintyQuantifier` — scaffolded, not implemented

`uncertainty/base.py` defines the frozen contract and a worked example,
but no concrete method exists yet. This is deliberate scope for the
current sprint — see `docs/refactor.md` for the plan.

---

## The synthetic drift benchmark

`src/drift_lab/synthetic/generator.py` is the single source of the
synthetic data every detector is validated against before it ever touches
AEMO. Not a class — just one function:

```python
from drift_lab.synthetic.generator import make_series

y, changepoints = make_series(kind="sudden", n=20_000, noise=1.0, seed=1)
```

It's a daily seasonal sine wave (period 48, half-hourly) plus Gaussian
noise, with a level shift injected according to `kind`. `changepoints` is
already in the exact coordinate system `evaluation.evaluate_detections`
expects for that `kind` — nothing needs reshaping between the two:

| `kind` | What happens | `changepoints` |
|---|---|---|
| `"none"` | no shift — pure seasonal + noise, a control | `[]` |
| `"sudden"` | one step shift at `n // 4` | `[cp]` — one point event |
| `"gradual"` | the shift ramps in linearly over a window starting at `n // 4` | `[start, end]` — one transition **interval**, not two points |
| `"recurring"` | shifts up at `n // 3`, back down at `2n // 3` (A → B → A) | `[cp1, cp2]` — two independent point events |

`seed` controls only the noise draw (`np.random.default_rng(seed)`) — the
changepoint positions themselves are deterministic given `kind`/`n`, which
is what lets `run_synthetic_detectors.py` build a `split_id` from the
changepoints directly (see below) rather than from the seed.

Two things read `make_series`, for two different purposes:
- `run_synthetic_detectors.py` — runs the three detectors against it and
  logs metrics to `runs.csv` (the actual Step 4 deliverable).
- `run_synthetic_generator.py` — just plots it (one figure per `kind`,
  one panel per seed, true changepoints marked) as a visual sanity check;
  it doesn't touch `runs.csv`.

---

## Evaluation — the one place metrics are computed

`src/drift_lab/evaluation/evaluation.py` is the **only** module allowed
to compute a metric. Every other file — every `run_*.py`, every model,
every detector — calls into it rather than reimplementing `.rolling()`,
mean absolute error, or detection matching by hand. This is enforced by
`tests/test_evaluation.py::test_no_metric_code_outside_evaluation_module`,
which greps the whole `src/drift_lab/` and `experiments/` trees for
`.rolling(`/`.ewm(` outside this one file and fails the build if it finds
any.

```python
from drift_lab.evaluation import (
    calculate_mae,               # MAE(y_true, y_pred)
    calculate_rolling_mae,       # rolling MAE, default 7-day (336 obs) window
    evaluate_detections,         # match detected vs true changepoints -> the 3 metrics below + diagnostics
    calculate_detection_delay,
    calculate_false_alarms_per_10000,
    calculate_missed_detections,
)
```

`evaluate_detections` is drift-type-aware: `sudden`/`recurring` match
point events within a tolerance window (default 336 obs = 7 days);
`gradual` matches against a `[start, end]` interval instead; `none`
treats every detection as a false alarm. See the module's own docstring
for the full matching protocol — it's long and precise on purpose,
because this is the part every table in the report depends on.

### AEMO documented-event matching

AEMO has no ground-truth changepoints the way the synthetic benchmark
does, so a different set of functions handles it — matching a detector's
output timestamps against the documented events in `aemo/events.csv`
instead of scoring against a known drift point:

```python
from drift_lab.evaluation import (
    match_unmatch,             # classify each detection as Match / Unmatch against events
    calculate_event_metrics,   # precision, recall, unmatched counts, from match_unmatch's output
    build_event_windows,       # turn each event into a pre-drift / drift / post-drift date range
    assign_regime,             # label each detection with which of those windows it falls in
    calculate_regime_metrics,  # match/unmatch counts broken down by regime
    evaluate_aemo_detections,  # runs everything above together, returns it all in one dict
)
```

**Matching** (`match_unmatch`): a detection matches when it falls within an
event's matching window. For a single-day event, `POINT_WINDOW` covers the
seven days after the event starts. For a multi-day event, `INTERVAL_GRACE`
covers the seven days after it ends. Tier 1 events are checked first, followed
by Tier 2 events. A detection outside these windows is labelled `Unmatch`, not
a false positive, because the documented events are not complete ground truth.

**Regime labelling** (`assign_regime`) describes where a detection falls in an
event's lifecycle. Each detection is labelled `drift` (during an event),
`pre_drift` (before an event), or `post_drift` (after an event). Across all
events, the priority is `drift > pre_drift > post_drift`.

For example, assume one event occurs on 1 March and another on 10 March, each
with a seven-day pre-drift window. A detection on 5 March is after the first
event but also before the second. It is labelled `pre_drift` and attributed to
the second event because `pre_drift` has priority over `post_drift`.

Each `drift` or `pre_drift` label records the event whose window produced it.
`post_drift` detections and detections before any event use
`event_id="unassigned"`. This text value behaves consistently when results are
grouped, joined or sorted.

One known gap is that `post_drift` currently has no end. After an event starts,
later detections remain `post_drift` unless another event's `drift` or
`pre_drift` window applies. This remains an open team decision in
`DECISIONS.md` and is recorded as an `xfail` test so it stays visible.

---

## The experiment harness

`experiments/run_harness.py::record_run(...)` is the only function that appends
to `results/runs.csv`. A `run_*.py` script fits, predicts or detects, then sends
the resulting arrays to `record_run`. It calculates metrics through
`evaluation/` and writes rows using the shared schema.

```python
record_run(
    *, method, dataset, seed, config, wall_clock_s, split_id,
    region=None, train_samples=0, n_retrains=0,
    forecast=None,      # (y_true, y_pred, index)        -> group "baseline"
    detection=None,     # (detected, truth, n_samples)    -> group "detection"
    changepoints=None,  # optional: split forecast metrics into pre/drift/post-drift rows
)
```

### `runs.csv` schema (long format — one row per method × dataset × seed × metric × regime)

| Column | Meaning |
|---|---|
| `group` | `"baseline"` (forecasting) or `"detection"` |
| `method` | `model.name` / `detector.name` |
| `dataset` | `"aemo"` or `"synthetic_<drift_type>"` |
| `region` | `"SA1"` / `"NSW1"` / `"-"` (synthetic) |
| `seed` | experiment seed, or NaN for a deterministic model (never fabricated) |
| `config_hash` | hash of the model/detector's public (non-underscore) hyperparameters |
| `split_id` | which **data-partitioning scheme** produced the arrays (e.g. `"aemo_frozen_v1"`) — not "this batch of rows" |
| `metric_name` / `metric_value` | e.g. `"mae"`, `"rolling_mae_7d_mean"`, `"detection_delay"` |
| `regime` | `"full"`, or `"pre-drift"`/`"drift"`/`"post-drift"` when `changepoints=` is passed |
| `n_retrains`, `train_samples`, `wall_clock_s`, `timestamp` | run bookkeeping |

`config_hash` is traceable: `results_io.dump_config()` writes the actual
hyperparameter values to `results/runs/<config_hash>/config.json` the
first time that hash is seen, so a table can show `"delta = 0.002"`
instead of an opaque hash. Forecast runs also dump their full rolling-MAE
curve to `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv`
for the figure scripts to read.

`config_of(model_or_detector)` builds the config dict from an object's
public (non-underscore) instance attributes — which is exactly why every
model/detector keeps its fitted state in `self._x`-prefixed attributes
and its constructor hyperparameters unprefixed: the underscore is what
keeps fitted arrays out of `config_hash`/`config.json`.

`runs.csv` is append-only — re-running a script adds new rows rather
than overwriting old ones. `produce_*.py` scripts take the latest row per
`(method, region)` when duplicates exist (see `produce_figure_f1.py`'s
`load_model_curves()`).

---

## Running experiments

Each of these is a standalone script — run with `python -m experiments.run.forecasting.<name>` or `python -m experiments.run.detection.<name>`, whichever folder it's in below:

| Script | What it does |
|---|---|
| `run_aemo_baselines.py` | Fits `SeasonalNaive`, `XGBoostForecaster`, `DHRArima` on both AEMO regions, frozen split (`split_id="aemo_frozen_v1"`), `seed=None` (deterministic). |
| `run_synthetic_detectors.py` | Runs `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` against `make_series` for every `(drift_type, seed)` in `{none, sudden, gradual, recurring} × SEEDS`. `split_id` embeds the actual changepoint positions, so a future change to the generator's drift geometry can't silently mix with old rows. |
| `run_synthetic_generator.py` | Plots the synthetic benchmark itself (one figure per drift type, one panel per seed) — visual sanity check, not a `runs.csv` producer. |
| `run_aemo_nhits.py` | NHITS on both AEMO regions, its own `split_id` (`aemo_nhits_block7d_v1` / `aemo_nhits_blind_v1` depending on the `BLIND` flag) — kept separate from `run_aemo_baselines.py` since its evaluation protocol isn't the same yet. |
| `run_aemo_detectors.py` | Runs `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` on each region's full raw half-hourly demand (2018-2023, so detectors are warmed up before the test window), scores only test-window detections against the *full* event catalogue (Tier 1 **and** Tier 2). `split_id="aemo_detect_full_v1"`. Feeds Figure F2 only — no longer Table T2's source, see `run_aemo_detectors_daily_frozen.py`. |
| `run_aemo_error_stream_detectors.py` | Same three detectors, fed each frozen baseline's 7-day rolling-MAE *error* curve instead of raw demand — sparser, more drift-shaped signal. One `split_id` per baseline (`aemo_errstream_<baseline>_v1`). |
| `run_aemo_detectors_daily_frozen.py` | Fed `aggregate_daily_demand(test)` (see `aemo/deseasonalise.py`) instead of raw half-hourly demand or an error stream — one point per complete calendar day, test split only, no warmup (`train_samples=0`), which removes the intraday seasonality that makes raw-demand detection fire so often. Each detector uses the frozen config chosen by `run_detector_budget_tuning.py`'s synthetic-only sweep, and matches against the *full* event catalogue (Tier 1 **and** Tier 2), not Tier 1 only — the input `produce_table_t2.py` actually reads. Three detectors, three independent `record_run` calls per region — never pooled into one combined detection stream. `split_id="aemo_detect_daily_frozen_v1"`. |
| `run_aemo_nhits_retrain3mo_kswin_nsw.py` / `..._sa.py` | Adaptation-arm smoke test: `NHITSForecaster` + `KSWINDetector` + `RetrainUsing3MonthWindows`, one file per region. Detection runs once, fully upfront, on `aggregate_daily_demand(test)` (not raw half-hourly demand — every firing here costs a full retrain, see `DECISIONS.md`). Same BLOCK rolling-forecast protocol as `run_aemo_nhits.py`, except a block is cut short exactly at a changepoint's day when one falls inside what would otherwise be a full 7-day block — the model retrains there, then resumes a normal 7-day cadence from the next day. Logged under its own `split_id="aemo_nhits_retrain3mo_kswin_v1"`, with `changepoints=` set so `runs.csv` gets pre-drift/drift/post-drift regime rows too, and `n_retrains` = the adapter's `retrain_count`. |

---

## Producing tables & figures

Every `produce_*.py` script is a pure read of `results/runs.csv` (plus,
for figures, the curve dumps and `docs/event_tiering_criteria.md`-tiered
events) — no metric is computed here that `evaluation.py` didn't already
produce.

| Script | Output |
|---|---|
| `produce_table_t1.py` | `results/tables/table_t1_synthetic_detection.csv` — one row per (detector, drift type): delay (mean ± sd), false alarms/10k, missed, threshold, seed count. |
| `produce_table_t2.py` | `results/tables/table_t2_aemo_events.csv` — one row per (detector, region), one column per Tier 1 event **by name**, in date order (delay, or `"not detected"`), then `tier1_unmatched` (Tier 1 events this detector missed), `unmatched` (unmatched events across **both** tiers), and `precision`. Built with `groupby(["method","region","metric_name"]).unstack()`, not manual filtering. Pinned to `run_aemo_detectors_daily_frozen.py`'s `split_id`. |
| `produce_figure_f1.py` | `results/figures/f1_degradation_{2020,2021,2022,2023}.png` — one file per year, both regions stacked, event markers **and** names. (There used to also be a `f1_degradation_{SA1,NSW1}.png` full-test-period figure with markers only; it was dropped as redundant with the per-year ones.) |
| `produce_figure_f2_aemo.py` | `results/figures/f2_detection_<detector>_<region>.png` — detector flags plotted against the raw AEMO series with documented events marked. Reads `run_aemo_detectors.py`'s run. Palette/layout matches the briefing's Figure D (light-blue raw + blue smoothed demand, black event lines, orange period shading, bold green `MATCHED` / red `FALSE ALARM` labels) — wording says "documented event", not "true changepoint", since AEMO events are not ground truth. |
| `produce_figure_f2_aemo_daily_frozen.py` | Same figure, same formatting, different source: `results/figures/f2_detection_daily_frozen_<detector>_<region>.png`, plotted against `aggregate_daily_demand` (the exact series the detector saw) and reading `run_aemo_detectors_daily_frozen.py`'s frozen-config, both-tier-matched run. A sibling of the original, not a replacement — neither script's output overwrites the other's. |

To add a new one: read `runs.csv`, `groupby` what you need, write a table
to `results/tables/` or a figure to `results/figures/`. Never re-derive a
metric by hand here — if
`evaluation.py` doesn't expose what you need yet, that's a new function
there, not inline code in a `produce_*.py`.

---

## Tests

```bash
pytest              # full suite
pytest -q tests/test_evaluation.py   # one file
RUN_BASELINE_REPRO=1 pytest tests/test_xgboost_forecaster.py   # opt-in: reproduces a committed baseline CSV, slow
```

Notes:
- `tests/test_evaluation.py::test_no_metric_code_outside_evaluation_module`
  is the grep-guard described above — if it fails, you computed a metric
  somewhere you shouldn't have.
- A couple of `xgboost`/`dhr_arima` tests are gated behind
  `RUN_BASELINE_REPRO=1` because they reproduce a committed baseline CSV
  against real AEMO data and take minutes; they're skipped by default.

---

## Where do I put my code?

| You're building... | File goes in | Subclass | Read first |
|---|---|---|---|
| A forecasting model | `forecasting/<your_model>.py` | `Forecaster` | `forecasting/base.py` |
| A drift detector | `detection/<your_detector>.py` | `DriftDetector` | `detection/base.py` |
| A retraining policy (adaptation arm) | `adaptation/<your_arm>.py` | `Adapter` | `adaptation/base.py` |
| A UQ method | `uncertainty/<your_method>.py` | `UncertaintyQuantifier` | `uncertainty/base.py` |
| A metric used anywhere in the pipeline | add it to `evaluation/evaluation.py`, re-export from `evaluation/__init__.py` | plain function | the grep-guard test above |
| A script that runs an experiment | `experiments/run/forecasting/run_<name>.py` or `experiments/run/detection/run_<name>.py`, whichever the script exercises | — | an existing `run_*.py` for the pattern |
| A table/figure derived from `runs.csv` | `experiments/produce/produce_<name>.py` | — | an existing `produce_*.py` for the pattern |

A few rules that keep several people from stepping on each other:

- **One implementation per file.** One model/detector = one file.
- **Never edit a `base.py` to fit your implementation.** If a frozen
  contract genuinely doesn't fit, that's a team conversation (see
  `docs/interfaces.md`), not a solo edit.
- **No registries.** Import the class you want and instantiate it
  directly — don't add a `get_forecaster(name)` lookup function.
- **`src/drift_lab/` never runs anything and never touches `results/`.**
  Scripts that run experiments or write figures/tables live in
  `experiments/` only.
- **Metrics only come from `evaluation/`.** See the grep-guard test.
- **Import via the package**, e.g.
  `from drift_lab.detection.adwin import ADWINDetector` — never a
  relative or local path.

---

## Further reading

- [`DECISIONS.md`](DECISIONS.md) — **why** the codebase looks like this:
  per-component design decisions, the full `runs.csv`/regime/`split_id`
  reasoning, and what's still open. Read this before changing something
  that seems arbitrary — it probably isn't.
- [`docs/interfaces.md`](docs/interfaces.md) — full rationale for the
  frozen contracts and where escalation logic lives.
- [`docs/refactor.md`](docs/refactor.md) — the living design doc this
  codebase's current shape was built from.
- [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md) — what `aemo/loader.py`'s
  cleaning stage checks and why.
- [`docs/event_tiering_criteria.md`](docs/event_tiering_criteria.md) —
  how documented AEMO/AER events are tiered for detector-event matching.
