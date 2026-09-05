# Detecting and Adapting to Concept Drift in Time Series Forecasting

Design and implement an automated software pipeline that detects when
concept drift occurs in a time series and adapts the underlying
forecasting model in response, while also reporting how confident its
predictions are and flagging cases where a human should review the
output.

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

This installs the `drift_lab` package (from `src/`) in editable mode, so
`import drift_lab...` works from anywhere — notebooks, tests, your own
scripts — without path hacks. It also pulls in `pytest`/`ruff` (`dev`
extra).

The AEMO raw CSVs are already committed under `data/raw/{SA1,NSW1}/` —
there's no download step. `data/processed/`, `data/synthetic/` and
`results/runs/` are write targets for cached/generated output and are
gitignored (see [`.gitignore`](.gitignore)); `results/runs.csv` itself
*is* tracked, since it's the shared experiment ledger.

---

## Quickstart

```bash
# Fit the three frozen baselines on both AEMO regions, log to runs.csv
python -m experiments.run_aemo_baselines

# Run the three detectors against the synthetic benchmark (5 seeds x 4 drift types)
python -m experiments.run_synthetic_detectors

# Turn runs.csv into the deliverables
python -m experiments.produce_table_t1     # results/figures/table_t1_synthetic_detection.csv
python -m experiments.produce_figure_f1    # results/figures/f1_degradation_*.png

# Run the test suite
pytest
```

Everything under `results/figures/` and `results/runs/` is *derived* —
delete it and regenerate it from `results/runs.csv` (or regenerate
`runs.csv` itself by re-running the `run_*.py` scripts) at any time.

---

## Repository layout

```
.
├── src/drift_lab/                  # the installable package — implementation only, nothing runs here
│   ├── config.py                    # paths, REGIONS, SEEDS, train/calibration/test SPLIT, shared constants
│   │
│   ├── aemo/                        # AEMO demand data
│   │   ├── loader.py                  load(region) -> (train, calibration, test); load_processed(region)
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
│   │   ├── nhits_forecaster.py         NHITS (Nixtla neuralforecast) — see caveats below
│   │   └── nixtla_common.py            shared rolling-forecast adapter for Nixtla-backed models
│   │
│   ├── detection/                   # DriftDetector implementations (detect(stream) -> indices)
│   │   ├── base.py                    the frozen contract + detect_with_river() shared helper
│   │   ├── adwin.py, kswin.py, page_hinkley.py    river-backed detectors
│   │
│   ├── adaptation/                  # Adapter contract — scaffold only, no arms implemented yet
│   │   └── base.py
│   │
│   ├── uncertainty/                 # UncertaintyQuantifier contract — scaffold only
│   │   └── base.py
│   │
│   └── evaluation/                  # the ONE place every metric is computed
│       ├── evaluation.py              calculate_mae, calculate_rolling_mae, evaluate_detections, ...
│       └── __init__.py                 re-exports the public functions
│
├── experiments/                     # the only place that runs anything or writes to results/
│   ├── run_harness.py                 record_run() — turns one run's arrays into runs.csv rows
│   ├── results_io.py                   runs.csv / config.json / curve-dump filesystem plumbing
│   ├── run_aemo_baselines.py           fits seasonal_naive + xgboost + dhr_arima on AEMO
│   ├── run_aemo_nhits.py               NHITS pilot on AEMO — run separately, own split_id (see below)
│   ├── run_synthetic_detectors.py      runs adwin/kswin/page_hinkley on the synthetic benchmark
│   ├── run_synthetic_generator.py      plots the synthetic benchmark itself (no runs.csv row)
│   ├── produce_table_t1.py             synthetic detection table, grouped from runs.csv
│   └── produce_figure_f1.py            AEMO rolling-MAE degradation figures, grouped from runs.csv
│
├── results/
│   ├── runs.csv                       the experiment ledger — tracked in git
│   ├── runs/<config_hash>/             config.json + per-run curve dumps — gitignored
│   └── figures/                       generated tables/figures — gitignored
│
├── tests/                           # pytest, one file per package/module it covers
├── data/                            # raw/ (committed), processed/ + synthetic/ (generated, gitignored)
├── docs/                            # interfaces.md, refactor.md, DATA_QUALITY.md, event_tiering_criteria.md
├── notebooks/exploration/            # scratch EDA — not imported by anything, not production code
└── pyproject.toml                    # package metadata + dependencies
```

---

## The interfaces

Three contracts were frozen early so multiple people could build against
them in parallel without breaking each other's code:

```python
detector(stream) -> changepoint indices
adapter(changepoints, model, data) -> updated model
uq(point_forecast, calibration_residuals) -> lower bound, upper bound, escalate flag
```

A fourth, `Forecaster` (`fit`/`predict`), isn't on the original slide but
is required by `adapter`'s signature — every model needs the same shape
so an adapter can retrain any of them interchangeably. Full rationale in
[`docs/interfaces.md`](docs/interfaces.md).

**There is no registry.** To use a model or detector, just import the
class and instantiate it — no `get_forecaster("xgboost")` lookup
anywhere:

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
| `NHITSForecaster` | `nhits_forecaster.py` | neural, direct multi-step (Nixtla) — see caveats below |

To add a model: new file next to these, subclass `Forecaster`, read
`forecasting/base.py`'s docstring first (it has a worked example).

**NHITS caveats** (read before trusting its numbers): it's run by its own
script (`run_aemo_nhits.py`), separately from the other three, under its
own `split_id`. Running it fully frozen/blind like xgboost/dhr_arima
causes it to diverge (a direct multi-step model fed its own compounding
forecasts for years with zero correction runs away numerically), so it's
evaluated with **periodic weekly re-grounding** instead — a materially
different, more forgiving protocol than the other three. See the
docstring at the top of `run_aemo_nhits.py` for the full story. Also
note: sharing a process with `XGBoostForecaster` requires the
`OMP_NUM_THREADS=1` workaround already set at the top of
`nhits_forecaster.py` — XGBoost's and PyTorch's OpenMP runtimes deadlock
(occasionally segfault) otherwise. Don't remove it.

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

### `Adapter` and `UncertaintyQuantifier` — scaffolded, not implemented

`adaptation/base.py` and `uncertainty/base.py` define the frozen
contracts and a worked example each, but no concrete arm/method exists
yet. This is deliberate scope for the current sprint (forecasting +
detection first) — see `docs/refactor.md` for the plan.

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

---

## The experiment harness

`experiments/run_harness.py::record_run(...)` is the **only** function
allowed to append to `results/runs.csv`. It doesn't run anything itself —
a `run_*.py` script fits/predicts or detects, then hands the resulting
arrays to `record_run`, which routes them through `evaluation/` and
writes the schema rows.

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

Each of these is a standalone script — run with `python -m experiments.<name>`:

| Script | What it does |
|---|---|
| `run_aemo_baselines.py` | Fits `SeasonalNaive`, `XGBoostForecaster`, `DHRArima` on both AEMO regions, frozen split (`split_id="aemo_frozen_v1"`), `seed=None` (deterministic). |
| `run_aemo_nhits.py` | NHITS on both AEMO regions, its own `split_id="aemo_frozen_v1_nhits_pilot"`, **not** part of `run_aemo_baselines.py`'s model list — see the NHITS caveats above before reading its numbers alongside the other three. |
| `run_synthetic_detectors.py` | Runs `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` against `make_series` for every `(drift_type, seed)` in `{none, sudden, gradual, recurring} × SEEDS`. `split_id` embeds the actual changepoint positions, so a future change to the generator's drift geometry can't silently mix with old rows. |
| `run_synthetic_generator.py` | Plots the synthetic benchmark itself (one figure per drift type, one panel per seed) — visual sanity check, not a `runs.csv` producer. |

---

## Producing tables & figures

Every `produce_*.py` script is a pure read of `results/runs.csv` (plus,
for figures, the curve dumps and `docs/event_tiering_criteria.md`-tiered
events) — no metric is computed here that `evaluation.py` didn't already
produce.

| Script | Output |
|---|---|
| `produce_table_t1.py` | `results/figures/table_t1_synthetic_detection.csv` — one row per (detector, drift type): delay (mean ± sd), false alarms/10k, missed, threshold, seed count. |
| `produce_figure_f1.py` | `results/figures/f1_degradation_{SA1,NSW1}.png` (full test period, event markers) and `f1_degradation_{2020,2021,2022,2023}.png` (one file per year, both regions stacked, event markers **and** names). |

To add a new one: read `runs.csv`, `groupby` what you need, write to
`results/figures/`. Never re-derive a metric by hand here — if
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
- On macOS, don't remove the `OMP_NUM_THREADS=1` line at the top of
  `nhits_forecaster.py` — without it, running the XGBoost and NHITS test
  files together deadlocks (occasionally segfaults) from two competing
  OpenMP runtimes in one process.

---

## Where do I put my code?

| You're building... | File goes in | Subclass | Read first |
|---|---|---|---|
| A forecasting model | `forecasting/<your_model>.py` | `Forecaster` | `forecasting/base.py` |
| A drift detector | `detection/<your_detector>.py` | `DriftDetector` | `detection/base.py` |
| A retraining policy (adaptation arm) | `adaptation/<your_arm>.py` | `Adapter` | `adaptation/base.py` |
| A UQ method | `uncertainty/<your_method>.py` | `UncertaintyQuantifier` | `uncertainty/base.py` |
| A metric used anywhere in the pipeline | add it to `evaluation/evaluation.py`, re-export from `evaluation/__init__.py` | plain function | the grep-guard test above |
| A script that runs an experiment | `experiments/run_<name>.py` | — | an existing `run_*.py` for the pattern |
| A table/figure derived from `runs.csv` | `experiments/produce_<name>.py` | — | an existing `produce_*.py` for the pattern |

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
  reasoning, the NHITS case study, and what's still open. Read this
  before changing something that seems arbitrary — it probably isn't.
- [`docs/interfaces.md`](docs/interfaces.md) — full rationale for the
  frozen contracts and where escalation logic lives.
- [`docs/refactor.md`](docs/refactor.md) — the living design doc this
  codebase's current shape was built from.
- [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md) — what `aemo/loader.py`'s
  cleaning stage checks and why.
- [`docs/event_tiering_criteria.md`](docs/event_tiering_criteria.md) —
  how documented AEMO/AER events are tiered for detector-event matching.
