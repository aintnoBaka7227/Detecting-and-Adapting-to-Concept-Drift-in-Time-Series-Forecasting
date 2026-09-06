# Codebase refactor plan

Goal: every model and detector is one class in one file behind its frozen
interface; every metric comes from one module; every experiment run writes
one long-format row per metric to `results/runs.csv`; every table is a
group-by of that file.

Scope: **forecasting + detection** end to end; adaptation + uncertainty
**scaffolded** (base classes, registries, harness hooks, no implementations).

### Hard rules

1. `src/drift_lab/` is importable library code only — returns objects
   and numbers, no working `if __name__ == "__main__"`, the string `results/`
   never appears in it.
2. **Compose in memory within a stage.** The data -> model -> metric path
   passes return values, never re-reading a file an earlier function wrote.
   `aemo.loader.load(region)` returns the splits directly — no processed CSV
   at all. The per-run curve dumps are write-only housekeeping. The one
   sanctioned file hand-off is `runs.csv`: `run_*.py` append to it,
   `produce_*.py` group-by it.
3. Comments are precise and minimal: the non-obvious *why*, never the *what*.

---

## 1. Target layout

```
src/drift_lab/
  config.py                     paths, regions, SPLIT, seeds        (unchanged)
  viz.py                    NEW plot helpers that draw on an Axes (no savefig)

  aemo/
    loader.py               ONE file: load(region) -> (train, calibration, test)
                             + internal read / clean / standardise / split; no file output
  synthetic/
    generator.py            make_series(kind, n, noise, seed) -> (y, changepoints); no file output

  forecasting/
    base.py                     Forecaster: fit / predict  (+ observe(), name)
    seasonal_naive.py       NEW SeasonalNaive
    dhr_arima.py            MOVED+WRAPPED  DHRArima            (from forecasting/classical-model/)
    xgboost_forecaster.py       XGBoostForecaster              (drop the _history hack)
    __init__.py             NEW FORECASTERS = {...}; get_forecaster(name, **cfg)

  detection/
    base.py                     DriftDetector: detect  (+ optional reset())
    adwin.py                NEW ADWINDetector
    kswin.py                NEW KSWINDetector
    page_hinkley.py         NEW PageHinkleyDetector
    __init__.py             NEW DETECTORS = {...}; get_detector(name, **cfg)

  adaptation/  base.py  __init__.py   ADAPTERS = {} + get_adapter()   (scaffold)
  uncertainty/ base.py  __init__.py   QUANTIFIERS = {} + get_quantifier()  (scaffold)

  evaluation/
    __init__.py
    evaluation.py           NEW the ONLY place metrics are computed


results/                    NEW  written by record_run + produce_*; the run/produce hand-off
  runs.csv                       append-only ledger, schema in section 4
  runs/<config_hash>/            per-run rolling-MAE curve dumps
  figures/                       every table/figure produce_*.py emits

experiments/               NEW  the ONLY place that writes to results/
  _harness.py             NEW  record_run(): the shared row-writer (section 5)
  _io.py                  NEW  locked runs.csv append; figure / curve-dump paths
  run_*.py                     author-written; call models via the interface,
                               write rows only via record_run(), one method per call
  produce_*.py                 author-written; group-by runs.csv -> results/figures/
  main.py                NEW  optional: run every run_*, then every produce_*

notebooks/exploration/           scratch only — no committed figure/table originates here
```

**Fixed vs. free.** No registries. A `run_*.py` author imports the model /
detector class and instantiates it directly
(`SeasonalNaive(48)`, `ADWINDetector(delta=0.002)`), runs it however that
model needs to be run, and passes the arrays to `record_run(...)`. Two rules,
regardless of who writes the script:

1. Models / detectors are classes satisfying the frozen ABC (`fit`/`predict`,
   `detect`) plus a `name`. Reached by import, not a lookup.
2. `record_run(...)` is the only way a row reaches `runs.csv`, and the only
   place metrics are turned into rows — via `evaluation.py`, never inline.
   One call records **one** method on one `(dataset, region, seed)`.

Everything else in `run_*.py` / `produce_*.py` (loop structure, argparse,
which methods/seeds, how `predict` is called, table shape, plot style) is the
author's. `produce_*.py` read `runs.csv` and group-by — that file is the
deliberate hand-off between the two halves.

### Files deleted / consolidated

| Remove | Reason |
|---|---|
| `forecasting/classical-model/` (whole dir) | hyphen = not importable; model logic in `forecasting/dhr_arima.py` |
| `aemo/generator.py`, `synthetic/generator_example.py` | duplicate / stub of `synthetic/generator.py` |
| `aemo/csv_output/` (whole dir) | 3rd DHR+ARIMA copy + regenerable synthetic CSVs |
| `aemo/cleaning.py`, `aemo/splits.py` | merged into `aemo/loader.py` |
| every `__main__` block in `src/` + generated `plot_output/` / `*.png` | src is library-only; artifacts regenerate |
| `data/raw/{REGION}_..._native.csv`, `data/processed/*` (splits + figures) | `loader.load(region)` returns the splits in memory; no processed file |

---

## 2. Interface changes (additive — bases stay backward compatible)

`forecasting/base.py` — add `name` and `observe()`:
```python
class Forecaster(ABC):
    name: str = ""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...   # point forecasts, unchanged

    def observe(self, y: pd.Series) -> None:
        """Add observed values to history without retraining. Default no-op."""
```
`observe()` replaces the `model._history = pd.concat(...)` hack: it lets
`predict()` resolve lags across the train->test boundary. `SeasonalNaive` and
`XGBoostForecaster` override it; `DHRArima` keeps the no-op (Fourier phase
comes from `X.index`).

`detection/base.py` — add `name` and `reset()`:
```python
class DriftDetector(ABC):
    name: str = ""

    @abstractmethod
    def detect(self, stream: np.ndarray | pd.Series) -> list[int]: ...

    def reset(self) -> None: ...          # default no-op
```

---

## 3. `evaluation/evaluation.py` — the only metric module

The team's shared module (its own long docstring covers the matching protocol).
Public surface:

```python
def calculate_mae(y_true, y_pred) -> float
def calculate_rolling_mae(y_true, y_pred, window=48*7) -> pd.Series
def evaluate_detections(detected_changepoints, true_changepoints, n_observations,
                        drift_type, tolerance=48*7, gradual_grace=None) -> dict
#   -> detection_delay, false_alarms_per_10000, missed_detections (+ matching diagnostics)
def calculate_detection_delay / calculate_false_alarms_per_10000 / calculate_missed_detections
```

- `calculate_mae` / `calculate_rolling_mae` **raise** on length mismatch or NaN
  inputs; `calculate_rolling_mae` is count-based (`.rolling(window).mean()`),
  336 = 7 days of half-hourly data, leading NaN warm-up.
- `evaluate_detections` is `drift_type`-aware: `none` → every detection is a
  false alarm; `sudden`/`recurring` → point events, first unused detection in
  `[cp, cp + tolerance]`; `gradual` → truth is `[drift_start, drift_end]`, one
  interval, delay from `drift_start`. `synthetic.generator.make_series` returns
  changepoints in exactly this shape.
- coverage / interval-width / documented-event matching are *planned* in that
  module.

Enforced by a test that greps `src/` and `experiments/` for `.rolling(` /
`.ewm(` outside this file.

---

## 4. `results/runs.csv` schema (frozen)

One row per **(method x dataset x region x seed x metric x regime)**. Long,
not wide. Appended **only** by `record_run` (§5); read only by `produce_*.py`,
which group-by it into every table and figure.

| column | meaning |
|---|---|
| `group` | `baseline` \| `detection` \| `adaptation` \| `uq` |
| `method` | `obj.name`, e.g. `seasonal_naive`, `adwin` |
| `dataset` | `aemo` \| `synthetic_sudden` \| `synthetic_gradual` \| `synthetic_recurring` \| `synthetic_none` |
| `region` | `SA1` \| `NSW1` for aemo; `-` for synthetic |
| `seed` | int, or NaN for a deterministic method (no seed fabricated just to fill the column) |
| `config_hash` | `sha1(json.dumps(config, sort_keys=True, default=str))[:12]` |
| `split_id` | the data-partitioning version, not "this batch of rows": `aemo_frozen_v1` (config.SPLIT), `synth_n20000` (generator config). Required — no generic default, so a synthetic run can't silently inherit an AEMO-shaped id |
| `metric_name` | baseline: `mae`, `rolling_mae_7d_mean`, `rolling_mae_7d_max`. detection: `detection_delay`, `false_alarms_per_10000`, `missed_detections` |
| `metric_value` | float scalar |
| `regime` | `full` (whole stream) or `pre-drift` \| `drift` \| `post-drift` — see below |
| `n_retrains` | int (0 for baseline/detection) |
| `train_samples` | rows the method fit on (detectors: warm-up count) |
| `wall_clock_s` | `time.perf_counter` around the run |
| `timestamp` | ISO-8601, run time |

**Regime, decided 2026-09 (cross-checked against Sahar's team's chat + the
required-outputs deck — T3/T5/T6 group by regime, so it is not optional):**

- **detection, synthetic** (`truth` given) → **always `full`**. Team decision:
  splitting delay/false-alarms/missed by regime was judged not worth the
  complexity, and `full` is already presented for T1 — kept for consistency
  with the other team's evaluation bed.
- **detection, AEMO** (`truth=None`) → **`full`**, one `n_detections` row.
  There is no ground truth and no `drift_type` to score against yet — a
  detector only ever says "something changed here". Placeholder until the
  Tier 1/2 documented-event matcher (T2) exists; re-visitable without a
  schema change.
- **forecast** → `record_run(..., changepoints=[...])` splits into
  `pre-drift` / `drift` / `post-drift`; omitting `changepoints` gives one
  `full` set. `changepoints` is **never ground truth for AEMO** — it is
  either the synthetic generator's known changepoints, or a detector's own
  output, passed in by the `run_*.py` author. Regime boundary:
  `config.REGIME_DRIFT_MARGIN` (7 days, an assumption — widen it if a
  detector's drift band turns out too narrow to see in the MAE).

Scalar metrics only. The full rolling-MAE curve is summarised into
`rolling_mae_7d_mean` / `_max` rows here (via `np.nanmean` / `np.nanmax` of the
`calculate_rolling_mae` output), and the whole series is dumped to
`results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` for the curve
figure to pick up. Append is lock-guarded for parallel runs.

---

## 5. `experiments/_harness.py` — `record_run`

The one shared row-writer. It does **not** run the model — the author's
`run_*.py` does that (fit/observe/predict, or detect) — it takes the raw
arrays, computes metrics through `evaluation.py`, and appends the schema rows.

```python
def record_run(
    *,
    method: str,              # obj.name
    dataset: str,             # "aemo" | "synthetic_sudden" | ...
    seed: int | None,         # None for a deterministic run -> written as NaN
    config: dict,             # hashed -> config_hash; config_of(obj) is a helper
    wall_clock_s: float,
    split_id: str,            # the DATA version, e.g. "aemo_frozen_v1" / "synth_n20000" — required, no generic default
    region: str | None = None,
    train_samples: int = 0,
    n_retrains: int = 0,
    forecast: tuple | None = None,     # (y_true, y_pred, index)      -> baseline rows
    detection: tuple | None = None,    # (detected, truth, n_samples) -> detection rows
    changepoints: list[int] | None = None,   # forecast only, see regime note above
) -> pd.DataFrame
```

Exactly one of `forecast` / `detection` is given; `group` follows from it.

- **forecast** → `evaluation.calculate_rolling_mae(y_true, y_pred)` (index
  re-attached for the dump) + `evaluation.calculate_mae`. No `changepoints`:
  one `mae` / `rolling_mae_7d_mean` / `rolling_mae_7d_max` set, `regime=full`.
  `changepoints` given: `_label_regimes` (in `_harness.py` — a candidate to
  move into the shared `evaluation.py`) slices `y_true`/`y_pred`/the curve
  into `pre-drift`/`drift`/`post-drift` first, same 3 metrics each. Curve
  dumped either way.
- **detection, `truth` given** → `dataset` must be `synthetic_<drift_type>`;
  `evaluation.evaluate_detections(detected, truth, n_observations=n,
  drift_type=<from dataset>, tolerance=336)`; emit `detection_delay`,
  `false_alarms_per_10000`, `missed_detections`, regime `full`.
- **detection, `truth=None`** (AEMO) → one row, `metric_name="n_detections"`,
  `metric_value=len(detected)`, regime `full`.

`config_hash = sha1(json.dumps(config, sort_keys=True, default=str))[:12]`.
`experiments/_io.py` owns the locked `runs.csv` append, the `config_hash`
helper, and the curve-dump path.

A `run_*.py` loop body — author's code, one `record_run` per method:

```python
for region in REGIONS:
    for seed in SEEDS:
        m = SeasonalNaive()
        m.fit(pd.DataFrame(index=train.index), train_y)
        m.observe(cal_y)
        preds = m.predict(pd.DataFrame({"TOTALDEMAND": test_y}, index=test.index))
        t0 = time.perf_counter()  # (timed around the actual run in real code)
        record_run(method=m.name, dataset="aemo", region=region, seed=seed,
                   config=config_of(m), wall_clock_s=..., train_samples=len(train_y),
                   forecast=(test_y, preds, test.index))
```

---

## 6. Data pipeline — one file, one function

### `aemo/loader.py`

```python
def load(region: str) -> tuple[DataFrame, DataFrame, DataFrame]:
    raw = _read_monthly(region)   # concat the monthly raw CSVs, chronological
    return split(standardise(clean(raw)))

def clean(df) -> DataFrame   # prints the full original 11-stage audit + cleaning sequence
def standardise(df) -> DataFrame  # 30-min grid, interval-end, raw MW, gaps -> NaN
def split(df) -> (train, calibration, test)   # frozen config.SPLIT boundaries
```

`load` takes only the region ("SA1" / "NSW1"), memoised per region, returns
the three splits in memory. No combined frame, no cross-region merge, **no
CSV output**. Any module needing real splits calls `load`; synthetic data
comes from `synthetic.generator.make_series`. There is no `Bundle` / dataset
wrapper — `record_run` (§5) branches on `dataset`; the `run_*.py` author
loads splits with `loader.load` or a series with `make_series` and hands the
arrays to `record_run`.

---

## 7. Migration order — each step ships green, reproduces current numbers

| # | Step | Must reproduce |
|---|---|---|
| 1 | `evaluation/evaluation.py` (lift from the notebooks); grep test | notebook CSVs / Table 1 unchanged |
| 2 | `SeasonalNaive` + `observe()` on base + tests | `{SA1,NSW1}_seasonal_naive_test.csv` to float tol |
| 3 | `XGBoostForecaster.observe()`, drop `_history` hack | `{SA1,NSW1}_xgboost_test.csv` to float tol |
| 4 | `forecasting/dhr_arima.py` (`DHRArima(Forecaster)`); delete `classical-model/` | `{SA1,NSW1}_rolling_mae.csv` to float tol |
| 5 | `aemo/loader.py` (one file: read/clean/standardise/split); delete `cleaning.py`/`splits.py`/`datasets.py`, `data/csv_output/`, generator dup+stub, all `src` `__main__` blocks + generated artifacts; retire the raw/processed CSVs | split row counts + dates match the old split CSVs exactly |
| 6 | team `evaluation.py` swapped in (`calculate_mae` / `calculate_rolling_mae` / `evaluate_detections`); `detection/{adwin,kswin,page_hinkley}.py` + `name`/`reset()` on base; `experiments/_io.py` + `_harness.py` (`record_run`); `results/` schema; grep guard covers `experiments/` | detector wrappers match raw river; DHR curve still matches its CSV |
| 7 | first `run_*.py` per group -> populate `runs.csv` | baseline MAEs match steps 2-4 |
| 8 | first `produce_*.py` + `viz.py` + `main.py` | `f1_comparison_*.png`, degradation bands, detector table |

Both regions (SA1 **and** NSW1) in every step that touches baselines.

---

## 8. Open items for the team

- `evaluate_detections` tolerance — `record_run` uses `336`; the team module
  documents a documented-event rule for AEMO once detectors run on real data.
- `.gitignore` policy for `results/` (commit `runs.csv`? figures?).
- The frozen-vs-walk-forward choice per baseline: seasonal-naive consumes the
  test actuals (lag-48), XGBoost + DHR-ARIMA forecast without feedback. The
  `run_*.py` author calls `predict` accordingly; not `record_run`'s concern.

### Settled

- No registries. Models/detectors are imported and instantiated directly.
- `run_*.py` / `produce_*.py` are author-written and free-form; the fixed
  contract is: models satisfy the frozen ABC + `name`; rows reach `runs.csv`
  only via `record_run` (one method per call); tables/figures are group-bys
  of `runs.csv`.
- `predict()` returns point forecasts and computes no metrics; `record_run`
  gets the arrays and calls `evaluation.py`.
- `record_run` does not run the model — the author's `run_*.py` does, and
  passes `forecast=(y_true, y_pred, index)` or `detection=(detected, truth, n)`.
- Standardisation = 30-min grid only; raw MW; gaps stay NaN.
- `regime`: detection always `full` (both datasets); forecast `full` unless
  `changepoints` passed, then `pre-drift`/`drift`/`post-drift`. See §4.
- `_harness.py` lives in `experiments/`, not `src/`.
- Scope: forecasting + detection implemented; adaptation + uq scaffolded.
- `notebooks/` kept for scratch only.
- `data/baseline/**` historical CSVs stay untouched (user decides later).
- Comments: precise, minimal, why-not-what.

### Deferred — needs the team

- **T2 documented-event matcher.** `src/drift_lab/aemo/events.csv`
  (`config.DOCUMENTED_EVENTS_CSV`; Tier 1: 5 primary AEMO/AER events; Tier 2:
  the rest — see `docs/event_tiering_criteria.md`, frozen before matching)
  exists but nothing scores against it yet. Needs a
  `match_detections_to_events(detections, events_df, tolerance)`-shaped
  function: Tier 1 first, then Tier 2, else `unmatched` — never
  auto-classified as a false alarm. Coordinate with the team on where it
  lives (`evaluation.py`) and the temporal matching rule before building it.
- **`_label_regimes`** currently lives in `_harness.py`, not `evaluation.py`
  — raise with the team whether the shared module should own it, since both
  groups need to carve the AEMO timeline identically for T3/T7.
- **Recovery regime.** Between two changepoints the error can fall back to
  the pre-drift level (most visible for `recurring`); right now that segment
  is still labelled `post-drift`. If it needs its own label later, that's a
  change inside `_label_regimes`, not a schema change.
