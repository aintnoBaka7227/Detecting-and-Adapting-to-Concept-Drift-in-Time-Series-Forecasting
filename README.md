# Detecting and Adapting to Concept Drift in Time Series Forecasting

This project builds an automated pipeline that detects concept drift in a
time series and adapts the forecasting model when drift occurs. It also
reports prediction confidence and flags forecasts that need human review.

It runs on two AEMO electricity-demand regions (**SA1**, **NSW1**,
half-hourly, 2018–2023). It also runs on a synthetic benchmark with known
drift points, used to tune and validate the detectors.

---

## The pipeline

```
data -> detection -> forecasting -> adaptation -> uncertainty -> escalation
```

| Stage | Question it answers | Deliverables |
|---|---|---|
| **1. Data** | Is the demand series clean, and does the TRAIN-fitted seasonal profile generalise to later splits? | QC figures and table |
| **2. Detection** | Which detector finds real changes without false alarms, and when does it fire on AEMO? | T1, F2 (synthetic), T2, F2 (AEMO), change-point files, periodicity check |
| **3. Forecasting** | How fast do frozen forecasters degrade, and which forecaster is selected? | F1, forecaster-selection table, calibration residuals |
| **4. Adaptation** | Does retraining on drift alarms beat never or scheduled retraining, and at what cost? | T3, T4, F3 |
| **5. Uncertainty** | Do the 90% intervals hold their coverage through drift? | T5, F4 |
| **6. Escalation** | How much human review keeps coverage on target? | T6, F5 |
| *Comparison* | Distribution-shift trigger (us) vs error-based trigger (Team 42) | T7, F6 |

**Each stage reads only what the earlier stages saved.** The links are:
- **detection → adaptation:** the change points saved by detection are the
  days arms C and D retrain on;
- **forecasting → uncertainty:** the calibration residuals saved by
  forecasting calibrate both conformal methods;
- **adaptation → uncertainty:** uncertainty replays adaptation's saved
  forecasts;
- **uncertainty → escalation:** escalation sweeps uncertainty's saved
  intervals.

Every later stage takes the **latest** run of the stage before it. Rerunning
a stage, then every stage after it, refreshes the whole chain.

### Execution order: what to run for each deliverable

Run the commands top to bottom. `run_*` / `check_*` scripts compute results
and log them. `produce_*` scripts only read those results and draw the
tables and figures.

| # | Stage | Run | Then produce | Deliverable |
|---|---|---|---|---|
| 1 | Data | `experiments.run.detection.check_seasonal_residuals` | — | `results/figures/qc_*_half_hourly.png` |
| 2 | Data | `experiments.run.detection.check_residual_generalisation` | — | `results/tables/qc_residual_generalisation_half_hourly.{csv,png}` |
| 3 | Detection | `experiments.run.detection.run_fine_tune_on_synthetic` | — | `results/tables/fine_tune_on_synthetic_winners.csv` (the frozen detector configs, read by every later detection script) |
| 4 | Detection | `experiments.run.detection.run_post_tune_on_synthetic` | `produce_table_t1_post_tune`, `produce_figure_f2_synthetic_post_tune` | **T1**, **F2** (synthetic) |
| 5 | Detection | `experiments.run.detection.run_aemo_detectors_standard_half_hourly_post_tune` | `produce_table_t2_standard_half_hourly_post_tune`, `produce_figure_f2_aemo_standard_half_hourly_post_tune` | **T2**, **F2** (AEMO) |
| 6 | Detection | `experiments.run.detection.run_detector_changepoints` | — | `results/runs/changepoints/<detector>_standard_half_hourly_post_tune_<region>.csv` |
| 7 | Detection | `experiments.run.detection.check_detection_periodicity` | — | `results/tables/detection_periodicity/` |
| 8 | Forecasting | `experiments.run.forecasting.run_aemo_baselines`, `experiments.run.forecasting.run_aemo_nhits` | `produce_figure_f1` | **F1** |
| 9 | Forecasting | `experiments.run.forecasting.run_aemo_holdout_selection`, `experiments.run.forecasting.run_aemo_calibration_forecasts` | `produce_table_forecaster_selection` | forecaster-selection tables (held-out Nov–Dec 2019 from TRAIN; Calibration window); `calibration_forecast_aemo_<region>_<seed>.csv` residuals |
| 10 | Adaptation | `experiments.run.adaptation.run_aemo_adaptation_arms` (once per detector × model × region × seed) | — | adaptation forecasts per run |
| 11 | Adaptation | `experiments.run.adaptation.rescore_adaptation_runs` | `produce_table_t3`, `produce_table_t4`, `produce_figure_f3`, `produce_table_arm_d_sweep` | **T3**, **T4**, **F3**, arm D sweep table (all four windows, winner per model × region) |
| 12 | Uncertainty | `experiments.run.uncertainty.run_aemo_uq_replay` | `produce_table_t5`, `produce_figure_f4` | **T5**, **F4** |
| 13 | Escalation | `experiments.run.uncertainty.run_aemo_escalation_sweep` | `produce_table_t6`, `produce_figure_f5` | **T6**, **F5** |
| 14 | Comparison | — | `produce_table_t7`, `produce_figure_f6` | **T7**, **F6** |

All producers are `experiments.produce.<name>`.

Steps 1–2 sit in `experiments/run/detection/` because they validate the
seasonal residual stream the detectors read. Steps 10–13 loop over both
trigger detectors (ADWIN, Page-Hinkley) unless `--detector` is given.

### Full run

```bash
# 1. Data
python -m experiments.run.detection.check_seasonal_residuals
python -m experiments.run.detection.check_residual_generalisation

# 2. Detection
python -m experiments.run.detection.run_fine_tune_on_synthetic
python -m experiments.run.detection.run_post_tune_on_synthetic
python -m experiments.run.detection.run_aemo_detectors_standard_half_hourly_post_tune
python -m experiments.run.detection.run_detector_changepoints
python -m experiments.run.detection.check_detection_periodicity
python -m experiments.produce.produce_table_t1_post_tune
python -m experiments.produce.produce_figure_f2_synthetic_post_tune
python -m experiments.produce.produce_table_t2_standard_half_hourly_post_tune
python -m experiments.produce.produce_figure_f2_aemo_standard_half_hourly_post_tune

# 3. Forecasting
python -m experiments.run.forecasting.run_aemo_baselines
python -m experiments.run.forecasting.run_aemo_nhits
python -m experiments.run.forecasting.run_aemo_holdout_selection
python -m experiments.run.forecasting.run_aemo_calibration_forecasts
python -m experiments.produce.produce_figure_f1
python -m experiments.produce.produce_table_forecaster_selection

# 4. Adaptation -- one run per detector x model x region x seed
#    ADWIN runs all arms (A, B, C, D); other detectors run C and D only (A and B use no detector)
for detector in adwin page_hinkley; do
  for region in SA1 NSW1; do
    python -m experiments.run.adaptation.run_aemo_adaptation_arms --detector $detector --model dhr_arima --region $region
    for model in nhits xgboost; do
      for seed in 1 2 3 4 5; do
        python -m experiments.run.adaptation.run_aemo_adaptation_arms --detector $detector --model $model --region $region --seed $seed
      done
    done
  done
done
python -m experiments.run.adaptation.rescore_adaptation_runs
python -m experiments.produce.produce_table_t3
python -m experiments.produce.produce_table_t4
python -m experiments.produce.produce_figure_f3
python -m experiments.produce.produce_table_arm_d_sweep

# 5. Uncertainty (needs R + opera, see Setup)
python -m experiments.run.uncertainty.run_aemo_uq_replay
python -m experiments.produce.produce_table_t5
python -m experiments.produce.produce_figure_f4

# 6. Escalation
python -m experiments.run.uncertainty.run_aemo_escalation_sweep
python -m experiments.produce.produce_table_t6
python -m experiments.produce.produce_figure_f5

# Comparison with Team 42
python -m experiments.produce.produce_table_t7
python -m experiments.produce.produce_figure_f6
```

**NHITS runs in parallel.** Each parallel NHITS job writes to
`lightning_logs/version_*` in its working directory, so give every job its
own working directory. Run them with
`PYTHONPATH=<repo> uv run --project <repo> python -m ...` so jobs don't
collide.

---

## Setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

This installs `drift_lab` (from `src/`) and `experiments` in editable mode,
plus `pytest` and `ruff`.

AgACI's online aggregation calls the R package opera (stage 5 only):

```bash
Rscript -e 'install.packages(c("remotes", "jsonlite")); remotes::install_github("Dralliag/opera@24070db")'
```

**Data:** the raw AEMO CSVs are committed under `data/raw/{SA1,NSW1}/`, so
there is no download step.

**What is tracked in git:**
- `results/runs.csv` (the experiment ledger) is tracked.
- `results/runs/` (per-run outputs) and `lightning_logs/` are gitignored.
- `docs/requirements/` (the project briefing and required-outputs PDFs) is
  gitignored.

---

## Repository layout

```
.
├── src/drift_lab/                    # the installable package -- implementation only, never writes to results/
│   ├── config.py                       paths, REGIONS, SEEDS, SPLIT (train 2018-19 / calibration Jan-Feb 2020 / test Mar 2020-), constants
│   ├── aemo/
│   │   ├── loader.py                   load(region) -> (train, calibration, test)
│   │   ├── deseasonalise.py            TRAIN-fitted seasonal profile: fit / apply
│   │   └── events.csv                  documented AEMO/AER events, Tier 1 / Tier 2
│   ├── synthetic/generator.py          make_series(kind, n, noise, seed) -> (y, changepoints)
│   ├── detection/                      DriftDetector: detect(stream) -> indices
│   │   ├── base.py                     contract + detect_with_river()
│   │   └── adwin.py, kswin.py, page_hinkley.py
│   ├── forecasting/                    Forecaster: fit / predict / observe
│   │   ├── base.py                     contract
│   │   ├── seasonal_naive.py           lag-48 baseline (comparison only, not a candidate)
│   │   ├── xgboost_forecaster.py       recursive XGBoost on lag + calendar features
│   │   ├── dhr_arima.py                Fourier terms + SARIMAX errors
│   │   ├── nhits_forecaster.py         NHITS, 7 days in -> 1 day ahead (the selected forecaster)
│   │   └── nixtla_common.py            Forecaster <-> neuralforecast adapter
│   ├── adaptation/                     Adapter: adapt(changepoints, model, data) -> model
│   │   ├── base.py                     contract
│   │   ├── retrain_accounting.py       shared retrain count / samples / wall clock
│   │   ├── never_retrain.py            arm A
│   │   ├── retrain_every_30_days.py    arm B
│   │   ├── retrain_using_full_history.py   arm C
│   │   └── retrain_using_recent_window.py  arm D (30 / 60 / 120 / 180 days)
│   ├── uncertainty/                    UncertaintyQuantifier: quantify -> (lower, upper, escalate); observe(actual)
│   │   ├── base.py                     contract
│   │   ├── fixed_split_conformal.py    one radius, calibrated once
│   │   ├── agaci.py                    AgACI (online aggregation of ACI candidates)
│   │   ├── opera_boa.py, opera_boa.R   AgACI's aggregator via R opera
│   │   └── conformal_common.py         shared conformal pieces
│   └── evaluation/evaluation.py        EVERY metric (forecast, detection, regimes, coverage, escalation, periodicity)
│
├── experiments/                      # the only code that runs things or writes to results/
│   ├── run_harness.py                  record_run(): arrays -> runs.csv rows (metrics via evaluation.py)
│   ├── results_io.py                   runs.csv, hash folders, output paths, calibration_path()
│   ├── trigger_detectors.py            trigger detectors (ADWIN, Page-Hinkley) and their split_ids
│   ├── run/
│   │   ├── detection/                  stages 1-2
│   │   │   ├── check_seasonal_residuals.py, check_residual_generalisation.py   data QC
│   │   │   ├── run_fine_tune_on_synthetic.py      detector config sweep (synthetic only)
│   │   │   ├── post_tune_detector_configs.py      reads the sweep winners
│   │   │   ├── run_post_tune_on_synthetic.py      -> T1, F2 synthetic
│   │   │   ├── run_aemo_detectors_standard_half_hourly_post_tune.py   -> T2, F2 AEMO
│   │   │   ├── run_detector_changepoints.py       accepted change points for the arms
│   │   │   ├── check_detection_periodicity.py     do alarms follow the clock/calendar?
│   │   │   ├── standard_stream_common.py, detection_artifacts.py   shared helpers
│   │   │   └── ⚠ run_pre_tune_on_synthetic.py, run_aemo_detectors_{raw,standard_daily}_*.py,
│   │   │       run_aemo_detectors_standard_half_hourly_pre_tune.py, run_synthetic_generator.py
│   │   │                                          not in the report pipeline (retired comparisons / visual check)
│   │   ├── forecasting/                stage 3
│   │   │   ├── run_aemo_baselines.py              seasonal naive, XGBoost, DHR + ARIMA -> F1
│   │   │   ├── run_aemo_nhits.py                  NHITS -> F1
│   │   │   ├── run_aemo_holdout_selection.py      forecaster selection on Nov–Dec 2019 held out from TRAIN
│   │   │   ├── run_aemo_calibration_forecasts.py  calibration residuals (+ selection on the Calibration window)
│   │   │   └── ⚠ run_aemo_nhits_retrain3mo_kswin_{nsw,sa}.py   commented out, superseded by arm D
│   │   ├── adaptation/                 stage 4
│   │   │   ├── run_aemo_adaptation_arms.py        arms A-D -> T3, T4, F3
│   │   │   └── rescore_adaptation_runs.py         regimes + pinball loss from saved forecasts
│   │   └── uncertainty/                stages 5-6
│   │       ├── run_aemo_uq_replay.py              intervals -> T5, F4
│   │       ├── add_rolling_coverage.py            one-off back-fill of F4's rolling coverage
│   │       └── run_aemo_escalation_sweep.py       threshold sweep -> T6, F5
│   └── produce/                        one script per deliverable, reads runs.csv + saved files only
│       ├── produce_table_t1_post_tune.py ... produce_table_t7.py
│       ├── produce_figure_f1.py ... produce_figure_f6.py   (F2 has a synthetic and an AEMO producer)
│       ├── produce_table_forecaster_selection.py
│       ├── table_t1_common.py, table_t2_common.py, table_adaptation_common.py,
│       │   figure_f2_*_common.py, chance_baseline.py        shared logic
│       ├── table_image.py, figure_text.py                    shared rendering / text wrapping
│       └── ⚠ produce_adwin_results.py, produce_kswin_results.py, produce_aemo_clock_diagnostics.py,
│           produce_agaci_width_diagnostic.py                 pre-refactor summaries / diagnostics, not deliverables
│
├── results/
│   ├── runs.csv                        the ledger (append-only, tracked)
│   ├── runs/<config_hash>/             config.json + per-run files (forecasts, curves, intervals) -- gitignored
│   ├── runs/changepoints/              accepted change points per detector x region
│   ├── tables/tableN/<detector>/[<model>/[<signal>/]]   T1-T7 (+ forecaster_selection/, detection_periodicity/, QC)
│   └── figures/figureN/<detector>/[<model>/[<signal>/]] F1-F6 (+ QC figures)
│
├── tests/                            # pytest
├── data/raw/{SA1,NSW1}/              # committed source CSVs
├── notebooks/                        # research scratch, not imported anywhere
├── TEAM_DECISIONS.md, SHARED_DECISIONS.md   # recorded decisions (team / shared with Team 42)
└── pyproject.toml
```

`⚠` marks files that are kept but are not part of the report pipeline. Their
earlier `runs.csv` rows remain.

---

## The interfaces

```python
detector(stream) -> changepoint indices
adapter(changepoints, model, data) -> updated model
uq.quantify(point_forecast, calibration_residuals) -> UQResult(lower, upper, escalate)
uq.observe(actual)
```

`Forecaster` (`fit` / `predict` / `observe`) completes the set. Every model
shares it, so any adapter can retrain any model.

**There is no registry.** Import the class you need and create it directly.

| Contract | File | Implementations |
|---|---|---|
| `Forecaster` | `forecasting/base.py` | `SeasonalNaive`, `XGBoostForecaster`, `DHRArima`, `NHITSForecaster` |
| `DriftDetector` | `detection/base.py` | `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` (river) |
| `Adapter` | `adaptation/base.py` | arms A–D, see the table below |
| `UncertaintyQuantifier` | `uncertainty/base.py` | `FixedSplitConformal`, `AgACI` |

**Adaptation arms:**

| Arm | Policy |
|---|---|
| A | Never retrain (control) |
| B | Retrain every 30 days on the last 30 days (scheduled control) |
| C | Retrain on each accepted drift alarm, on all history |
| D | Retrain on each accepted drift alarm, on the last 30 / 60 / 120 / 180 days (180 reported) |

**Accepted alarms** are what remains after a 14-day refractory filter.

**The uncertainty loop:**
- `quantify` issues one 90% interval.
- `observe` feeds back that half-hour's actual value. Fixed split conformal
  ignores it. AgACI updates its candidates and weights from it.
- The `escalate` flag is set when the escalation signal exceeds the
  threshold.

**Escalation signals:**
- interval width;
- a nonconformity score, the lagged rolling mean |error| over 30 minutes,
  1 day or 7 days.

---

## Evaluation: the one place metrics are computed

`src/drift_lab/evaluation/evaluation.py` is the **only** module that
computes a metric. `tests/test_evaluation.py::test_no_metric_code_outside_evaluation_module`
enforces this: it fails if `.rolling(` or `.ewm(` appears anywhere else in
`src/drift_lab/` or `experiments/`.

| Section | What it covers |
|---|---|
| Forecast | MAE, 7-day rolling MAE |
| Synthetic detection | delay, false alarms per 10,000, missed detections |
| AEMO events | `match_unmatch`, event precision / recall, 14-day refractory |
| Regimes | `assign_timestamp_global_regime` labels each TEST timestamp as pre-drift, drift (event + 7-day buffer) or post-drift |
| Uncertainty | coverage, width, worst 24h coverage, pinball loss |
| Escalation | review rate, retained coverage, escalation curves, operating points, nonconformity score |
| Periodicity | alarm gaps, gap CV, Rayleigh phase concentration, random-alarm baseline |

**AEMO alarm matching:** documented events are not complete ground truth.
A detection outside every event window is labelled `Unmatch`, not counted
as a false positive.

---

## The experiment ledger

`experiments/run_harness.py::record_run(...)` is the only function that
writes to `results/runs.csv`. A `run_*.py` script produces the arrays and
`record_run` turns them into metric rows via `evaluation.py`.

`runs.csv` is long format: one row per run × metric × regime.

| Column | Meaning |
|---|---|
| `group` | baseline / detection / adaptation / uncertainty / escalation |
| `method` | model, detector, or `<model>/<arm>` |
| `dataset`, `region`, `seed` | `seed` is empty for deterministic models |
| `config_hash` | hash of the run's settings; `results/runs/<config_hash>/config.json` holds them |
| `split_id` | the experiment / data-partitioning scheme. Detector-specific stages name the detector, e.g. `aemo_uq_page_hinkley_half_hourly_v1` |
| `metric_name`, `metric_value`, `regime` | `regime` is `full`, `pre_drift`, `drift` or `post_drift` |
| `n_retrains`, `train_samples`, `wall_clock_s`, `timestamp` | run bookkeeping |

**Rules:**
- **`runs.csv` is append-only.** A rerun adds rows, and producers take the
  latest row per run key.
- **Never edit or delete rows.**
- **Old split_ids stay readable.** ADWIN's earlier uncertainty and
  escalation rows keep their original split_ids. `trigger_detectors.LEGACY_SPLIT_IDS`
  maps them to the current names, so producers still find them.

**Per-run outputs** go in `results/runs/<config_hash>/`:
- forecasts;
- rolling-MAE curves;
- calibration residuals;
- intervals;
- escalation curves.

The next stage finds these through the hash recorded in `runs.csv`.

---

## Tests

```bash
uv run python -m pytest -q
RUN_BASELINE_REPRO=1 uv run python -m pytest tests/test_xgboost_forecaster.py   # opt-in, slow
```

---

## Where do I put my code?

| You're building... | File goes in |
|---|---|
| A forecasting model | `src/drift_lab/forecasting/<model>.py` (subclass `Forecaster`) |
| A drift detector | `src/drift_lab/detection/<detector>.py` (subclass `DriftDetector`) |
| An adaptation arm | `src/drift_lab/adaptation/<arm>.py` (subclass `Adapter`) |
| A UQ method | `src/drift_lab/uncertainty/<method>.py` (subclass `UncertaintyQuantifier`) |
| A metric | `src/drift_lab/evaluation/evaluation.py`, re-exported from `evaluation/__init__.py` |
| An experiment | `experiments/run/<stage>/run_<name>.py`, logging through `record_run` |
| A table or figure | `experiments/produce/produce_<name>.py`, reading `runs.csv` and saved files only |

**Rules:**
- **One implementation per file.**
- **Don't change a `base.py` on your own.** A change to a contract is a team
  decision.
- **`src/drift_lab/` never runs anything or writes to `results/`.**
- **Metrics only come from `evaluation.py`.**
- **Each stage reads the latest saved output of the stage before it.** Never
  recompute an earlier stage's result.

---

## Further reading

- `docs/requirements/`: project briefing, required outputs (T1–T7, F1–F6)
  and sprint plan.
- [`TEAM_DECISIONS.md`](TEAM_DECISIONS.md),
  [`SHARED_DECISIONS.md`](SHARED_DECISIONS.md): why the codebase looks like
  this.
