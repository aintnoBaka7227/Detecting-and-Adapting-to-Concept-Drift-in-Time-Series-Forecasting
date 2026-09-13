# Design decisions

This file explains the main design decisions in the codebase and why each one was chosen. It is organised by component. [`README.md`](README.md) explains how to use the codebase; this file explains why it is structured this way, so future changes remain consistent.

[`docs/refactor.md`](docs/refactor.md) is the original refactor plan. Some names in that document, including `_harness.py` and registries, are now out of date. Follow this file wherever the two documents differ.

---

## Cross-cutting decisions (apply everywhere in `src/drift_lab/`)

**No registries.** Each `run_*.py` script imports and creates the class it needs directly, such as `XGBoostForecaster(max_depth=4)` or `ADWINDetector(delta=0.002)`. There is no `get_forecaster("xgboost")`-style lookup. This lets a team member add a model, detector or adaptation arm in its own file without also editing a central list, which reduces shared-file changes and merge conflicts.

**One implementation per file.** Each model or detector belongs in its own file, even when an implementation is small. This makes each class easier to find, review and test. It also keeps ownership clear when several team members are working in parallel.

**Two underscore conventions apply in different places:**

- **Instance attributes** such as `self._history`, `self._model` and `self._nf` use an underscore for private or fitted state. `experiments/run_harness.py::config_of()` reads `vars(obj)` and keeps only attributes without an underscore. This means constructor settings appear in `config_hash`, while fitted arrays and model state do not. The same hyperparameters therefore produce the same configuration identity across runs.
- **Files and functions in `experiments/`** do not use a leading underscore. Use names such as `run_harness.py`, `results_io.py` and `label_regimes` so their purpose is clear.

**`src/drift_lab/` is for implementation only.** It must not contain working `if __name__ == "__main__":` blocks, write files, or refer to `"results/"`. Experiment scripts and table or figure generation belong in `experiments/`.

**Pass data in memory within each stage.** Each function passes its result directly to the next. For example, `aemo.loader.load(region)` returns the three splits without writing an intermediate processed CSV. The only file used between experiment and production stages is `results/runs.csv`: `run_*.py` scripts append results and `produce_*.py` scripts read them. This gives the pipeline one clear hand-off point and prevents a later step from accidentally reading an older intermediate file.

**Keep comments precise and minimal.** Comments should explain a useful, non-obvious reason rather than repeat the code.

**Package naming.** The package is named `drift_lab` (pip name `drift-lab`), and AEMO-specific data code is under `aemo/`. These names clearly describe their purpose.

---

## `config.py`

`config.py` is the single source of truth for shared values, preventing the same setting from being repeated across files:

- `REGIONS = ("SA1", "NSW1")` — every required figure and table covers both regions.
- `SEEDS = (1, 2, 3, 4, 5)` — all experiments use the agreed seeds so results are comparable.
- `SPLIT` — train (2018–2019), calibration (January–February 2020), and test (March 2020 onwards). These frozen boundaries are imported wherever needed.
- `SEASON_LENGTH = 48`, `ROLLING_WINDOW_DAYS = 7`, and `REGIME_DRIFT_MARGIN = 48 * 7`. The seven-day drift margin is an assumption and can be adjusted if it is too narrow to appear clearly in the rolling-MAE curve.
- `DOCUMENTED_EVENTS_CSV` lives in `src/drift_lab/aemo/` because it is a pipeline input, not temporary notebook output.

---

## `aemo/` — AEMO demand data

- **One file, one function:** `loader.py` reads, cleans, standardises and splits the data. `load(region)` returns `(train, calibration, test)` in memory, and no other module should create its own date split. Keeping these operations together means a change to a cleaning rule or split boundary is applied consistently everywhere.
- **`clean()` always prints the full audit.** It does not have a `verbose=False` option. The audit is inexpensive and makes missing values, duplicates or other data problems visible whenever the loader runs, instead of relying on a team member to remember to enable logging.
- **`standardise()` resamples to a 30-minute grid and stops there.** Raw MW/$ values are not clipped, replaced or imputed. Missing intervals become explicit `NaN` rows. Imputation is handled as a modelling decision, so every downstream model receives the same honestly gapped data and can decide how to handle a missing lag.
- **`events.csv`** keeps the 34 rows and `tier` and `tier_reason` columns from the team's `events.xlsx`. It is the agreed source for detector–event matching, with tiering rules in `docs/event_tiering_criteria.md`.

---

## `synthetic/` — the drift benchmark

- **`make_series(kind, n, noise, seed)`** returns `(y, changepoints)` in memory. Regenerating it when needed ensures the data matches the current generator.
- **The shape of `changepoints` depends on `kind`:** `none` → `[]`, `sudden` → `[cp]`, `recurring` → `[cp1, cp2]`, and `gradual` → `[start, end]`. This matches `evaluation.evaluate_detections`; gradual drift is an interval because it develops over time.
- **Changepoint positions:** `sudden` and `gradual` begin at `n // 4`, leaving enough post-drift data for evaluation. `recurring` changes at `n // 3` and `2n // 3`, representing A → B → A.
- **`seed` controls only the noise.** Changepoint locations remain fixed for each `kind` and `n`, allowing `split_id` to record the actual changepoints.

---

## `forecasting/` — Forecaster implementations

- **`Forecaster.observe(y)`** adds known calibration history without retraining. Its default is a no-op. `SeasonalNaive` and `XGBoostForecaster` use it to extend lag history across the train → calibration → test boundary. This allows their first test forecasts to look up the required earlier values without changing model weights. `DHRArima` does not need stored lag history because its Fourier phase comes from the timestamp index (`X.index`).
- **Each `run_*.py` script defines whether a model is frozen or walk-forward.** `record_run` only records the supplied results; it does not decide what information a model can observe. In `run_aemo_baselines.py::predict_input()`, `seasonal_naive` receives real test demand because lag-48 forecasting is defined using yesterday's observed value. `xgboost` and `dhr_arima` receive no test target values and recursively use their forecasts after calibration. This makes the evaluation protocol explicit and lets F1 show how frozen models degrade without test-period feedback.

---

## `detection/` — DriftDetector implementations

- `adwin.py`, `kswin.py` and `page_hinkley.py` wrap `river` detectors. They share `detect_with_river()` in `detection/base.py`, which sends one value at a time through a fresh detector and records flagged indices. This keeps streaming behaviour consistent.
- **KSWIN's `seed` is its internal reservoir-sampling seed.** It stays fixed at the `river` default of 42. The outer experiment seed only controls the noise produced by `make_series`. Keeping the two seeds separate means a changed result can be traced either to a different input series or to detector sampling, rather than changing both at once.
- **Page-Hinkley's current defaults are the agreed tuning baseline.** They reproduce the notebook settings and currently produce about 311–312 false alarms per 10,000 observations, including for `"none"`. This gives the research task a measured starting point to improve after tuning.

---

## `adaptation/` — Adapter implementations

`uncertainty/` currently contains only the frozen `base.py` contract and a worked example. No arm is implemented because this work belongs to a later sprint.

`adaptation/` contains its first arm: **`RetrainUsing3MonthWindows`** (`retrain_using_3_month_windows.py`). It implements the Step 5 approach of retraining after detected drift with recent data.

- **Use the frozen `adapt(changepoints, model, data)` signature as written.** The caller supplies history available up to the current point, and the arm decides whether to retrain and how much history to use. This keeps all adapters interchangeable.
- **The retraining window ends on the latest changepoint date.** `adapt()` uses the previous `window_months` calendar months (three by default), ending at `data.index[max(changepoints)]`. Any rows after that date are removed, even if the caller supplied a larger block containing later observations. This places the no-leakage check inside the adapter instead of depending on every caller to slice the data correctly. If a call contains several changepoints, the latest one sets the endpoint, as documented in `test_retrain_using_3_month_windows.py`.
- **The window may cross earlier split boundaries.** It can include training or calibration data because those values were already available at the changepoint. It never includes later observations.
- **The arm has no retraining cooldown.** Every call containing at least one changepoint retrains the model. Detection frequency is managed by the detector input and experiment design. This keeps the adapter responsible only for how retraining is performed and allows the same policy to work with different detectors and detection cadences.
- **`data` must have a `DatetimeIndex`.** The target must be its only column or be selected through `target_column`. This matches existing `Forecaster.fit(X, y)` calls and works with `SeasonalNaive`, `XGBoostForecaster`, `DHRArima`, and `NHITSForecaster` without model-specific feature columns.
- **`retrain_count` is a read-only property backed by `self._retrain_count`.** The underscore keeps this run outcome out of `config_hash` and `config.json`.

**For AEMO experiments, detectors use the demand stream itself, with optional daily aggregation.** A detector does not need a forecasting model's rolling-error curve before it can run. This keeps each adaptation experiment self-contained and avoids depending on a separate baseline run and its saved curve files.

Raw half-hourly demand contains strong daily and weekly patterns, which can be incorrectly flagged as drift. This becomes costly when every flag triggers retraining. **`aemo/deseasonalise.py`** prepares the signal watched by the detector, while forecasting and retraining remain on the original half-hourly data. It belongs in `aemo/` because it shapes AEMO data rather than calculating metrics or implementing a pipeline interface.

**`aggregate_daily_demand` is the standard daily aggregation function.** It calculates a mean only for complete days with exactly 48 half-hourly observations and validates inputs using `TypeError` and `ValueError`. Requiring a complete day prevents a partial day's mean from being treated as normal daily demand. **`remove_daily_weekly_profile`** is also kept in `aemo/deseasonalise.py`; it learns the expected demand for each weekday and half-hour from a separate historical `reference` period, then subtracts that profile. This produces a less seasonal detector input while keeping half-hourly resolution.

**`run_aemo_detectors_daily_frozen.py`** applies the same three detectors to `aggregate_daily_demand(test)`, using the frozen configuration each detector was given by `run_detector_budget_tuning.py`'s synthetic-only sweep rather than a plain class default, and matches against the region's *full* event catalogue — Tier 1 **and** Tier 2 — rather than Tier 1 only. Matching both tiers is what lets `produce_table_t2.py` report a Tier-1-specific unmatched count next to an overall one from a single run, instead of needing two differently-scoped runs to say both things. The detector receives only the processed test split, with no training or calibration warm-up. Its own `split_id` (`aemo_detect_daily_frozen_v1`) and `train_samples=0` prevent these rows from being grouped with full-stream or adaptation results. Each of the three detectors is run and logged as its own, independent `record_run` call — their detections are never pooled or unioned before matching, so a detection one detector raised can never "help" another detector's match count.

An earlier `run_aemo_detectors_daily.py` (plain class defaults, Tier-1-only matching, `split_id="aemo_detect_daily_test_v1"`) served as the one-off sanity check for how much daily aggregation cuts detection volume, before daily aggregation was adopted as the adaptation arm's default detector input. That question is answered and recorded here; the script itself was removed once `run_aemo_detectors_daily_frozen.py` existed to replace it as the actual source for Table T2.

**`produce_figure_f2_aemo_daily_frozen.py` is a sibling of `produce_figure_f2_aemo.py`, not a replacement.** Redirecting the original script to `run_aemo_detectors_daily_frozen.py`'s data instead of adding a new script would have orphaned `run_aemo_detectors.py` — nothing else reads its `aemo_detect_full_v1` rows. Both scripts and both their `runs.csv` sources are kept: the original still plots default-config detections against raw half-hourly demand from `run_aemo_detectors.py`; the new one plots frozen-config, both-tier-matched detections against `aggregate_daily_demand` — the exact series the detector actually saw, not a separately-computed resample. Output filenames differ (`f2_detection_daily_frozen_<method>_<region>.png` vs. `f2_detection_<method>_<region>.png`) so neither run can overwrite the other's figures. Visual styling is identical between the two on purpose, so a reader can compare them directly — only the title text and data source differ.

**`run_aemo_detectors.py` was later brought up to the same both-tier matching as `run_aemo_detectors_daily_frozen.py`** — its `tier1_events()` helper became `region_events()`, matching the full catalogue (Tier 1 and Tier 2) instead of Tier 1 only, so its `runs.csv` rows carry the same `n_matched_t1`/`n_matched_t2`/`event_recall` breakdown. It's still the raw half-hourly, default-hyperparameter run (that hasn't changed) and it's still Figure F2's only source, not Table T2's.

**Both F2 scripts' palette (light-blue raw demand, blue smoothed trend, black event lines, orange period shading, bold green `MATCHED`/red `FALSE ALARM` labels) matches the original briefing's Figure D exactly, on request.** The wording deliberately does not: Figure D is an illustrative mockup where "true changepoint" and "true gradual drift" assume known ground truth. AEMO's documented events are explicitly not ground truth — a project-wide rule stated repeatedly elsewhere in this file — so the black event line keeps its existing event-name annotation instead of "true changepoint", and an unmatched detection reads "FALSE ALARM / no known event" rather than borrowing Figure D's "true" language. Same colors and layout, wording that stays honest about what AEMO data actually is. One known limitation carried over from the original F2 script, not introduced by this change: text labels are placed at fixed data-derived positions with no collision detection, so a `FALSE ALARM` note and a `MATCHED` note can visually overlap when they land close together in time (seen on `page_hinkley`/NSW1's daily-frozen figure) — a real layout gap, not something worth a bespoke fix without seeing it matter for the actual report.

**Unmatched detections get a full-height dashed line each (`ax.vlines`), matching Figure D and the matched-detection style, on purpose — even though this makes the raw-half-hourly-stream figures (`run_aemo_detectors.py`, hundreds-to-thousands of false alarms) look like a solid red barcode that drowns out the demand curve.** A density-based fallback (short lines, or the old top-of-axis rug, past some detection-count threshold) was considered and explicitly declined — every figure uses the identical convention rather than switching styles depending on how many detections happen to fall in a given run. The daily-aggregated figures (`run_aemo_detectors_daily_frozen.py`, tens of false alarms at most) are the ones this styling was designed to read well on; the raw-stream ones inherit it for consistency, not because they're expected to be legible at that density.

---

## `evaluation/` — the one place metrics are computed

- **Sahar's implementation is the shared evaluation module.** One team-agreed implementation ensures every experiment calculates metrics consistently. The only code-level addition is one `# noqa: TRY004` comment.
- **A test enforces this rule.** `tests/test_evaluation.py::test_no_metric_code_outside_evaluation_module` checks for `.rolling(` or `.ewm(` outside `evaluation.py`, preventing slightly different metric calculations in separate scripts.
- **`evaluate_detections` handles each drift type:** `none` treats all detections as false alarms; `sudden` and `recurring` match point events within a tolerance; and `gradual` matches `[start, end]`, measuring delay from `start`.
- **AEMO evaluation uses documented-event matching.** `match_unmatch` and `calculate_event_metrics` evaluate matches. `build_event_windows`, `assign_regime` and `calculate_regime_metrics` describe where detections occur in an event's lifecycle. `evaluate_aemo_detections` runs all five. An unmatched detection is `Unmatch`, not a false alarm, because the event list is not complete ground truth.
- **Point and interval events use separate windows.** `POINT_WINDOW` applies to a single-date event, while `INTERVAL_GRACE` applies after a multi-day event ends. A point event needs a short window after one date, whereas a long event already covers a date range and only needs a grace period after its end. Keeping both settings configurable allows the team to adjust them without changing the matching logic.
- **`assign_regime` uses the global priority `drift > pre_drift > post_drift`.** All event windows are checked together rather than processing one event at a time. For example, if E1 is on 1 March and E2 is on 10 March, a detection on 5 March falls in E1's aftermath and E2's lead-up. It is labelled `pre_drift` and attributed to E2 because `pre_drift` has priority over `post_drift`. If several windows with the same priority match, the relevant event-order rule in the implementation selects the attribution. The overlapping-window behaviour is covered by `test_assign_regime_overlapping_events`.
- **Use `event_id="unassigned"` when no specific event window supplies the label.** This applies to `post_drift` detections and detections before any event. A literal text category works consistently in `groupby`, joins and sorting. It also prevents summary rows from being silently dropped through the default handling of `None` or `NaN` keys.
- **Known gap: `post_drift` has no upper limit.** After an event begins, later detections remain `post_drift` unless another drift or pre-drift window claims them. `build_event_windows` calculates `post_drift_end`, but `assign_regime` does not yet use it. The `xfail` test `test_assign_regime_detection_outside_all_windows_is_pre_drift` keeps this visible until the team decides whether to close the window.
- **A day-precision event covers the full calendar day.** Its drift window is `[start_of_day, start_of_next_day)`, ensuring a detection at any time that day counts as occurring during the event.

---

## The experiment harness and `results/runs.csv`

This section explains the design used by every table and figure.

### The core principle

`experiments/run_harness.py::record_run(...)` is the only function that appends rows to `runs.csv` and turns raw arrays into metrics through `evaluation/`. It does not fit a model or run a detector. A `run_*.py` script first forecasts or detects, then sends its arrays and run information to `record_run`. A `produce_*.py` script reads stored results and groups them without recalculating metrics. The flow is: run → record → produce. This ensures a metric is calculated once and the same stored result is used by every table and figure.

**`experiments/run/` is split into `forecasting/` and `detection/`.** The rule is mechanical, not a judgment call: whichever `record_run` group the script produces decides the folder — `forecast=(...)` goes in `forecasting/`, `detection=(...)` goes in `detection/`. This kept the split unambiguous even for a script that uses both a forecaster and a detector, such as the adaptation-arm scripts (`run_aemo_nhits_retrain3mo_kswin_*.py`): they trigger a retrain from a detector, but the thing actually being scored and logged is the forecast, so they live under `forecasting/`. `run_harness.py` and `results_io.py` stay directly under `experiments/`, not inside either subfolder, since they're shared plumbing every script imports rather than an experiment themselves.

### Where experiment output lives — and why nowhere else

Each experiment writes three types of output to fixed locations:

```
results/
├── runs.csv                 the ledger — scalar metrics only, one row per
│                             (method × dataset × region × seed × metric × regime)
├── runs/<config_hash>/
│   ├── config.json          the actual hyperparameter values behind that hash
│   └── curve_<dataset>_<region>_<seed>.csv    the full rolling-MAE curve, one
│                             per forecast run — not squeezed into runs.csv
└── figures/                  every table/figure a produce_*.py script emits
```

- **`runs.csv` stores scalar metrics only:** `mae`, `rolling_mae_7d_mean`, `rolling_mae_7d_max`, `detection_delay`, `false_alarms_per_10000`, `missed_detections`, and `n_detections`. Time-indexed curves are stored separately because each long-format row holds one scalar value.
- **Figure scripts read saved curve files.** `produce_figure_f1.py` reads the exact rolling-MAE curve saved by the experiment through `results_io.dump_curve()`. The path is keyed by `config_hash`, `dataset`, `region`, and `seed`, and `results_io.curve_path()` creates the same path on both the write and read sides. The figure therefore does not recalculate rolling MAE from forecasts. `produce_table_t1.py` only groups scalar values already stored in `runs.csv`.
- **`produce_figure_f1.py` no longer writes a full-test-period, one-figure-per-region output.** That figure (markers only, no event names, ~4 years squeezed onto one x-axis) was redundant with the per-year figures, which show the same curves in more legible detail with event names attached. `plot_full_period`, `annotate_events`, and `format_full_period_axis` were removed along with it rather than left as dead code behind an unused call.
- **`produce_*.py` scripts write tables to `results/tables/` and figures to `results/figures/`.** They use `TABLES_DIR` and `FIGURES_DIR` from `experiments/results_io.py`, keeping output organised and reproducible.
- **`run_*.py` scripts do not write result files directly.** They call `record_run(...)`, which manages `append_runs`, `dump_config`, and `dump_curve`. A script should not add its own `df.to_csv(...)` result path. Using one writer gives every experiment the same schema, folder layout, configuration record and curve naming rule.
- **Only `runs.csv` is tracked in git.** Curve files and figures can be regenerated, so `runs/` and `figures/` remain gitignored.

These locations make results easy to trace through the scalar ledger, saved configuration, and full curve where required.

### Column-by-column reasoning

| Column | Why it's there / non-obvious rule |
|---|---|
| `seed` | Use `None` (stored as `NaN`) for a deterministic method. A stochastic model records its actual seed, clearly separating random repeats from deterministic results. |
| `config_hash` | Calculated as `sha1(json.dumps(config, sort_keys=True, default=str))[:12]`, where `config_of(obj)` includes only public attributes. Private fitted state therefore does not change the hash. |
| `split_id` | Identifies the data-partitioning scheme, not a batch of result rows, and is always required. For synthetic detection, it includes the actual changepoints (`synth_n{N}_cp{cp1-cp2-...}`), preventing results from different generator layouts from being grouped together under the same dataset name. |
| `regime` | Records whether a metric covers the full series or a pre-drift, drift, or post-drift period. |
| `n_retrains`, `train_samples`, `wall_clock_s` | Records data and computation used, allowing retraining cost and runtime to be considered with accuracy. |

### The regime decision

The `regime` column follows these agreed rules:

- **Detection, synthetic:** use `regime="full"`. Table 1 reports delay, false alarms and missed detections across the full synthetic series.
- **Detection, AEMO:** use `regime="full"` for now, and record one match-summary row (`n_detections`, `n_matched`, and so on). AEMO has documented events, but these are not complete ground-truth changepoints known beforehand. The detector first identifies possible changes, and its timestamps are compared with documented events afterwards. The `full` value therefore describes the detector run without treating its own outputs as known truth. `evaluation.py` already supports going further than `full` here, though nothing calls it yet: `calculate_regime_metrics` computes the exact same match/unmatch counts and precision as `calculate_event_metrics` does for `full`, just split per `assign_regime`'s `drift`/`pre_drift`/`post_drift` label instead of pooled across the whole run. So moving AEMO detection rows from one `full` row to a per-regime breakdown is a case of `record_run` calling an existing function differently, not inventing a new metric.
- **Forecast rows:** passing `changepoints=[...]` to `record_run` divides metrics into `pre-drift`, `drift`, and `post-drift`. Without `changepoints`, it records one `regime="full"` set. For synthetic data, changepoints may come from the generator's known truth. For AEMO, they come from a detector and must not be described as ground truth. The drift band uses the assumed seven-day `config.REGIME_DRIFT_MARGIN`, allowing rolling MAE to be examined around a detected change. The value can be adjusted if the band is too narrow to show the model's response.

### A bug worth remembering (fixed, but easy to reintroduce)

- **`curve_path(..., seed=None)` uses the token `"none"`.** Empty seed cells in `runs.csv` are read by pandas as `NaN`, so the explicit `seed_token = "none" if seed is None else str(seed)` must be used in both `dump_curve` and `produce_figure_f1.py` to keep paths consistent.
- **`runs.csv` is append-only.** Re-running a script adds rows rather than overwriting earlier results, including after an interrupted run. This preserves the experiment history. Where duplicate method and region results exist, `produce_*.py` scripts take the latest row, as shown by `rows.iloc[-1]` in `produce_figure_f1.py::load_model_curves()`.

---

## Testing conventions

- **Use one test file for each package or module**, such as `test_detectors.py`, `test_dhr_arima.py`, and `test_evaluation.py`. This mirrors `src/` and makes tests easy to find.
- **The grep guard** (`test_no_metric_code_outside_evaluation_module`) checks that metric code stays in `evaluation.py`.
- **`RUN_BASELINE_REPRO=1`** enables tests that reproduce committed AEMO baseline CSVs for `XGBoostForecaster` and `DHRArima`. They are skipped by default because they take several minutes. If approved research changes the frozen defaults, the team should agree on the new baseline and regenerate the reference CSV.

---

## Per-method research notes (`docs/methods/`)

As required by `docs/PG-S2-41-method-tickets.md`, each method research task creates a `docs/methods/<name>.md` note. It records the architecture, current wrapper behaviour, suitability for synthetic drift and AEMO, and at least two peer-reviewed references from 2016 onwards. Keeping these notes in the repository preserves the research beyond a pull request or ticket comment.

---

## Deferred — needs the team before building

- **T2 documented-event matcher — done.** It is implemented through `match_unmatch` and `calculate_event_metrics` in `evaluation.py`. The `evaluation/` section explains its matching and regime rules.
- **`label_regimes` ownership.** It currently lives in `experiments/run_harness.py`. The team still needs to decide whether it should move to `evaluation.py` so forecasting regimes and detection event windows use one shared timeline implementation.
- **Recovery regime.** Between two changepoints, forecast error may return to its pre-drift level, especially for recurring drift. This period is currently labelled `post-drift`. Adding a recovery label requires a team decision and a change inside `label_regimes`, but not a schema change.
