# Team 41 Implementation Decisions

This document records Team 41's implementation decisions for the `drift_lab` pipeline. Joint settings are recorded in a separate shared-decision file.

## Project pipeline

`Data → Detection (synthetic, then AEMO) → Forecasting → Adaptation → Uncertainty → Escalation`, then the T7/F6 comparison with Team 42.

`README.md` lists the exact run order. Team 41's own results chain is Page-Hinkley + NHITS (T1–T6, F1–F5). T7/F6 are the shared experiment with Team 42 and use XGBoost.

Evaluation, result logging, testing and quality assurance apply across all pipeline stages.

---

# Cross-cutting decisions

## Result storage

| Result type | File/location | Why |
|---|---|---|
| Official scalar metrics | `results/runs.csv` | Provides one long-format ledger for every table. |
| Run configuration | `results/runs/<config_hash>/config.json` | Connects results to the exact preprocessing, model and detector settings. |
| Detection timestamps | `results/runs/<config_hash>/detections_<dataset>_<region>_<seed>.csv` | Timestamp-level results do not belong in the scalar ledger. |
| Retraining timestamps | `results/runs/<config_hash>/retrains_<dataset>_<region>_<seed>.csv` | Figure 3 needs the exact retraining times for each adaptation arm. |
| Forecast/error curves | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Figures read the exact curve produced by the experiment. |
| Accepted change points | `results/runs/changepoints/<detector>_standard_half_hourly_post_tune_<region>.csv` | One accepted-alarm list read by arms C and D, T7 and F6. |
| Calibration residuals | `results/runs/<config_hash>/calibration_forecast_aemo_<region>_<seed>.csv` | Calibrate both conformal methods; found through `results_io.calibration_path()`. |
| Prediction intervals | `results/runs/<adaptation hash>/intervals_aemo_<region>_<seed>.csv` | Both UQ methods side by side, read by F4 and escalation. |
| Escalation curves | `results/runs/<config_hash>/escalation_curve_aemo_<region>_<seed>.csv` | Review rate and retained coverage at every threshold, per regime. |
| Final tables | `results/tables/tableN/<detector>/[<model>/[<signal>/]]` | Separates report outputs from experiment records. |
| Final figures | `results/figures/figureN/<detector>/[<model>/[<signal>/]]` | Separates report outputs from experiment records. |
| Metric calculation | `src/drift_lab/evaluation/` | Gives every experiment one trusted metric implementation. |
| Run logging | `experiments/run_harness.py` | Enforces the shared result schema. |
| Result paths/readers | `experiments/results_io.py` | Keeps writer and reader filenames consistent. |

Additional rules:

- `results/runs.csv` is append-only.
- Producers select the latest completed row matching method, dataset, region, seed, configuration, split, regime and metric.
- Applicable forecast runs record `n_retrains`, `train_samples` and `wall_clock_s`.
- Track `results/runs.csv` and the report tables and figures; `results/runs/` is gitignored and regenerated.
- ADWIN's earlier uncertainty and escalation rows keep their original split_ids; `trigger_detectors.LEGACY_SPLIT_IDS` lets readers accept both. Never rename ledger rows.
- Known exceptions to the storage rules: tuning winners (`results/tables/fine_tune_on_synthetic_winners.csv`) and accepted change points have no `runs.csv` row or hash, and intervals are written into the adaptation run's folder rather than the UQ run's. An adaptation run's hash records the change-point filename, not its contents.

## Code organisation

| Decision | Location | Why |
|---|---|---|
| No central registry | Experiment scripts import implementations directly | Reduces shared-file merge conflicts. |
| One implementation per file | `forecasting/`, `detection/`, `adaptation/`, `uncertainty/` | Keeps ownership and testing clear. |
| Configuration identity | Public constructor settings define `config_hash`; fitted/private state does not | The same hyperparameters must identify the same configuration. |
| Reusable code | `src/drift_lab/` does not write results | Prevents incompatible output formats. |
| Experiment orchestration | `experiments/run/` | Makes the causal execution order explicit. |
| Report production | `experiments/produce/` | Tables and figures read recorded outputs only. |
| Shared constants | `src/drift_lab/config.py` | Prevents regions, seeds, splits or rolling windows from differing between scripts. |

---

# 1. Data

## 1.1 Dataset and splits

| Decision | Team 41 choice | Main file | Recorded result | Why |
|---|---|---|---|---|
| Dataset | AEMO `TOTALDEMAND` | `src/drift_lab/aemo/loader.py` | Dataset/region in `runs.csv` | Primary variable required by the project. |
| Regions | `SA1`, `NSW1` | `src/drift_lab/config.py` | `region` | Provides two demand systems with different scales and behaviour. |
| Period | 1 Jan 2018–31 Dec 2023 | `config.py` | `split_id` | Covers the required study period. |
| Resolution | Average native 5-minute demand to 30-minute intervals | `loader.py` | Passed in memory | Gives all forecasters one time grid. |
| TRAIN | 1 Jan 2018–31 Dec 2019 | `config.py` | `split_id` | Fits models and preprocessing. |
| Calibration | 1 Jan–29 Feb 2020 | `config.py` | `split_id` | Supplies pre-test history and calibration residuals. |
| TEST | From 1 Mar 2020 | `config.py` | `split_id` | Only period used for final AEMO scoring. |
| Missing intervals | Preserve explicit `NaN` rows | `loader.py` | Data-quality audit | Avoids hidden imputation. |
| Duplicates | Detect and report | `loader.py` | Data-quality audit | Prevents repeated values affecting results. |
| Time windows | Left-inclusive, right-exclusive on raw AEMO stamps: a day or event from `s` to `e` is `[s 00:00, (e+1) 00:00)`; no 00:30 shift | `loader.py`, `evaluation.py` | None | One team-wide rule for splits, events, days and adaptation. Each day window therefore holds the previous day's 23:30–24:00 interval and not its own. |
| TEST length | 67,249 half-hours, 1 Mar 2020 00:00 to 1 Jan 2024 00:00; no missing demand in any split | `loader.py` | Curve row count | TEST has no end date, so it keeps the final interval-end stamp. Check this count if a curve looks truncated. |
| Intermediate data | Pass in memory; no processed dataset CSV | Loader and runners | None | Avoids stale processed files. |

`load(region)` is the only function that creates TRAIN, Calibration and TEST.

## 1.2 Documented events

| Decision | Team 41 choice | Main file | Why |
|---|---|---|---|
| Catalogue | One frozen documented-event file | `src/drift_lab/aemo/events.csv` | All detector runs use identical reference events. |
| Event identity | Unique `event_id` | `events.csv` | Prevents ambiguous attribution. |
| Tiers | Preserve Tier 1 (4 events) and Tier 2 (29 events) | `events.csv` | Prevents the larger Tier 2 catalogue dominating Tier 1. |
| COVID | 16 Mar–17 May 2020 | `events.csv` | Includes the late-March demand change. |
| Five-Minute Settlement | Removed from the catalogue, so it is not a positive target | `events.csv` | It is a price-market rule change, not an expected demand shift. |
| Freeze rule | No changes after final detector runs begin | File version in `config.json` | Prevents changing targets after seeing results. |

---

# 2. Synthetic benchmark

## 2.1 Generator

| Decision | Team 41 choice | Recorded result | Why |
|---|---|---|---|
| Drift types | No drift, sudden, gradual, recurring | Dataset name/`split_id` | Covers the required drift behaviours. |
| Length | 20,000 observations | `config.json` | Long enough to measure false-alarm behaviour. |
| Noise | Standard deviation 1.0 | `config.json` | Keeps runs comparable. |
| Seeds | `1, 2, 3, 4, 5` | `seed` in `runs.csv` | Supports mean ± standard deviation. |
| Storage | Regenerate deterministically | Generator configuration | Avoids stale cached synthetic files. |
| No-drift truth | Empty changepoint list | `split_id` | Every detection is a measurable false alarm. |
| Sudden truth | One fixed point | `split_id` | Keeps delays comparable across seeds. |
| Gradual truth | Fixed start/end interval | `split_id` | Represents a transition rather than one instant. |
| Recurring truth | Two fixed points, A → B → A | `split_id` | Tests return to an earlier regime. |
| Seed effect | Changes noise only | `seed` + `split_id` | Keeps ground-truth locations fixed. |

## 2.2 Detector calibration

| Decision | Team 41 choice | Main file | Why |
|---|---|---|---|
| Tuning runner | Synthetic only | `experiments/run/detection/run_fine_tune_on_synthetic.py` | Prevents AEMO TEST tuning. |
| Final input frequency | Half-hourly, 17,520 observations/year | Runner configuration | Matches the final half-hourly residual stream. |
| False-alarm budget | ≤2 per simulated year | Evaluation module | Rejects continuous firing. |
| Missed-drift target | Zero | Evaluation module | Rejects insensitive settings. |
| Eligibility | On every seed: no false alarm on the no-drift series, at most 2 false alarms per simulated year on every row, no missed drift | `run_fine_tune_on_synthetic.py` | A config that fails one seed is not reliable. |
| Selection rule | Fewest false alarms per year, then shortest mean delay, then parameter order | `run_fine_tune_on_synthetic.py` | The false-alarm count decides first; delay only separates ties. |
| Frozen ADWIN | `delta=1e-8`, `clock=48`, `min_window_length=336` | `results/tables/fine_tune_on_synthetic_winners.csv` | Winner of the synthetic sweep. |
| Frozen KSWIN | `alpha=5e-5`, `window_size=1344`, `stat_size=96`, `seed=42` | Same file | Winner of the synthetic sweep. |
| Frozen Page-Hinkley | `alpha=0.9995`, `delta=0.0001`, `min_instances=20`, `threshold=600`, `mode=both` | Same file | Winner of the synthetic sweep, inside a grid whose bounds were informed by AEMO results (see below). |

| KSWIN internal seed | Fixed separately from generator seed | Detector configuration | Separates detector and input randomness. |
| Final validation | Rerun frozen settings | `run_post_tune_on_synthetic.py` | Produces official T1 inputs. |
| T1 producer | Read `runs.csv` | `produce_table_t1_post_tune.py` | T1 is not calculated manually. |

Detector settings are frozen after synthetic calibration and applied to AEMO unchanged.

**Disclosure:** the Page-Hinkley winner was picked on synthetic data, but its grid bounds (`alpha`, `threshold`, `delta`) were narrowed after seeing AEMO results, as recorded in `run_fine_tune_on_synthetic.py`. Report this alongside every Page-Hinkley result.

## 2.3 T1 — synthetic detection table

T1 is produced in the Synthetic Benchmark stage, before any AEMO detection result is accepted.

| T1 item | Required value | Existing evaluation support | Result location |
|---|---|---|---|
| Row identity | Detector × drift type | Supplied by the synthetic runner | `method`, `dataset`, `split_id` and `seed` in `results/runs.csv` |
| Detection delay | Mean ± standard deviation across five seeds | `calculate_detection_delay()` through `evaluate_detections()` | `detection_delay` rows |
| False alarms | False alarms per 10,000 observations | `calculate_false_alarms_per_10000()` through `evaluate_detections()` | `false_alarms_per_10000` rows |
| Missed changes | Number of planted changes not detected | `calculate_missed_detections()` through `evaluate_detections()` | `missed_detections` rows |
| Detector setting | Frozen threshold/configuration | Supplied by `config.json`; not calculated by evaluation | `results/runs/<config_hash>/config.json` |
| Seeds | Five fixed seeds, `1–5` | Aggregated by the T1 producer | `seed` in `runs.csv`; count shown in T1 |
| No-drift control | One row for every detector | Already supported by an empty changepoint list | Same T1 file |

Producer: `experiments/produce/produce_table_t1_post_tune.py`  
Saved output: `results/tables/table1/table_t1_post_tune_synthetic_detection.csv`

---

# 3. Baseline

## 3.1 Baseline and reference forecasting models

| Model | Role | Implementation file | Experiment runner | Scalar results | Detailed results | Why |
|---|---|---|---|---|---|---|
| Seasonal naïve | Frozen baseline; not a selection candidate | `src/drift_lab/forecasting/seasonal_naive.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Provides a simple seasonal benchmark. |
| DHR + ARIMA | Frozen baseline and forecaster candidate | `src/drift_lab/forecasting/dhr_arima.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | Same | Provides the required classical statistical benchmark. |
| XGBoost | Frozen baseline and forecaster candidate; forecaster for T7/F6 | `src/drift_lab/forecasting/xgboost_forecaster.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | Same | Provides the required boosted-tree benchmark. |
| NHITS | Team 41's forecaster (see 3.4) | `src/drift_lab/forecasting/nhits_forecaster.py` | `experiments/run/forecasting/run_aemo_nhits.py` | `results/runs.csv` | Same | Provides the advanced forecasting reference used by the adaptation experiments. |

Each model's hyperparameters and constructor defaults are documented in its own implementation file. The exact configuration used by an experiment is saved in `results/runs/<config_hash>/config.json`; this decision document does not duplicate model-specific hyperparameters.

## 3.2 Shared baseline protocol and result storage

| Decision | Team 41 choice | Recorded result | Why |
|---|---|---|---|
| Fit | Fit each baseline on TRAIN only | `results/runs/<config_hash>/config.json` | Establishes a pre-drift model without TEST leakage. |
| Calibration | Add known history through `observe`; do not refit model weights | Run state described by `config.json` | Supplies realistic lag history while keeping the baseline frozen. |
| Test metric | MAE | Metric row in `results/runs.csv` | Provides the shared scalar accuracy measure used in tables. |
| Forecast and error series | Save timestamped actual values, forecasts and errors | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Supports degradation analysis, figures and reproducibility. |
| Degradation | Calculate 7-day rolling MAE from the saved curve | Same curve file; derived summaries in `results/runs.csv` | Shows when predictive performance worsens. |
| F1 production | Read saved baseline curves and documented events with `experiments/produce/produce_figure_f1.py` | `results/figures/figure1/f1_degradation_<year>.png` | Ensures the figure is generated from recorded results rather than recalculated manually. |

## 3.3 F1 — degradation curve

F1 is produced in the Baseline stage. It must contain:

- one rolling-MAE line per frozen baseline model;
- the rolling window stated as 7 days / 336 half-hourly observations;
- documented events annotated;
- axes with units;
- values read from the saved baseline curve files rather than recalculated in the plotting notebook.

`calculate_rolling_mae()` already provides the required series. The runner saves it in each baseline curve file, and `produce_figure_f1.py` writes `results/figures/figure1/f1_degradation_<year>.png`.

## 3.4 Forecaster selection and protocol

| Decision | Team 41 choice | Why |
|---|---|---|
| Selection rule | Lowest 7-day rolling MAE; settled on a window held out from the end of TRAIN (Nov–Dec 2019), with the same comparison on the Calibration window | Neither window touches TEST. |
| Result | NHITS has the lowest rolling MAE in SA1 and NSW1 on both windows | `results/tables/forecaster_selection/`. |
| NHITS protocol | 7 days in, 1 day out; 500 steps; 48-step blocks with real context between blocks | A block's actuals are revealed only after it is scored. |
| XGBoost and DHR + ARIMA protocol | Blind roll-forward from the latest fit | NHITS's protocol is more forgiving, so F1 is not a like-for-like comparison. State this wherever they are compared. |
| Calibration forecasts | Fit on TRAIN, forecast Calibration with each model's TEST protocol, save the residuals | `run_aemo_calibration_forecasts.py`; the residuals calibrate both UQ methods and carry in the nonconformity signal. |

---

# 4. Detection

## 4.1 Input streams

Team 41's detector variable is AEMO `TOTALDEMAND`. Raw demand, daily aggregation and seasonal residual are not three different datasets; they are three alternative ways of processing the same demand stream before it enters ADWIN, KSWIN or Page-Hinkley.

All three are retained for comparison. The corrected half-hourly demand residual is the primary final input.

| Stream | Calculation | Function/file | Use | Why |
|---|---|---|---|---|
| Raw 30-minute demand | `raw_t = TOTALDEMAND_t`; TRAIN z-score | `loader.py`, `deseasonalise.py` | Diagnostic comparison | Preserves the original half-hourly stream, including daily, weekly and annual seasonality. |
| Daily aggregation | Mean of 48 complete half-hours; TRAIN z-score | `aggregate_daily_demand()` | Comparison | Removes the within-day cycle but leaves annual and day-of-week effects. |
| Half-hourly demand residual | `residual_t = TOTALDEMAND_t − fitted_profile(DOY_t, DOW_t, TOD_t)`; TRAIN-residual z-score | `fit_seasonal_profile()` and `apply_seasonal_profile()` in `deseasonalise.py`, `build_standard_stream()` in `standard_stream_common.py` | Primary final input | Follows the supervisor's instruction to fit day-of-year, day-of-week and, for half-hourly data, time-of-day seasonality on TRAIN only. |

Daily calculation:

`daily_d = (x_d,1 + ... + x_d,48) / 48`

Only complete days are used.

Half-hourly residual calculation:

1. Fit a multiplicative profile on TRAIN: `base demand × day-of-year factor × weekday/half-hour factor`, with the annual factor smoothed over 31 days.
2. Keep the fitted profile unchanged for Calibration and TEST.
3. For every half-hour timestamp `t`, obtain the fitted normal demand for its day-of-year, day-of-week and time-of-day:

   `seasonal_t = fitted_profile(DOY_t, DOW_t, TOD_t)`

4. Subtract the fitted profile from the observed half-hourly `TOTALDEMAND`:

   `residual_t = TOTALDEMAND_t - seasonal_t`

5. Standardise with TRAIN residual statistics:

   `z_t = (residual_t - mean(TRAIN residual)) / std(TRAIN residual)`

Fit the seasonal profile, residual mean and residual standard deviation on TRAIN only. Apply all three unchanged to Calibration and TEST. The residual remains at 30-minute resolution.

## 4.2 Detectors

| Detector | Implementation | Result | Why |
|---|---|---|---|
| ADWIN | `detection/adwin.py` | Metrics + timestamp dump | Required adaptive-window detector. |
| KSWIN | `detection/kswin.py` | Metrics + timestamp dump | Required distribution detector. |
| Page-Hinkley | `detection/page_hinkley.py` | Metrics + timestamp dump | Required mean-shift detector. |
| Shared loop | `detection/base.py` | Common timestamps | Ensures one-point-at-a-time processing. |

Rules:

- One continuous detector instance per run.
- Do not pool detections from different detectors.
- Use River public attributes only.
- Do not reset automatically after an alarm unless the frozen configuration explicitly requires it.

## 4.3 AEMO runs

| Stream | Status | Runner | Table producer | Figure producer |
|---|---|---|---|---|
| Half-hourly demand residual | Report stream | `run_aemo_detectors_standard_half_hourly_post_tune.py` | `produce_table_t2_standard_half_hourly_post_tune.py` | `produce_figure_f2_aemo_standard_half_hourly_post_tune.py` |
| Raw, daily | Retired comparisons; earlier `runs.csv` rows remain | `run_aemo_detectors_raw_post_tune.py`, `run_aemo_detectors_standard_daily_post_tune.py` | None | None |

ADWIN and Page-Hinkley are the trigger detectors for adaptation. KSWIN is evaluated in T1 and T2 only, with 48 accepted alarms in each region, and has no adaptation runs.

`run_detector_changepoints.py` reruns each trigger detector and writes the accepted change points. It must give the same alarms as the T2 run.

## 4.4 Warm-up and accepted alarms

| Decision | Team 41 choice | Why |
|---|---|---|
| Warm-up | Process TRAIN then Calibration | Gives a pre-COVID baseline. |
| Warm-up alarms | Ignore | T2 scores TEST only. |
| Recording | Begin 1 Mar 2020 | Matches the fixed TEST split. |
| Refractory | 14 days after every accepted alarm | Collapses repeat alarms. |
| Updates during refractory | Continue | Preserves detector state. |
| Retraining condition | Any accepted alarm; no event match required | Deployment does not know event labels. |

`raw alarms → refractory filter → accepted alarms → event matching and adaptation`

## 4.5 T2 — AEMO detection table

T2 is produced in the Detection stage after T1 is complete and the detector configuration is frozen.

| T2 item | Source in the current evaluation flow | Result location |
|---|---|---|
| Event-specific delay or `not detected` | `match_unmatch()` matching assignments | Timestamp assignment file plus scalar rows in `runs.csv` |
| Accepted detections | `calculate_event_metrics()` effective-detection count | `n_effective_detections` row |
| Unmatched accepted detections | `calculate_event_metrics()` | `n_unmatched_detections` row |
| Precision | `calculate_event_metrics()` | `precision`, `precision_t1`, `precision_t2` rows |
| Tier 1/2 event coverage | `tier1_matched` / `tier1_unmatched`, `tier2_matched` | T2 CSV |
| Mean delay | Per named event in the T2 producer | T2 CSV |
| Chance baseline | Closed form in `experiments/produce/chance_baseline.py`; Tier 1 only | `chance_tier1` in the T2 CSV |

Producer: `produce_table_t2_standard_half_hourly_post_tune.py`  
Saved output: `results/tables/table2/table_t2_aemo_events_standard_half_hourly_post_tune.csv`

T2 must describe documented events as historical reference points, not ground truth. An accepted detection that matches none of them is `unmatched`, not a false positive.

## 4.6 F2 — detection panel

F2 is also produced in the Detection stage. For each region it must show:

- the selected processed demand stream and documented event periods;
- every accepted alarm labelled matched or unmatched;
- a lower detector-diagnostic panel using public River information only;
- a real threshold line only when that detector exposes a valid statistic/threshold pair;
- ADWIN window width, if retained, labelled only as `adaptive window width`, without a fake threshold.

The timestamps and match labels must come from the same saved accepted-alarm and assignment files used for T2. The producer is `produce_figure_f2_aemo_standard_half_hourly_post_tune.py`, and outputs go in `results/figures/figure2/`.

---

# 5. Adaptation

## 5.1 Four arms

| Arm | Decision | Main implementation/runner | Why |
|---|---|---|---|
| A | Never retrain | `run_aemo_adaptation_arms.py` | No-adaptation control. |
| B | Retrain every 30 days, on the last 30 days | Same runner | Scheduled control. |
| C | Accepted alarm → full available history | `retrain_using_full_history.py` | Tests drift-triggered full history. |
| D | Accepted alarm → recent window | `retrain_using_recent_window.py` | Tests recent-data adaptation. |

All four arms run for NHITS, XGBoost and DHR + ARIMA. Arms A and B use no detector, so they run once, in the ADWIN batch, and every detector's tables share those runs. Page-Hinkley and ADWIN each have their own arms C and D, logged as `aemo_adapt_<detector>_std_hh_v1`.

The project does not pick a winning arm. All four are studied as results. Arm D is the arm of most interest because it is the hypothesis being tested: a drift trigger plus recent data only should adapt best. Results are framed against that hypothesis.

Arm D runs a fixed sweep over 30, 60, 120 and 180 days, reported in full in `results/tables/arm_d_sweep/<detector>/`, with the window of lowest mean drift-regime MAE marked per model × region and a tie noted when the runner-up is within one seed standard deviation. That mark uses TEST data, so it is a best observed TEST result, not an independently validated optimum.

T3, T4, uncertainty, escalation and T7 report Arm D at 180 days. F3 instead draws each model × region's marked window (see Open gaps).

## 5.2 Retraining boundary

For an accepted alarm on day `d`:

1. Old model forecasts through `d 23:30`.
2. Retraining uses data available through day `d`.
3. New model starts on `d + 1`.

The recent window may reach into TRAIN or Calibration because those observations were already available. It must exclude every observation later than the alarm boundary.

Adapters do not add a second cooldown; the refractory filter controls retraining frequency.

## 5.3 Adaptation regimes

Regimes are decided in `SHARED_DECISIONS.md` section 14.

| Rule | Decision | Function |
|---|---|---|
| Global (headline) | Every TEST timestamp gets one regime: `pre_drift` before the first event; `drift` inside `[drift_start, drift_end + 7 days)`; `post_drift` after that until the next event | `assign_timestamp_global_regime()` |
| Event-local (comparison) | 7 days before an event, the event itself and 7 days after it, with no drift-end buffer; timestamps outside every window are left out | `assign_event_local_regime()` |
| Overlaps | `drift > pre_drift > post_drift`; earliest event for `drift` and `pre_drift`, most recently started event for `post_drift` | Both functions |

## 5.4 Reporting rules

| Decision | Team 41 choice | Why |
|---|---|---|
| Aggregation | Pool all half-hours within each regime; one MAE per arm × region × seed × regime | Each half-hour counts once, so a one-day event does not weigh the same as a long one. |
| Seeds | Five seeds `1–5` for NHITS and XGBoost; DHR + ARIMA is deterministic and runs once | Report mean ± standard deviation. |
| Regions | Report SA1 and NSW1 separately | Their demand scales differ too much to pool raw MW. |
| Cost timer | `wall_clock_s` is the time spent on TEST-period retrain fits only | Every arm measures the same cost. |
| MAE comparison sign | `drift_mae_arm - drift_mae_arm_a`; negative means better than never retraining | Matches the required-output example. |
| T3 pinball loss | The 90% fixed split conformal interval, averaged at `q=0.05` and `q=0.95` over all TEST timestamps, added by `rescore_adaptation_runs.py` | The rescore must follow any new adaptation run, or T3 shows `pending UQ`. |

## 5.5 Outputs

| Output | Producer | Saved output |
|---|---|---|
| T3, forecast accuracy by regime | `produce_table_t3.py` | `results/tables/table3/<detector>/table_t3_forecast_accuracy[_event_local]_<detector>.csv` |
| T4, adaptation cost | `produce_table_t4.py` | `results/tables/table4/<detector>/table_t4_adaptation_cost[_event_local]_<detector>.csv` |
| Arm D sweep | `produce_table_arm_d_sweep.py` | `results/tables/arm_d_sweep/<detector>/` |
| F3, four-arm comparison | `produce_figure_f3.py` | `results/figures/figure3/<detector>/` |

Global-regime tables are the headline results; event-local tables sit alongside them for comparison.

---

# 6. Uncertainty

Methods are decided in `SHARED_DECISIONS.md` section 15. `run_aemo_uq_replay.py` replays each latest adaptation run's saved forecasts, one half-hour at a time. No model is refitted.

| Decision | Team 41 choice | Why |
|---|---|---|
| Methods | Fixed split conformal and AgACI, at 90% (`alpha = 0.10`) | The non-adaptive baseline against an online method. |
| Calibration | Each model × region × seed's Calibration residuals | Uses only information from before TEST. |
| AgACI buffer | Latest 336 `|actual - issued forecast|` values, 7 days; window and `gammas` are provisional | Adapts to recent errors. Candidate aggregation uses the R package `opera`. |
| Scope | Arms A, B, C and Arm D at 180 days | The other Arm D windows have no intervals. |
| Normalised width | Mean interval width divided by mean TRAIN demand | Compares regions on one scale without using TEST. |
| Worst period | Worst rolling 24-hour coverage over complete 48-half-hour windows, with the window count logged | A window with a timestamp gap is not counted. |

T5 is written by `produce_table_t5.py` to `results/tables/table5/<detector>/`. F4 is written by `produce_figure_f4.py` and plots 7-day rolling coverage against 0.90.

---

# 7. Flag/Escalation

`run_aemo_escalation_sweep.py` sweeps the saved intervals; no interval is re-issued. A forecast is escalated when its signal is strictly above `τ`, and every distinct signal value is tried as `τ`.

| Decision | Team 41 choice | Why |
|---|---|---|
| Signals | Interval width, and lagged mean `|actual - forecast|` over 30 minutes, 1 day and 7 days | The shared decision names width and the 30-minute score; 1-day and 7-day are Team 41 additions. |
| Lag | The score for forecast `t` sees errors up to `t-1`; the first TEST forecasts carry in Calibration errors | Keeps the signal causal. |
| Fixed conformal under width | Only 0% or 100% review is possible | Its width is constant. Nonconformity signals give it a full curve. |
| Operating points | Review rate for 0.90 and 0.95 retained coverage, and coverage at 5%, 10% and 20% review, per regime and overall | Required by the brief. `unreachable` is a valid result. |
| Minimum retained forecasts | `MIN_RETAINED = 1`, provisional | Any non-empty retained set may reach a target. |

T6 is written by `produce_table_t6.py` to `results/tables/table6/<detector>/`, one file per signal. F5 is written by `produce_figure_f5.py`.

---

# 8. Comparison with Team 42

T7 and F6 are the shared experiment, not Team 41's own findings.

| Setting | Team 41 (Group A) | Team 42 (Group B) |
|---|---|---|
| Trigger | Page-Hinkley on the demand residual | Page-Hinkley on forecast error |
| Forecaster | XGBoost | `HistGradientBoostingRegressor` |
| Arm, UQ | Arm D at 180 days, AgACI | Arm D at 180 days, AgACI |
| Escalation | Interval width; 30-minute nonconformity | Interval width; lagged 30-minute error |

Team 42's numbers come from `results/tables/table7/group_b_team42.csv` and `results/figures/figure6/group_b_team42_alarms.csv`. The forecasters are the same family but different implementations, and AgACI equivalence is unverified; `produce_table_t7.py` prints these differences in the settings table.

---

# Cross-cutting evaluation and metrics

## Evaluation functions

| Task | Function | Result |
|---|---|---|
| Synthetic detection | `evaluate_detections()` | Delay, false alarms, missed detections |
| Event windows | `build_event_windows()`, `calculate_match_window()` | Matching and regime windows |
| AEMO matching | `match_unmatch()` | Match, Unmatch, Ignored labels, 14-day refractory |
| Overall event metrics | `calculate_event_metrics()` | Counts, Tier 1/2 precision, event recall, mean delay |
| Regime labels | `assign_timestamp_global_regime()`, `assign_event_local_regime()` | Pre-drift, drift, post-drift |
| Point-forecast metrics | `calculate_mae()`, `calculate_rolling_mae()`, `calculate_forecast_metrics_by_regime()` | T3, F1, F3 |
| Adaptation benefit | `calculate_adaptation_gain()` | Paired drift-MAE difference vs Arm A, and per retrain |
| Quantile metrics | `calculate_pinball_loss()`, `calculate_interval_pinball_loss()` | T3 and T5 pinball loss |
| Uncertainty metrics | `calculate_uq_metrics_by_regime()` and the coverage and width functions | T5, F4 |
| Escalation metrics | `calculate_escalation_curve()`, `calculate_escalation_operating_points()`, `calculate_nonconformity_score()` | T6, F5 |
| Periodicity | `calculate_alarm_gaps()`, `calculate_gap_cv()`, `calculate_phase_concentration()` | Do alarms follow the clock or calendar? |

## Matching and attribution

| Decision | Team 41 choice |
|---|---|
| Cardinality | One detection matches at most one event; one event receives coverage once. |
| Single-date event | Covers the full calendar day. |
| Regime overlap priority | `drift > pre_drift > post_drift`. |
| Same-priority attribution | Apply one fixed event-order rule. |
| Missing attribution | Use `event_id="unassigned"` before the first event outside explicit pre-drift windows, and for the general post-drift fallback outside all bounded event windows. |
| Event-linked post-drift attribution | Within `drift_end <= timestamp < post_drift_end`, use the applicable event ID; if bounded post windows overlap, use the most recently started applicable event. |
| AEMO non-match | Report `unmatched`, not false alarm. |

## Chance baseline

Implemented as a closed form in `experiments/produce/chance_baseline.py`. Use the accepted alarm count `N`, identical TEST dates and identical event windows:

`expected matches = Σ_j [1 − (1 − w_j / T)^N]`

The value is computed by the T2 producer, not logged in `runs.csv`, and is Tier 1 only (see Open gaps).

---

# Testing and quality assurance

| Decision | Location | Why |
|---|---|---|
| Mirrored tests | `tests/` | Keeps tests close to their implementation. |
| Metric-code guard | `tests/test_evaluation.py` | Blocks duplicate metric implementations. |
| No-drift test | Detector tests | Detects continuous firing. |
| Event-overlap tests | Evaluation tests | Locks matching and priority rules. |
| Leakage tests | Preprocessing/adaptation tests | Prevents future TEST use. |
| Refractory tests | Evaluation tests | Locks 14-day suppression. |
| Slow baseline reproduction | `RUN_BASELINE_REPRO=1` | Keeps default tests fast. |

# Documentation and open gaps

- `README.md` explains how to run the repository.
- `docs/DATA_QUALITY.md` and `docs/EVENT_TIERING_CRITERIA.md` are maintained, and `tests/test_docs_in_sync.py` guards them. Other markdown under `docs/` is stale; this file and `SHARED_DECISIONS.md` take precedence.

Open gaps, each needing a team decision:

1. **Retrain budget:** the shared limit is 8 retrains a year for arms C and D, and nothing enforces it. Page-Hinkley stays within it (at most 6 in a calendar year). ADWIN exceeds it in SA1 in 2021 (9) and in NSW1 in 2022 and 2023 (9 each).
2. **Page-Hinkley grid:** the bounds were informed by AEMO results, while the shared rule is never to tune against AEMO. See 2.2.
3. **Chance baseline:** the shared rule asks for at least 1,000 simulations reported as mean ± standard deviation, for each tier. The code uses a closed form, Tier 1 only.
4. **Negative controls:** the shared rule retains Five-Minute Settlement as a negative control reported separately. It is not in `events.csv`, and T2 has no such row.
5. **3-day sensitivity:** the shared appendix analysis with a 3-day grace period has not been implemented.
6. **F3 Arm D window:** F3 draws each model × region's lowest-MAE window, while T3, T4, T5 and T6 report 180 days. They differ for DHR + ARIMA in SA1 and XGBoost in SA1 under Page-Hinkley, and for more combinations under ADWIN.
7. **Tuning-rule wording:** the shared rule names shortest delay as the tie-breaker; Team 41 ranks fewest false alarms first.
8. **Forecasting wording:** the shared rule says "Model seed 1" and identical forecasters across teams, while runs use seeds 1–5 and T7 compares XGBoost with `HistGradientBoostingRegressor`.

---

# Final causal rule

`available TOTALDEMAND → synthetic validation → demand residual → accepted detection → causal adaptation → uncertainty interval → escalation flag`

No TEST observation may fit preprocessing, tune detectors, trigger an earlier retraining event or calibrate an earlier prediction interval.
