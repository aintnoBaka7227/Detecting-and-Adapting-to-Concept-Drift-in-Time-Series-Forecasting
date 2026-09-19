# Team 41 Implementation Decisions

This document records Team 41's implementation decisions for the `drift_lab` pipeline. Joint settings are recorded in a separate shared-decision file.

## Project pipeline

`Data → Synthetic Benchmark → Baseline → Detection → Adaptation → Uncertainty → Flag/Escalation`

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
| Final tables | `results/tables/` | Separates report outputs from experiment records. |
| Final figures | `results/figures/` | Separates report outputs from experiment records. |
| Metric calculation | `src/drift_lab/evaluation/` | Gives every experiment one trusted metric implementation. |
| Run logging | `experiments/run_harness.py` | Enforces the shared result schema. |
| Result paths/readers | `experiments/results_io.py` | Keeps writer and reader filenames consistent. |

Additional rules:

- `results/runs.csv` is append-only.
- Producers select the latest completed row matching method, dataset, region, seed, configuration, split, regime and metric.
- Applicable forecast runs record `n_retrains`, `train_samples` and `wall_clock_s`.
- Track `results/runs.csv`; regenerate detailed curves, tables and figures.

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
| Intermediate data | Pass in memory; no processed dataset CSV | Loader and runners | None | Avoids stale processed files. |

`load(region)` is the only function that creates TRAIN, Calibration and TEST.

## 1.2 Documented events

| Decision | Team 41 choice | Main file | Why |
|---|---|---|---|
| Catalogue | One frozen documented-event file | `src/drift_lab/aemo/events.csv` | All detector runs use identical reference events. |
| Event identity | Unique `event_id` | `events.csv` | Prevents ambiguous attribution. |
| Tiers | Preserve Tier 1 and Tier 2 | `events.csv` | Prevents the larger Tier 2 catalogue dominating Tier 1. |
| COVID | 16 Mar–17 May 2020 | `events.csv` | Includes the late-March demand change. |
| Five-Minute Settlement | Remove from positive targets | `events.csv` | It is a price-market rule change, not an expected demand shift. |
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
| Selection rule | Shortest mean delay among passing configurations | Tuning results | Balances delay and false alarms. |
| KSWIN internal seed | Fixed separately from generator seed | Detector configuration | Separates detector and input randomness. |
| Final validation | Rerun frozen settings | `run_post_tune_on_synthetic.py` | Produces official T1 inputs. |
| T1 producer | Read `runs.csv` | `produce_table_t1_post_tune.py` | T1 is not calculated manually. |

Detector settings are frozen after synthetic calibration and applied to AEMO unchanged.

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
Saved output: `results/tables/table_t1_post_tune.csv`

---

# 3. Baseline

## 3.1 Baseline and reference forecasting models

| Model | Role | Implementation file | Experiment runner | Scalar results | Detailed results | Why |
|---|---|---|---|---|---|---|
| Seasonal naïve | Frozen baseline | `src/drift_lab/forecasting/seasonal_naive.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Provides a simple seasonal benchmark. |
| DHR + ARIMA | Frozen baseline | `src/drift_lab/forecasting/dhr_arima.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Provides the required classical statistical benchmark. |
| XGBoost | Frozen baseline | `src/drift_lab/forecasting/xgboost.py` | `experiments/run/forecasting/run_aemo_baselines.py` | `results/runs.csv` | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Provides the required boosted-tree benchmark. |
| NHITS | Reference forecaster used before adaptation | `src/drift_lab/forecasting/nhits.py` | `experiments/run/forecasting/run_aemo_nhits.py` | `results/runs.csv` | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Provides the advanced forecasting reference used by the adaptation experiments. |

Each model's hyperparameters and constructor defaults are documented in its own implementation file. The exact configuration used by an experiment is saved in `results/runs/<config_hash>/config.json`; this decision document does not duplicate model-specific hyperparameters.

## 3.2 Shared baseline protocol and result storage

| Decision | Team 41 choice | Recorded result | Why |
|---|---|---|---|
| Fit | Fit each baseline on TRAIN only | `results/runs/<config_hash>/config.json` | Establishes a pre-drift model without TEST leakage. |
| Calibration | Add known history through `observe`; do not refit model weights | Run state described by `config.json` | Supplies realistic lag history while keeping the baseline frozen. |
| Test metric | MAE | Metric row in `results/runs.csv` | Provides the shared scalar accuracy measure used in tables. |
| Forecast and error series | Save timestamped actual values, forecasts and errors | `results/runs/<config_hash>/curve_<dataset>_<region>_<seed>.csv` | Supports degradation analysis, figures and reproducibility. |
| Degradation | Calculate 7-day rolling MAE from the saved curve | Same curve file; derived summaries in `results/runs.csv` | Shows when predictive performance worsens. |
| F1 production | Read saved baseline curves and documented events with `experiments/produce/produce_figure_f1.py` | `results/figures/f1_degradation_<year>.png` | Ensures the figure is generated from recorded results rather than recalculated manually. |

## 3.3 F1 — degradation curve

F1 is produced in the Baseline stage. It must contain:

- one rolling-MAE line per frozen baseline model;
- the rolling window stated as 7 days / 336 half-hourly observations;
- documented events annotated;
- axes with units;
- values read from the saved baseline curve files rather than recalculated in the plotting notebook.

`calculate_rolling_mae()` already provides the required series. The runner saves it in each baseline curve file, and `produce_figure_f1.py` writes `results/figures/f1_degradation_<year>.png`.

---

# 4. Detection

## 4.1 Input streams

Team 41's detector variable is AEMO `TOTALDEMAND`. Raw demand, daily aggregation and seasonal residual are not three different datasets; they are three alternative ways of processing the same demand stream before it enters ADWIN, KSWIN or Page-Hinkley.

All three are retained for comparison. The corrected half-hourly demand residual is the primary final input.

| Stream | Calculation | Function/file | Use | Why |
|---|---|---|---|---|
| Raw 30-minute demand | `raw_t = TOTALDEMAND_t`; TRAIN z-score | `loader.py`, `deseasonalise.py` | Diagnostic comparison | Preserves the original half-hourly stream, including daily, weekly and annual seasonality. |
| Daily aggregation | Mean of 48 complete half-hours; TRAIN z-score | `aggregate_daily_demand()` | Comparison | Removes the within-day cycle but leaves annual and day-of-week effects. |
| Half-hourly demand residual | `residual_t = TOTALDEMAND_t − fitted_profile(DOY_t, DOW_t, TOD_t)`; TRAIN-residual z-score | Extend the merged `remove_daily_weekly_profile()` in `deseasonalise.py` | Primary final input | Follows the supervisor's instruction to fit day-of-year, day-of-week and, for half-hourly data, time-of-day seasonality on TRAIN only. |

Daily calculation:

`daily_d = (x_d,1 + ... + x_d,48) / 48`

Only complete days are used.

Half-hourly residual calculation:

1. Start from the already merged `remove_daily_weekly_profile()` function, which models the normal day-of-week and half-hour/time-of-day pattern.
2. Extend that TRAIN-fitted seasonal profile with the day-of-year component required by the supervisor.
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

| Stream | Runner | Table producer | Figure producer |
|---|---|---|---|
| Raw | `run_aemo_detectors_raw_post_tune.py` | `produce_table_t2_raw_post_tune.py` | `produce_figure_f2_aemo_raw_post_tune.py` |
| Daily | `run_aemo_detectors_daily_post_tune.py` | `produce_table_t2_daily_post_tune.py` | `produce_figure_f2_aemo_daily_post_tune.py` |
| Half-hourly demand residual | `run_aemo_detectors_deseasonalized_post_tune.py` | `produce_table_t2_deseasonalized_post_tune.py` | `produce_figure_f2_aemo_deseasonalized_post_tune.py` |
| Combined comparison | No new run | `produce_table_t2_all_streams_post_tune.py` | Not the final F2 |

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
| Tier 1/2 event coverage | Requires explicit tier denominators in the T2 producer; `calculate_event_metrics()` currently returns only combined `event_recall` | Add tier-specific coverage/recall values to `calculate_event_metrics()` |
| Mean delay | `calculate_event_metrics()` | `mean_delay_days` row |
| Chance baseline | Not implemented in `evaluation.py` | Add the agreed event-specific chance calculation or calculate it in one shared producer function |

Producer: the selected input stream's `produce_table_t2_*_post_tune.py`  
Saved output: `results/tables/table_t2_<stream>_post_tune.csv`

T2 must describe documented events as historical reference points, not ground truth. An accepted detection that matches none of them is `unmatched`, not a false positive.

## 4.6 F2 — detection panel

F2 is also produced in the Detection stage. For each region it must show:

- the selected processed demand stream and documented event periods;
- every accepted alarm labelled matched or unmatched;
- a lower detector-diagnostic panel using public River information only;
- a real threshold line only when that detector exposes a valid statistic/threshold pair;
- ADWIN window width, if retained, labelled only as `adaptive window width`, without a fake threshold.

The timestamps and match labels must come from the same saved accepted-alarm and assignment files used for T2. The final producer is `produce_figure_f2_aemo_<stream>_post_tune.py`, and outputs belong in `results/figures/`.

---

# 5. Adaptation

## 5.1 Four arms

| Arm | Decision | Main implementation/runner | Why |
|---|---|---|---|
| A | Never retrain | `run_aemo_adaptation_arms.py` | No-adaptation control. |
| B | Retrain every 30 days | Same runner | Scheduled control. |
| C | Accepted alarm → full available history | Full-history adapter | Tests drift-triggered full history. |
| D | Accepted alarm → recent window | `retrain_using_recent_window.py` | Tests recent-data adaptation. |

Arm D performs a fixed sweep over 30, 60, 120 and 180 days. Run all four window lengths on the same data and select the window with the best forecast performance. The official selection rule is the lowest mean drift-regime MAE across the fixed seeds. Keep the results for all four windows in T3/T4 and mark the selected window; do not discard the non-selected sweep results.

If the sweep and selection both use the final TEST period, describe the selected Arm D window as the best observed TEST configuration rather than an independently validated optimum. Reporting all four window results keeps that post-hoc selection transparent.

## 5.2 Retraining boundary

For an accepted alarm on day `d`:

1. Old model forecasts through `d 23:30`.
2. Retraining uses data available through day `d`.
3. New model starts on `d + 1`.

The recent window may reach into TRAIN or Calibration because those observations were already available. It must exclude every observation later than the alarm boundary.

Adapters do not add a second cooldown; the refractory filter controls retraining frequency.

## 5.3 Adaptation regime decisions and current implementation status

The current `evaluation.py` is not final, but it already establishes the following effective regime policy. These rules should be reused for T3 rather than introducing a second, inconsistent regime implementation.

| Regime item | Current implemented rule | Function/value |
|---|---|---|
| Event source | Use every `NEM` event and every event for the selected region that is present in the supplied catalogue. The function does not filter by tier. | `build_event_windows()` |
| Pre-drift window | Begin 7 days before each documented event. Timestamps before the first event that are outside an explicit window are also labelled `pre_drift`. | `REGIME_PRE_DRIFT_DAYS = 7`; `assign_regime()` |
| Drift window | From the documented start date through the complete documented end date. | `drift_start = start_date`; `drift_end = end_date + 1 day` |
| Event-linked post-drift window | From an event's `drift_end` to its bounded `post_drift_end`. A timestamp inside this window is `post_drift` with that event's ID. | `REGIME_POST_DRIFT_DAYS = 7`; `build_event_windows()`; missing check in the current `assign_regime()` |
| General post-drift fallback | If a timestamp is outside every event's drift, pre-drift and bounded post-drift windows, but at least one event has already occurred, label it `post_drift` with `event_id="unassigned"`. | Existing fallback in `assign_regime()` |
| Overlap priority | `drift > pre_drift > post_drift` globally. | `assign_regime()` |
| Same-priority overlap | For overlapping drift or pre-drift windows, select the earliest event after sorting by `drift_start` and `event_id`. For overlapping bounded post-drift windows, select the most recently started applicable event. | `assign_regime()` plus required bounded post-drift check |
| Event attribution | Drift uses the matching event ID; explicit pre-drift uses the upcoming event ID; bounded post-drift uses the applicable event ID. The general post-drift fallback and the period before the first event remain `unassigned`. | Decision fixed; current `assign_regime()` still needs bounded post-drift attribution. |
| Full-stream classification | Every supplied timestamp receives one of `pre_drift`, `drift` or `post_drift`; there is no separate `stable` label. | `assign_regime()` |

For T3, pass the complete forecast timestamp index—not only detection timestamps—through this same regime logic. The parameter name and documentation should be generalised from `detected_timestamps` to `timestamps`, but the project should not create a second conflicting regime definition.

`post_drift_end` is required to distinguish two different cases:

1. **Event-linked post-drift:** the timestamp is between an event's `drift_end` and `post_drift_end`. Label it `post_drift` and record that event's ID.
2. **General post-drift fallback:** the timestamp is outside every event's pre-drift, drift and bounded post-drift window, but at least one event occurred previously. Label it `post_drift` with `event_id="unassigned"`.

The current `assign_regime()` checks drift, then pre-drift, then jumps directly to the general unassigned fallback. It never checks `drift_end <= timestamp < post_drift_end`, so event-linked post-drift detections are currently losing their event ID. Add the bounded post-drift check before the general fallback.

If a later event starts during an earlier event's post-drift window, `drift` has higher priority and the timestamp is attributed to the later event. If it is in the later event's pre-drift window, `pre_drift` also outranks the older event's post-drift window.

## 5.4 Decisions still required before T3, T4 and F3

The four arms, Arm D sweep/selection, causal retraining boundary and regime assignment above are already decided. The following output-design choices still need to be frozen.

| Decision still needed | Recommendation | Why it is needed |
|---|---|---|
| Aggregation across events | Pool all timestamp-level forecast errors within each regime and calculate one MAE per arm × region × seed × regime. Do not average event-level MAEs equally. | Pooling makes each half-hour forecast contribute once and avoids a one-day event receiving the same weight as a long event. |
| Common forecast timestamps | Score only timestamps with an actual value and valid outputs from all four compared arms; log the count excluded from each arm. | Different missing forecasts would otherwise make the arm comparison unfair. |
| Seeds | Use the same five seeds `1–5` for all four arms and report mean ± standard deviation for stochastic results. | The required-output brief requires at least five seeds and error bars. |
| T4 region handling | Report T4 separately for SA1 and NSW1. | Their demand scales differ too much for raw-MW gains to be pooled. |
| Cost timer boundary | Define `wall_clock_s` as cumulative TEST-period model fitting/retraining time; exclude the common initial TRAIN fit and forecast-generation time. | Every arm must measure the same cost. |
| MAE comparison sign | Use `drift_mae_arm - drift_mae_arm_a`; negative means improvement over never retraining. | The required-output example uses negative values for an improvement, so the sign must be stated. |
| Pinball loss | Finalise after the UQ layer exists. Recommended 90% interval score: calculate pinball loss at `q=0.05` and `q=0.95`, then average the two over the scored TEST timestamps. | Pinball loss cannot be calculated from a point forecast alone. |

## 5.5 T3 — forecast accuracy by regime

Required columns from the required-output specification:

`arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds`

| T3 requirement | Present in `evaluation.py`? | Recommended addition/action | Logged result |
|---|---|---|---|
| Point MAE | Yes: `calculate_mae()` | Reuse without duplicating the formula. | `metric_name=mae`, one row per regime |
| Pre-/drift/post assignment for every forecast timestamp | Partly. `assign_regime()` already assigns the agreed three-regime policy to any supplied timestamps, but its name, parameter and documentation are detection-oriented. | Generalise the existing function to accept a full timestamp index. Preserve the intentional unbounded post-drift fallback and do not create a second regime algorithm. | `regime` column in the curve file; counts logged for QA |
| MAE by regime | No. `calculate_regime_metrics()` calculates detection counts and precision, not forecast error. | Add `calculate_forecast_metrics_by_regime(y_true, y_pred, regime_labels)` returning MAE and observation count for each regime. | Three MAE rows per arm × region × seed |
| Pinball loss | No. | Add `calculate_pinball_loss(y_true, y_quantile, quantile)` with `0 < quantile < 1` validation. Add `calculate_interval_pinball_loss(y_true, lower, upper, alpha=0.10)` to average the 0.05 and 0.95 losses once the UQ layer outputs interval bounds. | `pinball_loss` row; quantiles/alpha in `config.json` |
| Five-seed summary | No; correctly belongs in the producer rather than the metric function. | `produce_table_t3.py` groups the five per-seed rows and reports mean ± standard deviation plus seed count. | `results/tables/table_t3_forecast_accuracy.csv` |

T3 must not replace the three regime MAEs with one overall TEST MAE. Pinball loss is a UQ-dependent final column: T3 can be produced provisionally with MAE first, but it is not complete until the interval/quantile forecasts are available.

## 5.6 T4 — adaptation cost

Required columns from the required-output specification:

`arm | region | retrains | cumulative train samples | wall clock (s) | MAE difference vs Arm A | difference per retrain`

| T4 requirement | Present now? | Recommended addition/action | Logged result |
|---|---|---|---|
| Retrain count | Schema field exists; not calculated by `evaluation.py` | Runner increments only after a TEST-period retrain completes. Initial TRAIN fit is excluded; Arm A is zero. | `n_retrains` |
| Cumulative training samples | Schema field exists; not calculated by `evaluation.py` | At each TEST retrain, record the number of observations passed to `fit()` and sum them. Repeated observations count again because they incur training cost. | `train_samples` |
| Wall-clock seconds | Schema field exists; not calculated by `evaluation.py` | Time each TEST retraining call with one shared boundary and sum the durations. | `wall_clock_s` |
| MAE difference vs Arm A | Missing | Add `calculate_adaptation_gain(arm_drift_mae, arm_a_drift_mae, n_retrains)` or one equivalent shared pure function. Pair by region and seed before subtracting. | `drift_mae_difference_vs_arm_a` |
| Difference per retrain | Missing | Same function returns the difference divided by `n_retrains`; return `NaN/not applicable` when `n_retrains == 0`. | `drift_mae_difference_per_retrain` |
| Five-seed summary | Producer responsibility | `produce_table_t4.py` reports mean ± standard deviation for stochastic cost/benefit values and the seed count. | `results/tables/table_t4_adaptation_cost.csv` |

Cost counters belong to the adaptation runner because `evaluation.py` cannot infer how many samples a model was trained on or how long fitting took. Only the relative MAE calculations belong in shared evaluation code.

## 5.7 F3 — four-arm comparison

F3 is produced in the Adaptation stage and requires:

- exactly four main lines on one axis: Arms A, B, C and the Arm D window selected by the 30/60/120/180-day sweep;
- 7-day / 336-observation rolling MAE, using the existing `calculate_rolling_mae()`;
- controls A and B in muted grey and drift-triggered arms in colour;
- the pre-drift Arm A MAE as a horizontal reference line;
- one retraining rug row per arm beneath the curve;
- retrain count included in each legend label;
- separate figures for SA1 and NSW1 because their MW scales differ;
- mean rolling MAE across seeds with a ±1 standard-deviation band if NHITS remains stochastic;
- axis units and the rolling window stated in the caption.

The curve file for every arm × region × seed must contain `timestamp`, `actual`, `forecast`, `absolute_error`, `rolling_mae`, `regime`, `arm` and `seed`. Retraining rugs come from `retrains_<dataset>_<region>_<seed>.csv`, not from inferred changes in the forecast curve.

Producer: `experiments/produce/produce_figure_f3.py`  
Saved output: `results/figures/f3_four_arm_comparison_<region>.png`

## 5.8 Recommended `evaluation.py` adaptation API

| Function | Status | Purpose |
|---|---|---|
| `calculate_mae()` | Exists | Point accuracy for each scored regime. |
| `calculate_rolling_mae()` | Exists | F3 time series; default 336 observations. |
| `assign_regime()` | Exists; correct and generalise | Reuse it for forecast timestamps; add event-linked bounded post-drift attribution before retaining the unassigned general post-drift fallback. |
| `calculate_regime_metrics()` | Exists, detection only | Keep for detection summaries; do not use it as T3 forecast evaluation. |
| `calculate_forecast_metrics_by_regime()` | Add | Return MAE and `n_observations` for each regime. |
| `calculate_pinball_loss()` | Add with UQ | Score one quantile forecast. |
| `calculate_interval_pinball_loss()` | Add with UQ | Combine lower/upper quantile losses for the 90% interval. |
| `calculate_adaptation_gain()` | Add | Produce paired drift-MAE difference vs Arm A and per-retrain difference. |
| `evaluate_adaptation()` | Add last | Orchestrate regime labels, regime MAE, rolling MAE and optional pinball loss without calculating runner-owned costs. |

## 5.9 Minimum recorded data for T3, T4 and F3

For every `(arm, region, seed, config_hash)` run:

- `results/runs.csv` records regime MAE, pinball loss when available, `n_retrains`, `train_samples`, `wall_clock_s`, and the paired Arm A differences;
- the curve file records the complete F3 plotting fields and interval bounds once UQ is added;
- the retraining file records alarm time, retraining boundary, model-active time, training-window start/end, sample count and fit duration;
- `config.json` records the exact model, detector, adapter, split, Arm D candidate window, whether it was selected by the sweep, the sweep selection metric, regime policy and interval quantiles.

---

# 6. Uncertainty

The uncertainty method and final metric implementation are not yet decided. The following are project requirements, not frozen Team 41 implementation decisions.

| Required item | Project requirement | Current status / future evaluation plan |
|---|---|---|
| Target interval | Produce a 90% prediction interval for every forecast. | Method not selected. Save lower/upper bounds in the adaptation curve file so T3 pinball can be completed later. |
| Calibration | Use only information available before each forecast. | Procedure not selected; requires a separate causal-leakage decision. |
| Pinball dependency | T3 requires pinball loss. | Add the two pinball functions described in Section 5 only after interval semantics are frozen. |
| Coverage | Report empirical coverage by regime. | Add `calculate_empirical_coverage()`. |
| Width | Report mean and normalised interval width by regime. | Add `calculate_mean_interval_width()` and `calculate_normalised_interval_width()`; the normalising denominator still needs a decision. |
| Worst period | Report worst rolling 24-hour coverage. | Add `calculate_rolling_coverage(window=48)` and `calculate_worst_rolling_coverage(window=48)`. |
| UQ orchestration | Produce T5-ready values consistently. | Add `calculate_uq_metrics_by_regime()` after the UQ output format is selected. |
| F4 | Plot rolling empirical coverage with the 0.90 target and drift periods marked. | Producer not finalised. It should reuse the same forecast-regime labels as T3. |

Pinball loss belongs at the boundary between adaptation and uncertainty: it evaluates quantile forecasts, not ordinary point forecasts. Team 41 should not report it until the interval or quantile method and its output format are agreed.

---

# 7. Flag/Escalation

The escalation method is not yet selected. The items below are future project requirements from the brief, not frozen Team 41 implementation decisions.

| Future requirement | Required output | Why |
|---|---|---|
| Mark low-confidence forecasts for human review | Escalation result | Converts uncertainty into action. |
| Sweep the escalation threshold | Threshold-level results in `runs.csv` or a detailed escalation curve file | Measures the review-rate/coverage trade-off. |
| Find the review rate needed for coverage 0.90 and 0.95 | T6 | Required operating points. |
| Report coverage at 5%, 10% and 20% review | T6 | Compares methods at equal human workload. |
| Record `unreachable` if no threshold reaches a target | T6 | Failure to reach the target is a valid result. |
| Plot review rate against retained-set coverage | F5 through `produce_figure_f5.py` | Shows the operational trade-off. |

T5 is produced by `produce_table_t5.py`; T6 by `produce_table_t6.py`.

---

# Cross-cutting evaluation and metrics

## Evaluation functions

| Task | Function | Result |
|---|---|---|
| Synthetic detection | `evaluate_detections()` | Delay, false alarms, missed detections |
| Event windows | `build_event_windows()` | Matching/regime windows |
| AEMO matching | `match_unmatch()` | Match, Unmatch, Ignored labels |
| Overall event metrics | `calculate_event_metrics()` | Counts, Tier 1/2 precision, combined event recall and mean delay; tier-specific event coverage is still missing |
| Regime labels | `assign_regime()` | Pre-drift, drift, post-drift |
| Regime metrics | `calculate_regime_metrics()` | Regime-specific detection summaries |
| AEMO orchestration | `evaluate_aemo_detections()` | Official T2 metric rows |
| Point-forecast metrics | `calculate_mae()`, `calculate_rolling_mae()` | Existing inputs for T3 and F3 |
| Forecast regime labels | Existing `assign_regime()` after correcting/generalising its interface | Add bounded event-linked post-drift first, then retain the unassigned general post-drift fallback |
| Forecast metrics by regime | Proposed `calculate_forecast_metrics_by_regime()` | Missing workflow for T3 |
| Quantile metrics | Proposed `calculate_pinball_loss()`, `calculate_interval_pinball_loss()` | Missing; requires quantile/interval forecasts |
| Relative adaptation benefit | Proposed `calculate_adaptation_gain()` | Missing paired drift-MAE difference vs Arm A |
| Uncertainty metrics | Coverage, interval-width and worst-24-hour functions | Planned but not implemented |

`evaluate_aemo_detections()` orchestrates:

`build_event_windows() → assign_regime() and match_unmatch() → calculate_event_metrics() and calculate_regime_metrics()`

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

Use the accepted alarm count `N`, identical TEST dates and identical event windows:

`expected matches = Σ_j [1 − (1 − w_j / T)^N]`

Record `chance_matches_t1` and `chance_matches_t2` in `runs.csv`.

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

- Method notes live in `docs/methods/<method>.md`.
- Each note records architecture, wrapper behaviour, suitability, limitations and at least two peer-reviewed references.
- `README.md` explains how to run the repository.
- `docs/refactor.md` is historical; this decision file takes precedence.

Open gaps:

- `assign_regime()` is documented for detections even though T3 needs the same rules applied to all forecast timestamps;
- the current `assign_regime()` does not yet check the bounded `post_drift_end`, so event-linked post-drift timestamps are incorrectly returned with `event_id="unassigned"`;
- after the bounded event-linked post-drift window ends, the general post-drift regime remains unbounded but deliberately carries no event ID.

---

# Final causal rule

`available TOTALDEMAND → synthetic validation/baseline → selected demand processing → accepted detection → causal adaptation → uncertainty interval → escalation flag`

No TEST observation may fit preprocessing, tune detectors, trigger an earlier retraining event or calibrate an earlier prediction interval.
