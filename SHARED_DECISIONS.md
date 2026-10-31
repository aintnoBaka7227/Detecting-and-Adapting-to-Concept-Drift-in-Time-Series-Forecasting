# Project Decision Register (Both team 41 and 42) 

This is the copy for team 41

This file records agreed research and experiment decisions. It does not explain code structure or implementation history.

## 1. Project scope

| Decision | Agreed value |
|---|---|
| Primary task | Detect and adapt to concept drift in electricity-demand forecasting |
| Primary dataset | AEMO `TOTALDEMAND` |
| Regions | `SA1`, `NSW1` |
| Data resolution | Native 5-minute records aggregated to 30-minute intervals |
| Data period | 1 January 2018 to 31 December 2023 |
| Team 41 trigger | AEMO `TOTALDEMAND`, processed as raw 30-minute demand, daily aggregation, or a seasonal residual |
| Team 42 trigger | Processed forecast-error stream |
| Final comparison | The trigger stream is the intended difference between Teams 41 and 42; forecasters come from the same model family |

## 2. Fixed data split

| Split | Dates | Use |
|---|---|---|
| Training | 1 January 2018 to 31 December 2019 | Fit forecasting models and preprocessing parameters; detector warm-up |
| Calibration | 1 January 2020 to 29 February 2020 | Calibration, pre-test observations and detector warm-up |
| Test | 1 March 2020 onward | Record and evaluate forecasts and detections |

The split is fixed for all experiments and must not be changed between methods.

## 3. Shared experiment settings

| Decision | Agreed value |
|---|---|
| Regions | `SA1`, `NSW1` |
| Random seeds | `1, 2, 3, 4, 5` |
| Seasonal period | 48 half-hourly observations per day |
| Rolling forecast-error window | 7 days = 336 half-hourly observations |
| Official event file | One shared frozen `events.csv` |
| Official evaluation | Shared `drift_lab.evaluation` functions |
| Official results ledger | `results/runs.csv` |
| Minimum repetitions | 5 seeds for stochastic experiments |
| Reporting | Mean ± standard deviation across seeds |

## 4. Synthetic drift benchmark

| Decision | Agreed value |
|---|---|
| Series length | 20,000 observations |
| Noise standard deviation | 1.0 |
| Seeds | `1, 2, 3, 4, 5` |
| Drift types | No drift, sudden, gradual and recurring |
| No-drift changepoints | None |
| Sudden changepoint | `n // 4` |
| Gradual drift interval | Starts at `n // 4`; report the full planted interval |
| Recurring changepoints | `n // 3` and `2n // 3` |
| Required metrics | Detection delay, false alarms, missed drifts |
| Calibration order | Complete synthetic validation before AEMO evaluation |
| Parameter selection | Synthetic data only; never tune against AEMO event matches |
| False-alarm budget | At most 2 false alarms per simulated year |
| Missed-drift target | 0 missed planted drifts |
| Tie-breaker | Among passing configurations, select the shortest mean detection delay |
| Final settings | Freeze the complete detector configuration before AEMO |

False alarms per year must use the detector's actual input frequency:

| Input frequency | Samples per year |
|---|---:|
| Half-hourly | 17,520 |
| Three-hourly | 2,920 |
| Daily | 365 |

## 5. AEMO preprocessing

### Team 41 demand stream

| Decision | Agreed value |
|---|---|
| Base detector variable | AEMO `TOTALDEMAND` |
| Processing choices | Raw 30-minute demand; daily aggregated demand; half-hourly seasonal residual |
| Seasonal fitting period | Training split only |
| Residual seasonal components | Day-of-year, day-of-week and, because the residual stays half-hourly, time-of-day |
| Primary corrected detector input | Half-hourly seasonal residual, not the fitted seasonal component |
| Standardisation | Z-score residuals using training-residual mean and standard deviation |
| Test leakage | Test data must not fit or update the seasonal profile or standardiser |
| Forecasting data | Forecasting and retraining continue to use the original half-hourly demand values |

The three Team 41 choices are alternative transformations of the same `TOTALDEMAND` stream:

1. **Raw 30-minute demand**

   `raw_t = TOTALDEMAND_t`

2. **Daily aggregation**

   `daily_d = mean(x_d,1, ..., x_d,48)`

   Only complete days with 48 half-hourly observations are used.

3. **Half-hourly seasonal residual**

   Start from Team 41's merged `remove_daily_weekly_profile()` function, which already represents the normal day-of-week and half-hour/time-of-day pattern. Extend its TRAIN-fitted profile with the supervisor-required day-of-year component:

   `seasonal_t = fitted_profile(DOY_t, DOW_t, TOD_t)`

   `residual_t = TOTALDEMAND_t - seasonal_t`

   `z_t = (residual_t - mean(TRAIN residual)) / std(TRAIN residual)`

The fitted profile and z-score statistics are learned from TRAIN only and applied unchanged to Calibration and TEST. Daily aggregation remains a comparison input. The half-hourly residual is the primary corrected input because it removes annual, weekly and within-day seasonality while retaining 30-minute resolution.

### Team 42 forecast-error stream

Team 42 must freeze its own causal error preprocessing before final calibration. The final decision must state:

- error definition and forecast horizon;
- aggregation frequency;
- autocorrelation treatment;
- standardisation period;
- timestamp assigned to an aggregated error;
- time at which the error becomes available.

## 6. Detector evaluation lifecycle

| Decision | Agreed value |
|---|---|
| Detectors | ADWIN, KSWIN and Page-Hinkley |
| Detector parameters | Each team tunes and freezes its own; parameters need not match across teams |
| Detector state | One continuous detector instance per independent run |
| Test boundary | Do not recreate the detector on 1 March 2020 |
| Warm-up | TRAIN + Calibration, for both teams |
| Warm-up alarms | Ignore all pre-test alarms |
| Recorded alarms | Test-period alarms only |
| Refractory processing | Apply before event matching and before adaptation |
| Detector updates during refractory period | Continue updating the detector |
| Saved outputs | Preserve both raw alarms and accepted alarms |
| T2 input | Accepted alarms only |
| Adaptation input | The same accepted alarms used in T2 |
| AEMO terminology | Use `unmatched`, not `false positive`, when no documented event is matched |

## 7. Refractory period

| Decision | Agreed value |
|---|---|
| Refractory period | 14 calendar days, for both teams |
| Start | Time of an accepted alarm |
| Rule | Suppress later accepted alarms during the following 14 days |
| Matching dependency | Refractory filtering does not depend on whether an alarm matches an event |
| Retraining dependency | Only accepted alarms can trigger retraining |

## 8. AEMO event catalogue

| Decision | Agreed value |
|---|---|
| Catalogue | One version of `events.csv` shared by both teams |
| Event identity | Every event has a unique `event_id` |
| Required fields | Event name, start, end, region, tier, event type and source |
| Positive-event tiers | Tier 1 and Tier 2 |
| Tier reporting | Report Tier 1 and Tier 2 separately |
| COVID | 16 March – 17 May 2020 (`EVT-2020-02`, NEM-wide, Tier 1), following AEMO Quarterly Energy Dynamics Q2 2020 |
| Catalogue freeze | Freeze dates and classifications before the final AEMO run |

## 9. Event-matching rules

| Decision | Agreed value |
|---|---|
| Single-date event window | Event date through event date + 7 days |
| Multi-day event window | Event start through event end + 7 days |
| Pre-event detections | Do not match detections before the documented event start |
| Detection-to-event matching | One detection can match at most one event |
| Event coverage | One event receives credit at most once |
| Detection delay | Detection time minus documented event start |
| Remaining detections | Label `unmatched` |
| Remaining positive events | Label `missed` |
| Overlapping event windows | Same rule for both teams: Tier 1 events are matched first, then Tier 2, each in chronological order; each event takes the first unused accepted detection in its window |

### Regime attribution

| Decision | Agreed value |
|---|---|
| Regime priority | `drift > pre_drift > post_drift` |
| Event-linked post-drift | If a detection falls from an event's `drift_end` up to but not including its `post_drift_end`, label it `post_drift` and record that event's `event_id` |
| Overlapping post-drift windows | Attribute the detection to the most recently started applicable event |
| General post-drift fallback | If a detection is outside every event's pre-drift, drift and bounded post-drift window, but at least one event occurred previously, label it `post_drift` with `event_id="unassigned"` |
| Before all events | If no event has occurred and the detection is outside every explicit pre-drift window, label it `pre_drift` with `event_id="unassigned"` |

The evaluation order is:

`drift → pre_drift → bounded event-linked post_drift → general unassigned post_drift`

## 10. Table T1 decisions

T1 reports synthetic detector validation.

| Required column | Decision |
|---|---|
| Detector | ADWIN, KSWIN or Page-Hinkley |
| Drift type | No drift, sudden, gradual or recurring |
| Delay | Mean ± standard deviation |
| False alarms | Per 10,000 samples and per simulated year |
| Missed | Number of planted drifts missed |
| Configuration | Full frozen detector settings |
| Seeds | `1, 2, 3, 4, 5` |

T1 must be completed before T2 is presented as a valid result.

## 11. Table T2 decisions

T2 reports AEMO detections matched against documented events. Documented events are reference events, not complete ground truth.

| Required result | Decision |
|---|---|
| Tier 1 event coverage | Report separately |
| Tier 2 event coverage | Report separately |
| Tier 1 detection precision | Report separately |
| Tier 2 detection precision | Report separately |
| Unmatched detections | Report count |
| Missed documented events | Report count |
| Detection delays | Report in days |
| Chance baseline | Required for every detector and region |

Metric definitions:

`event coverage = events matched at least once / target events`

`detection precision = matched accepted detections / accepted detections`

Tier-specific precision uses the same pooled denominator as the overall `detection precision` above, applied per tier:

`tier X detection precision = tier X matched accepted detections / accepted detections`

A detection matched to the other tier still counts in the denominator even though it is not a hit for the tier being scored, so `precision_t1` and `precision_t2` are both bounded by the overall `detection precision` and share its zero-detections `nan` condition.

Tier-specific event coverage keeps the pooled-looking formula above applied per tier (`tier X events matched at least once / tier X target events`), since coverage is defined over each tier's own event set, not over a shared detection pool.

## 12. Chance-matching baseline

| Decision | Agreed value |
|---|---|
| Detection count | Use the same accepted count `N` as the real detector row |
| Test period | Same dates as the real detector run |
| Refractory rule | Same rule as the real detector run |
| Matching rule | Same event matcher as the real detector run |
| Event catalogue | Same frozen `events.csv` |
| Method | Closed form: `expected matches = Σ_k [1 − (1 − w_k / T)^N]`, with `w_k` each event's match window and `T` the test days |
| Scope | Tier 1 events |

Calculate the chance baseline separately for every detector × region × input-stream result.

## 13. Figure F2 decisions

| Decision | Agreed value |
|---|---|
| Regions | Produce figures for SA1 and NSW1 |
| Upper panel | Detector input, documented events, matched alarms and unmatched alarms |
| Event wording | Do not call AEMO events `true changepoints` |
| Unmatched wording | Use `unmatched`, not `false alarm` |
| Detector diagnostics | Use documented public values only |
| River private attributes | Do not use attributes beginning with `_` |
| ADWIN width | May be shown only as adaptive-window width, not as a test statistic |
| ADWIN threshold | Do not draw a threshold for width |

## 14. Forecasting and adaptation

### Shared forecasting settings

| Decision | Agreed value |
|---|---|
| Forecast horizon | 48 half-hourly steps |
| Input size | 336 half-hourly steps |
| Training steps | 500 |
| Model seeds | `1, 2, 3, 4, 5` |
| Compute device | CPU |
| Comparison requirement | Teams 41 and 42 use forecasters from the same model family; the implementation and forecast values may differ |

### Four adaptation arms

| Arm | Decision |
|---|---|
| A | Never retrain |
| B | Retrain every 30 days |
| C | Accepted drift alarm → retrain using all available history |
| D | Accepted drift alarm → retrain using only the most recent window |

Arm D's window is swept over 30, 60, 120 and 180 days. The best window per model × region has the lowest drift-regime MAE and is selected on the TEST period; 180 days is the window carried forward to T7.

### Retraining boundary

For an accepted alarm on day `d`:

- the old model forecasts through the end of day `d`;
- retraining uses observations available through day `d`;
- the new model begins forecasting on day `d + 1`.

Retraining budget: about 8 retrains per year for the drift-triggered arms (C and D), the same guide for both teams. It is not a strict limit: a yearly average under 9 over the test period is accepted. Arms A and B are control arms with a fixed schedule, so the budget does not apply to them.

Record for every arm:

- MAE by pre-drift, drift and post-drift regime;
- number of retrains;
- cumulative training samples;
- wall-clock time;
- MAE gain relative to Arm A;
- gain per retrain.

Do not run final Arms C and D until the detector configuration and accepted-alarm stream are frozen.

### Adaptation regimes

| Decision | Agreed value |
|---|---|
| Primary regime | Global: every TEST timestamp gets one regime. `pre_drift` before the first event; `drift` inside an event window `[drift_start, drift_end + 7 days)`; `post_drift` after that window until the next event |
| Comparison regime | Event-local, an extra experiment: 7 days before the event (`pre_drift`), the event itself (`drift`), 7 days after it (`post_drift`), with no drift-end buffer; timestamps outside every window are left out |
| Overlaps | `drift > pre_drift > post_drift`; earliest event for `drift` and `pre_drift`, most recently started event for `post_drift` |
| Reporting | Headline results use the global regime; event-local results are reported alongside for comparison |

## 15. Uncertainty quantification

| Decision | Agreed value |
|---|---|
| Methods | Fixed split conformal and AgACI, both run by both teams |
| Target | 90% prediction interval (`alpha = 0.10`) |
| Calibration | Absolute Calibration-window residuals `|actual - forecast|` |
| Fixed split conformal | One radius from the conformal quantile of the Calibration residuals; never updated on TEST. The non-adaptive baseline |
| AgACI | Online aggregation of ACI candidates; updates after each observed outcome; one instance per forecaster × region × arm × seed, not reset on retraining |
| Results | Coverage and width by regime; worst rolling 24-hour coverage |

## 16. Escalation

| Decision | Agreed value |
|---|---|
| Method | Threshold sweep: escalate a forecast for human review when its signal is strictly above threshold `τ`; every distinct signal value is tried as `τ` |
| Signals | Both teams sweep the same two signals: the 90% interval width (`upper - lower`), and the 30-minute nonconformity score (the previous half-hour's `|actual - forecast|`, known before the forecast is issued) |
| UQ methods swept | Both fixed split conformal and AgACI. Under width, fixed split conformal's constant width only allows 0% or 100% review |
| Results | Review rate needed for 0.90 and 0.95 retained coverage (`unreachable` if no threshold reaches it), and retained coverage at 5%, 10% and 20% review, by regime |

## 17. Results and reproducibility

| Decision | Agreed value |
|---|---|
| Official ledger | `results/runs.csv` |
| Format | Long format: one scalar metric per row |
| Required identifiers | Group, method, dataset, region, seed, `config_hash`, `split_id`, metric and regime |
| Required cost fields | Retrains, training samples and wall-clock time |
| Detailed outputs | Store timestamped curves and detections under the run's `config_hash` directory |
| Result writing | Experiment runners write through the shared harness |
| Tables and figures | Read recorded outputs; do not rerun experiments or recalculate official metrics |
| Ledger policy | Append-only; never manually edit historical results |
| Duplicate rows | Producers select the latest matching completed run |

## 18. Final cross-team comparison

For T7, both teams must use the same:

- AEMO regions and dates;
- train, calibration and test split;
- forecaster family;
- forecasting features and schedule;
- event catalogue;
- matching rules;
- refractory period;
- random seeds;
- four adaptation arms;
- recent-window length;
- retraining boundary and budget;
- uncertainty method;
- evaluation functions;
- logging format.

The intended comparison is:

| Team | Trigger stream |
|---|---|
| Team 41 | `TOTALDEMAND`-based trigger; primary corrected input is the half-hourly DOY/DOW/TOD seasonal residual |
| Team 42 | Processed forecast error |

No other material experimental setting should differ. Team 41 uses XGBoost and Team 42 uses `HistGradientBoostingRegressor`, both gradient boosting.
