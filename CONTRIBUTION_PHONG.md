# Contribution: Phong

Branch: `docs/S6-contribution-phong`

Format: one table per pipeline stage, then that stage's final deliverables. Only work merged into `main` is listed (68 commits). A file named on its own is all mine. For shared files, the row names the functions that are mine. Line counts are from `git blame -w -M -C` on the final `main`, and from plain `git blame` for result CSVs.

**Origin column**
- **From scratch**: no teammate had a version before me.
- **Built on [name]**: a teammate had a version first. The row names it and what I changed. Most are on `main`; exceptions say where they are.

**Built on teammates' work** (every other row is from scratch)

| Teammate | Their original | What I did | Stage |
|---|---|---|---|
| Ishan | `generator.py` (179 lines) | Rewrote it as a 60-line function that returns change points | Detection |
| Ishan | `dhr_arima_real.py` (297 lines) | Rewrote it as a model behind the shared interface | Forecasting |
| Ishan | Streamlit app | Added the Data page, extended the other pages | Results app |
| Aditya | `assign_regime` (#28) | Rebuilt it as global and event-local regime rules | Detection, used by later stages |
| Aditya | `match_unmatch`, `calculate_event_metrics` (#28) | Rebuilt event matching and precision | Detection |
| Aditya | XGBoost model (#15) | Adapted it to the shared interface, added `observe()` | Forecasting |
| Aditya | Adaptation metrics (unmerged branch `feature/s4-t3-t4-evaluation-metrics`) | Integrated and extended them | Adaptation |
| Aditya | Synthetic detection and uncertainty metrics (developed locally, shared in the group chat, not in git) | Implemented and extended them | Detection, Uncertainty |
| Antony | Detector tuning and synthetic runs (#26) | Grid search, a third acceptance rule, automatic winner selection | Detection |
| Antony | Daily aggregation in `data/deseasonalise.py` (#26) | Adopted it in place of my own | Data |
| Antony | `run_detector_input_experiments.py` (#26) | Replaced it with per-stream runners and standardised streams | Detection |
| Bingqi | F2 synthetic script (#24, 914 lines) | Replaced it with a shared figure module | Detection |
| Bingqi | T7 and F6 (unmerged branch `feature/t7-f6-bingqi-final`) | Validated his results and produced the versions on `main` | Comparison |

**To check a row:** run `gh pr view <number>` for a PR, `git show <hash>` for a commit pushed straight to `main` (marked "main"), or `git blame <file>`.

**Deliverables** are under `results/`. Every file there was created and last written by me. "Final version from" is the PR or commit that last wrote it. Most come in one version per trigger detector (`adwin/`, `page_hinkley/`) and forecast model (`dhr_arima/`, `nhits/`, `xgboost/`), for NSW1 and SA1.

## Setup and shared infrastructure

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #1 | Repository structure by pipeline stage, with fixed interfaces for forecaster, detector, adaptation and uncertainty | `src/drift_lab/*/base.py`, `config.py` | From scratch |
| #20, #23, #36, #38 | Rewrite into `src/drift_lab`: one interface for all models, one `evaluation.py` for all metrics, and a run harness that logs every result to the `runs.csv` ledger | `experiments/run_harness.py`, `experiments/results_io.py` | From scratch (Bingqi later added file locking to `results_io.py`) |
| #27, #29, `8f697f1` (main) | `experiments/` split into `run/` and `produce/` folders per stage; stale scripts removed | `experiments/run/`, `experiments/produce/` | From scratch |
| #21, #34, #38, `574fa9c` (main) | README, team decisions, and decisions shared with Team 42 | `README.md`, `TEAM_DECISIONS.md`, `SHARED_DECISIONS.md` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| Experiment ledger, one row per run (35 of 80,187 rows are Antony's) | `runs.csv` | `13a6e07` (main) |

## 1. Data

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #10, #20 | AEMO NSW1 and SA1 data from monthly CSVs (not `nemosis`, which has duplicate timestamps), cleaning, 30-minute resample, and the fixed train / calibration / test split | `src/drift_lab/aemo/loader.py` | From scratch |
| #27, #34 | Deseasonalising: a seasonal profile (base × day-of-year × weekday/half-hour) fitted on training data only and removed from later splits. `remove_daily_weekly_profile` keeps Antony's function name but is my reimplementation | `deseasonalise.py`: `fit_seasonal_profile`, `apply_seasonal_profile`, `remove_daily_seasonal_profile`, `remove_daily_weekly_profile` | From scratch |
| `16f7406` (main) | Daily aggregation: replaced my own version from #27 with Antony's, and merged his module into mine | `deseasonalise.py`: `aggregate_daily_demand` | Built on Antony (#26) |
| #34, `70c7335` (main) | Checks that the training-fit profile holds on later splits; data audit | `experiments/run/detection/check_*.py` | From scratch |
| #10, #20, `574fa9c` (main) | Data quality and event tiering docs, with a sync test | `docs/`, `tests/test_docs_in_sync.py` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| Data audit and split tables | `tables/data/` | `13a6e07` (main) |
| Residual QC figures and generalisation table | `figures/qc_seasonal_residual_*`, `figures/qc_standardised_residual_*`, `tables/qc_residual_generalisation_half_hourly.*` | #36 |

## 2. Detection

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #20 | Synthetic generator as an in-memory function that returns change points. Kept Ishan's signal design; removed CSV and plotting code | `src/drift_lab/synthetic/generator.py` | Built on Ishan (#7 to #9) |
| #20 | ADWIN, KSWIN and Page-Hinkley as `river` wrappers: the first detector modules in the package. Antony's versions arrived later in #26 | `src/drift_lab/detection/` | From scratch |
| #20 | Synthetic detection metrics: delay, false alarms per 10,000 steps, missed detections (Aditya later edited 17 of 315 lines) | `evaluation.py`: `_match_detections`, `calculate_detection_delay`, `calculate_false_alarms_per_10000`, `calculate_missed_detections`, `evaluate_detections` | Built on Aditya (developed locally, shared in the group chat, not in git). Barathkumar's notebook on `main` also computed them for a first T1 |
| #32, #34, #36, `4fc8f4e` (main) | Synthetic tuning. Antony's had two acceptance rules (≤ 2 false alarms a year, no missed drift) and a one-parameter sweep. I added a third rule (no alarms on no-drift series), a full grid search, automatic winner selection, and a later recalibration | `run_fine_tune_on_synthetic.py`, `run_post_tune_on_synthetic.py`, `post_tune_detector_configs.py`, `produce_fine_tune_calibration_summary.py` | Built on Antony (#26) |
| #20, #32 | Table 1, built from the ledger | `table_t1_common.py`, `produce_table_t1_post_tune.py` | From scratch |
| #32 | Figure 2 (synthetic) as one module for pre-tune and post-tune | `figure_f2_synthetic_common.py` | Built on Bingqi (#24) |
| #23, #27, #29 | AEMO detector runs on raw and daily streams; Table 2 and Figure 2 (AEMO) | `run_aemo_detectors_raw_post_tune.py`, `table_t2_common.py`, `figure_f2_aemo_common.py` | From scratch |
| #32, #34 | Final streams: raw, standardised daily and standardised half-hourly, one runner each. Replaced Antony's three-stream comparison and added standardisation | `run_aemo_detectors_standard_*_post_tune.py`, `standard_stream_common.py` | Built on Antony (#26) |
| #29, #34, #38 | Regimes rebuilt as two rules. **Global** (headline): every test timestamp is pre-drift, drift (event window + 7 days) or post-drift (until the next event). **Event-local** (comparison): only timestamps inside an event's windows, no buffer. Aditya's original labelled detections only. Used in T3 to T7 and F4 to F5 | `evaluation.py`: `assign_timestamp_global_regime`, `assign_event_local_regime` | Built on Aditya (#28) |
| #29, #34, #36, #38 | Event matching rebuilt: 14-day refractory debounce (repeats labelled Ignored), separate 7-day tolerances for single-date and long events, and precision over accepted detections per tier | `evaluation.py`: `match_unmatch`, `calculate_event_metrics`, `calculate_match_window`. `build_event_windows` is still Aditya's | Built on Aditya (#28) |
| #34, #36, #38 | Chance baseline, change-point files, periodicity check, alarm-timing diagnostics | `chance_baseline.py`, `run_detector_changepoints.py`, `check_detection_periodicity.py`. `evaluation.py`: `calculate_alarm_gaps`, `calculate_gap_cv`, `calculate_gap_share`, `calculate_phase_concentration`, `simulate_random_alarm_days` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| T1: synthetic detector results | `tables/table1/` | #38 |
| F2 (synthetic): sudden, gradual, recurring drift | `figures/figure2/f2_synthetic_post_tune_*` | #38 |
| T2: AEMO detections against documented events | `tables/table2/` | #38 |
| F2 (AEMO): alarms per detector and region | `figures/figure2/f2_detection_standard_half_hourly_post_tune_*` | #38 |
| Tuning grid, winners and calibration summary | `tables/fine_tune_*.csv` | #36, `4fc8f4e` (main) |
| Alarm periodicity | `tables/detection_periodicity/` | `219fa22` (main) |

## 3. Forecasting

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #14, #16, #20 | Seasonal naive (lag 48) with 7-day rolling MAE; notebook, then module | `seasonal_naive.py` | From scratch |
| #20 | DHR+ARIMA behind the shared interface. Kept Ishan's Fourier + SARIMAX method; phase now counts real elapsed time across the calibration gap | `dhr_arima.py` | Built on Ishan (#13) |
| #20 | XGBoost adapted to the shared interface, plus `observe()` so lags reach across the train-to-test boundary without refitting (286 of 305 lines are Aditya's) | `xgboost_forecaster.py`, `tests/test_xgboost_forecaster.py` | Built on Aditya (#15) |
| #20, #23, #25 | NHITS forecaster and a research notebook on its architecture | `nhits_forecaster.py`, `notebooks/research/nhits.ipynb` | From scratch |
| #17, #18, #20, #23, #29 | Frozen-baseline runs, MAE metrics, Figure 1 (Aditya later edited 8 of 71 metric lines) | `run_aemo_baselines.py`, `run_aemo_nhits.py`, `produce_figure_f1.py`. `evaluation.py`: `calculate_mae`, `calculate_rolling_mae` | From scratch |
| #36, #38, #40 | Calibration-window forecasts, forecaster selection by accuracy and run time, and a train-only hold-out check | `run_aemo_calibration_forecasts.py`, `produce_table_forecaster_selection.py`, `run_aemo_holdout_selection.py` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| F1: frozen-baseline degradation, 2020 to 2023 and per year | `figures/figure1/` | #38 |
| Forecaster selection, with the Nov–Dec 2019 hold-out | `tables/forecaster_selection/` | #40 |

## 4. Adaptation

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #27, #36, #38 | Four arms: A never, B every 30 days, C full history after an alarm, D recent window after an alarm (30, 60, 120 or 180 days). C and D triggered by ADWIN, KSWIN or Page-Hinkley. Includes retrain accounting, runner and re-scoring | `src/drift_lab/adaptation/`, `run_aemo_adaptation_arms.py`, `rescore_adaptation_runs.py`, `experiments/trigger_detectors.py` | From scratch (Bingqi's separate C/D in #35 was not merged) |
| #36 | Adaptation metrics: error by regime, adaptation gain, pinball loss | `evaluation.py`: `calculate_forecast_metrics_by_regime`, `calculate_adaptation_gain`, `calculate_pinball_loss` | Built on Aditya (unmerged branch `feature/s4-t3-t4-evaluation-metrics`, 29 Sep; my version landed the same day with the same signatures) |
| #36, #38, #43 | T3, T4, F3, arm D sweep | `experiments/produce/adaptation/` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| T3: forecast accuracy per arm, global and event-local | `tables/table3/<detector>/` | #38 |
| T4: adaptation cost per arm (retrains, samples, wall-clock, gain over arm A, gain per retrain) | `tables/table4/<detector>/` | #38 |
| F3: four-arm comparison per model and region | `figures/figure3/<detector>/<model>/` | `8f697f1` (main) |
| Arm D sweep: best window per model and region | `tables/arm_d_sweep/<detector>/` | `13a6e07` (main) |

## 5. Uncertainty

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #36 | Split conformal intervals from calibration residuals | `fixed_split_conformal.py`, `conformal_common.py` | From scratch |
| #36, #38 | AgACI (Zaffran et al. 2022); the aggregation step calls the authors' R package `opera` | `agaci.py`, `opera_boa.py`, `opera_boa.R` | From scratch |
| #36, #38 | Coverage and width metrics | `evaluation.py`: `calculate_interval_pinball_loss`, `calculate_empirical_coverage`, `calculate_mean_interval_width`, `calculate_width_scale`, `calculate_normalised_interval_width`, `calculate_rolling_coverage`, `calculate_complete_window_count`, `calculate_worst_rolling_coverage`, `calculate_uq_metrics_by_regime` | Built on Aditya (developed locally, shared in the group chat, not in git) |
| #38, `219fa22` (main) | Interval replay on test data, with the calibration run hash stored | `run_aemo_uq_replay.py` | From scratch |
| #38, #43 | T5, F4, AgACI width diagnostic | `experiments/produce/uncertainty/` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| T5: coverage and width, split conformal and AgACI, per region | `tables/table5/<detector>/` | #43 |
| F4: rolling coverage over the test period | `figures/figure4/<detector>/<model>/` | #38 |
| AgACI width diagnostic | `figures/qc_agaci_width_*` | #38 |

## 6. Escalation

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #38, #43 | Two review rules: interval-width threshold, and nonconformity-score threshold with 30-minute, 1-day and 7-day look-backs. Threshold sweep, T6, F5 | `run_aemo_escalation_sweep.py`, `experiments/produce/escalation/`. `evaluation.py`: `calculate_nonconformity_score`, `calculate_review_rate`, `calculate_retained_coverage`, `calculate_escalation_curve`, `calculate_escalation_curves_by_regime`, `calculate_review_rate_for_coverage`, `calculate_coverage_at_review_rate`, `calculate_escalation_operating_points` | From scratch |

| Deliverable | Files | Final version from |
|---|---|---|
| T6: escalation operating points per signal | `tables/table6/<detector>/<model>/<signal>/` | #38 |
| F5: escalation curves per signal | `figures/figure5/<detector>/<model>/<signal>/` | #38 |

## Comparison with Team 42 and results app

| PR / commit | What I built | Main files | Origin |
|---|---|---|---|
| #38, #40 | T7 and F6: our distribution-shift trigger against Team 42's error-based trigger. I validated Bingqi's comparison; the files on `main` come from my scripts | `produce_table_t7.py`, `produce_figure_f6.py` | Built on Bingqi (unmerged branch `feature/t7-f6-bingqi-final`) |
| `70c7335`, `587c3f0`, `b15361b`, `13a6e07`, `e66b1a5` (main) | Data page, additions to every other page, NHITS + Page-Hinkley default view, fixes | `streamlit_app.py`: `tab_data`, `synthetic_preview`, `method`, `pick_models` (new), plus parts of the other pages | Built on Ishan (#37, #41) |

| Deliverable | Files | Final version from |
|---|---|---|
| T7: Page-Hinkley vs Team 42 trigger, with settings | `tables/table7/page_hinkley/` | `13a6e07` (main) |
| F6: trigger timeline for both teams | `figures/figure6/page_hinkley/` | #40 |
| Team 42's results (their numbers, added by me as inputs) | `tables/table7/group_b_team42.csv`, `figures/figure6/group_b_team42_alarms.csv` | #40 |

PRs #15, #22 and #28 were opened from my account, but the commits are Aditya's. I merged #26, which is Antony's work. None of these is counted above.

## Line counts on final main

| Area | Mine / total |
|---|---|
| `src/drift_lab/` | 4,205 / 4,751 lines |
| `experiments/` | 8,109 / 8,672 lines |
| `tests/` | 2,811 / 3,762 lines |
| `app/streamlit_app.py` | 394 / 961 lines (rest is Ishan's) |
| `results/runs.csv` | 80,152 / 80,187 lines (rest is Antony's) |
| Other result CSVs (37 files) | 11,407 / 11,407 lines |
| Result images | 156 / 156 PNG files |
| `README.md` | 412 / 432 lines (rest is Ishan's) |
| `TEAM_DECISIONS.md`, `SHARED_DECISIONS.md` | 874 / 874 lines |
| `docs/DATA_QUALITY.md`, `docs/EVENT_TIERING_CRITERIA.md` | 121 / 121 lines |

`git blame` counts lines I rewrote in "built on" files as mine. The original approach in those files belongs to the teammate named in that row.
