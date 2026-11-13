# Contribution - Barathkumar Palanisamy

## Completed work not merged into final main

### `feature/table1-synthetic-detection-barath`

**Stage:** Detection / Synthetic benchmark

Independently implemented and evaluated drift detection methods using synthetic time series datasets. The work included generating synthetic datasets, testing different drift detectors, tuning their parameters, and comparing their performance.

Generated 20 synthetic datasets with 20,000 observations each, using five random seeds for sudden, gradual, recurring, and no-drift scenarios. Implemented and tested ADWIN, KSWIN, and Page-Hinkley detectors to identify drift in the data.

Performed hyperparameter tuning to improve detector performance. Tested 36 configurations for ADWIN by changing delta, clock, and minimum window length. Also tested six configurations for KSWIN using different significance levels and window sizes.

Evaluated the detectors using detection delay, false alarms per 10,000 observations, and missed detections. Compared the results across different drift scenarios and checked false alarms on datasets without drift.

**Work completed:**
- Generated synthetic datasets for four drift scenarios.
- Implemented ADWIN, KSWIN, and Page-Hinkley detection experiments.
- Performed hyperparameter tuning for ADWIN and KSWIN.
- Evaluated detection delay, false alarms, and missed detections.
- Compared detector performance across five random seeds.
- Generated evaluation results and supporting Python scripts for Table 1.

**Status:** Completed independently, but not merged into final main.

---

### `feature/table2-aemo-detection-barath`

**Stage:** Detection / AEMO validation and visualisation

Independently implemented drift detection experiments using AEMO electricity demand data from NSW1 and SA1. The work included preparing the datasets, applying deseasonalisation, running drift detectors, and evaluating the detection results.

Applied train-only deseasonalisation to remove regular seasonal patterns from the electricity demand data. Tested ADWIN, KSWIN, and Page-Hinkley detectors on the processed data to identify possible drift points.

Used a 14-day refractory period to filter alarms that occurred too close to each other. Compared the accepted alarms with documented events to evaluate detector performance.

Generated Table 2 results to compare drift detection performance across NSW1 and SA1. Also produced Figure 2 visualisations to show the detected drift points and understand detector behaviour.

**Work completed:**
- Prepared AEMO electricity demand datasets for NSW1 and SA1.
- Applied train-only deseasonalisation.
- Implemented ADWIN, KSWIN, and Page-Hinkley detection experiments.
- Applied a 14-day refractory period to filter repeated alarms.
- Compared detected alarms with documented events.
- Generated Table 2 evaluation results.
- Created Figure 2 detection visualisations.
- Organised the Python scripts, datasets, and results in the GitHub branch.

**Status:** Completed independently and uploaded to the branch, but not merged into final main.

---

### `feature/table5-fixed-conformal-barath`

**Stage:** Uncertainty Quantification / Coverage and Interval Width Evaluation

Independently implemented Fixed Split Conformal Prediction for Table 5 to evaluate the reliability of prediction intervals under concept drift. The work included preparing forecast data, calculating conformal prediction intervals, evaluating coverage and interval widths, and summarising the results.

Used XGBoost forecasts from the AEMO NSW1 and SA1 datasets with Page-Hinkley drift detection. The evaluation included four adaptation arms (A, B, C, and D), five random seeds, and three drift regimes (pre-drift, drift, and post-drift).

Calculated the conformal radius using absolute calibration residuals with a target coverage of 90%. The radius was calculated separately for each region and seed and applied consistently across all four adaptation arms.

Implemented four evaluation metrics: coverage, mean interval width, normalised interval width, and worst 24-hour coverage. For worst 24-hour coverage, used rolling windows of 48 consecutive half-hourly observations within each drift regime.

Processed 40 adaptation forecast files and 10 calibration files. Generated 120 seed-level evaluation results and summarised them into 24 final rows using mean and standard deviation across five seeds.

Compared the results with the team's Table 5 and observed similar overall patterns. Also performed separate verification of the conformal radius, interval widths, coverage, worst 24-hour coverage, window counts, and final summary calculations.

**Work completed:**
- Implemented Fixed Split Conformal Prediction with 90% target coverage.
- Prepared and validated XGBoost forecasts for NSW1 and SA1.
- Used Page-Hinkley drift regimes for the evaluation.
- Calculated conformal radii using calibration residuals.
- Generated prediction intervals for adaptation arms A, B, C, and D.
- Implemented coverage, mean interval width, and normalised interval width calculations.
- Implemented worst 24-hour coverage using 48 consecutive half-hourly observations.
- Evaluated results separately for pre-drift, drift, and post-drift regimes.
- Generated 120 seed-level results and 24 final summary rows.
- Calculated mean and standard deviation across five random seeds.
- Independently verified the evaluation calculations and compared the results with the team's Table 5.
- Uploaded the Python scripts, input datasets, evaluation results, and README documentation to the GitHub branch.

**Status:** Completed independently and uploaded to the branch, but not merged into final main.
