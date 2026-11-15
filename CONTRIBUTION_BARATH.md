# Contribution - Barathkumar Palanisamy

## `feature/table1-synthetic-detection-barath`

**Stage:** Detection / Synthetic Benchmark

Independently implemented and evaluated ADWIN, KSWIN, and Page-Hinkley drift detectors on synthetic time series data.

- Performed hyperparameter tuning with 36 ADWIN and six KSWIN configurations.
- Evaluated detection delay, false alarms per 10,000 observations, and missed detections.
- Compared detector performance across four drift scenarios and five random seeds.
- Generated Table 1 evaluation results and supporting Python scripts.

**Status:** Completed, not merged into main.

---

## `feature/table2-aemo-detection-barath`

**Stage:** Detection / AEMO Validation

Independently implemented drift detection experiments on NSW1 and SA1 electricity demand data.

- Implemented train-only deseasonalisation.
- Tested ADWIN, KSWIN, and Page-Hinkley detectors.
- Applied a 14-day refractory period and evaluated detections against documented events.
- Generated Table 2 evaluation results and Figure 2 visualisations.

**Status:** Completed, not merged into main.

---

## `feature/table5-fixed-conformal-barath`

**Stage:** Uncertainty Quantification

Independently implemented Fixed Split Conformal Prediction with 90% target coverage for Table 5.

- Calculated conformal radii and prediction intervals for adaptation arms A-D.
- Implemented coverage, mean width, normalised width, and worst 24-hour coverage metrics.
- Evaluated pre-drift, drift, and post-drift regimes across NSW1 and SA1.
- Generated 120 seed-level results and 24 summary rows across five seeds.
- Verified the calculations and uploaded scripts, results, and documentation.

**Status:** Completed, not merged into main.
