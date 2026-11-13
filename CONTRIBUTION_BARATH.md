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
