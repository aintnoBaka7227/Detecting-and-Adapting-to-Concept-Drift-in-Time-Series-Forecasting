# AEMO Drift Detection - Table 2 and Figure 2

## 1. Overview

This folder contains my implementation of concept drift detection on AEMO electricity demand data for NSW1 and SA1.

I used three drift detection methods:

- ADWIN
- KSWIN
- Page-Hinkley

The main aim was to check how these detectors perform on real electricity demand data and whether their alarms match documented events in the electricity market.

I used the detector parameters selected from the synthetic experiments in Table 1. I kept these parameters fixed when running the AEMO experiments.

The results are presented in Table 2 and Figure 2.

## 2. Dataset

I used AEMO electricity demand data from 2018 to 2023 for two regions:

- NSW1 - New South Wales
- SA1 - South Australia

The data contains electricity demand observations recorded at 30-minute intervals.

I divided the data into three parts:

- Training data: 2018 to 2019
- Calibration data: January to February 2020
- Testing data: March 2020 to December 2023

The training and calibration data were used before the testing stage. The final detector alarms and evaluation results were collected from the testing period.

## 3. Data Preprocessing

Electricity demand has regular seasonal patterns. These patterns can affect drift detection, so I applied deseasonalisation before running the detectors.

I fitted the seasonal profile using the training data only. The same profile was then applied to the calibration and testing data.

I also standardised the residuals using statistics calculated from the training data.

This helped me use the same preprocessing method across both regions without using information from the testing period to fit the seasonal profile.

## 4. Drift Detection

I applied ADWIN, KSWIN and Page-Hinkley to the standardised residuals.

The detector settings were taken from the synthetic experiments and were not tuned again using the AEMO testing data.

For each detector and region, I saved the raw alarm timestamps.

I then used a 14-day refractory period during evaluation to avoid counting repeated alarms too close to each other.

The evaluation separates the alarms into accepted and ignored alarms.

## 5. Event Matching and Evaluation

I used the documented event catalogue in `data/events.csv` to evaluate the detector alarms.

The catalogue contains Tier 1 and Tier 2 events for NSW1, SA1 and the wider National Electricity Market.

I used the shared evaluation functions to compare accepted alarms with the documented event windows.

The evaluation includes:

- Raw alarm counts
- Accepted alarm counts
- Tier 1 matched events
- Tier 2 matched events
- Unmatched accepted alarms
- Detection delays
- Tier 1 precision

I also excluded the negative control event from the positive Tier 1 event matching.

The matched events and evaluation results were saved as CSV files for checking.

## 6. Chance Baseline

I calculated a random-chance baseline for the Tier 1 events.

This baseline estimates how many events could be matched if the accepted alarms were placed randomly across the testing period.

I used the number of accepted alarms and the documented event matching windows to calculate the expected number of matches.

This is an approximate baseline. It assumes random alarms are independently and uniformly distributed and does not directly simulate the 14-day refractory period.

## 7. Table 2

I created Table 2 using the saved evaluation results.

The table includes the detection delays for four selected Tier 1 events:

- COVID-19 demand change
- December solar and mild weather
- 2022 market suspension
- System-security directions

It also includes the raw and accepted alarm counts, Tier 1 and Tier 2 matches, unmatched alarms, Tier 1 precision and chance baseline.

The final results are saved in:

`outputs/table2_results.csv`

The detailed evaluation results are saved in:

`outputs/table2_detailed_results.csv`

## 8. Figure 2

I created Figure 2 to show the drift detection results for NSW1 and SA1.

The figure contains six panels, with one panel for each detector and region.

Each panel shows:

- Daily mean standardised residuals
- 15-day moving average
- Documented Tier 1 event periods
- Tier 1 matched alarms
- Tier 2 matched alarms
- Unmatched accepted alarms
- Ignored alarms
- Detection counts and event delays

The detectors were run on the half-hourly residual data. I used daily averages only to make the figure easier to read.

The combined figure is saved in:

`outputs/figure/figure2_detection_panel.png`

Individual detector figures are also saved in the `nsw1` and `sa1` folders.

## 9. Folder Structure

```text
table2_aemo_detection/
    README.md
    data/
        events.csv
        NSW.csv
        SA.csv
    experiment/
        analyse_unmatched_alarms.py
        calculate_chance_baseline.py
        check_aemo_alarms.py
        check_alarm_residuals.py
        check_tier1_matches.py
        create_figure2.py
        create_table2.py
        run_aemo_detectors.py
        run_aemo_evaluation.py
    src/
        aemo_detectors.py
        aemo_preprocessing.py
        deseasonalise.py
        evaluation.py
    outputs/
        figure/
        *.csv
```

## 10. Running the Experiments

The implementation uses Python with packages including pandas, numpy, matplotlib and river.

After installing the required packages, the scripts can be run from the `table2_aemo_detection` folder in the following order:

```bash
python experiment/run_aemo_detectors.py
python experiment/run_aemo_evaluation.py
python experiment/check_tier1_matches.py
python experiment/calculate_chance_baseline.py
python experiment/create_table2.py
python experiment/create_figure2.py
```

The first script runs the detectors and saves their alarms.

The evaluation script compares the alarms with the documented events.

The next two scripts prepare the Tier 1 matching results and chance baseline.

The final two scripts generate Table 2 and Figure 2.

## 11. Final Results

I completed the detection and evaluation experiments for all three detectors on NSW1 and SA1.

The final outputs include:

- Table 2 with detection delays and evaluation metrics
- Figure 2 with the detection results for both regions
- Raw alarm timestamps for each detector
- Event matching results
- Random-chance baseline results
- Additional alarm analysis files

These results were generated using the same detector parameters selected from the synthetic experiments and the shared event evaluation method.
