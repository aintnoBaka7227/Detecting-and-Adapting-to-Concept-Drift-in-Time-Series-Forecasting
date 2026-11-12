# Synthetic Drift Detection - Table 1

## About this work

In this part of the project, I worked on detecting concept drift using three detectors: ADWIN, KSWIN and Page-Hinkley.

I used synthetic time series data because we already know where the drift happens. This helps us check whether the detectors can find the changes correctly.

I also tested different parameter settings for the detectors and used the selected settings for the final experiment.

## 1. Synthetic Data

I generated four types of synthetic data:

- Sudden - The data changes suddenly at observation 5000.
- Gradual - The data slowly changes between observations 5000 and 6000.
- Recurring - The data changes at observation 6666 and returns to its original level at observation 13333.
- None - The data does not have any drift.

Each dataset contains 20000 observations.

The data also has a seasonal pattern of 48 observations, which represents one day of half-hourly data.

I used two sets of random seeds:

- Calibration: 101, 102, 103, 104, 105
- Testing: 1, 2, 3, 4, 5

The calibration datasets were used for selecting the detector parameters. The test datasets were used for the final results.

## 2. Preprocessing

Before running the detectors, I removed the seasonal pattern from the data.

I used the first 4000 observations to calculate the seasonal pattern, mean and standard deviation.

The same values were then used to preprocess the full dataset.

This is done to avoid using future data when calculating the preprocessing values.

The preprocessing code is in `src/preprocessing.py`.

## 3. Drift Detectors

I used three detectors:

- ADWIN
- KSWIN
- Page-Hinkley

The detector code is in `src/detectors.py`.

Each detector reads the data one observation at a time. When it detects a change, the observation number is saved.

## 4. Parameter Tuning

I tested different parameter settings for each detector using the calibration datasets.

The tuning scripts are:

- `experiment/tune_detectors.py`
- `experiment/tune_detectors_expand.py`

After tuning, I selected one configuration for each detector.

The selected parameters are saved in:

`outputs/expanded_best_parameters.csv`

These saved parameters are used for the final test, so there is no need to run the tuning again just to reproduce Table 1.

## 5. Testing and Evaluation

I ran the three detectors on the 20 test datasets using the selected parameters.

The test script is:

`experiment/run_synthetic_test.py`

I used the following metrics:

- Detection delay - How many observations the detector takes to detect the drift after it starts.
- False alarms per 10000 observations - How often the detector reports drift that does not match a known drift event.
- Missed detections - Number of drift events the detector did not detect.

I used a matching window of 336 observations.

For gradual drift, the change from observation 5000 to 6000 is treated as one event. Detections are matched during this period and up to 336 observations after it ends.

The evaluation code is in `src/evaluation.py`.

## 6. Table 1 Results

After testing, I used `experiment/run_table1.py` to generate Table 1.

The results are saved in:

- `outputs/synthetic_test_results.csv` - Results for each detector and dataset.
- `outputs/synthetic_test_summary.csv` - Average results for each detector and drift type.
- `outputs/table/table_1_synthetic_detection.csv` - Final Table 1.

Table 1 contains the detection delay, false alarms, missed detections, detector settings and number of seeds.

## 7. How to Run the Code

If you want to reproduce Table 1, first download this folder and open a terminal inside `barath_table1`.

Make sure Python is installed.

### Step 1 - Install the packages

```bash
python -m pip install -r requirements.txt
```

### Step 2 - Generate the synthetic datasets

```bash
python experiment/generate_synthetic_data.py
```

This generates 40 synthetic datasets in total.

The datasets are saved in:

- `data/synthetic/calibration/`
- `data/synthetic/test/`

### Step 3 - Run the final test

```bash
python experiment/run_synthetic_test.py
```

This runs ADWIN, KSWIN and Page-Hinkley on the test datasets using the saved parameters.

The results are saved in the `outputs` folder.

### Step 4 - Generate Table 1

```bash
python experiment/run_table1.py
```

This generates the final Table 1 CSV file in:

`outputs/table/table_1_synthetic_detection.csv`

You do not need to run the tuning scripts again because the selected parameters are already saved.

## 8. Files in This Folder

- `src/synthetic_generator.py` - Generates synthetic data.
- `src/preprocessing.py` - Removes seasonality and standardises the data.
- `src/detectors.py` - Runs the three drift detectors.
- `src/evaluation.py` - Calculates the detection results.
- `experiment/generate_synthetic_data.py` - Creates the calibration and test datasets.
- `experiment/run_baseline.py` - Runs the baseline experiments.
- `experiment/tune_detectors.py` - Tests different detector parameters.
- `experiment/tune_detectors_expand.py` - Tests additional parameter settings.
- `experiment/run_synthetic_test.py` - Runs the final synthetic test.
- `experiment/run_table1.py` - Creates Table 1.
- `outputs/` - Contains the saved parameters and results.
- `requirements.txt` - Contains the required Python packages.
