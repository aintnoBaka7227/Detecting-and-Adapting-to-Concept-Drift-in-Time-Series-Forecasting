# Data Quality and Preprocessing

## 1. Overview

This document records how the AEMO electricity demand and price data for
**NSW1** and **SA1** are loaded, quality checked, cleaned, standardised
to a common sampling interval, and divided into train, calibration, and
test datasets.

The preprocessing pipeline is:

``` text
Monthly AEMO CSV files
        ↓
      loader.py
        ↓
Combined native-frequency regional datasets
        ↓
     cleaning.py
        ↓
Data-quality checks and safe cleaning
        ↓
Standardisation to 30-minute intervals
        ↓
Cleaned 30-minute regional datasets
        ↓
      splits.py
        ↓
Train / Calibration / Test datasets
```

The two regions are processed independently throughout the pipeline.

------------------------------------------------------------------------

## 2. Raw Data Loading

The original AEMO data are stored as separate monthly files for NSW1 and
SA1. `loader.py` reads the monthly files for each region, retains the
required columns, concatenates the files in chronological order, and
saves one combined raw dataset per region.

The retained columns are:

  Column             Description
  ------------------ ------------------------------------------
  `REGION`           AEMO region identifier (`NSW1` or `SA1`)
  `SETTLEMENTDATE`   Timestamp of the trading interval
  `TOTALDEMAND`      Regional electricity demand
  `RRP`              Regional Reference Price
  `PERIODTYPE`       AEMO period type

No cleaning or frequency standardisation is performed by the loader.
This preserves the source data before the quality-checking stage.

### Loader output

  Region        Rows   Columns Output
  -------- --------- --------- ------------------------------------------
  NSW1       302,448         5 `data/raw/NSW1_201801_202312_native.csv`
  SA1        302,448         5 `data/raw/SA1_201801_202312_native.csv`

The combined datasets preserve AEMO's native sampling frequencies:

-   **2018-01 to 2021-09:** 30-minute intervals
-   **2021-10 onward:** 5-minute intervals

------------------------------------------------------------------------

## 3. Data Quality Checks and Cleaning

`cleaning.py` reads each combined regional dataset independently.
Quality checks are performed before the data are standardised to a
common frequency.

### 3.1 Timestamp quality

Both datasets contained valid and chronologically ordered timestamps.

  Check                                      NSW1                   SA1
  ------------------------- --------------------- ---------------------
  Invalid timestamps                            0                     0
  Out-of-order timestamps                       0                     0
  First timestamp             2018-01-01 00:30:00   2018-01-01 00:30:00
  Last timestamp              2024-01-01 00:00:00   2024-01-01 00:00:00

No timestamp rows needed to be removed.

Although the filenames identify the source period as 2018-01 to 2023-12,
the final settlement timestamp is `2024-01-01 00:00:00`, representing
the end boundary of the final interval in the supplied data.

### 3.2 Duplicate observations

No duplicate observations were found.

  Check                                    NSW1   SA1
  -------------------------------------- ------ -----
  Exact duplicate rows                        0     0
  Conflicting duplicate timestamp rows        0     0

Therefore, no rows were removed during duplicate handling.

### 3.3 Missing values

Neither regional dataset contained missing values in any of the retained
columns.

  Column               NSW1 missing   SA1 missing
  ------------------ -------------- -------------
  `REGION`                        0             0
  `SETTLEMENTDATE`                0             0
  `TOTALDEMAND`                   0             0
  `RRP`                           0             0
  `PERIODTYPE`                    0             0

No interpolation or missing-value replacement was required.

### 3.4 Native sampling frequency

The timestamp intervals were checked separately on each side of AEMO's
frequency change.

  -------------------------------------------------------------------------
  Period          Expected             NSW1 unexpected       SA1 unexpected
                  frequency                  intervals            intervals
  --------------- --------------- -------------------- --------------------
  Before          30 minutes                         0                    0
  2021-10-01                                           

  From 2021-10-01 5 minutes                          0                    0
  -------------------------------------------------------------------------

The results confirm that both regional datasets are continuous at their
expected native frequencies.

### 3.5 Period type

All observations had the same period type:

  Region     `TRADE` rows
  -------- --------------
  NSW1            302,448
  SA1             302,448

No other `PERIODTYPE` values were present.

### 3.6 Demand and price ranges

Basic range checks were performed on `TOTALDEMAND` and `RRP`.

  Measure                               NSW1         SA1
  ------------------------------ ----------- -----------
  Minimum `TOTALDEMAND`             3,664.34      -46.35
  Maximum `TOTALDEMAND`            13,700.90    3,141.66
  Negative demand observations             0          87
  Minimum `RRP`                    -1,000.00   -1,000.00
  Maximum `RRP`                    16,599.89   16,600.00

NSW1 contained no negative demand observations. SA1 contained **87
negative `TOTALDEMAND` observations**. These observations were
identified by the quality check but were not automatically removed by
the current cleaning process.

Negative and extreme `RRP` observations were also retained. The cleaning
stage reports these values rather than treating them automatically as
invalid observations.

### 3.7 Result after quality checks

No rows were removed from either dataset during the current cleaning
stage.

  ---------------------------------------------------------------------------
  Region          Rows before cleaning  Rows after cleaning Chronologically
                                                            sorted
  --------------- -------------------- -------------------- -----------------
  NSW1                         302,448              302,448 Yes

  SA1                          302,448              302,448 Yes
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Sampling-Interval Standardisation

The raw datasets cannot be passed directly to a forecasting pipeline as
one uniformly sampled series because the sampling frequency changes
during the study period:

``` text
2018-01                     2021-10                     2024-01
|-----------------------------|----------------------------|
          30 minutes                     5 minutes
```

A consistent interval is required for subsequent time-series modelling.
Therefore, after the quality checks, both datasets are standardised to
**30-minute intervals**.

The 30-minute portion is already at the target frequency. The 5-minute
portion is aggregated into 30-minute intervals.

For each 30-minute interval:

-   `TOTALDEMAND` is aggregated using the mean.
-   `RRP` is aggregated using the mean.

Conceptually:

``` text
Native data

30 min   30 min   30 min      5 min  5 min  5 min  5 min  5 min  5 min
  |        |        |           \      |      |      |      |      /
  |        |        |            \----- 30-minute aggregation -----/
  ↓        ↓        ↓                         ↓

Standardised data

30 min   30 min   30 min                    30 min
```

This produces one consistent sampling frequency across the complete time
series.

### Standardised outputs

  -----------------------------------------------------------------------------------------------------------------
  Region                   Native rows Rows after 30-minute Output
                                            standardisation 
  --------------- -------------------- -------------------- -------------------------------------------------------
  NSW1                         302,448              105,168 `data/processed/NSW1_201801_202312_cleaned_30min.csv`

  SA1                          302,448              105,168 `data/processed/SA1_201801_202312_cleaned_30min.csv`
  -----------------------------------------------------------------------------------------------------------------

The reduction from 302,448 to 105,168 rows is expected because the
5-minute observations from October 2021 onward are aggregated into
30-minute intervals.

------------------------------------------------------------------------

## 5. Train, Calibration, and Test Split

The standardised 30-minute datasets are passed to `splits.py`.

The split is **chronological rather than random**. This is necessary for
time-series forecasting because future observations must not be used to
train a model that is intended to predict earlier observations.

The frozen split is:

  Split         Period                     Purpose
  ------------- -------------------------- -----------------------
  Train         2018-01-01 to 2019-12-31   Model training
  Calibration   2020-01-01 to 2020-02-29   Calibration stage
  Test          2020-03-01 onward          Sequential evaluation

The same boundaries are applied independently to NSW1 and SA1.

### NSW1 split

  Split             Rows First timestamp       Last timestamp
  ------------- -------- --------------------- ---------------------
  Train           35,039 2018-01-01 00:30:00   2019-12-31 23:30:00
  Calibration      2,880 2020-01-01 00:00:00   2020-02-29 23:30:00
  Test            67,249 2020-03-01 00:00:00   2024-01-01 00:00:00

### SA1 split

  Split             Rows First timestamp       Last timestamp
  ------------- -------- --------------------- ---------------------
  Train           35,039 2018-01-01 00:30:00   2019-12-31 23:30:00
  Calibration      2,880 2020-01-01 00:00:00   2020-02-29 23:30:00
  Test            67,249 2020-03-01 00:00:00   2024-01-01 00:00:00

For each region:

``` text
35,039 + 2,880 + 67,249 = 105,168 rows
```

Therefore, every row in the standardised dataset is assigned to exactly
one of the three splits.

The resulting files are:

``` text
data/processed/
├── NSW1_train.csv
├── NSW1_calibration.csv
├── NSW1_test.csv
├── SA1_train.csv
├── SA1_calibration.csv
└── SA1_test.csv
```

------------------------------------------------------------------------

## 6. Final Data Pipeline

The complete preprocessing process is:

``` text
data/raw/nsw1/ monthly files
data/raw/sa1/  monthly files
              │
              ▼
          loader.py
              │
              ├── NSW1_201801_202312_native.csv
              └── SA1_201801_202312_native.csv
                       │
                       ▼
                   cleaning.py
                       │
              Data-quality checks
              ├── timestamps
              ├── duplicates
              ├── missing values
              ├── native frequency
              ├── period type
              └── demand / price ranges
                       │
                       ▼
              30-minute standardisation
                       │
              ├── NSW1_..._cleaned_30min.csv
              └── SA1_..._cleaned_30min.csv
                       │
                       ▼
                    splits.py
                       │
              ┌────────┼────────────┐
              ▼        ▼            ▼
            Train   Calibration    Test
```

## 7. Data Quality Summary

The source data were highly complete: both NSW1 and SA1 contained no
missing values, invalid timestamps, duplicate observations, out-of-order
timestamps, or unexpected native-frequency intervals.

The main quality item identified was **87 negative `TOTALDEMAND`
observations in SA1**. These values are retained by the current pipeline
rather than silently removed, allowing their treatment to be considered
explicitly during later analysis.

After quality checking, both regional datasets are standardised to a
common **30-minute frequency**, reducing each dataset from **302,448
native-frequency observations to 105,168 observations**. The resulting
series are then divided chronologically into fixed train, calibration,
and test periods, with no random shuffling.
