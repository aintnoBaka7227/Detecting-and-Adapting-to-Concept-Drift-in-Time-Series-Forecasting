# Data Quality and Preprocessing

How AEMO demand data for **SA1** and **NSW1** is read, checked, cleaned, put on a 30-minute grid and split.

This file describes rules only. Counts and ranges live in the generated tables, so they cannot drift from the data:

| Numbers | File | Refresh with |
|---|---|---|
| Every audit and cleaning check, per region | `results/tables/data/data_audit.csv` | `python -m experiments.run.detection.check_data_audit` |
| Split dates and half-hour counts | `results/tables/data/data_splits.csv` | Same command |

`tests/test_docs_in_sync.py` fails if this file names a path or function that no longer exists.

## Pipeline

```text
data/raw/<region>/*.csv  ->  _read_monthly  ->  clean  ->  standardise  ->  split
```

All of this is in `src/drift_lab/aemo/loader.py` and runs in memory. Nothing is written to `data/`: `load_processed()` returns the cleaned 30-minute series and `load()` returns `(train, calibration, test)`, both memoised per region. Each region is processed independently.

## Reading

`_read_monthly()` joins a region's monthly `PRICE_AND_DEMAND_*.csv` files in filename order and keeps five columns: `REGION`, `SETTLEMENTDATE`, `TOTALDEMAND`, `RRP` and `PERIODTYPE`. A file missing any of them raises an error.

AEMO's native sampling is 30-minute before 1 October 2021 and 5-minute from then on.

## Cleaning

`clean()` prints the audit and never alters a demand or price value.

| Check | Action |
|---|---|
| Required columns | Stop if any is missing |
| Non-numeric demand or price | Coerced to `NaN`, so it shows up as missing |
| Invalid timestamps | Row dropped, as it cannot be placed in time |
| Out-of-order timestamps | Counted, then the series is sorted by time |
| Exact duplicate rows | Dropped |
| Same timestamp and region with different values | First row kept |
| Missing values | Reported |
| Sampling gaps against the expected 30-minute or 5-minute interval | Reported |
| Period type | Reported |
| Negative demand | Reported and kept |
| Demand and price ranges | Reported and kept |

Two results need interpreting:

- **Out-of-order rows are real.** Some raw 2018 months are not in time order within the file. Sorting fixes this, and no duplicates are involved.
- **SA1 has negative demand.** It is kept as recorded. Negative operational demand is a genuine SA1 event, and `EVT-2023-07` documents one in the catalogue.

Price (`RRP`) is audited but not used anywhere else in the project.

## 30-minute grid

`standardise()` resamples to 30 minutes with `label="right", closed="right"` and takes the mean of `TOTALDEMAND` and `RRP`. The 5-minute values in `(00:00, 00:30]` become one value stamped `00:30`, and existing 30-minute values keep their stamp. A half-hour with no data becomes `NaN` and is never imputed. `PERIODTYPE` is dropped here.

## Splits

`split()` applies the frozen `SPLIT` dates in `src/drift_lab/config.py`. The split is chronological, with no shuffling.

Windows are left-inclusive and right-exclusive on the raw AEMO stamps: a split from `s` to `e` is `[s 00:00, (e+1) 00:00)`, with no 00:30 shift. AEMO stamps mark the end of an interval, so each day window holds the previous day's 23:30–24:00 interval and not its own. The team chose this convention knowingly.

Two consequences:

- TRAIN starts at the first stamp in the data, `2018-01-01 00:30`.
- TEST has no end date, so it keeps the last stamp in the data, `2024-01-01 00:00`, which closes the final half-hour of 2023.

## After the split

`src/drift_lab/aemo/deseasonalise.py` handles the detector input. `fit_seasonal_profile()` is fitted on TRAIN only and `apply_seasonal_profile()` applies it unchanged to later splits. Forecasters use the demand values directly.
