"""Single source of truth for paths, regions and dataset boundaries."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT_DIR / "reports"

# NEMOSIS cache for raw AEMO MMS files.
NEMOSIS_CACHE = RAW_DATA_DIR / "nemosis_cache"

REGIONS = ("SA1", "NSW1")

# Native 30-minute trading-data period.
AEMO_30MIN_START = "2018/01/01 00:00:00"
AEMO_30MIN_END = "2021/09/30 23:59:59"

# Native 5-minute dispatch-data period.
AEMO_5MIN_START = "2021/10/01 00:00:00"
AEMO_5MIN_END = "2023/12/31 23:59:59"

# Train / calibration / test split.
SPLIT = {
    "train": ("2018-01-01", "2019-12-31"),
    "calibration": ("2020-01-01", "2020-02-29"),
    "test": ("2020-03-01", None),
}

# Used later by forecasting models if the modelling data is standardised
# to half-hourly intervals.
SEASON_LENGTH = 48

ROLLING_WINDOW_DAYS = 7
INTERVAL_COVERAGE_TARGET = 0.90