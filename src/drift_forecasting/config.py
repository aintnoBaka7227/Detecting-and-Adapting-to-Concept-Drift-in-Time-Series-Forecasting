"""Single source of truth for paths, regions and the train/calibration/test split.

Fixed in Sprint 1 (Step 1) and frozen after Checkpoint 1 — every rolling-MAE
number, detector metric and coverage table downstream depends on these
boundaries never moving. If a split needs to change, it changes here and
only here, and everyone re-runs from Step 3.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT_DIR / "reports"

# Raw AEMO files downloaded/cached by NEMOSIS.
NEMOSIS_CACHE = RAW_DATA_DIR / "nemosis_cache"

REGIONS = ("SA1", "NSW1")

# Half-hourly trading-data period.
# Stop before the NEM five-minute settlement transition in October 2021.
AEMO_START = "2018/01/01 00:00:00"
AEMO_END = "2021/09/30 23:59:59"

# See drift-project-briefing, Step 1. Test end date is open — consume
# in time order only, never slice past "now" in the simulated stream.
SPLIT = {
    "train": ("2018-01-01", "2019-12-31"),
    "calibration": ("2020-01-01", "2020-02-29"),
    "test": ("2020-03-01", None),
}

SEASON_LENGTH = 48  # half-hourly intervals per day
ROLLING_WINDOW_DAYS = 7
INTERVAL_COVERAGE_TARGET = 0.90