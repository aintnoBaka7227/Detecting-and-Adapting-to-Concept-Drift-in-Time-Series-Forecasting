"""Load raw AEMO price & demand CSVs into a single, time-sorted DataFrame.

Signature is fixed so every other module can call it the same way
regardless of who wrote the body.
"""

from pathlib import Path

import pandas as pd

from drift_forecasting.config import RAW_DATA_DIR


def load_aemo_demand(region: str, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate monthly AEMO CSVs for one region (e.g. "SA1").

    Returns a DataFrame sorted ascending by SETTLEMENTDATE, with at least
    a `TOTALDEMAND` column. Does not clean or split the result — see
    `cleaning.py` and `splits.py` for those steps.
    """
    raise NotImplementedError("TODO: Data owner — Sprint 1, Step 1")
