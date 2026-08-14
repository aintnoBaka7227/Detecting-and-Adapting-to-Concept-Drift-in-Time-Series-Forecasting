"""Data-quality checks and fixes: DST gaps, duplicate intervals, missing half-hours.

Every fix applied here must be recorded in the data-quality note that is
the Step 1 deliverable — this is where those fixes actually happen, so
keep the note in sync with what this function does.
"""

import pandas as pd


def clean_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of `df` with gaps/duplicates handled or flagged.

    Must only use information available at or before each row's own
    timestamp — no full-series statistics — or the Step 7 leakage audit
    will fail on this function.
    """
    raise NotImplementedError("TODO: Data owner — Sprint 1, Step 1")
