"""Enforce the frozen train / calibration / test split from `config.SPLIT`.

No other module should slice by date directly — always go through here, so
the split boundaries can only ever be changed in one place (and, per the
project brief, never after Checkpoint 1).
"""

import pandas as pd

from drift_forecasting.config import SPLIT


def train_calibration_test_split(
    df: pd.DataFrame, date_col: str = "SETTLEMENTDATE"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train, calibration, test) using `config.SPLIT`.

    The returned `test` frame must still be consumed in time order only by
    callers — this function guarantees the boundary, not the consumption
    order.
    """
    raise NotImplementedError("TODO: Data owner — Sprint 1, Step 1")
