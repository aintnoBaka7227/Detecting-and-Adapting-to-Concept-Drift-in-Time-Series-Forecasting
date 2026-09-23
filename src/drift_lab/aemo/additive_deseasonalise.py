"""
Shared preprocessing for concept-drift experiments.

This module extends the daily/weekly deseasonalisation approach used by
Team 41.

Two preprocessing stages are provided:

1. remove_daily_weekly_profile()
   Removes the expected seasonal demand profile from a half-hourly
   demand series.

   The seasonal profile is learned from a historical reference period
   only and contains:

       - overall mean demand
       - smoothed day-of-year effect
       - weekday / half-hour effect

   This extends Team 41's original weekday / half-hour profile with the
   day-of-year component requested in the shared supervisor protocol.

2. ErrorZScorePreprocessor
   Standardises the absolute forecast-error stream using statistics
   learned from a TRAIN-only forecast-error reference.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


HALF_HOURS_PER_DAY = 48

DOY_SMOOTHING_DAYS = 14

_EPS = 1e-12

# Demand-series validation
def _validate_datetime_series(
    series: pd.Series,
) -> None:
    """Validate a time-indexed numerical demand series."""

    if not isinstance(series, pd.Series):
        raise TypeError(
            "Demand input must be a pandas Series."
        )

    if not isinstance(
        series.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Demand series must use a pandas DatetimeIndex."
        )

    if series.empty:
        raise ValueError(
            "Demand series must not be empty."
        )

    if series.isna().any():
        raise ValueError(
            "Demand series contains missing values."
        )

    values = series.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Demand series contains non-finite values."
        )


# Circular DOY smoothing
def _smooth_day_of_year_profile(
    profile: pd.Series,
    window: int = DOY_SMOOTHING_DAYS,
) -> pd.Series:
    """Smooth a TRAIN-derived day-of-year profile circularly.

    Day-of-year is cyclic, so values near the end of the year
    contribute to smoothing values near the beginning of the year
    and vice versa.

    Only values already estimated from the reference period are used.
    No target, Calibration, or Test observations are used.
    """

    if profile.empty:
        raise ValueError(
            "Day-of-year profile must not be empty."
        )

    if window < 1:
        raise ValueError(
            "DOY smoothing window must be at least 1."
        )

    profile = (
        profile
        .sort_index()
        .astype(float)
    )

    values = profile.to_numpy(
        dtype=float
    )

    n_values = len(values)

    if window > n_values:
        raise ValueError(
            "DOY smoothing window cannot exceed "
            "the number of available DOY values."
        )

    # Use an odd window so the rolling mean is centred exactly
    # on the target day.
    if window % 2 == 0:
        window = window + 1

    half_window = (
        window // 2
    )

    extended = np.concatenate(
        [
            values[
                -half_window:
            ],
            values,
            values[
                :half_window
            ],
        ]
    )

    smoothed_extended = (
        pd.Series(
            extended
        )
        .rolling(
            window=window,
            center=True,
            min_periods=window,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    smoothed = (
        smoothed_extended[
            half_window:
            half_window + n_values
        ]
    )

    if not np.isfinite(
        smoothed
    ).all():
        raise ValueError(
            "Circular DOY smoothing produced "
            "non-finite values."
        )

    return pd.Series(
        smoothed,
        index=profile.index,
        name=profile.name,
    )


#  Seasonal demand profile
def remove_daily_weekly_profile(
    series: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """Remove the normal seasonal demand profile.

    This function starts from Team 41's weekday / half-hour
    deseasonalisation approach and extends it with a smoothed
    day-of-year component.

    The final seasonal expectation is:

        overall mean
        + smoothed day-of-year effect
        + weekday / half-hour effect

    All components are estimated from ``reference`` only.

    Estimation order
    ----------------
    1. Estimate an initial weekday / half-hour profile.
    2. Remove that profile from TRAIN demand.
    3. Estimate the day-of-year effect from the residual.
    4. Smooth the TRAIN-derived day-of-year effect circularly.
    5. Remove the annual component from TRAIN demand.
    6. Re-estimate the final weekday / half-hour profile.

    This ordering prevents weekday/weekend and time-of-day structure
    from being absorbed into the annual day-of-year profile.

    Parameters
    ----------
    series:
        Half-hourly demand series to adjust.

    reference:
        Historical half-hourly TRAIN demand used to estimate the
        seasonal profile.

    Returns
    -------
    pandas.Series
        Seasonally adjusted demand residuals.
    """

    _validate_datetime_series(
        series
    )

    _validate_datetime_series(
        reference
    )

    #  Build seasonal reference

    reference_frame = pd.DataFrame(
        {
            "demand":
                reference.astype(float),
        }
    )

    reference_frame[
        "day_of_year"
    ] = (
        reference_frame
        .index
        .dayofyear
    )

    reference_frame[
        "weekday"
    ] = (
        reference_frame
        .index
        .dayofweek
    )

    reference_frame[
        "half_hour"
    ] = (
        reference_frame.index.hour * 2
        + reference_frame.index.minute // 30
    )

    #  Overall TRAIN demand level
    overall_mean = float(
        reference_frame[
            "demand"
        ].mean()
    )

    # Initial weekday / half-hour profile
    # 
    # Start from Team 41's joint weekday / half-hour profile.
    #
    # This first estimate is used only to remove daily/weekly
    # structure before estimating the annual DOY component.

    reference_frame[
        "demand_centered"
    ] = (
        reference_frame[
            "demand"
        ]
        - overall_mean
    )

    initial_profile = (
        reference_frame
        .groupby(
            [
                "weekday",
                "half_hour",
            ]
        )[
            "demand_centered"
        ]
        .mean()
    )

    initial_expected = (
        pd.MultiIndex.from_arrays(
            [
                reference_frame[
                    "weekday"
                ],
                reference_frame[
                    "half_hour"
                ],
            ],
            names=[
                "weekday",
                "half_hour",
            ],
        )
        .map(
            initial_profile
        )
    )

    if pd.isna(
        initial_expected
    ).any():
        raise ValueError(
            "Initial weekday/half-hour profile "
            "contains missing combinations."
        )

    reference_frame[
        "demand_without_daily_weekly"
    ] = (
        reference_frame[
            "demand"
        ].to_numpy(
            dtype=float
        )
        - overall_mean
        - np.asarray(
            initial_expected,
            dtype=float,
        )
    )

    #  Raw day-of-year profile
    #
    # Daily/weekly structure has already been removed, so the
    # DOY profile is less contaminated by weekday/weekend and
    # intraday effects.

    day_of_year_profile = (
        reference_frame
        .groupby(
            "day_of_year"
        )[
            "demand_without_daily_weekly"
        ]
        .mean()
    )

    # Smooth annual profile circularly
    # 
    # Annual seasonality changes gradually through the year.
    #
    # Circular smoothing also ensures that the end and beginning
    # of the year are treated as neighbouring points.

    day_of_year_profile = (
        _smooth_day_of_year_profile(
            day_of_year_profile,
            window=DOY_SMOOTHING_DAYS,
        )
    )

    # Re-centre after smoothing so the annual component has
    # approximately zero mean.

    day_of_year_profile = (
        day_of_year_profile
        - float(
            day_of_year_profile.mean()
        )
    )

    #  Map annual effect back to TRAIN 
    reference_frame[
        "day_of_year_effect"
    ] = (
        reference_frame[
            "day_of_year"
        ]
        .map(
            day_of_year_profile
        )
    )

    if (
        reference_frame[
            "day_of_year_effect"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Could not map TRAIN day-of-year effects."
        )

    #  Remove annual component from TRAIN
  
    reference_frame[
        "demand_without_annual"
    ] = (
        reference_frame[
            "demand"
        ]
        - overall_mean
        - reference_frame[
            "day_of_year_effect"
        ]
    )

    #  Final weekday / half-hour profile
    # 
    # Re-estimate Team 41's joint weekday / half-hour profile
    # after the annual component has been removed.

    profile = (
        reference_frame
        .groupby(
            [
                "weekday",
                "half_hour",
            ]
        )[
            "demand_without_annual"
        ]
        .mean()
    )

    # Centre the final profile to avoid changing the overall
    # TRAIN demand level.

    profile = (
        profile
        - float(
            profile.mean()
        )
    )

    #  Build target frame
   
    target = pd.DataFrame(
        {
            "demand":
                series.astype(float),
        }
    )

    target[
        "day_of_year"
    ] = (
        target
        .index
        .dayofyear
    )

    target[
        "weekday"
    ] = (
        target
        .index
        .dayofweek
    )

    target[
        "half_hour"
    ] = (
        target.index.hour * 2
        + target.index.minute // 30
    )

    #  Look up annual component
    
    expected_day_of_year = (
        target[
            "day_of_year"
        ]
        .map(
            day_of_year_profile
        )
    )

    #  Handle unseen DOY values
    # Example:
    # TRAIN may contain no leap year, so day 366 may be absent
    # from the frozen TRAIN profile.
    #
    # No target information is used to estimate the missing
    # value. Instead, the nearest available TRAIN DOY effect is
    # used.

    if (
        expected_day_of_year
        .isna()
        .any()
    ):

        available_days = (
            day_of_year_profile
            .index
            .to_numpy(
                dtype=int
            )
        )

        if len(
            available_days
        ) == 0:
            raise ValueError(
                "Seasonal reference produced an empty "
                "day-of-year profile."
            )

        missing_days = (
            target.loc[
                expected_day_of_year.isna(),
                "day_of_year",
            ]
            .unique()
        )

        for missing_day in missing_days:

            nearest_day = int(
                available_days[
                    np.argmin(
                        np.abs(
                            available_days
                            - int(
                                missing_day
                            )
                        )
                    )
                ]
            )

            nearest_effect = float(
                day_of_year_profile.loc[
                    nearest_day
                ]
            )

            missing_mask = (
                target[
                    "day_of_year"
                ]
                == missing_day
            )

            expected_day_of_year.loc[
                missing_mask
            ] = nearest_effect

    if (
        expected_day_of_year
        .isna()
        .any()
    ):
        raise ValueError(
            "Could not assign a TRAIN-derived "
            "day-of-year effect to every target observation."
        )

    #  Look up final weekday / half-hour component
   
    expected_daily_weekly = (
        pd.MultiIndex.from_arrays(
            [
                target[
                    "weekday"
                ],
                target[
                    "half_hour"
                ],
            ],
            names=[
                "weekday",
                "half_hour",
            ],
        )
        .map(
            profile
        )
    )

    if pd.isna(
        expected_daily_weekly
    ).any():
        raise ValueError(
            "Seasonal reference does not contain every "
            "weekday/half-hour combination required by "
            "the target series."
        )

    #  Complete frozen seasonal expectation

    expected = (
        overall_mean
        + expected_day_of_year.to_numpy(
            dtype=float
        )
        + np.asarray(
            expected_daily_weekly,
            dtype=float,
        )
    )

    #  Seasonal residual
    adjusted = pd.Series(
        target[
            "demand"
        ].to_numpy(
            dtype=float
        )
        - expected,
        index=series.index,
        name=(
            f"{series.name or 'demand'}"
            "_seasonally_adjusted"
        ),
    )

    return adjusted


#  Project dataframe helper
def deseasonalise_demand_frame(
    data: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Deseasonalise TOTALDEMAND using a frozen reference period.

    Parameters
    ----------
    data:
        Project dataframe containing SETTLEMENTDATE and TOTALDEMAND.

    reference:
        TRAIN dataframe used to estimate the seasonal profile.

    Returns
    -------
    pandas.DataFrame
        Copy of ``data`` with TOTALDEMAND replaced by the seasonal
        residual.
    """

    required = {
        "SETTLEMENTDATE",
        "TOTALDEMAND",
    }

    missing_data = (
        required.difference(
            data.columns
        )
    )

    missing_reference = (
        required.difference(
            reference.columns
        )
    )

    if missing_data:
        raise ValueError(
            "Data is missing required columns: "
            f"{sorted(missing_data)}"
        )

    if missing_reference:
        raise ValueError(
            "Reference is missing required columns: "
            f"{sorted(missing_reference)}"
        )

    transformed = (
        data.copy()
        .reset_index(
            drop=True
        )
    )

    series = pd.Series(
        transformed[
            "TOTALDEMAND"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            pd.to_datetime(
                transformed[
                    "SETTLEMENTDATE"
                ]
            )
        ),
        name="TOTALDEMAND",
    )

    reference_series = pd.Series(
        reference[
            "TOTALDEMAND"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            pd.to_datetime(
                reference[
                    "SETTLEMENTDATE"
                ]
            )
        ),
        name="TOTALDEMAND",
    )

    adjusted = (
        remove_daily_weekly_profile(
            series=series,
            reference=reference_series,
        )
    )

    transformed[
        "TOTALDEMAND"
    ] = (
        adjusted.to_numpy(
            dtype=float
        )
    )

    return transformed


#  Forecast-error standardisation

@dataclass
class ErrorZScorePreprocessor:
    """TRAIN-only z-score for the forecast-error stream.

    This stage performs no seasonal adjustment.

    It is fitted on a chronological TRAIN-only absolute
    forecast-error reference after demand has already been
    deseasonalised.
    """

    mean_: float | None = field(
        default=None,
        init=False,
    )

    std_: float | None = field(
        default=None,
        init=False,
    )

    is_fitted_: bool = field(
        default=False,
        init=False,
    )

    def fit(
        self,
        train_errors,
    ) -> "ErrorZScorePreprocessor":

        values = pd.to_numeric(
            pd.Series(
                train_errors
            ).reset_index(
                drop=True
            ),
            errors="raise",
        ).astype(float)

        if values.empty:
            raise ValueError(
                "Training error stream must not be empty."
            )

        if values.isna().any():
            raise ValueError(
                "Training error stream contains NaN values."
            )

        array = values.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                "Training error stream contains non-finite values."
            )

        self.mean_ = float(
            np.mean(
                array
            )
        )

        self.std_ = float(
            np.std(
                array,
                ddof=0,
            )
        )

        if (
            not np.isfinite(
                self.std_
            )
            or self.std_ <= _EPS
        ):
            raise ValueError(
                "Training error standard deviation is "
                "zero or too small."
            )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        errors,
    ) -> pd.Series:

        if not self.is_fitted_:
            raise RuntimeError(
                "ErrorZScorePreprocessor must be fitted first."
            )

        values = pd.to_numeric(
            pd.Series(
                errors
            ).reset_index(
                drop=True
            ),
            errors="raise",
        ).astype(float)

        if values.isna().any():
            raise ValueError(
                "Error stream contains NaN values."
            )

        array = values.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                "Error stream contains non-finite values."
            )

        z = (
            array
            - self.mean_
        ) / self.std_

        return pd.Series(
            z,
            index=values.index,
            name="processed_error",
        )

    def fit_transform(
        self,
        train_errors,
    ) -> pd.Series:

        self.fit(
            train_errors
        )

        return self.transform(
            train_errors
        )

    def get_fitted_statistics(
        self,
    ) -> dict:

        if not self.is_fitted_:
            raise RuntimeError(
                "ErrorZScorePreprocessor has not been fitted."
            )

        return {
            "fit_target":
                "absolute_forecast_error",

            "mean":
                self.mean_,

            "std":
                self.std_,
        }