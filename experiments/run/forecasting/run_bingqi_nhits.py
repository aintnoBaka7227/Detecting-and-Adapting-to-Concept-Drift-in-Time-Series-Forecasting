from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.bingqi import BingqiNHITS
from experiments.run_harness import config_of, record_run


SPLIT_ID = "aemo_bingqi_nhits_v1"
SEEDS = (1, 2, 3, 4, 5)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)

        train_y = demand_series(train)
        cal_y = demand_series(calibration)
        test_y = demand_series(test)

        for seed in SEEDS:
            print(f"\n=== {region} seed={seed} ===")

            model = BingqiNHITS(random_seed=seed)

            t0 = time.perf_counter()

            # Fit only on training data
            model.fit(
                pd.DataFrame(index=train_y.index),
                train_y,
            )

            # Calibration data is observed before test
            model.observe(cal_y)

            # During rolling forecasting, actual test values are only
            # used as context after each forecast block has been issued.
            predict_input = pd.DataFrame(
                {"TOTALDEMAND": test_y.to_numpy()},
                index=test_y.index,
            )

            preds = model.predict(predict_input)

            wall_clock_s = time.perf_counter() - t0

            record_run(
                method=model.name,
                dataset="aemo",
                region=region,
                seed=seed,
                config=config_of(model),
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=len(train_y),
                forecast=(
                    test_y.to_numpy(),
                    preds,
                    test_y.index,
                ),
            )

            print(
                f"{region} seed={seed}: "
                f"wall_clock={wall_clock_s:.1f}s"
            )


if __name__ == "__main__":
    main()