
import os
import sys

import numpy as np
import pandas as pd

# Import files from the project folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.synthetic_generator import make_series


# Data settings
n = 20000
noise = 1.0

drift_types = ["sudden", "gradual", "recurring", "none"]

# Seeds for calibration and testing
datasets = {
    "calibration": [101, 102, 103, 104, 105],
    "test": [1, 2, 3, 4, 5]
}


for split, seeds in datasets.items():

    folder = f"data/synthetic/{split}"
    os.makedirs(folder, exist_ok=True)

    for kind in drift_types:

        for seed in seeds:

            # Generate the data
            values, changepoints = make_series(
                kind=kind,
                n=n,
                noise=noise,
                seed=seed
            )

            # Create the CSV data
            df = pd.DataFrame({
                "t": np.arange(n),
                "y": values,
                "is_changepoint": np.zeros(n, dtype=int)
            })

            # Mark drift points
            for cp in changepoints:
                df.loc[cp, "is_changepoint"] = 1

            # Save the file
            filename = f"series_{kind}_n{n}_seed{seed}.csv"
            file_path = os.path.join(folder, filename)

            df.to_csv(file_path, index=False)

            print(
                split,
                kind,
                "seed:", seed,
                "changepoints:", changepoints
            )


print("\nSynthetic dataset generation completed.")
