"""T3 — forecast accuracy by regime.

Consumes forecast metrics already recorded through the shared run harness.
This producer does not calculate MAE or fabricate quantile forecasts.

Required columns:
    arm
    region
    MAE pre-drift
    MAE drift
    MAE post-drift
    pinball loss
    seeds
"""

from __future__ import annotations

import json

import pandas as pd

from experiments import results_io
from experiments.produce.table_image import save_table_image

ARM_ORDER = ("A", "B", "C", "D")

ARM_LABELS = {
    "A": "A — never retrain",
    "B": "B — retrain every 30 days",
    "C": "C — on drift, full history",
    "D": "D — on drift, recent window",
}

REGIMES = ("pre-drift", "drift", "post-drift")

TABLE_COLUMNS = [
    "arm",
    "region",
    "MAE pre-drift",
    "MAE drift",
    "MAE post-drift",
    "pinball loss",
    "seeds",
]


def _load_config(config_hash: str) -> dict:
    path = results_io.RUNS_DIR / config_hash / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(config_hash: str) -> str | None:
    config = _load_config(config_hash)

    for key in ("arm", "adaptation_arm"):
        value = config.get(key)
        if value is not None:
            candidate = str(value).strip().upper()
            if candidate in ARM_ORDER:
                return candidate

    return None


def _mean_sd(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return "—"

    mean = values.mean()

    if len(values) == 1:
        return f"{mean:.2f}"

    sd = values.std(ddof=1)
    return f"{mean:.2f} ± {sd:.2f}"


def _latest_batches(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest complete record_run batch per arm/region/seed."""
    latest = runs.groupby(
        ["arm", "region", "seed"],
        dropna=False,
    )["timestamp"].transform("max")

    return runs[runs["timestamp"] == latest].copy()


def _regime_mae(
    rows: pd.DataFrame,
    arm: str,
    region: str,
    regime: str,
) -> str:
    values = rows[
        (rows["arm"] == arm)
        & (rows["region"] == region)
        & (rows["metric_name"] == "mae")
        & (rows["regime"] == regime)
    ]["metric_value"]

    if values.empty:
        raise SystemExit(
            f"T3 missing MAE for Arm {arm}, {region}, regime={regime}. "
            "The reviewed runner must record all three regime MAEs."
        )

    return _mean_sd(values)


def _pinball(rows: pd.DataFrame, arm: str, region: str) -> str:
    """Return genuine recorded pinball loss, or an em dash if unavailable.

    Point forecasts are not treated as quantile forecasts.
    """
    values = rows[
        (rows["arm"] == arm)
        & (rows["region"] == region)
        & (rows["metric_name"] == "pinball_loss")
    ]["metric_value"]

    if values.empty:
        return "—"

    return _mean_sd(values)


def _seed_count(rows: pd.DataFrame, arm: str, region: str) -> int:
    subset = rows[
        (rows["arm"] == arm)
        & (rows["region"] == region)
    ]

    # Deterministic runs use seed=None, which round-trips through CSV as NaN.
    # Such a run still represents one completed run, not zero runs.
    non_null = subset["seed"].dropna().nunique()

    if non_null:
        return int(non_null)

    return 1 if not subset.empty else 0


def build_table(split_id: str) -> pd.DataFrame:
    if not results_io.RUNS_CSV.exists():
        raise SystemExit(
            f"{results_io.RUNS_CSV} not found"
        )

    runs = pd.read_csv(results_io.RUNS_CSV)

    runs = runs[
        (runs["group"] == "baseline")
        & (runs["dataset"] == "aemo")
        & (runs["split_id"] == split_id)
    ].copy()

    if runs.empty:
        raise SystemExit(
            f"no AEMO forecast rows for split_id={split_id!r}"
        )

    runs["arm"] = runs["config_hash"].map(_arm)
    runs = runs[runs["arm"].isin(ARM_ORDER)].copy()

    if runs.empty:
        raise SystemExit(
            "no adaptation runs with config arm=A/B/C/D were found"
        )

    runs = _latest_batches(runs)

    output = []

    for arm in ARM_ORDER:
        for region in ("SA1", "NSW1"):
            subset = runs[
                (runs["arm"] == arm)
                & (runs["region"] == region)
            ]

            if subset.empty:
                continue

            output.append(
                {
                    "arm": ARM_LABELS[arm],
                    "region": region,
                    "MAE pre-drift": _regime_mae(
                        runs,
                        arm,
                        region,
                        "pre-drift",
                    ),
                    "MAE drift": _regime_mae(
                        runs,
                        arm,
                        region,
                        "drift",
                    ),
                    "MAE post-drift": _regime_mae(
                        runs,
                        arm,
                        region,
                        "post-drift",
                    ),
                    "pinball loss": _pinball(
                        runs,
                        arm,
                        region,
                    ),
                    "seeds": _seed_count(
                        runs,
                        arm,
                        region,
                    ),
                }
            )

    table = pd.DataFrame(output, columns=TABLE_COLUMNS)

    if table.empty:
        raise SystemExit("no T3 rows could be produced")

    return table


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-id",
        required=True,
        help="Reviewed four-arm adaptation experiment split_id.",
    )
    args = parser.parse_args()

    table = build_table(args.split_id)

    results_io.TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        results_io.TABLES_DIR
        / "table_t3_forecast_accuracy_by_regime.csv"
    )
    png_path = (
        results_io.TABLES_DIR
        / "table_t3_forecast_accuracy_by_regime.png"
    )

    table.to_csv(csv_path, index=False)

    save_table_image(
        table,
        png_path,
        title="T3: Forecast accuracy by regime",
        subtitle=(
            "MAE in MW; values show mean ± SD when multiple seeds exist. "
            "Pinball loss is shown only for genuine quantile forecasts."
        ),
    )

    print(table.to_string(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
