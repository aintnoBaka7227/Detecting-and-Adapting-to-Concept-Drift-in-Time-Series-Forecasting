# Synthetic Drift Data Generator

`data_generator.py` creates fake time series data with **known, labelled concept drift** — used to test whether our drift-detection and forecasting components actually work, since we need ground truth to score them against.

## Requirements

```
pip install numpy pandas matplotlib
```

## Quick Start

```python
from data_generator import make_series

df = make_series(
    n_samples=2000,
    drift_type="sudden",
    drift_point=1000,
    seed=42,
)

print(df.head())
```

That's it — `df` is a pandas DataFrame ready to feed into a drift detector or forecasting model.

## Output Columns

| Column | Type | Meaning |
|---|---|---|
| `t` | int | Time index (0, 1, 2, ...) |
| `value` | float | The generated data point — this is the actual "signal" your model consumes |
| `regime` | str | `"A"` or `"B"` — which underlying regime generated this point (useful for debugging) |
| `is_drift_region` | bool | `True` during the transition window (for gradual/incremental/recurring drift) |
| `true_drift_point` | bool | `True` only at the exact drift onset index(es) — **this is the ground truth to score detectors against** |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_samples` | `2000` | Total number of time steps to generate |
| `drift_type` | `"sudden"` | One of `"sudden"`, `"gradual"`, `"incremental"`, `"recurring"`, `"none"` (see below) |
| `drift_point` | midpoint | Time index where drift begins (ignored for `"recurring"`) |
| `drift_width` | `100` | How many steps the transition takes (only used by `"gradual"` and `"incremental"`) |
| `n_recurring_cycles` | `4` | Number of A/B alternations (only used by `"recurring"`) |
| `regime_a`, `regime_b` | sensible defaults | Custom `Regime` objects if you want to control level/trend/seasonality/noise directly |
| `noise_std` | `None` | If set, overrides noise level for both regimes |
| `seed` | `None` | Set this for reproducible output |

## Drift Types — What Each One Looks Like

- **`"none"`** — no drift at all. Use this as a control case: a good detector should stay silent here.
- **`"sudden"`** — an instant step change at `drift_point`. The simplest, sharpest case.
- **`"gradual"`** — old and new regimes are randomly interleaved with increasing probability of the new regime over `drift_width` steps. Looks noisy/flickery during the transition — this is intentional, it mimics real gradual drift.
- **`"incremental"`** — a smooth, continuous ramp from old to new regime over `drift_width` steps. No flickering, just a steady slope.
- **`"recurring"`** — the series alternates back and forth between regime A and B every `n_samples / n_recurring_cycles` steps. Good for testing whether a detector can catch repeated drift events, not just one.

## Examples

Generate one of each drift type:

```python
for dtype in ["none", "sudden", "gradual", "incremental", "recurring"]:
    df = make_series(n_samples=2000, drift_type=dtype, seed=7)
    df.to_csv(f"series_{dtype}.csv", index=False)
```

Check where the ground-truth drift point landed:

```python
df = make_series(n_samples=2000, drift_type="sudden", drift_point=1000, seed=42)
drift_at = df.loc[df["true_drift_point"], "t"].tolist()
print(drift_at)  # -> [1000]
```

Run the file directly to generate a preview plot of all five drift types:

```
python3 data_generator.py
```

This saves `drift_examples.png` showing all five series stacked, with red dashed lines marking the true drift points.

## Notes for the Team

- Column names (`t`, `value`, `regime`, `is_drift_region`, `true_drift_point`) are the current interface. If this needs to change to match what the forecasting/detection components expect, flag it before building on top of it — happy to adjust.
- `regime_a` / `regime_b` are configurable, so if we want a series that mimics a specific scenario (e.g. an energy price shock), we can tune `level`, `trend`, `seasonality_amp`, and `noise_std` rather than hardcoding new logic.
