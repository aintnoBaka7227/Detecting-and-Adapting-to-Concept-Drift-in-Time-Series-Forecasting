# Synthetic Drift Data Generator

`data_generator.py` creates fake time series data with **known, labelled concept drift** — used to test whether our drift-detection and adaptation components actually work, since we need ground truth to score them against.

This matches the interface frozen in the supervisor's project brief.

## Requirements

```
pip install -r requirements.txt
```

(installs `numpy` and `matplotlib`)

## Quick Start

```python
from data_generator import make_series

y, changepoints = make_series(kind="sudden", n=20000, noise=1.0, seed=42)

print(y.shape)         # (20000,)
print(changepoints)    # [10000]
```

## Interface

```python
make_series(kind, n, noise, seed=None) -> (y, changepoints)
```

| Parameter | Type | Description |
|---|---|---|
| `kind` | str | One of `"none"`, `"sudden"`, `"gradual"`, `"recurring"` — see below |
| `n` | int | Number of time steps to generate |
| `noise` | float | Standard deviation of additive Gaussian noise |
| `seed` | int, optional | Random seed — set this for reproducible output |

**Returns:**

| Value | Type | Meaning |
|---|---|---|
| `y` | `np.ndarray`, shape `(n,)` | The generated series — this is the actual signal fed into detectors/models |
| `changepoints` | `list[int]` | The true time indices where drift begins — **this is the ground truth used to score a detector's delay, false alarms, and misses** |

## Drift Types (`kind`)

- **`"none"`** — no drift at all. Control case: a detector run on this should find nothing. `changepoints` will be `[]`.
- **`"sudden"`** — an instant step change at the series midpoint. One changepoint.
- **`"gradual"`** — the new pattern is interpolated in over ~1000 steps (per the brief). One changepoint, marking where the transition *begins*.
- **`"recurring"`** — the series shifts to a second regime and then returns to the original — an old pattern coming back. Two changepoints (`[cp1, cp2]`).

## Reproducibility

- **`changepoints` are deterministic** — they're calculated from `n`, not randomness, so they never change between runs.
- **The noisy values in `y` will differ each run unless you pass a `seed`.** Always pass an explicit `seed=` for anything you're reporting results on, so results can be reproduced exactly by teammates or your supervisor.

## Example: Generate All Four Types

```python
for kind in ["none", "sudden", "gradual", "recurring"]:
    y, cps = make_series(kind=kind, n=20000, noise=1.0, seed=7)
    print(kind, cps)
```

## Run the Demo / Preview Plot

```
python data_generator.py
```

This will:
- Run a **sanity check**: confirm `kind="none"` returns zero changepoints (required test from the brief)
- Generate and save `drift_examples.png` in the same folder — a 4-panel plot of all drift types, with true changepoints marked as red dashed lines
- Print an example `(y, changepoints)` call to the console

To also pop the plot up in a window when you run it (instead of only saving the file), add `plt.show()` right after the `fig.savefig(...)` line near the bottom of the script.

## Notes for the Team

- This interface is **frozen** per the project brief — if it needs to change, flag it with the team before building on top of it, since detector/adaptation components will import this directly.
- Gradual drift's transition width is currently fixed at ~1000 steps regardless of `n`. Worth confirming with the supervisor whether this should scale with series length for very short or very long series.
- For `"recurring"`, the series currently splits evenly into three segments (A → B → A). Let us know if a different cycle structure is needed.
