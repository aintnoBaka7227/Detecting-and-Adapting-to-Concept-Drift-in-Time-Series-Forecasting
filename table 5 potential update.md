## Table 5 — Required Implementation

### 1. Evaluate coverage and width by regime

Produce one row per UQ method × region × regime.

Keep the three regime labels:

- `pre_drift`
- `drift`
- `post_drift`

Evaluate the full test period, assigning every timestamp a regime.
Here, `full` describes the evaluation scope; it does not replace the
three regime labels.

Use the forecast regime boundaries agreed with Team 42 and record them
in `DECISIONS.md`. Keep forecast regime assignment separate from the
existing detection regime assignment if their definitions differ.

### 2. Required columns

| Column | Calculation or content |
|---|---|
| UQ method | Split Conformal, ACI, AgACI, etc. |
| Region | SA1 or NSW1; optional if separate regional tables are produced. |
| Regime | `pre_drift`, `drift`, or `post_drift`. |
| Empirical coverage | Number of actual values inside their prediction intervals ÷ number of evaluated observations in the regime. |
| Mean interval width (MW) | Mean of `upper_bound - lower_bound` within the regime. |
| Normalised interval width | Mean interval width ÷ mean actual TRAIN demand for the region. |
| Worst 24-hour coverage | Minimum coverage among eligible complete rolling 24-hour windows within the regime. |
| Complete 24-hour windows | Number of eligible rolling windows used for the worst-coverage calculation. |

### 3. Use a fixed regional denominator

Supervisor instruction:

> Use the mean actual demand in the training window, calculated separately
> for SA1 and NSW1, as the denominator.

Implementation:

1. Calculate mean actual demand using TRAIN data only, separately for SA1
   and NSW1.
2. Calculate these values once.
3. Pass the same regional value to every UQ method and every regime.
4. Do not recalculate the denominator from calibration, test, or regime data.
5. State the actual denominator values in the table caption.

Formula:

    normalised_interval_width = mean_interval_width / regional_train_mean

The other team's synthetic scale of `10` is acceptable for their calculation
checks. It is not the denominator to use for real AEMO results.

Existing code already provides:

- `calculate_width_scale(y_train)`
- `calculate_normalised_interval_width(lower_bound, upper_bound, scale)`

Check that the calling script supplies the correct regional TRAIN mean.

### 4. Keep rolling coverage within each regime

Supervisor instruction:

> Points 2 and 3 are correct — keep both as they are.

For half-hourly data:

- Use 48 consecutive observations per 24-hour window.
- Move the window forward one observation at a time.
- Accept a window only when every observation has the selected regime label.
- Reject windows crossing a regime boundary or a gap in timestamps.
- Keep the original chronological order.
- Do not concatenate separate regime periods before calculating rolling coverage.

For each eligible window:

    window_coverage = number_of_covered_observations / 48

Then:

    worst_24h_coverage = minimum eligible window_coverage

If no complete window qualifies, return `NaN` internally and display `N/A`.

Existing code already restricts windows to the same regime label.
Timestamp continuity still needs to be checked or guaranteed by the input pipeline.

### 5. Add a complete-window counter

Supervisor instruction:

> Report how many complete 24-hour windows each regime contributes.

Add an output such as:

    n_complete_windows

Count windows using exactly the same eligibility rules as the
worst-coverage calculation.

For one continuous segment with `n` half-hourly observations:

    n_complete_windows = max(0, n - 48 + 1)

Sum the counts across separate continuous segments of the regime.

Examples:

| Observations in one segment | Complete windows |
|---:|---:|
| 30 | 0 |
| 48 | 1 |
| 60 | 13 |
| 336 | 289 |

The existing `n_observations` output is not a substitute for this count.

### 6. Identify results with limited window support

Supervisor instruction:

> If a short regime only has a handful, the worst-window figure isn't
> meaningful and should be marked as such.

Implementation:

- Zero eligible windows: display worst coverage as `N/A`.
- Few eligible windows: flag the result as `Limited window support`.
- Report the eligible window count alongside the result.

No minimum count was specified by the supervisor.
Inspect the counts first, agree a common threshold with Team 42 and the
supervisor, and record it in `DECISIONS.md`.

Rolling windows overlap. Keep this behaviour for Table 5, but do not
describe their count as the number of independent days or independent
failure events.

### 7. Caption and reporting notes

State:

- SA1 and NSW1 TRAIN mean demand denominators, in MW.
- The target interval coverage, e.g. 90%.
- The rolling window size: 48 half-hourly observations.
- Windows must remain within the same regime and use consecutive timestamps.
- Rolling windows overlap.
- The agreed rule for flagging limited window support.

For stochastic methods, report mean ± standard deviation across at least
five seeds, following the required-outputs document. Calculate worst-window
coverage separately for each seed before summarising across seeds.

### Implementation checklist

- [ ] Use the agreed forecast regime labels for all test timestamps.
- [ ] Calculate and consistently pass each region's TRAIN mean demand.
- [ ] Retain the existing within-regime rolling-window restriction.
- [ ] Validate or guarantee consecutive half-hourly timestamps.
- [ ] Add `n_complete_windows` to the returned metrics.
- [ ] Add a limited-support flag once the threshold is agreed.
- [ ] Export the new count and flag through the run logging and table script.
- [ ] Include denominator values and window rules in the caption.