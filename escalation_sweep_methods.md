# Escalation for the concept-drift forecasting project

This note specifies two escalation scores, how to sweep their thresholds, and how to produce Table 6 and Figure 5. Method 1 uses interval width and is the direct starting point for AgACI. Method 2 uses recent interval misses and can also be applied to constant-width split conformal intervals.

Escalation means handing a forecast to a human for review. The sweep evaluates the original prediction intervals on the forecasts retained by the system. It does not assume that humans correct rejected forecasts, change intervals, or trigger retraining.

## 1. Method 1: sweep interval width for AgACI

### Score and decision

For each half-hourly forecast, retain the bounds as originally issued:

\[
s_t = U_t-L_t.
\]

Escalate when:

\[
E_t(\tau)=\mathbf{1}\{s_t>\tau\}.
\]

A lower threshold sends more forecasts for review. A higher threshold retains more forecasts. Keep the forecasts, AgACI updates, and interval bounds fixed throughout the sweep: do not rerun AgACI for each threshold.

Use width in MW within each region and configuration. If using normalised width, divide by a positive demand scale learned from TRAIN only. Within a configuration, division by a constant changes the threshold units, not the ranking or attainable operating points.

### Sweep procedure

1. Select one forecaster, adaptation arm, region, and UQ method.
2. Calculate the width of every issued interval.
3. Evaluate every distinct width as a threshold, keeping ties together.
4. Include the no-review operating point. Record reject-all separately with undefined retained coverage.
5. At every threshold, record review rate, retained coverage, number retained, and number reviewed.
6. Evaluate the same threshold grid separately in the required regimes.

For a regime containing N eligible forecasts:

\[
R(\tau)=\frac{\sum_t E_t(\tau)}{N},
\qquad
C_{\mathrm{kept}}(\tau)=
\frac{\sum_t(1-E_t(\tau))\mathbf{1}\{L_t\leq y_t\leq U_t\}}
{\sum_t(1-E_t(\tau))}.
\]

When nothing is retained, coverage is undefined, not 100%.

Example: reviewing 100 of 1,000 forecasts leaves 900 retained. If 828 retained intervals contain actual demand, the review rate is 10% and retained coverage is 92%.

Width is a candidate uncertainty signal, not a guaranteed predictor of failure. Wide AgACI intervals may already cover actual demand successfully. Consequently, retained coverage need not increase monotonically as more forecasts are reviewed.

### Why a constant-width split conformal method cannot use this ranking

If every interval within a configuration has the same width, a strict width threshold keeps everything or reviews everything. It cannot identify the most uncertain 5%, 10%, or 20%. Check the exported bounds: some split conformal implementations have horizon-specific or input-dependent widths, in which case width may still vary.

## 2. Method 2: sweep the trailing interval-miss rate

This alternative identifies periods following recent coverage failures. It can be used with fixed split conformal and AgACI.

### Score and decision

Once actual demand is observed, record whether the interval missed:

\[
m_i=\mathbf{1}\{y_i<L_i\ \text{or}\ y_i>U_i\}.
\]

For sequential half-hourly issuance, a candidate score before forecast t is:

\[
s_t=\frac{1}{48}\sum_{i=t-48}^{t-1}m_i,
\qquad
E_t(\tau)=\mathbf{1}\{s_t>\tau\}.
\]

The window covers the previous 48 half-hours, excluding the current outcome. Choosing 48 is a project design choice; the brief does not prescribe it. Fix this window before inspecting test outcomes, or select it using an earlier validation period.

| Threshold | Escalation rule |
| --- | --- |
| 0.10 | Review when more than 10% of the previous 48 intervals missed |
| 0.20 | Review when more than 20% missed |
| 0.30 | Review when more than 30% missed |

The score takes discrete values such as 4/48 or 5/48. Sweep its distinct values with the same evaluation procedure as Method 1. Ties may cause jumps in review rate.

### Timing and data requirements

- Use only outcomes available at the decision's issue time. A simple one-row shift is valid only when the previous half-hour's outcome is already available.
- For a day-ahead batch, calculate the score from outcomes available when that batch is issued. Do not use later within-day actuals to decide escalation at the original issue time.
- Compute the score on the full chronological stream, then slice by regime. Do not join disjoint drift windows and treat adjacent rows as adjacent half-hours.
- Carry in earlier valid forecast outcomes when available; otherwise exclude and count the initial 48-score warm-up.
- Require a regular half-hourly index. Missing actuals or bounds must remain unknown, rather than being labelled as misses.
- Continue observing outcomes after review so the monitoring stream remains complete, if those outcomes are available in deployment. Human decisions should not retrospectively alter the original interval-hit indicator.

This rule reacts after failures begin. It can miss the start of a sudden shift and continue escalating after conditions improve. Its effect on retained coverage must be measured.

### Research basis and an optional weighted version

Gibbs and Candès (2021), Section 2, page 3, Equation (3), explicitly consider a weighted average of historical interval-miss indicators in an adaptive calibration update. Their example uses normalised exponentially decaying weights proportional to 0.95 raised to the age of each observation [1].

That supports using recent miscoverage as a monitoring signal. It does **not** establish the proposed 48-point threshold rule as a published escalation algorithm. Their method updates calibration; this project's proposed extension uses a past-miscoverage score to request review while leaving the intervals unchanged. Their coverage guarantees do not transfer to this extension.

For a score closer to their weighting example, optionally replace the fixed window with:

\[
s_t=\frac{\sum_{i=1}^{t-1}0.95^{t-1-i}m_i}
{\sum_{i=1}^{t-1}0.95^{t-1-i}}.
\]

The weighted score is another candidate policy, not a requirement. Choose and document one version before test evaluation.

## 3. Implementation example

Assumptions: one configuration per DataFrame; sorted, unique, regular half-hourly timestamps; numeric columns `actual`, `lower`, `upper`, and a `regime` column. Bounds must be those available when each forecast was issued.

```python
import numpy as np
import pandas as pd


def add_scores(df, window=48):
    """For sequential issuance with previous outcomes already available."""
    df = df.sort_index().copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required")
    if not df.index.is_unique:
        raise ValueError("Duplicate forecast timestamps")
    gaps = df.index.to_series().diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=30)).all():
        raise ValueError("Reindex to a regular half-hourly grid first")

    valid = df[["actual", "lower", "upper"]].notna().all(axis=1)
    if (df.loc[valid, "lower"] > df.loc[valid, "upper"]).any():
        raise ValueError("Inverted interval bounds")

    covered = df["actual"].between(df["lower"], df["upper"])
    missed = (~covered).astype(float).where(valid)
    df["width_score"] = df["upper"] - df["lower"]
    df["miss_rate_score"] = (
        missed.shift(1).rolling(window, min_periods=window).mean()
    )

    # Optional exponentially weighted alternative:
    # Only use after declaring the missing-data and warm-up policy.
    # df["weighted_miss_score"] = (
    #     missed.ewm(alpha=0.05, adjust=True, min_periods=window)
    #     .mean().shift(1)
    # )
    return df


def exact_sweep(df, score_col, thresholds=None):
    """Evaluate tied scores together without a quadratic threshold loop.

    Optional thresholds let every regime use the same threshold grid.
    Missing rows are excluded and counted. Non-finite values raise an
    error so infinite intervals are not silently discarded.
    """
    cols = ["actual", "lower", "upper", score_col]
    d = df.dropna(subset=cols).copy()
    if d.empty:
        raise ValueError("No eligible forecasts")
    if not np.isfinite(d[cols].to_numpy(dtype=float)).all():
        raise ValueError("Declare a policy for non-finite values first")
    if (d["lower"] > d["upper"]).any():
        raise ValueError("Inverted interval bounds")

    d["hit"] = d["actual"].between(d["lower"], d["upper"]).astype(int)
    groups = d.groupby(score_col, sort=True)["hit"].agg(["size", "sum"])
    scores = groups.index.to_numpy(dtype=float)
    cum_n = np.r_[0, groups["size"].cumsum().to_numpy()]
    cum_hits = np.r_[0, groups["sum"].cumsum().to_numpy()]

    if thresholds is None:
        thresholds = scores
    thresholds = np.unique(np.r_[-np.inf, thresholds, np.inf])
    positions = np.searchsorted(scores, thresholds, side="right")
    n_kept = cum_n[positions]
    hits_kept = cum_hits[positions]
    coverage = np.divide(
        hits_kept, n_kept,
        out=np.full(len(n_kept), np.nan), where=n_kept > 0,
    )
    return pd.DataFrame({
        "threshold": thresholds,
        "review_rate": (len(d) - n_kept) / len(d),
        "kept_coverage": coverage,
        "n_kept": n_kept,
        "n_reviewed": len(d) - n_kept,
        "n_eligible": len(d),
        "n_excluded": len(df) - len(d),
    })


# Run separately for each forecaster / arm / region / UQ method.
scored = add_scores(predictions)

# Method 1 for AgACI:
width_sweep = exact_sweep(scored, "width_score")

# Method 2 for split conformal, or AgACI:
miss_sweep = exact_sweep(scored, "miss_rate_score")

# Regime-specific results using one common threshold grid:
grid = miss_sweep["threshold"].to_numpy()
regime_tables = []
for regime, part in scored.groupby("regime"):
    eligible = part.dropna(
        subset=["actual", "lower", "upper", "miss_rate_score"]
    )
    if eligible.empty:
        continue  # Report this regime as having no evaluable forecasts.
    result = exact_sweep(part, "miss_rate_score", thresholds=grid)
    result["regime"] = regime
    regime_tables.append(result)
```

The grouped cumulative calculation avoids comparing the entire dataset against every distinct threshold. This matters for several years of half-hourly outputs.

If AgACI emits infinite bounds, explicitly report their frequency and define how their infinite widths enter the ordering. The example intentionally raises an error until that policy is implemented; do not silently remove those forecasts.

## 4. Table 6 and Figure 5

| Required result | Calculation / reporting convention |
| --- | --- |
| Review rate for 0.90 | Minimum observed review rate among thresholds with retained coverage at least 0.90 and a non-empty retained set |
| Review rate for 0.95 | Same calculation using 0.95 |
| Coverage at 5%, 10%, 20% | Use the attainable rate closest to the requested budget without exceeding it; report the actual rate as well |
| Unreachable | No evaluated threshold with a non-empty retained set reaches the target |
| No-review target already achieved | Report 0% review |
| No forecasts retained | Coverage is undefined |

For the fixed-budget columns, choose the threshold by its review rate, not by which retained coverage happens to be best. Do not break tied scores using actual errors. A constant-width rule may only offer 0% and 100% review; report the missing intermediate operating points explicitly rather than fabricating a curve.

Figure 5 uses review rate on x and retained interval coverage on y, with the required regimes shown separately and the 0.90 crossing marked where it exists. Use discrete operating points or state any interpolation. The curve need not be monotonic.

Log configuration identifiers, score definition, window/decay, threshold, regime, review rate, retained coverage, counts, and exclusions. Very small retained samples can produce misleadingly high coverage: always show counts and, if a minimum retained size is imposed, declare it in advance.

Table 6 is regime-level retained coverage. It does not guarantee 90% coverage in every rolling 24-hour window. Keep the worst-24-hour metric in Table 5 separate.

### Descriptive sweep versus deployment

A full test sweep describes operating points observed on test data. Selecting a threshold because its test coverage reaches 90% and then presenting that same result as independent validation would be optimistic.

For deployment, select the score design, window, and threshold on earlier validation data, freeze them, and evaluate once on the later test period. A threshold selected for 10% validation review need not yield 10% review on test data. Do not reuse the conformal calibration set for policy tuning without explaining the additional data reuse; preferably reserve a separate chronological validation segment or use an explicitly documented earlier-period design.

