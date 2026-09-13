# KSWIN

## Project role

KSWIN (Kolmogorov-Smirnov Windowing) is one of the online drift detectors
evaluated in this project.

The implementation wraps `river.drift.KSWIN` behind the frozen project
`DriftDetector` interface.

Implementation:

`src/drift_lab/detection/kswin.py`

The required public contract is:

    name = "kswin"
    detect(stream) -> list[int]

The returned values are positional stream indices where drift is detected.

## Detection mechanism

KSWIN maintains a sliding window of recent observations and compares two
samples from that window using the two-sample Kolmogorov-Smirnov test.

Conceptually, the comparison involves:

- a reference sample representing older observations;
- a recent sample containing the newest observations.

The KS statistic measures the maximum distance between the empirical
cumulative distribution functions of the two samples:

    D = sup_x |F_reference(x) - F_recent(x)|

A sufficiently small p-value provides evidence that the two samples are
unlikely to come from the same distribution.

The detector therefore reacts to distributional change rather than relying
only on a change in the mean.

## Current implementation

The project wrapper preserves:

    name = "kswin"

and:

    detect(stream) -> list[int]

The wrapper delegates sequential processing to the shared
`detect_with_river()` helper.

A new River KSWIN instance is created for every call to `detect()`, preventing
runtime detector state from leaking between independent experiment runs.

## Configuration

The canonical baseline configuration is:

    alpha = 0.005
    window_size = 100
    stat_size = 30
    seed = 42

`alpha` is the significance threshold used by the KS test.

`window_size` controls the number of observations retained by KSWIN.

`stat_size` controls the size of the most recent sample used in the
distribution comparison.

The remaining comparison sample is drawn from earlier observations in the
window.

KSWIN uses sampling internally. The project fixes the detector's internal
sampling seed at:

    seed = 42

This internal detector seed is distinct from the synthetic experiment seeds:

    1, 2, 3, 4, 5

The experiment seeds control generation of the synthetic streams. The KSWIN
seed controls its internal sampling behaviour. Keeping them separate prevents
the experiment seed from silently changing detector configuration.

All four detector parameters are public instance attributes and are therefore
included by `config_of()` in the recorded configuration and `config_hash`.

## Synthetic evaluation

KSWIN is evaluated using the canonical synthetic generator:

    make_series(kind, n, noise, seed)

The required benchmark uses:

- `none`
- `sudden`
- `gradual`
- `recurring`
- `n = 20,000`
- `noise = 1.0`
- experiment seeds `1, 2, 3, 4, 5`

The generator returns both the series and the ground-truth changepoints.
The runner uses those changepoints directly and does not hard-code them.

The experiment pipeline is:

    make_series()
        -> KSWINDetector.detect()
        -> config_of()
        -> record_run()
        -> results/runs.csv

Detection metrics are computed only by the shared evaluation workflow inside
`record_run()`.

## Expected behaviour

### No drift

Every detection in the stable no-drift stream is considered a false alarm.
This condition is important for measuring whether KSWIN is reacting to noise
rather than genuine distributional change.

### Sudden drift

A strong abrupt distributional change can create a clear difference between
the reference and recent samples, allowing the KS test to detect change after
enough post-change observations enter the window.

### Gradual drift

Gradual drift can be more difficult because the reference and recent samples
may overlap in distribution while the transition develops.

### Recurring drift

The recurring benchmark contains multiple regime transitions. KSWIN is
evaluated on whether it detects those transitions and on the number of
additional unmatched detections it produces.

## Runtime and parameter trade-offs

KSWIN performs repeated two-sample Kolmogorov-Smirnov comparisons as the
stream progresses.

Its computational behaviour depends mainly on:

- `window_size`;
- `stat_size`;
- the frequency of incoming observations;
- internal reference sampling.

Larger windows can provide more historical context but require more data to be
maintained. Larger statistical samples may strengthen the distribution
comparison but can increase computational cost.

The significance parameter `alpha` also affects sensitivity. A configuration
that reacts more readily may reduce detection delay but increase false alarms,
while a more conservative configuration may reduce false alarms at the cost of
delay or missed detections.

The canonical baseline remains:

    alpha = 0.005
    window_size = 100
    stat_size = 30
    seed = 42

Any change to these values must be supported by recorded experimental evidence.

## AEMO considerations and limitations

The real-data stage uses half-hourly AEMO demand data for SA1 and NSW1.

Electricity demand contains strong daily, weekly and seasonal patterns.
Because KSWIN compares empirical distributions, recurring seasonal behaviour
may appear as distributional change even when the forecasting relationship has
not degraded.

KSWIN can therefore be used on raw demand to identify distributional changes,
while monitoring forecast residuals can provide complementary evidence about
whether predictive performance has changed.

Unlike the synthetic benchmark, AEMO data does not contain complete
ground-truth labels for all concept-drift events. Unmatched detections in AEMO
data must therefore not automatically be classified as false alarms.

Important limitations include:

- sensitivity to the choice of `alpha`, `window_size`, and `stat_size`;
- dependence on internal sampling;
- possible sensitivity to seasonal distribution changes;
- raw-demand drift does not necessarily imply forecasting degradation;
- real AEMO streams do not provide complete drift ground truth.

## References

Raab, C., Heusinger, M., & Schleif, F.-M. (2020). Reactive soft prototype
computing for concept drift streams. Neurocomputing, 416, 340-351.

Oliveira, M. B., & Silva, L. A. (2026). Two-phase drift monitoring and
automatic retraining (TPDM-AR) for electric load forecasting. MethodsX, 16,
103966. https://doi.org/10.1016/j.mex.2026.103966

Rodríguez, F., Lobo, J. L., Laña, I., Álvarez, V., & Perea, E. (2026).
Intra-hour interval forecasting combined with concept drift adaptation for
electric energy demand at residential level. Knowledge-Based Systems, 116956.
https://doi.org/10.1016/j.knosys.2026.116956
