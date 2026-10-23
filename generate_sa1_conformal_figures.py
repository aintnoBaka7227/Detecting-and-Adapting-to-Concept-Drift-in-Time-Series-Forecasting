"""Generate independent SA1 Figure 4 and 5 results for XGBoost and N-HiTS.

Run from the repository root:
    python generate_sa1_conformal_figures.py

Uses existing drift_lab FixedSplitConformal and AdaptiveConformalInference.
Requires numpy, pandas, matplotlib and the project's dependencies.
No test outcomes are used for tuning gamma or for issuing their own intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {
    'xgboost': '68ebc0d52d05',
    'nhits': '291622879360',
}
ALPHA = 0.10
WINDOW = 336  # 7 days at 30-minute resolution
BLOCK = 48    # forecasts issued in one block before feedback
GAMMAS = (0.001, 0.005, 0.01, 0.02, 0.05)


def load_data(root: Path, model: str):
    cal_path = root / 'results' / 'runs' / 'calibration' / f'calibration_{model}_SA1_1.csv'
    run_dir = root / 'results' / 'runs' / RUNS[model]
    test_path = run_dir / 'curve_aemo_SA1_1.csv'
    config_path = run_dir / 'config.json'
    for p in (cal_path, test_path, config_path):
        if not p.is_file():
            raise FileNotFoundError(f'Missing input: {p}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('model') != model:
        raise ValueError(f'Run {run_dir} identifies as {config.get("model")}, not {model}')
    if config.get('arm') != 'A_never_retrain':
        raise ValueError(f'Unexpected adaptation arm for {model}: {config.get("arm")}')
    cal = pd.read_csv(cal_path)
    test = pd.read_csv(test_path)
    if 'residual' not in cal.columns:
        if not {'actual', 'forecast'}.issubset(cal.columns):
            raise ValueError(f'Calibration columns insufficient: {cal.columns.tolist()}')
        cal['residual'] = cal['actual'] - cal['forecast']
    if not {'timestamp', 'actual', 'forecast'}.issubset(test.columns):
        raise ValueError(f'Test columns insufficient: {test.columns.tolist()}')
    if 'timestamp' not in cal.columns:
        raise ValueError('Calibration must have timestamps for chronology check')
    cal['timestamp'] = pd.to_datetime(cal['timestamp'], errors='raise')
    test['timestamp'] = pd.to_datetime(test['timestamp'], errors='raise')
    for name, df in [('calibration', cal), ('test', test)]:
        if df['timestamp'].isna().any() or df['timestamp'].duplicated().any():
            raise ValueError(f'{model}: {name} has missing/duplicate timestamps')
        if not df['timestamp'].is_monotonic_increasing:
            raise ValueError(f'{model}: {name} timestamps are not sorted')
    if cal['timestamp'].max() >= test['timestamp'].min():
        raise ValueError(f'{model}: calibration overlaps test (data leakage)')
    for col in ('actual', 'forecast'):
        if not np.isfinite(test[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{model}: non-finite {col} in test')
    if not np.isfinite(cal['residual'].to_numpy(dtype=float)).all():
        raise ValueError(f'{model}: non-finite calibration residuals')
    if 'seed' in test and not (test['seed'].astype(str) == '1').all():
        raise ValueError(f'{model}: expected seed 1')
    if 'arm' in test and not (test['arm'] == 'A_never_retrain').all():
        raise ValueError(f'{model}: expected A_never_retrain')
    return cal, test, cal_path, test_path


def replay_aci(ACI, gamma, seed_residuals, forecasts, actuals):
    """Issue 48 forecasts using prior history, then observe their outcomes."""
    uq = ACI(gamma=gamma, window_size=WINDOW, target_alpha=ALPHA)
    uq.calibrate(seed_residuals)
    n = len(forecasts)
    lo = np.empty(n)
    hi = np.empty(n)
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        issued = uq.predict(forecasts[start:end])
        lo[start:end] = issued.lower
        hi[start:end] = issued.upper
        # Feedback is available only AFTER all forecasts in this block are issued.
        uq.observe(actuals[start:end], forecasts[start:end], issued.lower, issued.upper)
    return lo, hi


def evaluate(actual, lower, upper):
    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower
    finite = np.isfinite(width)
    return {
        'coverage': float(np.mean(covered)),
        'mean_width': float(np.mean(width)) if finite.all() else math.inf,
        'worst_7d_coverage': float(pd.Series(covered.astype(float)).rolling(WINDOW, min_periods=WINDOW).mean().min()),
        'infinite_interval_count': int(np.sum(~finite)),
    }


def choose_gamma(ACI, cal):
    # Strictly historical chronological validation; the test period is never consulted.
    n = len(cal)
    split = n // 2
    if split < WINDOW or n - split < BLOCK:
        raise ValueError('Not enough calibration rows for seed and validation')
    seed = cal['residual'].to_numpy(dtype=float)[:split]
    val = cal.iloc[split:]
    actual = val['actual'].to_numpy(dtype=float) if 'actual' in val else None
    forecast = val['forecast'].to_numpy(dtype=float) if 'forecast' in val else None
    if actual is None or forecast is None:
        raise ValueError('Calibration requires actual and forecast to tune gamma')
    candidates = []
    for gamma in GAMMAS:
        lower, upper = replay_aci(ACI, gamma, seed, forecast, actual)
        result = evaluate(actual, lower, upper)
        candidates.append({'gamma': gamma, **result})
    # Prefer finite intervals, then closeness to nominal 90%, then smaller width.
    ranked = sorted(candidates, key=lambda x: (
        x['infinite_interval_count'] > 0,
        abs(x['coverage'] - (1 - ALPHA)),
        x['mean_width'],
        x['gamma'],
    ))
    return ranked[0]['gamma'], pd.DataFrame(candidates)


def make_figures(root: Path, model: str, Fixed, ACI):
    cal, test, cal_path, test_path = load_data(root, model)
    print(f'\n[{model}] calibration={cal_path}, rows={len(cal)}')
    print(f'[{model}] test={test_path}, rows={len(test)}')
    gamma, tuning = choose_gamma(ACI, cal)
    print(f'[{model}] selected gamma={gamma}')

    residuals = cal['residual'].to_numpy(dtype=float)
    forecasts = test['forecast'].to_numpy(dtype=float)
    actuals = test['actual'].to_numpy(dtype=float)
    fixed = Fixed(alpha=ALPHA).calibrate(residuals)
    fixed_intervals = fixed.predict(forecasts)
    flo = np.asarray(fixed_intervals.lower)
    fhi = np.asarray(fixed_intervals.upper)
    alo, ahi = replay_aci(ACI, gamma, residuals, forecasts, actuals)
    fstats = evaluate(actuals, flo, fhi)
    astats = evaluate(actuals, alo, ahi)

    out = root / 'results' / 'figures' / 'independent_aci' / 'SA1' / model
    out.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(out / 'aci_gamma_validation.csv', index=False)
    full = test[['timestamp', 'actual', 'forecast']].copy()
    full['fixed_lower'] = flo
    full['fixed_upper'] = fhi
    full['aci_lower'] = alo
    full['aci_upper'] = ahi
    full['fixed_covered'] = ((actuals >= flo) & (actuals <= fhi)).astype(int)
    full['aci_covered'] = ((actuals >= alo) & (actuals <= ahi)).astype(int)
    full['aci_width'] = ahi - alo
    full['fixed_rolling_coverage'] = full['fixed_covered'].rolling(WINDOW, min_periods=WINDOW).mean()
    full['aci_rolling_coverage'] = full['aci_covered'].rolling(WINDOW, min_periods=WINDOW).mean()
    full.to_csv(out / 'figure4_issued_intervals_and_rolling_coverage.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full['timestamp'], 100 * full['fixed_rolling_coverage'], label='Fixed split conformal', alpha=.85, linewidth=1.0)
    ax.plot(full['timestamp'], 100 * full['aci_rolling_coverage'], label=f'ACI (gamma={gamma:g})', alpha=.85, linewidth=1.0)
    ax.axhline(90, linestyle='--', color='black', linewidth=1, label='Nominal 90%')
    ax.set(title=f'Figure 4 — Rolling 7-day coverage | {model.upper()} | SA1 | seed 1',
           xlabel='Forecast timestamp', ylabel='Rolling empirical coverage (%)', ylim=(0, 105))
    ax.legend(loc='lower right')
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / 'figure4_rolling_coverage.png', dpi=180)
    plt.close(fig)

    # Retrospective width ranking: issued widths known at issuance, but choosing
    # the widest global N% requires knowledge of all widths over the test period.
    # This is a retrospective budget curve, NOT a deployable online threshold policy.
    n = len(full)
    order = np.argsort(-full['aci_width'].to_numpy(dtype=float), kind='stable')
    covered = full['aci_covered'].to_numpy(dtype=bool)
    curve = []
    for rate in range(0, 101, 5):
        k = int(np.floor(n * rate / 100))
        retained = np.ones(n, dtype=bool)
        retained[order[:k]] = False
        remaining = int(retained.sum())
        curve.append({
            'review_rate_percent': rate,
            'review_count': k,
            'retained_count': remaining,
            'aci_retained_coverage': float(covered[retained].mean()) if remaining else np.nan,
            'fixed_full_sample_coverage_reference': fstats['coverage'],
            'nominal_coverage_reference': 0.90,
        })
    curve = pd.DataFrame(curve)
    curve.to_csv(out / 'figure5_retained_coverage_vs_review_rate.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve['review_rate_percent'], 100 * curve['aci_retained_coverage'], marker='o', markersize=3,
            label='ACI: review widest intervals first')
    ax.axhline(100 * fstats['coverage'], linestyle=':', label='Fixed split: full-sample coverage (reference)')
    ax.axhline(90, linestyle='--', color='black', label='Nominal 90%')
    ax.set(title=f'Figure 5 — Retained coverage vs review rate | {model.upper()} | SA1 | seed 1',
           xlabel='Review budget (% of test forecasts)', ylabel='Coverage among retained forecasts (%)',
           xlim=(0, 100), ylim=(0, 105))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'figure5_retained_coverage_vs_review_rate.png', dpi=180)
    plt.close(fig)

    summary = pd.DataFrame([
        {'model': model, 'region': 'SA1', 'seed': 1, 'arm': 'A_never_retrain',
         'method': 'fixed_split', 'gamma': np.nan, 'calibration_rows': len(cal),
         'test_rows': n, 'feedback_block': BLOCK, 'rolling_window': WINDOW, **fstats},
        {'model': model, 'region': 'SA1', 'seed': 1, 'arm': 'A_never_retrain',
         'method': 'aci', 'gamma': gamma, 'calibration_rows': len(cal),
         'test_rows': n, 'feedback_block': BLOCK, 'rolling_window': WINDOW, **astats},
    ])
    summary.to_csv(out / 'coverage_width_summary.csv', index=False)
    print(summary[['method', 'coverage', 'mean_width', 'worst_7d_coverage', 'infinite_interval_count']].to_string(index=False))
    print(f'[{model}] figures and CSVs saved to {out}')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root')
    parser.add_argument('--models', nargs='+', choices=tuple(RUNS), default=list(RUNS))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from drift_lab.uncertainty.fixed_split_conformal import FixedSplitConformal
    from drift_lab.uncertainty.adaptive_conformal import AdaptiveConformalInference
    summaries = [make_figures(root, model, FixedSplitConformal, AdaptiveConformalInference) for model in args.models]
    destination = root / 'results' / 'figures' / 'independent_aci' / 'SA1'
    pd.concat(summaries, ignore_index=True).to_csv(destination / 'combined_model_summary.csv', index=False)
    print(f'\nDONE. Combined summary: {destination / "combined_model_summary.csv"}')


if __name__ == '__main__':
    main()
