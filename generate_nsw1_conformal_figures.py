"""Generate independent NSW1 Figures 4/5 for DHR-ARIMA, XGBoost and N-HiTS.

Run from repository root after generating calibration and never-retrain forecasts:
    python generate_nsw1_conformal_figures.py

Automatically discovers model-labelled run configs; never guesses run hashes.
Requires project drift_lab package, pandas, numpy and matplotlib.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGION = 'NSW1'
MODELS = ('dhr_arima', 'xgboost', 'nhits')
ALPHA = 0.10
WINDOW = 336
BLOCK = 48
GAMMAS = (0.001, 0.005, 0.01, 0.02, 0.05)


def discover(root, model):
    seed = 'none' if model == 'dhr_arima' else '1'
    cal_path = root / 'results/runs/calibration' / f'calibration_{model}_{REGION}_{seed}.csv'
    if not cal_path.exists():
        raise FileNotFoundError(f'Missing calibration: {cal_path}\nRun: python -m experiments.run.forecasting.run_aemo_calibration_forecasts --model {model} --region {REGION}')
    candidates = []
    for config_path in (root / 'results/runs').glob('*/config.json'):
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if config.get('model') != model or config.get('arm') != 'A_never_retrain':
            continue
        if model != 'dhr_arima':
            config_seed = config.get('random_seed', config.get('random_state'))
            if str(config_seed) != seed:
                continue
        curve_path = config_path.parent / f'curve_aemo_{REGION}_{seed}.csv'
        if curve_path.exists():
            candidates.append(curve_path)
    if len(candidates) != 1:
        raise RuntimeError(f'{model} {REGION}: expected exactly one matching A_never_retrain forecast run, found {len(candidates)}: {candidates}. If no run exists, generate one; if multiple exist, select and archive duplicates before rerunning.')
    return cal_path, candidates[0]


def load(root, model):
    cp, tp = discover(root, model)
    cal = pd.read_csv(cp)
    test = pd.read_csv(tp)
    if not {'timestamp', 'actual', 'forecast'}.issubset(cal.columns):
        raise ValueError(f'Calibration must have timestamp, actual, forecast: {cp}')
    if not {'timestamp', 'actual', 'forecast'}.issubset(test.columns):
        raise ValueError(f'Test must have timestamp, actual, forecast: {tp}')
    for name, df in [('calibration', cal), ('test', test)]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
        if df['timestamp'].isna().any() or df['timestamp'].duplicated().any() or not df['timestamp'].is_monotonic_increasing:
            raise ValueError(f'{model} {name}: timestamps missing, duplicated or not sorted')
        for col in ('actual', 'forecast'):
            if not np.isfinite(df[col].to_numpy(dtype=float)).all():
                raise ValueError(f'{model} {name}: non-finite {col}')
    if cal['timestamp'].max() >= test['timestamp'].min():
        raise ValueError(f'{model}: calibration/test overlap')
    if 'arm' in test.columns and not (test['arm'] == 'A_never_retrain').all():
        raise ValueError(f'{model}: unexpected test arm')
    if model != 'dhr_arima' and 'seed' in test.columns and not (test['seed'].astype(str) == '1').all():
        raise ValueError(f'{model}: test seed mismatch')
    cal['residual'] = cal['actual'].to_numpy(dtype=float) - cal['forecast'].to_numpy(dtype=float)
    return cal, test, cp, tp


def replay(ACI, gamma, seed_residuals, forecasts, actuals):
    uq = ACI(gamma=gamma, window_size=WINDOW, target_alpha=ALPHA)
    uq.calibrate(seed_residuals)
    lower, upper = np.empty(len(forecasts)), np.empty(len(forecasts))
    for start in range(0, len(forecasts), BLOCK):
        end = min(start + BLOCK, len(forecasts))
        issued = uq.predict(forecasts[start:end])
        lower[start:end], upper[start:end] = issued.lower, issued.upper
        # Outcomes are observed after the entire issuance block, never before.
        uq.observe(actuals[start:end], forecasts[start:end], issued.lower, issued.upper)
    return lower, upper


def metrics(actual, lower, upper):
    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower
    return {'coverage': float(covered.mean()),
            'mean_width': float(width.mean()) if np.isfinite(width).all() else math.inf,
            'worst_7d_coverage': float(pd.Series(covered.astype(float)).rolling(WINDOW, min_periods=WINDOW).mean().min()),
            'infinite_interval_count': int((~np.isfinite(width)).sum())}


def tune(ACI, cal):
    half = len(cal) // 2
    if half < WINDOW or len(cal) - half < BLOCK:
        raise ValueError('Insufficient calibration history')
    residuals = cal['residual'].to_numpy(dtype=float)
    validation = cal.iloc[half:]
    actual = validation['actual'].to_numpy(dtype=float)
    forecast = validation['forecast'].to_numpy(dtype=float)
    rows = []
    for gamma in GAMMAS:
        lo, hi = replay(ACI, gamma, residuals[:half], forecast, actual)
        rows.append({'gamma': gamma, **metrics(actual, lo, hi)})
    ranked = sorted(rows, key=lambda x: (x['infinite_interval_count'] > 0,
                                       abs(x['coverage'] - 0.9), x['mean_width'], x['gamma']))
    return ranked[0]['gamma'], pd.DataFrame(rows)


def produce(root, model, Fixed, ACI):
    cal, test, cp, tp = load(root, model)
    gamma, validation = tune(ACI, cal)
    forecasts, actuals = test['forecast'].to_numpy(dtype=float), test['actual'].to_numpy(dtype=float)
    fixed = Fixed(alpha=ALPHA).calibrate(cal['residual'].to_numpy(dtype=float))
    issued = fixed.predict(forecasts)
    flo, fhi = np.asarray(issued.lower), np.asarray(issued.upper)
    alo, ahi = replay(ACI, gamma, cal['residual'].to_numpy(dtype=float), forecasts, actuals)
    fm, am = metrics(actuals, flo, fhi), metrics(actuals, alo, ahi)
    out = root / 'results/figures/independent_aci' / REGION / model
    out.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out / 'aci_gamma_validation.csv', index=False)
    full = test[['timestamp', 'actual', 'forecast']].copy()
    full['fixed_lower'], full['fixed_upper'] = flo, fhi
    full['aci_lower'], full['aci_upper'] = alo, ahi
    full['fixed_covered'] = ((actuals >= flo) & (actuals <= fhi)).astype(int)
    full['aci_covered'] = ((actuals >= alo) & (actuals <= ahi)).astype(int)
    full['aci_width'] = ahi - alo
    for method in ('fixed', 'aci'):
        full[f'{method}_rolling_coverage'] = full[f'{method}_covered'].rolling(WINDOW, min_periods=WINDOW).mean()
    full.to_csv(out / 'figure4_issued_intervals_and_rolling_coverage.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full['timestamp'], 100 * full['fixed_rolling_coverage'], label='Fixed split', linewidth=1)
    ax.plot(full['timestamp'], 100 * full['aci_rolling_coverage'], label=f'ACI (gamma={gamma:g})', linewidth=1)
    ax.axhline(90, color='black', linestyle='--', label='Nominal 90%')
    ax.set(title=f'Figure 4 — Rolling 7-day coverage | {model.upper()} | {REGION}', xlabel='Timestamp', ylabel='Coverage (%)', ylim=(0, 105))
    ax.grid(alpha=.2); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out / 'figure4_rolling_coverage.png', dpi=180); plt.close(fig)
    # Retrospective width ranking is NOT a deployable online escalation policy.
    n = len(full)
    order = np.argsort(-full['aci_width'].to_numpy(dtype=float), kind='stable')
    covered = full['aci_covered'].to_numpy(dtype=bool)
    curve = []
    for rate in range(0, 101, 5):
        k = int(np.floor(n * rate / 100))
        retained = np.ones(n, dtype=bool); retained[order[:k]] = False
        remaining = int(retained.sum())
        curve.append({'review_rate_percent': rate, 'review_count': k, 'retained_count': remaining,
                      'aci_retained_coverage': float(covered[retained].mean()) if remaining else np.nan,
                      'fixed_full_sample_coverage_reference': fm['coverage'], 'nominal_coverage_reference': .9})
    curve = pd.DataFrame(curve)
    curve.to_csv(out / 'figure5_retained_coverage_vs_review_rate.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve['review_rate_percent'], 100 * curve['aci_retained_coverage'], marker='o', markersize=3, label='ACI: review widest intervals')
    ax.axhline(100 * fm['coverage'], linestyle=':', label='Fixed split full-sample reference')
    ax.axhline(90, color='black', linestyle='--', label='Nominal 90%')
    ax.set(title=f'Figure 5 — Retained coverage vs review rate | {model.upper()} | {REGION}', xlabel='Review rate (%)', ylabel='Retained coverage (%)', xlim=(0, 100), ylim=(0, 105))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out / 'figure5_retained_coverage_vs_review_rate.png', dpi=180); plt.close(fig)
    rows = [{'model': model, 'region': REGION, 'seed': 'none' if model == 'dhr_arima' else 1,
             'arm': 'A_never_retrain', 'method': method, 'gamma': gamma if method == 'aci' else np.nan,
             'calibration_rows': len(cal), 'test_rows': len(test), 'feedback_block': BLOCK,
             'rolling_window': WINDOW, **result} for method, result in [('fixed_split', fm), ('aci', am)]]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'coverage_width_summary.csv', index=False)
    print(f'[{model}] calibration={cp} ({len(cal)} rows)')
    print(f'[{model}] test={tp} ({len(test)} rows); gamma={gamma}')
    print(summary[['method', 'coverage', 'mean_width', 'worst_7d_coverage', 'infinite_interval_count']].to_string(index=False))
    print(f'[{model}] saved: {out}')
    return summary


def main():
    root = Path.cwd()
    sys.path.insert(0, str(root / 'src'))
    from drift_lab.uncertainty.fixed_split_conformal import FixedSplitConformal
    from drift_lab.uncertainty.adaptive_conformal import AdaptiveConformalInference
    summaries = []
    failures = []
    for model in MODELS:
        try:
            summaries.append(produce(root, model, FixedSplitConformal, AdaptiveConformalInference))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append((model, str(exc)))
            print(f'\n[MISSING INPUT: {model}] {exc}\n')
    if summaries:
        dest = root / 'results/figures/independent_aci' / REGION / 'combined_model_summary.csv'
        pd.concat(summaries, ignore_index=True).to_csv(dest, index=False)
        print(f'Combined summary: {dest}')
    if failures:
        print('\nIncomplete models:', ', '.join(m for m, _ in failures))
        sys.exit(1)
    print('\nDONE: NSW1 Figure 4 and 5 complete for all three models.')

if __name__ == '__main__':
    main()
