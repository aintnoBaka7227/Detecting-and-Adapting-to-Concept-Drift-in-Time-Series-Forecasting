"""Cross-check independent fixed/ACI summaries against leader's Table 5 AgACI.

Run from repository root after extracting the leader's results/tables/table5:
    python compare_aci_agaci.py

The script never silently treats different detectors, adaptation arms or
escalation policies as equivalent. It outputs a report and source snapshots.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'results/figures/independent_aci/crosscheck'
LEADER = ROOT / 'leader_figures/results/tables/table5'
ALIASES = {
    'model': ['model', 'forecaster', 'forecast_model', 'base_model'],
    'region': ['region', 'aemo_region', 'dataset', 'area'],
    'method': ['method', 'uq_method', 'conformal_method', 'uncertainty_method', 'uq'],
    'coverage': ['coverage', 'empirical_coverage', 'marginal_coverage', 'picp', 'coverage_rate'],
    'mean_width': ['mean_width', 'mean_width_(mw)', 'mean_interval_width', 'avg_width', 'average_width', 'mpiw'],
    'arm': ['arm', 'adaptation_arm'],
    'seed': ['seed', 'random_seed'],
}

def find_col(df, key):
    mapping = {str(c).strip().lower().replace(' ', '_'): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in mapping:
            return mapping[alias]
    return None

def normalize(df, source):
    rename = {found: key for key in ALIASES if (found := find_col(df, key)) is not None}
    frame = df.rename(columns=rename).copy()
    frame['source_file'] = str(source)
    for field in ('model', 'region', 'method'):
        if field in frame:
            frame[field] = frame[field].astype(str).str.strip().str.lower()
    return frame

def numeric_estimate(value):
    """Read point estimate from e.g. '0.900 ± 0.012' or '90.0% ± 1.2%'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    value = str(value).strip().replace(',', '')
    match = re.match(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', value)
    if not match:
        return np.nan
    estimate = float(match.group(1))
    if '%' in value.split('±')[0].split('+/-')[0]:
        estimate /= 100
    return estimate

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    independent = []
    for region in ('SA1', 'NSW1'):
        base = ROOT / 'results/figures/independent_aci' / region
        # Prefer per-model summaries; SA1 DHR-ARIMA may have different older format.
        for path in sorted(base.glob('*/coverage_width_summary.csv')):
            independent.append(normalize(pd.read_csv(path), path))
        if not list(base.glob('*/coverage_width_summary.csv')) and (base / 'combined_model_summary.csv').exists():
            independent.append(normalize(pd.read_csv(base / 'combined_model_summary.csv'), base / 'combined_model_summary.csv'))
    if not independent:
        raise SystemExit('No independent summary files found; generate figures first.')
    ours = pd.concat(independent, ignore_index=True)
    ours.to_csv(OUT / 'independent_results.csv', index=False)

    sources = sorted(LEADER.glob('*/table_t5_coverage_and_width_*.csv')) + sorted(LEADER.glob('table_t5_coverage_and_width_*.csv'))
    if not sources:
        report = ('# Fixed Split / ACI versus AgACI cross-check\n\n'
                  'Independent results saved to `independent_results.csv`.\n\n'
                  '**AgACI comparison pending:** no leader Table 5 CSV files were found under '
                  '`leader_figures/results/tables/table5`. Extract them from the leader branch.\n')
        (OUT / 'crosscheck_report.md').write_text(report, encoding='utf-8')
        print(report)
        return
    leader_parts = []
    schema_lines = []
    for path in sources:
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            schema_lines.append(f'- {path}: READ ERROR {exc}')
            continue
        norm = normalize(raw, path)
        if 'detector' not in norm.columns:
            norm['detector'] = 'page_hinkley' if 'page_hinkley' in str(path).lower() else ('adwin' if 'adwin' in str(path).lower() else 'unknown')
        schema_lines.append(f'- `{path}`: {list(raw.columns)} ({len(raw)} rows)')
        leader_parts.append(norm)
    if not leader_parts:
        raise SystemExit('Could not read leader CSVs.')
    theirs = pd.concat(leader_parts, ignore_index=True)
    theirs.to_csv(OUT / 'leader_table5_raw.csv', index=False)
    needed = {'model', 'region', 'method', 'coverage', 'mean_width'}
    missing = needed - set(theirs.columns)
    report = ['# Independent Fixed Split / ACI versus AgACI', '',
              '## Experimental alignment', '',
              '- Independent runs: A_never_retrain; ACI updates after 48 issued half-hour forecasts; rolling score buffer 336; gamma tuned on historical calibration.',
              '- Figure 5 uses retrospective global ranking by interval width, **not** a deployable online threshold sweep.',
              '- Leader results may use different adaptation arms, detector triggers, feedback, escalation policies, seeds and/or AgACI implementations.',
              '- Do not interpret differences as a causal effect of the uncertainty method until those configurations are aligned.', '',
              '## Source CSV schemas', '', *schema_lines, '']
    if missing:
        report.extend(['## Comparison not yet computable', '',
                       f'Table 5 is missing recognizable columns: {sorted(missing)}.',
                       'Inspect `leader_table5_raw.csv` and map its schema explicitly before comparing.', ''])
    else:
        ag = theirs[theirs['method'].astype(str).str.contains('agaci', case=False, na=False)].copy()
        if ag.empty:
            report.extend(['## Comparison not yet computable', '',
                           'No rows in leader Table 5 have a method label containing `agaci`.',
                           'Inspect `leader_table5_raw.csv` to find how AgACI is encoded.', ''])
        else:
            ours['model'] = ours['model'].replace({'dhr-arima':'dhr_arima', 'n-hits':'nhits'})
            ag['model'] = ag['model'].replace({'dhr-arima':'dhr_arima', 'n-hits':'nhits'})
            for f in ('coverage', 'mean_width'):
                ag[f] = ag[f].map(numeric_estimate)
                ours[f] = ours[f].map(numeric_estimate)
            # Detect coverage represented as percentage and convert only when obvious.
            if len(ag['coverage'].dropna()) and ag['coverage'].dropna().gt(1).all() and ag['coverage'].dropna().le(100).all():
                ag['coverage'] = ag['coverage'] / 100
            compare = ours.merge(ag, on=['model','region'], how='inner', suffixes=('_ours', '_agaci'))
            if len(compare):
                compare['coverage_delta_ours_minus_agaci'] = compare['coverage_ours'] - compare['coverage_agaci']
                compare['width_delta_ours_minus_agaci'] = compare['mean_width_ours'] - compare['mean_width_agaci']
                compare.to_csv(OUT / 'descriptive_comparison_not_controlled.csv', index=False)
                # Retain only the same adaptation arm for a closer (still uncontrolled) check.
                if 'arm_ours' in compare and 'arm_agaci' in compare:
                    aligned = compare[compare['arm_ours'].astype(str).str.lower() == compare['arm_agaci'].astype(str).str.lower()].copy()
                    aligned.to_csv(OUT / 'same_arm_descriptive_comparison.csv', index=False)
                else:
                    aligned = compare.iloc[:0]
                report.extend(['', '## Parsed numerical comparison', '',
                               f'- AgACI rows with numeric coverage: {ag["coverage"].notna().sum()} / {len(ag)}',
                               f'- AgACI rows with numeric mean width: {ag["mean_width"].notna().sum()} / {len(ag)}',
                               f'- Same adaptation-arm comparison rows: {len(aligned)}',
                               '- Each AgACI value is the reported point estimate (the ± uncertainty is retained in the source CSV).',
                               '- The leader Table 5 contains regime-level and possibly multi-seed aggregates; independent results cover the full test series and one seed. **Do not compare coverage or width deltas as like-for-like performance.**',
                               '- For a valid controlled comparison, use matching full-test or matching regime windows, seeds, calibration periods, detector settings and adaptation arms.', ''])
                report.extend(['## Descriptive comparison (not controlled)', '',
                               f'{len(compare)} matched model-region-method-detector rows exported to `descriptive_comparison_not_controlled.csv`.',
                               'Multiple detector and/or arm rows can appear for the same model-region.',
                               'Check detector, adaptation arm, seed, calibration and feedback before making any performance claim.', ''])
            else:
                report.extend(['## Comparison not yet computable', '',
                               'No matching model/region pairs were found between independent results and AgACI Table 5.', ''])
    (OUT / 'crosscheck_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'\nOutputs: {OUT}')

if __name__ == '__main__':
    main()
