# Table 1 - synthetic drift detection
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

INPUT_FILE = Path('outputs/synthetic_test_results.csv')
OUTPUT_DIR = Path('outputs/table')
OUTPUT_FILE = OUTPUT_DIR / 'table_1_synthetic_detection.csv'

DETECTORS = ['ADWIN', 'KSWIN', 'Page-Hinkley']
DRIFT_TYPES = ['sudden', 'gradual', 'recurring', 'none']

if not INPUT_FILE.exists():
    raise FileNotFoundError(f'Missing file: {INPUT_FILE}. Run the synthetic test first.')

with INPUT_FILE.open('r', newline='', encoding='utf-8-sig') as f:
    test_results = list(csv.DictReader(f))

groups = defaultdict(list)
for result in test_results:
    groups[(result['detector'], result['drift_type'])].append(result)

output_rows = []
for detector in DETECTORS:
    for drift_type in DRIFT_TYPES:
        records = groups[(detector, drift_type)]
        if len(records) != 5:
            raise ValueError(f'Expected five seeds for {detector}, {drift_type}; got {len(records)}')

        # Check the five test datasets
        if len({r['dataset'] for r in records}) != 5:
            raise ValueError(f'Duplicate datasets found for {detector}, {drift_type}')

        delays = [float(r['mean_delay']) for r in records if r['mean_delay'].strip()]
        if drift_type == 'none':
            delay_display = '—'
        elif len(delays) == 5:
            delay_display = f'{mean(delays):.1f} ± {stdev(delays):.1f}'
        else:
            # Show how many seeds detected drift
            delay_display = f'{mean(delays):.1f} ± {stdev(delays):.1f} (detected {len(delays)}/5)' if len(delays) > 1 else 'Not enough detections'

        false_alarm_rate = mean(float(r['false_alarms_per_10k']) for r in records)
        false_alarms_year = false_alarm_rate * 17520 / 10000
        missed = sum(int(r['missed_detections']) for r in records)
        parameter_sets = {r['parameters'] for r in records}
        if len(parameter_sets) != 1:
            raise ValueError(f'Parameters vary between seeds for {detector}, {drift_type}')
        params = json.loads(records[0]['parameters'])
        config = ', '.join(f'{key}={value}' for key, value in sorted(params.items()))

        output_rows.append({
            'detector': detector,
            'drift type': drift_type,
            'delay (mean ± sd)': delay_display,
            'false alarms / 10k': f'{false_alarm_rate:.2f}',
            'false alarms / year (17520)': f'{false_alarms_year:.2f}',
            'missed': missed,
            'config': config,
            'seeds': len(records),
        })

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'detector', 'drift type', 'delay (mean ± sd)',
        'false alarms / 10k', 'false alarms / year (17520)',
        'missed', 'config', 'seeds'
    ])
    writer.writeheader()
    writer.writerows(output_rows)

print('Table 1 generated successfully:')
print(OUTPUT_FILE)
print(f'Rows: {len(output_rows)}')
