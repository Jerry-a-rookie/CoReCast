                                                                          
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
DATASETS = [('ETTh1', 'h'), ('ETTh2', 'h'), ('ETTm1', 't'), ('ETTm2', 't')]
OUT = ROOT / 'results' / 'component_raft_st_L512_H720'
OUT.mkdir(parents=True, exist_ok=True)
reports = []

for name, freq in DATASETS:
    stem = f'{name}_L512_H720'
    result_path = OUT / f'{stem}.json'
    if result_path.exists():
        print(f'SKIP {stem}: completed', flush=True)
        reports.append({'dataset': name, 'status': 'skipped-existing', 'result': str(result_path)})
        continue

    command = [
        sys.executable, '-u', str(ROOT / 'scripts' / 'run_component_raft.py'),
        '--data', name, '--model', 'ComponentRAFTSeasonTrend',
        '--variant', 'retrieval', '--seq-len', '512', '--pred-len', '720',
        '--freq', freq, '--epochs', '10', '--patience', '3', '--batch-size', '16',
        '--learning-rate', '0.001', '--topm', '20', '--groups', '4',
        '--candidate-stride', '1', '--max-candidates', '32',
        '--output', str(OUT),
    ]
    log_path = OUT / f'{stem}.log'
    print(f'START {stem}: {json.dumps(command)}', flush=True)
    started = time.time()
    with log_path.open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)

    report = {
        'dataset': name,
        'status': 'complete' if completed.returncode == 0 else 'failed',
        'returncode': completed.returncode,
        'seconds': time.time() - started,
        'result': str(result_path) if completed.returncode == 0 else None,
        'log': str(log_path),
    }
    reports.append(report)
    (OUT / 'progress.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(f'END {stem}: {json.dumps(report, ensure_ascii=False)}', flush=True)
    if completed.returncode:
        print('STOP: first failed dataset; inspect its log before continuing.', flush=True)
        break

(OUT / 'sweep_summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
subprocess.run([
    sys.executable, str(ROOT / 'scripts' / 'summarize_standard_results.py'),
    '--input', str(OUT), '--output', str(OUT / 'tables'),
], cwd=ROOT, check=True)
print('SWEEP_SUMMARY ' + json.dumps(reports, ensure_ascii=False), flush=True)
