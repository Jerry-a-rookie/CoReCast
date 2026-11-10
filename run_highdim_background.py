                      
                                                                          

                                                                           
                                                            
   
import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from ucast_paper_lengths import PAPER_LENGTHS

HIGH_FREQ = {
    'Atec': 't', 'Air Quality': 'h', 'Temp': 'h', 'Wind': 'h',
    'Traffic-CA': 'h', 'Traffic-GBA': 'h', 'Traffic-GLA': 'h',
    'Neurolib': 'h', 'SIRS': 'd', 'Meter': 't', 'SP500': 'd', 'Wiki-20k': 'd',
}
HIGH_CHANNELS = {
                                                                 
    'Atec': 1569, 'Air Quality': 1105, 'Temp': 3774, 'Wind': 3696,
    'Traffic-CA': 7491, 'Traffic-GBA': 2229, 'Traffic-GLA': 3376,
    'Neurolib': 2000, 'SIRS': 2994, 'Meter': 2898, 'SP500': 1475,
    'Wiki-20k': 20000,
}
HIGH_DATASETS = list(HIGH_FREQ)
MODEL_DATASETS = ['SIRS', 'Meter', 'SP500', 'Wiki-20k']
MODULE_MODELS = ['DLinear', 'iTransformer', 'DUET', 'PatchTST', 'UCast', 'CCM', 'ModernTCN', 'TSMixer']
DUAL_VARIANT_MODELS = ['ModernTCN', 'TSMixer']
CSV_FIELDS = [
    'record_type', 'scope', 'dataset', 'channels', 'groups', 'model', 'variant', 'batch_size', 'attempt',
    'seq_len', 'pred_len',
    'freq', 'seed', 'epoch', 'best_epoch',
    'epoch_seconds', 'epoch_peak_allocated_gib', 'epoch_peak_reserved_gib',
    'train_mse', 'train_mae', 'train_seconds', 'train_samples_per_second',
    'train_peak_allocated_gib', 'train_peak_reserved_gib',
    'val_mse', 'val_mae', 'val_seconds', 'val_samples_per_second',
    'val_peak_allocated_gib', 'val_peak_reserved_gib',
    'test_mse', 'test_mae', 'test_seconds', 'test_samples_per_second',
    'test_peak_allocated_gib', 'test_peak_reserved_gib', 'total_seconds',
    'parameter_count', 'status', 'failure_message', 'result_json',
]
LEGACY_CSV_FIELDS = [field for field in CSV_FIELDS if field not in ('batch_size', 'attempt')]
KEY_FIELDS = ('record_type', 'scope', 'dataset', 'channels', 'groups', 'model', 'variant',
              'batch_size', 'attempt', 'seq_len', 'pred_len', 'seed', 'epoch')


def fmt(value):
    return '' if value is None else f'{float(value):.3f}'


def gib(value):
    return '' if value is None else f'{float(value) / (1024 ** 3):.3f}'


def groups_for_dataset(dataset):
    pass                                                                        
    return max(1, math.ceil(HIGH_CHANNELS[dataset] / 10))


def empty_row(meta, record_type, result_file):
    row = {name: '' for name in CSV_FIELDS}
    row.update(meta)
    row['record_type'] = record_type
    row['result_json'] = str(result_file)
    return row


class ProgressCSV:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self.seen = set()
        with self.lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_file() and self.path.stat().st_size:
                with self.path.open('r', newline='', encoding='utf-8') as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames == LEGACY_CSV_FIELDS:
                        old_rows = list(reader)
                        temp_path = self.path.with_name(self.path.name + '.migrate.tmp')
                        with temp_path.open('w', newline='', encoding='utf-8') as migrated:
                            writer = csv.DictWriter(migrated, fieldnames=CSV_FIELDS)
                            writer.writeheader()
                            for row in old_rows:
                                row.setdefault('batch_size', '')
                                row.setdefault('attempt', '')
                                writer.writerow(row)
                            migrated.flush()
                            os.fsync(migrated.fileno())
                        os.replace(temp_path, self.path)
                    elif reader.fieldnames != CSV_FIELDS:
                        raise ValueError(f'Existing CSV header differs: {self.path}')
                with self.path.open('r', newline='', encoding='utf-8') as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        self.seen.add(tuple(row.get(field, '') for field in KEY_FIELDS))
            else:
                with self.path.open('w', newline='', encoding='utf-8') as handle:
                    csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
                    handle.flush()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, row):
        key = tuple(str(row.get(field, '')) for field in KEY_FIELDS)
        if key in self.seen:
            return
        with self.lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with self.path.open('a', newline='', encoding='utf-8') as handle:
                csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)
                handle.flush()
            self.seen.add(key)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def has_task_rows(self, meta, record_type=None):
        identity_fields = ('scope', 'dataset', 'channels', 'groups', 'model', 'variant',
                           'seq_len', 'pred_len', 'seed')
        for key in self.seen:
            row = dict(zip(KEY_FIELDS, key))
            if (record_type is not None and row.get('record_type') != record_type):
                continue
            if all(str(row.get(field, '')) == str(meta.get(field, '')) for field in identity_fields):
                return True
        return False


def make_epoch_row(meta, epoch, result_file, status='epoch_complete'):
    row = empty_row(meta, 'epoch', result_file)
    row['epoch'] = epoch.get('epoch', '')
    row['status'] = status
    train = epoch.get('train') or {}
    val = epoch.get('val') or {}
    for prefix, values in (('train', train), ('val', val)):
        for key in ('mse', 'mae', 'seconds', 'samples_per_second'):
            row[f'{prefix}_{key}'] = fmt(values.get(key))
        row[f'{prefix}_peak_allocated_gib'] = gib(values.get('peak_allocated_bytes'))
        row[f'{prefix}_peak_reserved_gib'] = gib(values.get('peak_reserved_bytes'))
    train_seconds = train.get('seconds')
    val_seconds = val.get('seconds')
    if train_seconds is not None and val_seconds is not None:
        row['epoch_seconds'] = fmt(float(train_seconds) + float(val_seconds))
    allocated = [v.get('peak_allocated_bytes') for v in (train, val) if v.get('peak_allocated_bytes') is not None]
    reserved = [v.get('peak_reserved_bytes') for v in (train, val) if v.get('peak_reserved_bytes') is not None]
    row['epoch_peak_allocated_gib'] = gib(max(allocated)) if allocated else ''
    row['epoch_peak_reserved_gib'] = gib(max(reserved)) if reserved else ''
    return row


def append_result(progress, meta, result, result_file, include_history=True, include_test=True):
    if include_history:
        for epoch_no, epoch in enumerate(result.get('history') or [], 1):
            epoch = dict(epoch)
            epoch.setdefault('epoch', epoch_no)
            progress.append(make_epoch_row(meta, epoch, result_file))

    if not include_test:
        return
    test = result.get('test') or {}
    performance = result.get('performance') or {}
    row = empty_row(meta, 'test', result_file)
    row.update({
        'best_epoch': result.get('best_epoch', ''),
        'test_mse': fmt(test.get('mse')),
        'test_mae': fmt(test.get('mae')),
        'test_seconds': fmt(performance.get('test_seconds_including_retrieval', test.get('seconds'))),
        'test_samples_per_second': fmt(test.get('samples_per_second')),
        'test_peak_allocated_gib': gib(performance.get('test_peak_allocated_bytes', test.get('peak_allocated_bytes'))),
        'test_peak_reserved_gib': gib(performance.get('test_peak_reserved_bytes', test.get('peak_reserved_bytes'))),
        'total_seconds': fmt(result.get('seconds')),
        'parameter_count': result.get('parameter_count', ''),
        'status': 'complete',
    })
    progress.append(row)


def batch_size_schedule(initial_batch_size):
    sizes = []
    current = initial_batch_size
    while current >= 1:
        sizes.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    return sizes


def is_oom_message(line):
    lowered = line.lower()
    markers = (
        'out of memory', 'cuda_error_out_of_memory', 'cudnn_status_alloc_failed',
        'cublas_status_alloc_failed', 'defaultcpuallocator: not enough memory',
    )
    return any(marker in lowered for marker in markers)


def archive_artifacts(target, paths, label):
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    archive = target / 'previous_runs' / f'{label}_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns() % 1000000:06d}'
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        os.replace(path, archive / path.name)
    return archive


def write_result_json(path, result):
    temp_path = path.with_name(path.name + '.tmp')
    with temp_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def run_job(job, args, output_base, progress):
    model, variant, data, channels, groups, seq_len, pred_len, freq = (
        job[k] for k in ('model', 'variant', 'dataset', 'channels', 'groups', 'seq_len', 'pred_len', 'freq')
    )
    scope = job['scope']
    scope_root = output_base / 'full' / scope
    target = (scope_root / model / variant / data.replace(' ', '_') /
              f'L{seq_len}_H{pred_len}' / f'seed{args.seed}')
    target.mkdir(parents=True, exist_ok=True)
    stem = f'{data.replace(" ", "_")}_L{seq_len}_H{pred_len}'
    result_file = target / f'{stem}.json'
    checkpoint = target / f'{stem}.pt'
    run_log = target / 'run.log'
    meta = {
        'scope': scope, 'dataset': data, 'channels': channels, 'groups': groups,
        'model': model, 'variant': variant,
        'batch_size': args.batch_size, 'attempt': '',
        'seq_len': seq_len, 'pred_len': pred_len, 'freq': freq, 'seed': args.seed,
    }

    expected = {
        'data': data, 'model': model, 'variant': variant, 'seq_len': seq_len,
        'pred_len': pred_len, 'freq': freq, 'topm': 10,
        'max_candidates': args.max_candidates,
        'candidate_stride': max(1, pred_len // 4), 'seed': args.seed,
        'device': args.device,
        'epochs': args.epochs, 'patience': args.patience,
        'learning_rate': args.learning_rate, 'groups': groups,
        'ema_alpha': 0.2, 'cluster_temperature': 0.1,
        'weight_temperature': 0.1, 'kmeans_iters': 3, 'cache_mb': 256,
        'checkpoint_in': None,
    }
    if args.resume and result_file.is_file():
        saved = None
        try:
            saved = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            pass
        config = (saved or {}).get('config') or {}
        differences = {k: (config.get(k), value) for k, value in expected.items()
                       if config.get(k) != value}
        batch_retry = (saved or {}).get('batch_retry') or {}
        saved_batch = config.get('batch_size')
        allowed_batches = batch_size_schedule(args.batch_size)
        if batch_retry:
            batch_matches = (
                batch_retry.get('initial_batch_size') == args.batch_size
                and batch_retry.get('effective_batch_size') == saved_batch
                and saved_batch in allowed_batches
                and batch_retry.get('oom_retries') == allowed_batches.index(saved_batch)
            )
        else:
            batch_matches = saved_batch == args.batch_size
        if not batch_matches:
            differences['batch_size'] = (saved_batch, args.batch_size)
        complete = (
            saved is not None and not differences and checkpoint.is_file()
            and 'mse' in (saved.get('test') or {})
        )
        if complete:
            retry_count = int(batch_retry.get('oom_retries', 0))
            effective_meta = dict(meta, batch_size=saved_batch, attempt=retry_count + 1)
            append_result(
                progress, effective_meta, saved, result_file,
                include_history=not progress.has_task_rows(effective_meta, 'epoch'),
                include_test=not progress.has_task_rows(effective_meta, 'test'),
            )
            print(f'SKIP complete {data} {model} {variant} batch={saved_batch}', flush=True)
            return True

        print(f'RERUN existing result {result_file}; config differences={differences}', flush=True)
        archive_artifacts(target, [result_file, checkpoint, run_log, *target.glob('run_attempt_*.log')], 'incompatible')
    elif args.resume:
        archive_artifacts(target, [checkpoint, run_log, *target.glob('run_attempt_*.log')], 'incomplete')
    elif not args.resume:
        archive_artifacts(target, [result_file, checkpoint, run_log, *target.glob('run_attempt_*.log')], 'rerun')

    batch_sizes = batch_size_schedule(args.batch_size)
    print(
        f'RUN {data} C{channels} groups{groups} {model} {variant} '
        f'L{seq_len} H{pred_len} batch_schedule={batch_sizes}', flush=True,
    )
    for attempt, current_batch_size in enumerate(batch_sizes, 1):
        attempt_meta = dict(meta, batch_size=current_batch_size, attempt=attempt)
        cmd = [
            sys.executable, '-u', str(SCRIPTS / 'run_component_raft.py'),
            '--data', data, '--model', model, '--variant', variant,
            '--seq-len', str(seq_len), '--pred-len', str(pred_len), '--freq', freq,
            '--topm', '10', '--max-candidates', str(args.max_candidates),
            '--candidate-stride', str(max(1, pred_len // 4)), '--seed', str(args.seed),
            '--batch-size', str(current_batch_size), '--device', args.device,
            '--epochs', str(args.epochs), '--patience', str(args.patience),
            '--learning-rate', str(args.learning_rate), '--groups', str(groups),
            '--ema-alpha', '0.2', '--cluster-temperature', '0.1',
            '--weight-temperature', '0.1', '--kmeans-iters', '3',
            '--cache-mb', '256', '--output', str(target),
        ]
        result_line = None
        oom_seen = False
        returncode = None
        with run_log.open('a', encoding='utf-8') as log:
            log.write(f'\n=== attempt {attempt}; batch_size={current_batch_size} ===\n')
            log.flush()
            process = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{data} {model} {variant} b{current_batch_size} a{attempt}] {line}', end='', flush=True)
                oom_seen = oom_seen or is_oom_message(line)
                stripped = line.strip()
                if stripped.startswith('RESULT '):
                    result_line = stripped[len('RESULT '):]
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and 'epoch' in event and 'train' in event and 'val' in event:
                    progress.append(make_epoch_row(attempt_meta, event, result_file))
            returncode = process.wait()

        result = None
        if result_file.is_file() and returncode == 0:
            try:
                result = json.loads(result_file.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                result = None
        if result is None and result_line:
            try:
                result = json.loads(result_line)
            except json.JSONDecodeError:
                result = None
        if result is not None and 'mse' in (result.get('test') or {}):
            result['batch_retry'] = {
                'initial_batch_size': args.batch_size,
                'effective_batch_size': current_batch_size,
                'oom_retries': attempt - 1,
            }
            write_result_json(result_file, result)
            append_result(progress, attempt_meta, result, result_file)
            print(f'DONE {data} {model} {variant} batch={current_batch_size} attempts={attempt}', flush=True)
            return True

        oom_exit = returncode in (-9, 137)
        if oom_seen or oom_exit:
            if attempt < len(batch_sizes):
                retry = empty_row(attempt_meta, 'retry', result_file)
                retry['status'] = 'oom_batch_halved'
                retry['failure_message'] = (
                    f'CUDA/system OOM at batch_size={current_batch_size}; '
                    f'retrying from scratch at batch_size={batch_sizes[attempt]}'
                )
                progress.append(retry)
                print(retry['failure_message'], flush=True)
                archive_artifacts(target, [result_file, checkpoint], f'oom_attempt_{attempt}')
                continue
            failure = empty_row(attempt_meta, 'failure', result_file)
            failure['status'] = 'oom_at_batch_size_1'
            failure['failure_message'] = f'Out of memory at minimum batch size 1; see {run_log}'
            progress.append(failure)
            print(f'FAILED {data} {model} {variant}; OOM persisted at batch size 1', flush=True)
            return False

        failure = empty_row(attempt_meta, 'failure', result_file)
        failure['status'] = f'failed_{returncode}'
        failure['failure_message'] = f'run_component_raft.py exited with {returncode}; see {run_log}'
        progress.append(failure)
        print(f'FAILED {data} {model} {variant}; see {run_log}', flush=True)
        return False

    return False


def main():
    parser = argparse.ArgumentParser(description='Run high-dimensional experiments with live CSV progress.')
    parser.add_argument('--scope', choices=['model', 'module', 'all'], required=True,
        help='all runs independent-model jobs, then the selected module jobs')
    parser.add_argument('--datasets', nargs='+', choices=HIGH_DATASETS, default=None,
                        help='filter datasets; independent model is restricted to SIRS, Meter, SP500 and Wiki-20k')
    parser.add_argument('--models', nargs='+', choices=MODULE_MODELS)
    parser.add_argument('--module-variants', nargs='+', choices=['original', 'retrieval'],
                        default=None,
                        help='override the default per-model variant policy for every selected module model')
    parser.add_argument('--both-variants-models', nargs='+', choices=MODULE_MODELS,
                        default=DUAL_VARIANT_MODELS,
                        help='models that run both original and retrieval; all other module models run retrieval only')
    parser.add_argument('--output', default='results/highdim_c10')
    parser.add_argument('--progress-csv',
                        help='optional shared CSV path; Linux file locking supports concurrent scopes')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--max-candidates', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')

    scopes = ['model', 'module'] if args.scope == 'all' else [args.scope]
    models = args.models or MODULE_MODELS
    selected_datasets = args.datasets if args.datasets is not None else HIGH_DATASETS
    if args.scope == 'model' and args.models:
        parser.error('--models is only used with --scope module')
    jobs = []
    for scope in scopes:
        scope_models = (models if scope == 'module' else ['ComponentRAFTSeasonTrend'])
        scope_datasets = selected_datasets if scope == 'module' else [d for d in selected_datasets if d in MODEL_DATASETS]
        if scope == 'model' and not scope_datasets:
            parser.error('no independent-model datasets selected; choose from ' + ', '.join(MODEL_DATASETS))
        for data in scope_datasets:
            horizon, factor = PAPER_LENGTHS[data]
            for model in scope_models:
                if scope != 'module':
                    variants = ['retrieval']
                elif args.module_variants is not None:
                    variants = args.module_variants
                elif model in args.both_variants_models:
                    variants = ['original', 'retrieval']
                else:
                    variants = ['retrieval']
                for variant in variants:
                    jobs.append({
                        'scope': scope, 'dataset': data, 'model': model, 'variant': variant,
                        'channels': HIGH_CHANNELS[data], 'groups': groups_for_dataset(data),
                        'seq_len': horizon * factor, 'pred_len': horizon, 'freq': HIGH_FREQ[data],
                    })

    output_base = Path(args.output)
    if not output_base.is_absolute():
        output_base = ROOT / output_base
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / 'matrix.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8')
    progress_path = Path(args.progress_csv) if args.progress_csv else output_base / 'highdim_progress.csv'
    if not progress_path.is_absolute():
        progress_path = ROOT / progress_path
    progress = ProgressCSV(progress_path)
    print(f'PLAN {len(jobs)} runs; scopes={scopes}; output={output_base}', flush=True)
    print(
        f'BATCH initial={args.batch_size}; OOM schedule={batch_size_schedule(args.batch_size)}; '
        f'patience={args.patience}', flush=True,
    )
    if args.scope in ('module', 'all'):
        print(
            'VARIANTS: ' + ', '.join(
                f'{model}={"/".join(args.module_variants) if args.module_variants is not None else ("original/retrieval" if model in args.both_variants_models else "retrieval")}'
                for model in models
            ), flush=True,
        )
    print(f'CSV {progress.path}', flush=True)
    print('GROUPS C/10: ' + ', '.join(
        f'{data}=C{HIGH_CHANNELS[data]}/G{groups_for_dataset(data)}' for data in selected_datasets
    ), flush=True)

    failures = 0
    for job in jobs:
        try:
            failures += int(not run_job(job, args, output_base, progress))
        except Exception as exc:
            meta = {
                'scope': job['scope'], 'dataset': job['dataset'],
                'channels': job['channels'], 'groups': job['groups'], 'model': job['model'],
                'variant': job['variant'], 'seq_len': job['seq_len'],
                'pred_len': job['pred_len'], 'freq': job['freq'], 'seed': args.seed,
            }
            row = empty_row(meta, 'failure', '')
            row['status'] = 'failed'
            row['failure_message'] = str(exc)
            progress.append(row)
            failures += 1
            print(f'FAILED {job}: {exc}', flush=True)
    print(f'FINISHED {len(jobs)-failures}/{len(jobs)} complete; failures={failures}; csv={progress.path}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
