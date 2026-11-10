                                                                                  
import argparse
import csv
import json
from pathlib import Path


STANDARD_DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Illness', 'Weather', 'ECL', 'Traffic')
DATA_ORDER = {name: i for i, name in enumerate(STANDARD_DATASETS)}


def fmt(value):
    return '' if value is None else f'{float(value):.3f}'


def gib(value):
    return '' if value is None else f'{float(value) / (1024 ** 3):.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Directory containing experiment result JSON files')
    parser.add_argument('--output', default=None, help='CSV output directory; defaults to INPUT/tables')
    args = parser.parse_args()
    root = Path(args.input).resolve()
    output = Path(args.output).resolve() if args.output else root / 'tables'
    output.mkdir(parents=True, exist_ok=True)

    metrics_rows, epoch_rows = [], []
    for result_file in root.rglob('*.json'):
        if result_file.parent == output or result_file.name in {
            'matrix.json', 'failures.json', 'progress.json', 'sweep_summary.json',
        }:
            continue
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        config = result.get('config', {})
        dataset = config.get('data', result.get('data'))
        test = result.get('test', {})
        if dataset not in DATA_ORDER or 'mse' not in test or 'mae' not in test:
            continue

        common = {
            '数据集': dataset,
            '模型': config.get('model', result.get('model', '')),
            '方案': config.get('variant', ''),
            '输入长度': result.get('seq_len', config.get('seq_len', '')),
            '预测长度': result.get('pred_len', config.get('pred_len', '')),
            '随机种子': config.get('seed', ''),
        }
        performance = result.get('performance', {})
        metrics_rows.append({
            **common,
            '测试MSE': fmt(test.get('mse')),
            '测试MAE': fmt(test.get('mae')),
            '最佳轮次': result.get('best_epoch', ''),
            '测试秒': fmt(performance.get('test_seconds_including_retrieval', test.get('seconds'))),
            '测试样本/秒': fmt(test.get('samples_per_second')),
            '测试峰值已分配GiB': gib(performance.get('test_peak_allocated_bytes')),
            '测试峰值预留GiB': gib(performance.get('test_peak_reserved_bytes')),
        })

        for epoch_no, epoch in enumerate(result.get('history', []), 1):
            train, val = epoch.get('train', {}), epoch.get('val', {})
            epoch_rows.append({
                **common,
                '轮次': epoch.get('epoch', epoch_no),
                '训练秒': fmt(train.get('seconds')),
                '训练样本/秒': fmt(train.get('samples_per_second')),
                '训练峰值已分配GiB': gib(train.get('peak_allocated_bytes')),
                '训练峰值预留GiB': gib(train.get('peak_reserved_bytes')),
                '验证秒': fmt(val.get('seconds')),
                '验证样本/秒': fmt(val.get('samples_per_second')),
                '验证峰值已分配GiB': gib(val.get('peak_allocated_bytes')),
                '验证峰值预留GiB': gib(val.get('peak_reserved_bytes')),
            })

    sort_key = lambda row: (
        DATA_ORDER[row['数据集']], int(row['预测长度'] or 0), row['模型'],
        row['方案'], int(row['随机种子'] or 0),
    )
    metrics_rows.sort(key=sort_key)
    epoch_rows.sort(key=lambda row: (*sort_key(row), int(row['轮次'])))

    tables = (
        ('standard_metrics.csv', metrics_rows, [
            '数据集', '模型', '方案', '输入长度', '预测长度', '随机种子',
            '测试MSE', '测试MAE', '最佳轮次', '测试秒', '测试样本/秒',
            '测试峰值已分配GiB', '测试峰值预留GiB',
        ]),
        ('epoch_efficiency.csv', epoch_rows, [
            '数据集', '模型', '方案', '输入长度', '预测长度', '随机种子', '轮次',
            '训练秒', '训练样本/秒', '训练峰值已分配GiB', '训练峰值预留GiB',
            '验证秒', '验证样本/秒', '验证峰值已分配GiB', '验证峰值预留GiB',
        ]),
    )
    for filename, rows, fields in tables:
        with (output / filename).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        'metrics_rows': len(metrics_rows),
        'epoch_rows': len(epoch_rows),
        'metrics_csv': str(output / 'standard_metrics.csv'),
        'efficiency_csv': str(output / 'epoch_efficiency.csv'),
    }, ensure_ascii=False))


if __name__ == '__main__':
    import json
    main()
