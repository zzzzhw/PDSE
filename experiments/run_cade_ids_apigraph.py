"""Run matched CADE and CADE+IDS experiments on the APIGraph stream."""

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / 'models' / 'experiments' / 'cade_ids_apigraph'
RESULT_ROOT = REPO_ROOT / 'experiments' / 'results' / 'cade_ids_apigraph'

IDS_PARAMETERS = {
    'ids_lambda': 0.001,
    'ids_gamma': 1.0,
    'ids_ema_decay': 0.6,
    'ids_proxy_lr': 1.0,
    'ids_robust_weight': 1.0,
    'ids_batch_size': 192,
    'ids_warmup': 5,
    'ids_grad_clip': 1.0,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for(name, smoke=False):
    suffix = 'smoke' if smoke else 'full'
    result_dir = RESULT_ROOT / suffix / name
    return {
        'model_dir': MODEL_ROOT / suffix / name / '200',
        'result_dir': result_dir,
        'result': result_dir / 'apigraph.csv',
        'log': result_dir / 'apigraph.log',
        'console': result_dir / 'console.log',
        'runtime': result_dir / 'runtime_seconds.txt',
    }


def result_complete(path, test_end):
    if not path.exists():
        return False
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    return any(row['date'] == test_end for row in rows)


def build_command(paths, ids=False, smoke=False):
    test_end = '2013-02' if smoke else '2018-12'
    initial_epochs = '2' if smoke else '250'
    monthly_epochs = '7' if smoke else '50'
    mlp_epochs = '2' if smoke else '50'
    command = [
        sys.executable,
        '-u',
        'base.py',
        '--method', 'cade',
        '--data', 'gen_apigraph_drebin',
        '--benign_zero',
        '--mdate', '20230501',
        '--train_start', '2012-01',
        '--train_end', '2012-12',
        '--test_start', '2013-01',
        '--test_end', test_end,
        '--encoder', 'cae',
        '--enc-hidden', '512-384-256-128',
        '--loss_func', 'triplet-mse',
        '--sampler', 'triplet',
        '--bsize', '1536',
        '--optimizer', 'adam',
        '--scheduler', 'cosine',
        '--learning_rate', '0.001',
        '--lr_decay_rate', '1',
        '--lr_decay_epochs', '10,500,10',
        '--epochs', initial_epochs,
        '--cae-lambda', '0.1',
        '--display-interval', '100',
        '--classifier', 'mlp',
        '--cls-feat', 'encoded',
        '--mlp-hidden', '100-100',
        '--mlp-dropout', '0.2',
        '--mlp-batch-size', '32',
        '--mlp-lr', '0.001',
        '--mlp-epochs', mlp_epochs,
        '--al',
        '--ood',
        '--count', '200',
        '--F1D', '6',
        '--encoder_retrain',
        '--warm_learning_rate', '0.00005',
        '--al_epochs', monthly_epochs,
        '--mlp-warm-lr', '0.00005',
        '--mlp-warm-epochs', mlp_epochs,
        '--seed', '1',
        '--model-dir', str(paths['model_dir']),
        '--result', str(paths['result']),
        '--log_path', str(paths['log']),
    ]
    if ids:
        command.append('--ids')
        for option, value in IDS_PARAMETERS.items():
            command.extend([f'--{option.replace("_", "-")}', str(value)])
    return command


def run_method(name, ids=False, smoke=False):
    paths = paths_for(name, smoke=smoke)
    paths['result_dir'].mkdir(parents=True, exist_ok=True)
    test_end = '2013-02' if smoke else '2018-12'
    if result_complete(paths['result'], test_end):
        print(f'Skipping completed {name} run.', flush=True)
        if paths['runtime'].exists():
            return float(paths['runtime'].read_text(encoding='utf-8').strip())
        return 0.0
    if paths['result'].exists():
        raise RuntimeError(f'Partial result exists: {paths["result"]}')

    command = build_command(paths, ids=ids, smoke=smoke)
    print(f'Running {name}: {" ".join(command)}', flush=True)
    started = time.time()
    with paths['console'].open('w', encoding='utf-8') as console:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.time() - started
    if not result_complete(paths['result'], test_end):
        raise RuntimeError(f'{name} completed without a result through {test_end}')
    paths['runtime'].write_text(f'{elapsed:.6f}\n', encoding='utf-8')
    print(f'{name} completed in {elapsed:.1f} seconds.', flush=True)
    return elapsed


def initial_checkpoints(name, smoke=False):
    model_dir = paths_for(name, smoke=smoke)['model_dir'] / '2012-01to2012-12'
    encoders = [path for path in model_dir.glob('cae_*.pth') if 'retrain' not in path.name]
    classifiers = [path for path in model_dir.glob('MLP_*.pth') if 'retrain' not in path.name]
    if len(encoders) != 1 or len(classifiers) != 1:
        raise RuntimeError(
            f'Expected one initial CAE and MLP checkpoint in {model_dir}; '
            f'found {len(encoders)} and {len(classifiers)}'
        )
    return encoders[0], classifiers[0]


def share_initial_checkpoints(smoke=False):
    sources = initial_checkpoints('cade', smoke=smoke)
    destination_dir = (
        paths_for('cade_ids', smoke=smoke)['model_dir'] / '2012-01to2012-12'
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = tuple(destination_dir / source.name for source in sources)
    for source, destination in zip(sources, destinations):
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f'Initial checkpoint differs: {destination}')
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise RuntimeError(f'Checkpoint verification failed: {destination}')
    return sources, destinations


def read_test_rows(path, test_end):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    selected = [row for row in rows if '2013-01' <= row['date'] <= test_end]
    expected = 2 if test_end == '2013-02' else 72
    if len(selected) != expected:
        raise RuntimeError(f'Expected {expected} test months in {path}, found {len(selected)}')
    return selected


def summarize(rows):
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in ['F1', 'FNR', 'FPR', 'ACC', 'PREC']
    }
    return {
        'mean_f1': float(values['F1'].mean()),
        'variance_f1': float(values['F1'].var()),
        'min_f1': float(values['F1'].min()),
        'max_f1': float(values['F1'].max()),
        'final_f1': float(values['F1'][-1]),
        'aut_f1': float(np.trapz(values['F1'], dx=1)),
        'mean_fnr': float(values['FNR'].mean()),
        'variance_fnr': float(values['FNR'].var()),
        'min_fnr': float(values['FNR'].min()),
        'max_fnr': float(values['FNR'].max()),
        'mean_fpr': float(values['FPR'].mean()),
        'variance_fpr': float(values['FPR'].var()),
        'min_fpr': float(values['FPR'].min()),
        'max_fpr': float(values['FPR'].max()),
        'mean_accuracy': float(values['ACC'].mean()),
        'mean_precision': float(values['PREC'].mean()),
    }


def pooled_summary(result_path, expected_months):
    stat_path = result_path.with_name(f'{result_path.stem}_stat.csv')
    with stat_path.open(newline='', encoding='utf-8') as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter='\t')
            if '2013-01' <= row['date'] <= '2018-12'
        ]
    if len(rows) != expected_months:
        raise RuntimeError(
            f'Expected {expected_months} statistic rows in {stat_path}, found {len(rows)}'
        )
    tp, tn, fp, fn = (
        sum(int(row[name]) for row in rows)
        for name in ['TP', 'TN', 'FP', 'FN']
    )
    return {
        'pooled_f1': 2 * tp / (2 * tp + fp + fn),
        'pooled_fnr': fn / (tp + fn),
        'pooled_fpr': fp / (tn + fp),
        'pooled_accuracy': (tp + tn) / (tp + tn + fp + fn),
        'pooled_precision': tp / (tp + fp),
    }


def write_comparison(runtimes, checkpoints, smoke=False):
    test_end = '2013-02' if smoke else '2018-12'
    output_dir = RESULT_ROOT / ('smoke' if smoke else 'full')
    baseline_rows = read_test_rows(paths_for('cade', smoke)['result'], test_end)
    ids_rows = read_test_rows(paths_for('cade_ids', smoke)['result'], test_end)
    baseline = summarize(baseline_rows)
    ids = summarize(ids_rows)
    expected_months = 2 if smoke else 72
    baseline.update(pooled_summary(
        paths_for('cade', smoke)['result'], expected_months
    ))
    ids.update(pooled_summary(
        paths_for('cade_ids', smoke)['result'], expected_months
    ))

    with (output_dir / 'comparison.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['metric', 'cade', 'cade_ids', 'difference'])
        for metric in baseline:
            writer.writerow([
                metric,
                f'{baseline[metric]:.8f}',
                f'{ids[metric]:.8f}',
                f'{ids[metric] - baseline[metric]:+.8f}',
            ])

    with (output_dir / 'per_month.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow([
            'date', 'cade_f1', 'cade_ids_f1', 'delta_f1',
            'cade_fnr', 'cade_ids_fnr', 'delta_fnr',
            'cade_fpr', 'cade_ids_fpr', 'delta_fpr',
        ])
        for base_row, ids_row in zip(baseline_rows, ids_rows):
            writer.writerow([
                base_row['date'], base_row['F1'], ids_row['F1'],
                f"{float(ids_row['F1']) - float(base_row['F1']):+.8f}",
                base_row['FNR'], ids_row['FNR'],
                f"{float(ids_row['FNR']) - float(base_row['FNR']):+.8f}",
                base_row['FPR'], ids_row['FPR'],
                f"{float(ids_row['FPR']) - float(base_row['FPR']):+.8f}",
            ])

    deltas = np.asarray([
        float(ids_row['F1']) - float(base_row['F1'])
        for base_row, ids_row in zip(baseline_rows, ids_rows)
    ])
    manifest_lines = [
        f'test_end\t{test_end}',
        f'cade_runtime_seconds\t{runtimes["cade"]:.3f}',
        f'cade_ids_runtime_seconds\t{runtimes["cade_ids"]:.3f}',
        f'initial_encoder_sha256\t{sha256(checkpoints[0][0])}',
        f'initial_classifier_sha256\t{sha256(checkpoints[0][1])}',
        f'ids_f1_wins\t{int((deltas > 0).sum())}',
        f'ids_f1_losses\t{int((deltas < 0).sum())}',
        f'ids_f1_ties\t{int((deltas == 0).sum())}',
    ]
    manifest_lines.extend(f'{key}\t{value}' for key, value in IDS_PARAMETERS.items())
    (output_dir / 'run_manifest.tsv').write_text(
        '\n'.join(manifest_lines) + '\n', encoding='utf-8'
    )
    print({'cade': baseline, 'cade_ids': ids, 'wins': int((deltas > 0).sum())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', choices=['both', 'cade', 'cade_ids'], default='both')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CADE APIGraph comparison requires CUDA')

    runtimes = {'cade': 0.0, 'cade_ids': 0.0}
    if args.only in ('both', 'cade'):
        runtimes['cade'] = run_method('cade', ids=False, smoke=args.smoke)
    checkpoints = share_initial_checkpoints(smoke=args.smoke)
    if args.only in ('both', 'cade_ids'):
        runtimes['cade_ids'] = run_method('cade_ids', ids=True, smoke=args.smoke)

    test_end = '2013-02' if args.smoke else '2018-12'
    if all(result_complete(paths_for(name, args.smoke)['result'], test_end)
           for name in ('cade', 'cade_ids')):
        write_comparison(runtimes, checkpoints, smoke=args.smoke)


if __name__ == '__main__':
    main()
