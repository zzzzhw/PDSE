"""Run matched HCL and HCL+IDS experiments on the KronoDroid 2012 stream."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prepare_kronodroid import REPO_ROOT, prepare_dataset


RAW_DIR = REPO_ROOT / 'data' / 'kronodroid_raw'
DATA_DIR = REPO_ROOT / 'data' / 'kronodroid_real_hybrid'
RESULT_ROOT = REPO_ROOT / 'experiments' / 'results' / 'kronodroid_real_2012'
MODEL_ROOT = REPO_ROOT / 'models' / 'experiments' / 'kronodroid_real_2012'

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


def result_complete(path):
    if not path.exists():
        return False
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    return any(row['date'] == '2012-12' for row in rows)


def build_command(model_dir, result_path, log_path, ids=False):
    command = [
        sys.executable,
        '-u',
        'base.py',
        '--method', 'pseudo',
        '--data', 'kronodroid_real_hybrid',
        '--benign_zero',
        '--mdate', '20260718',
        '--train_start', '2011-01',
        '--train_end', '2011-12',
        '--test_start', '2012-01',
        '--test_end', '2012-12',
        '--encoder', 'simple-enc-mlp',
        '--classifier', 'simple-enc-mlp',
        '--loss_func', 'hi-dist-xent',
        '--enc-hidden', '512-384-256-128',
        '--mlp-hidden', '100-100',
        '--mlp-dropout', '0.2',
        '--sampler', 'half',
        '--bsize', '1024',
        '--optimizer', 'sgd',
        '--scheduler', 'step',
        '--learning_rate', '0.001',
        '--lr_decay_rate', '0.5',
        '--lr_decay_epochs', '10,500,10',
        '--epochs', '200',
        '--encoder_retrain',
        '--al_optimizer', 'adam',
        '--warm_learning_rate', '0.00001',
        '--al_epochs', '50',
        '--xent-lambda', '100',
        '--display-interval', '180',
        '--al',
        '--count', '200',
        '--F1D', '6',
        '--local_pseudo_loss',
        '--reduce', 'none',
        '--sample_reduce', 'mean',
        '--seed', '1',
        '--model-dir', str(model_dir),
        '--result', str(result_path),
        '--log_path', str(log_path),
    ]
    if ids:
        command.append('--ids')
        for option, value in IDS_PARAMETERS.items():
            command.extend([f'--{option.replace("_", "-")}', str(value)])
    return command


def run_method(name, ids=False):
    run_dir = RESULT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = MODEL_ROOT / name / '200'
    result_path = run_dir / 'kronodroid.csv'
    log_path = run_dir / 'kronodroid.log'
    console_path = run_dir / 'console.log'
    if result_complete(result_path):
        print(f'Skipping completed {name} run.', flush=True)
        return 0.0
    if result_path.exists():
        raise RuntimeError(f'Partial result exists: {result_path}')

    command = build_command(model_dir, result_path, log_path, ids=ids)
    print(f'Running {name}: {" ".join(command)}', flush=True)
    started = time.time()
    with console_path.open('w', encoding='utf-8') as console:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.time() - started
    if not result_complete(result_path):
        raise RuntimeError(f'{name} completed without a full result')
    print(f'{name} completed in {elapsed:.1f} seconds.', flush=True)
    return elapsed


def share_initial_checkpoint():
    source_dir = MODEL_ROOT / 'hcl' / '200' / '2011-01to2011-12'
    matches = list(source_dir.glob('simple_enc_classifier_*.pth'))
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one HCL initial checkpoint in {source_dir}, found {len(matches)}'
        )
    source = matches[0]
    destination_dir = MODEL_ROOT / 'hcl_ids' / '200' / '2011-01to2011-12'
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists() and sha256(destination) != sha256(source):
        raise RuntimeError(f'IDS initial checkpoint differs: {destination}')
    if not destination.exists():
        shutil.copy2(source, destination)
    if sha256(destination) != sha256(source):
        raise RuntimeError('Initial checkpoint copy verification failed')
    return source, destination


def read_test_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    selected = [row for row in rows if '2012-01' <= row['date'] <= '2012-12']
    if len(selected) != 12:
        raise RuntimeError(f'Expected 12 test months in {path}, found {len(selected)}')
    return selected


def summarize(rows):
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in ['F1', 'FNR', 'FPR', 'ACC', 'PREC']
    }
    return {
        'mean_f1': float(values['F1'].mean()),
        'std_f1': float(values['F1'].std()),
        'min_f1': float(values['F1'].min()),
        'final_f1': float(values['F1'][-1]),
        'aut_f1': float(np.trapz(values['F1'], dx=1)),
        'mean_fnr': float(values['FNR'].mean()),
        'mean_fpr': float(values['FPR'].mean()),
        'mean_accuracy': float(values['ACC'].mean()),
        'mean_precision': float(values['PREC'].mean()),
    }


def write_comparison(runtimes, checkpoints):
    baseline_path = RESULT_ROOT / 'hcl' / 'kronodroid.csv'
    ids_path = RESULT_ROOT / 'hcl_ids' / 'kronodroid.csv'
    baseline_rows = read_test_rows(baseline_path)
    ids_rows = read_test_rows(ids_path)
    baseline = summarize(baseline_rows)
    ids = summarize(ids_rows)

    comparison_path = RESULT_ROOT / 'comparison.tsv'
    with comparison_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['metric', 'hcl', 'hcl_ids', 'difference'])
        for metric in baseline:
            writer.writerow([
                metric,
                f'{baseline[metric]:.8f}',
                f'{ids[metric]:.8f}',
                f'{ids[metric] - baseline[metric]:+.8f}',
            ])

    monthly_path = RESULT_ROOT / 'per_month.tsv'
    with monthly_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow([
            'date', 'hcl_f1', 'hcl_ids_f1', 'delta_f1',
            'hcl_fnr', 'hcl_ids_fnr', 'delta_fnr',
            'hcl_fpr', 'hcl_ids_fpr', 'delta_fpr',
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

    f1_deltas = np.asarray([
        float(ids_row['F1']) - float(base_row['F1'])
        for base_row, ids_row in zip(baseline_rows, ids_rows)
    ])
    run_manifest = {
        'runtimes_seconds': runtimes,
        'initial_checkpoint_hcl': str(checkpoints[0]),
        'initial_checkpoint_hcl_ids': str(checkpoints[1]),
        'initial_checkpoint_sha256': sha256(checkpoints[0]),
        'ids_parameters': IDS_PARAMETERS,
        'ids_f1_wins': int((f1_deltas > 0).sum()),
        'ids_f1_losses': int((f1_deltas < 0).sum()),
        'ids_f1_ties': int((f1_deltas == 0).sum()),
    }
    with (RESULT_ROOT / 'run_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(run_manifest, handle, indent=2)
    print(json.dumps({'hcl': baseline, 'hcl_ids': ids, **run_manifest}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-prepare', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('The HCL local pseudo-loss selector requires CUDA')
    if not args.skip_prepare:
        prepare_dataset(RAW_DIR, DATA_DIR)

    runtimes = {'hcl': run_method('hcl', ids=False)}
    checkpoints = share_initial_checkpoint()
    runtimes['hcl_ids'] = run_method('hcl_ids', ids=True)
    write_comparison(runtimes, checkpoints)


if __name__ == '__main__':
    main()
