"""Run reproducible staged hyperparameter tuning for HCL+IDS."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


BASELINE_RESULT = Path(
    'experiments/results/pseudo/200/05.19-10.36.58/'
    'gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_wlr0.00015_'
    'we100_test_2013-01_2018-12_cnt200.csv'
)
INITIAL_MODEL_DIR = Path('models/pseudo/200/2012-01to2012-12')
RESULT_ROOT = Path('experiments/results/hcl_ids_tuning')
MODEL_ROOT = Path('models/experiments/hcl_ids_tuning')


DEFAULT_IDS = {
    'ids_lambda': 1e-3,
    'ids_gamma': 1.0,
    'ids_ema_decay': 0.6,
    'ids_proxy_lr': 1.0,
    'ids_robust_weight': 1.0,
    'ids_batch_size': 192,
    'ids_warmup': 5,
    'ids_grad_clip': 1.0,
}


SCREEN_CONFIGS = {
    'gamma_1e4': {'ids_gamma': 1e4},
    'gamma_1e5': {'ids_gamma': 1e5},
    'lambda_1e4': {'ids_lambda': 1e-4},
    'lambda_2e4': {'ids_lambda': 2e-4},
    'lambda_3e4': {'ids_lambda': 3e-4},
    'lambda_5e4': {'ids_lambda': 5e-4},
    'lambda_7e4': {'ids_lambda': 7e-4},
    'lambda_3e3': {'ids_lambda': 3e-3},
    'proxy_lr_0p1': {'ids_proxy_lr': 0.1},
    'ema_0p9': {'ids_ema_decay': 0.9},
    'warmup_20': {'ids_warmup': 20},
}


def read_results(path, start, end):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    return [row for row in rows if start <= row['date'] <= end]


def initial_checkpoint():
    matches = list(INITIAL_MODEL_DIR.glob('simple_enc_classifier_*.pth'))
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one initial HCL checkpoint in {INITIAL_MODEL_DIR}, found {len(matches)}.'
        )
    return matches[0]


def prepare_model_dir(run_name):
    destination = MODEL_ROOT / run_name / '200' / '2012-01to2012-12'
    destination.mkdir(parents=True, exist_ok=True)
    source = initial_checkpoint()
    target = destination / source.name
    if not target.exists():
        shutil.copy2(source, target)
    return destination.parent


def result_complete(path, test_end):
    if not path.exists():
        return False
    rows = read_results(path, '2013-01', test_end)
    return bool(rows) and rows[-1]['date'] == test_end


def build_command(params, model_dir, result_path, log_path, test_end):
    command = [
        sys.executable,
        '-u',
        'base.py',
        '--method', 'pseudo',
        '--data', 'gen_apigraph_drebin',
        '--benign_zero',
        '--mdate', '20240312',
        '--train_start', '2012-01',
        '--train_end', '2012-12',
        '--test_start', '2013-01',
        '--test_end', test_end,
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
        '--learning_rate', '0.003',
        '--lr_decay_rate', '0.95',
        '--lr_decay_epochs', '10,500,10',
        '--epochs', '250',
        '--encoder_retrain',
        '--al_optimizer', 'adam',
        '--warm_learning_rate', '0.00015',
        '--al_epochs', '100',
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
        '--ids',
        '--ids-lambda', str(params['ids_lambda']),
        '--ids-gamma', str(params['ids_gamma']),
        '--ids-ema-decay', str(params['ids_ema_decay']),
        '--ids-proxy-lr', str(params['ids_proxy_lr']),
        '--ids-robust-weight', str(params['ids_robust_weight']),
        '--ids-batch-size', str(params['ids_batch_size']),
        '--ids-warmup', str(params['ids_warmup']),
        '--ids-grad-clip', str(params['ids_grad_clip']),
        '--result', str(result_path),
        '--log_path', str(log_path),
    ]
    return command


def evaluate(run_name, params, result_path, test_end):
    candidate = read_results(result_path, '2013-01', test_end)
    baseline = {
        row['date']: row for row in read_results(BASELINE_RESULT, '2013-01', test_end)
    }
    paired = [(row, baseline[row['date']]) for row in candidate]
    deltas = [float(ids['F1']) - float(base['F1']) for ids, base in paired]
    return {
        'run': run_name,
        'test_end': test_end,
        **params,
        'months': len(paired),
        'mean_f1': sum(float(ids['F1']) for ids, _ in paired) / len(paired),
        'mean_delta_f1': sum(deltas) / len(deltas),
        'mean_fnr': sum(float(ids['FNR']) for ids, _ in paired) / len(paired),
        'min_f1': min(float(ids['F1']) for ids, _ in paired),
        'worst_delta_f1': min(deltas),
        'wins': sum(delta > 0 for delta in deltas),
    }


def write_summary(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['mean_delta_f1'], reverse=True))


def run_config(run_name, overrides, test_end):
    params = {**DEFAULT_IDS, **overrides}
    run_dir = RESULT_ROOT / f'to_{test_end}' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'apigraph.csv'
    log_path = run_dir / 'apigraph.log'
    console_path = run_dir / 'console.log'
    model_dir = prepare_model_dir(f'to_{test_end}/{run_name}')

    if not result_complete(result_path, test_end):
        if result_path.exists():
            raise RuntimeError(
                f'Partial result exists at {result_path}; use a new run name after inspecting it.'
            )
        command = build_command(params, model_dir, result_path, log_path, test_end)
        print(f'Running {run_name}: {params}', flush=True)
        with console_path.open('w', encoding='utf-8') as console:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=console,
                stderr=subprocess.STDOUT,
                check=True,
            )
    else:
        print(f'Skipping completed run {run_name}.', flush=True)
    return evaluate(run_name, params, result_path, test_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-end', default='2013-07')
    parser.add_argument('--only', action='append', choices=sorted(SCREEN_CONFIGS))
    args = parser.parse_args()

    names = args.only or list(SCREEN_CONFIGS)
    summaries = []
    for name in names:
        summary = run_config(name, SCREEN_CONFIGS[name], args.test_end)
        summaries.append(summary)
        print(
            f"{name}: mean_f1={summary['mean_f1']:.6f}, "
            f"delta={summary['mean_delta_f1']:+.6f}, "
            f"fnr={summary['mean_fnr']:.6f}, wins={summary['wins']}",
            flush=True,
        )
        write_summary(
            summaries,
            RESULT_ROOT / f'to_{args.test_end}' / 'summary.tsv',
        )


if __name__ == '__main__':
    main()
