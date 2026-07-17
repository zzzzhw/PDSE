"""Plot monthly APIGraph performance for HCL and HCL+IDS."""

import argparse
import base64
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_IDS = Path('experiments/results/hcl_ids_full/apigraph.csv')
DEFAULT_BASELINE = Path(
    'experiments/results/pseudo/200/05.19-10.36.58/'
    'gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_wlr0.00015_'
    'we100_test_2013-01_2018-12_cnt200.csv'
)
DEFAULT_OUTPUT = Path('experiments/figures/hcl_ids_apigraph')


def read_metrics(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    return [row for row in rows if row['date'] >= '2013-01']


def build_comparison(ids_path, baseline_path):
    ids_rows = read_metrics(ids_path)
    baseline_rows = {row['date']: row for row in read_metrics(baseline_path)}
    dates = [row['date'] for row in ids_rows]
    if any(date not in baseline_rows for date in dates):
        raise ValueError('The IDS and baseline results do not cover the same months.')
    return dates, ids_rows, [baseline_rows[date] for date in dates]


def plot_metric(dates, ids_rows, baseline_rows, metric, label, output_path):
    baseline = [float(row[metric]) for row in baseline_rows]
    ids = [float(row[metric]) for row in ids_rows]
    x = list(range(len(dates)))

    fig, axis = plt.subplots(figsize=(14, 6.4), constrained_layout=True)
    axis.plot(x, baseline, color='#6B7280', linewidth=2.1, label='HCL baseline')
    axis.plot(x, ids, color='#007C7A', linewidth=2.4, label='HCL + IDS')
    axis.fill_between(x, baseline, ids, where=[a >= b for a, b in zip(ids, baseline)],
                      color='#007C7A', alpha=0.11, interpolate=True)
    axis.fill_between(x, baseline, ids, where=[a < b for a, b in zip(ids, baseline)],
                      color='#B45309', alpha=0.10, interpolate=True)

    year_ticks = [index for index, date in enumerate(dates) if date.endswith('-01')]
    axis.set_xticks(year_ticks, [dates[index][:4] for index in year_ticks])
    axis.set_xlim(0, len(dates) - 1)
    axis.set_ylim(0.70 if metric == 'F1' else 0.0, 1.0 if metric == 'F1' else 0.36)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel('Test month')
    axis.set_ylabel(label)
    axis.set_title(f'APIGraph monthly {label}: HCL + IDS vs HCL baseline', pad=12)
    axis.grid(axis='y', color='#D1D5DB', linewidth=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='lower right', frameon=False)

    baseline_mean = sum(baseline) / len(baseline)
    ids_mean = sum(ids) / len(ids)
    delta = ids_mean - baseline_mean
    axis.text(
        0.01,
        0.02,
        f'Mean: HCL {baseline_mean:.4f} | HCL + IDS {ids_mean:.4f} | Delta {delta:+.4f}',
        transform=axis.transAxes,
        fontsize=10,
        color='#374151',
        va='bottom',
    )
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_inline_fragment(image_path, visual_dir, title):
    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    alt_text = title.replace('-', ' ')
    fragment = (
        f'<div id="{title}">\n'
        f'  <img src="data:image/png;base64,{encoded}" '
        f'alt="{alt_text}" style="max-width:100%;height:auto;display:block">\n'
        '</div>\n'
    )
    (visual_dir / f'{title}.html').write_text(fragment, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', type=Path, default=DEFAULT_IDS)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--visual-dir', type=Path)
    args = parser.parse_args()

    dates, ids_rows, baseline_rows = build_comparison(args.ids, args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        'f1': args.output_dir / 'hcl_ids_vs_hcl_f1.png',
        'fnr': args.output_dir / 'hcl_ids_vs_hcl_fnr.png',
    }
    plot_metric(dates, ids_rows, baseline_rows, 'F1', 'F1', figures['f1'])
    plot_metric(dates, ids_rows, baseline_rows, 'FNR', 'False negative rate', figures['fnr'])

    if args.visual_dir:
        args.visual_dir.mkdir(parents=True, exist_ok=True)
        write_inline_fragment(figures['f1'], args.visual_dir, 'hcl-ids-f1-comparison')
        write_inline_fragment(figures['fnr'], args.visual_dir, 'hcl-ids-fnr-comparison')

    for figure in figures.values():
        print(figure)


if __name__ == '__main__':
    main()
