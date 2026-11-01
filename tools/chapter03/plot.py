"""Plot only completed, observed chapter-three results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {'mha': '#606A80', 'mla': '#A46B27', 'shared-dense': '#735C9E',
          'swa': '#277C9B', 'hybrid': '#BC5848'}


def loss(runs, output):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    shared = None
    for directory in runs:
        meta = json.loads((directory / 'training.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError('training is incomplete')
        matched = {k: meta[k] for k in ('steps', 'batch', 'context', 'seed', 'actual_token_bin_sha256',
            'validation_blocks', 'first_epoch_training_order_sha256', 'lr_peak', 'optimizer')}
        if shared is None:
            shared = matched
        elif matched != shared:
            raise ValueError('data or training schedule differs')
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        train = [r for r in rows if r['phase'] == 'train']
        val = [r for r in rows if r['phase'] == 'val']
        if [r['step'] for r in train] != list(range(1, meta['steps'] + 1)):
            raise ValueError('missing training records')
        expected = [0] + [s for s in range(1, meta['steps'] + 1) if s % meta['eval_every'] == 0 or s == meta['steps']]
        if [r['step'] for r in val] != expected:
            raise ValueError('missing validation records')
        values = [r['loss'] for r in train]
        smooth = [sum(values[max(0, i - 9):i + 1]) / min(i + 1, 10) for i in range(len(values))]
        color = COLORS[meta['variant']]
        label = f'{meta["variant"]} ({meta["parameters"] / 1e6:.1f}M)'
        axes[0].plot([r['step'] for r in train], values, color=color, alpha=.15, linewidth=.6)
        axes[0].plot([r['step'] for r in train], smooth, color=color, label=label)
        axes[1].plot([r['step'] for r in val], [r['loss'] for r in val], color=color, marker='o', markersize=4, label=label)
    axes[0].set_title('Train cross-entropy: raw steps and trailing 10-step mean')
    axes[1].set_title('Held-out validation cross-entropy (no auxiliary objective)')
    for ax in axes:
        ax.set_ylabel('Cross-entropy (nats/token)')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend()
        ax.set_xlim(0, meta['steps'])
    axes[1].set_xlabel('Optimizer step')
    fig.suptitle('V4-derived attention | Seq-Monkey subset | RTX 4090 | BF16 AMP\n'
                 'SWA control vs SWA+CSA+HCA; not quality/parameter-matched models')
    fig.savefig(output, dpi=180)
    plt.close(fig)


def capacity(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for variant in COLORS:
        samples = sorted([r for r in rows if r['status'] == 'completed' and r['variant'] == variant
                          and r['phase'] == 'prefill' and r['batch'] == 1], key=lambda r: r['context'])
        if not samples:
            continue
        x = [r['context'] for r in samples]
        y = [r['performance']['cache']['logical_bytes'] / 2**20 for r in samples]
        axes[0].plot(x, y, marker='o', color=COLORS[variant], label=variant)
        axes[1].plot(x, [r['performance']['wall_median_ms'] for r in samples], marker='o', color=COLORS[variant], label=variant)
    axes[0].set_ylabel('Persistent logical cache + compressor/indexer state (MiB)')
    axes[0].set_title('State after prefill, B=1')
    axes[1].set_ylabel('Synchronized wall time (ms)')
    axes[1].set_title('Prefill, B=1, last-position LM head')
    for ax in axes:
        ax.set_xlabel('Context length')
        ax.set_yscale('log')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle('Eager PyTorch references, RTX 4090 | no production-kernel comparison')
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path)
    parser.add_argument('--benchmarks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.runs:
        loss(args.runs, args.output)
    elif args.benchmarks:
        rows = [json.loads(line) for line in args.benchmarks.read_text().splitlines()]
        capacity(rows, args.output)
    else:
        parser.error('provide --runs or --benchmarks')
    print(args.output)


if __name__ == '__main__':
    main()
