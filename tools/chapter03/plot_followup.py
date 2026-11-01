"""Figures from saved follow-up records; no inferred or synthetic results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def save(fig, path):
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--moe', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = read(args.runs / 'supplement-continuous/continuous.json')['records']
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for r in rows:
        points = sorted({s['context'] for s in r['samples']})
        means = [np.mean([s['ms'] for s in r['samples'] if s['context'] == p]) for p in points]
        axes[0].plot(points, means, label=r['variant'], linewidth=1, alpha=.8)
        axes[1].bar(r['variant'], r['mean_ms'], color='#257d86' if r['variant'] != 'hybrid' else '#cb654f')
        axes[1].plot(r['variant'], r['p95_ms'], marker='_', markersize=16, color='#222222')
    for p in (640, 768):
        axes[0].axvline(p, color='#666666', linestyle='--', alpha=.4)
    axes[0].legend(ncol=5, fontsize=9)
    axes[0].set(xlabel='Total tokens after this decode', ylabel='Latency (ms)',
                title='Continuous cache advancement: mean of three repeats at each position')
    axes[1].set(ylabel='Latency (ms)', title='Bars: overall mean; black markers: sample P95')
    fig.suptitle('RTX 4090 | eager BF16 | B=1 | 512-token prefix + 256 steps\n'
                 'Pre-generated random tokens; per-token synchronization; not production-kernel ranking')
    save(fig, args.images / 'continuous-decode.png')

    rows = read(args.runs / 'supplement-ablation/ablation.json')['records']
    rows = [r for r in rows if r['context'] == 8192 and r['phase'] == 'decode']
    labels = [r['variant'] + (f" k={r['index_topk']}" if r['variant'] == 'hybrid' else '') for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].barh(labels, [r['performance']['wall_median_ms'] for r in rows], color='#257d86')
    axes[0].set(xlabel='Decode synchronized wall time (ms)', title='Block-closing position 8192')
    axes[1].barh(labels, [r['performance']['cache']['logical_bytes'] / 2**20 for r in rows], color='#cb654f')
    axes[1].set(xlabel='Persistent logical state (MiB)', xscale='log', title='Includes compressor and Indexer states')
    fig.suptitle('Mechanism ablation | random weights, no quality matching | eager PyTorch')
    save(fig, args.images / 'ablation.png')

    data = read(args.runs / 'supplement-history/history-eval.json')
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for variant, color, marker in [('hybrid', '#cb654f', 'o'), ('swa', '#257d86', 's')]:
        rows = [r for r in data['summary'] if r['variant'] == variant]
        ax.plot([r['context'] for r in rows], [r['mean_nll'] for r in rows],
                marker=marker, label=variant, color=color)
    ax.set(xscale='log', xlabel='Input length, ending at the same target suffix',
           ylabel='Held-out suffix cross-entropy (nats/token)',
           title='Same 8 target suffixes, 128 tokens each; 300-step checkpoints trained at length 512')
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle('Limited history-length probe, not a long-context capability benchmark')
    save(fig, args.images / 'history-loss.png')

    rows = [json.loads(s) for s in (args.moe / 'metrics.jsonl').read_text().splitlines()]
    train, val = [r for r in rows if r['phase'] == 'train'], [r for r in rows if r['phase'] == 'val']
    if [r['step'] for r in train] != list(range(1, 301)) or len(val) != 13:
        raise ValueError('incomplete MoE training records')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    axes[0].plot([r['step'] for r in train], [r['loss'] for r in train], color='#257d86', label='Train CE', alpha=.7)
    axes[0].plot([r['step'] for r in val], [r['loss'] for r in val], color='#cb654f', marker='o', label='Val CE')
    axes[0].set(xlabel='Optimizer step', ylabel='Cross-entropy (nats/token)', title='Real-text short training; auxiliary loss excluded')
    axes[0].legend()
    counts = np.asarray([r['expert_assignments'] for r in train[-25:]]).sum(0)
    share = counts / counts.sum(-1, keepdims=True)
    image = axes[1].imshow(share * 100, cmap='YlGnBu', aspect='auto', vmin=0)
    axes[1].set(xlabel='Expert ID', ylabel='Layer', title='Assignment share over the final 25 steps (%)',
                xticks=range(share.shape[1]), yticks=range(share.shape[0]))
    for y in range(share.shape[0]):
        for x in range(share.shape[1]):
            axes[1].text(x, y, f'{share[y,x]*100:.0f}', ha='center', va='center',
                         color='white' if share[y, x] > .22 else 'black', fontsize=9)
    fig.colorbar(image, ax=axes[1], shrink=.7)
    fig.suptitle('8-expert Top-2 MoE | RTX 4090 | Seq-Monkey subset | 300 steps')
    save(fig, args.images.parent / 'llm-attention/moe-training.png')


if __name__ == '__main__':
    main()
