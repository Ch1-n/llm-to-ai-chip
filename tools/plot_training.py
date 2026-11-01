#!/usr/bin/env python3
"""Plot observed training/validation losses without fabricating intermediate values."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='Seq-Monkey subset training')
    parser.add_argument('--subtitle', default='119.7M parameters | RTX 4090 D | BF16 AMP | B=12, S=512')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    train = [row for row in rows if row['phase'] == 'train']
    validation = [row for row in rows if row['phase'] == 'val']
    assert train and validation
    x = [row['step'] for row in train]
    y = [row['loss'] for row in train]
    window = 10
    smoothed = [sum(y[max(0, i - window + 1):i + 1]) / min(window, i + 1)
                for i in range(len(y))]
    fig, axis = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    axis.plot(x, y, color='#8B98A4', alpha=0.5, linewidth=0.9, label='Train loss per step')
    axis.plot(x, smoothed, color='#34699A', linewidth=2, label='Train trailing mean (10 steps)')
    axis.plot([row['step'] for row in validation], [row['loss'] for row in validation],
              color='#B34B3E', marker='o', markersize=4, linewidth=1.7,
              label='Held-out validation loss')
    axis.set_title(args.title + '\n' + args.subtitle)
    axis.set_xlabel('Optimizer step')
    axis.set_ylabel('Cross-entropy loss (nats/token)')
    axis.grid(alpha=0.18)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper right')
    axis.set_xlim(left=0, right=max(x))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
