#!/usr/bin/env python3
"""Compare measured training curves from matched Attention-backend runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    colors = ('#287AA0', '#C05746', '#33866C')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    reference = None
    last_step = 0
    for index, directory in enumerate(args.runs):
        config = json.loads((directory / 'config.json').read_text())
        matched = {key: config[key] for key in (
            'model', 'steps', 'batch', 'sequence', 'seed', 'actual_token_bin_sha256',
            'initial_parameter_sha256', 'first_epoch_training_order_sha256',
            'validation_block_ids', 'precision', 'optimizer', 'schedule', 'eval_every',
            'notebook_sha256', 'training_script_sha256', 'adapter_script_sha256',
            'pytorch', 'gpu')}
        if reference is None:
            reference = matched
        elif matched != reference:
            raise ValueError('comparison requires matched model, data, initialization and schedule')
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        train = [r for r in rows if r['phase'] == 'train']
        validation = [r for r in rows if r['phase'] == 'val']
        if [r['step'] for r in train] != list(range(1, config['steps'] + 1)):
            raise ValueError('training log is incomplete')
        expected_val = [0] + [step for step in range(1, config['steps'] + 1)
                              if step % config['eval_every'] == 0 or step == config['steps']]
        if [r['step'] for r in validation] != expected_val:
            raise ValueError('validation log is incomplete')
        loss = [r['loss'] for r in train]
        smooth = [sum(loss[max(0, i - 9):i + 1]) / min(i + 1, 10)
                  for i in range(len(loss))]
        steps = [r['step'] for r in train]
        label = {'fa2': 'FlashAttention-2', 'sdpa-flash': 'PyTorch SDPA Flash'}.get(
            config['backend'], config['backend'])
        color = colors[index % len(colors)]
        axes[0].plot(steps, loss, color=color, linewidth=.6, alpha=.18)
        style = '--' if index % 2 else '-'
        axes[0].plot(steps, smooth, color=color, linestyle=style, linewidth=1.8, label=label)
        axes[1].plot([r['step'] for r in validation], [r['loss'] for r in validation],
                     color=color, marker='o', markersize=4, linestyle=style,
                     markerfacecolor='white' if index % 2 else color,
                     linewidth=1.7, label=label)
        last_step = max(last_step, steps[-1])
    axes[0].set_title('Train loss: raw steps (faint) and trailing 10-step mean')
    axes[1].set_title('Held-out validation loss: measured every 25 steps')
    axes[1].set_xlabel('Optimizer step')
    for axis in axes:
        axis.set_ylabel('Cross-entropy (nats/token)')
        axis.grid(alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend()
        axis.set_xlim(0, last_step)
    fig.suptitle('Attention backend training | Seq-Monkey subset\n'
                 'Same initial weights and data order | 119.7M | RTX 4090 | BF16 AMP | B=12, S=512')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
