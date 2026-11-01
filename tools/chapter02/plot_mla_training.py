"""Plot complete real-text training runs for the matched MLA reference paths."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    reference = None
    gpu_names = []
    variants = set()
    for index, directory in enumerate(args.runs):
        cfg = json.loads((directory / 'training.json').read_text())
        variants.add(cfg['variant'])
        selected = {key: cfg[key] for key in ('config', 'steps', 'batch', 'context', 'eval_every',
            'lr_peak', 'warmup_steps', 'seed', 'initial_parameter_sha256',
            'first_epoch_training_order_sha256', 'validation_blocks', 'actual_token_bin_sha256',
            'optimizer', 'precision')}
        env = json.loads((directory / 'environment.json').read_text())
        selected['pytorch'] = env['pytorch']
        selected['sources'] = env['source_sha256']
        gpu_names.append(env['gpus'][0]['name'])
        if reference is None:
            reference = selected
        elif selected != reference or gpu_names[-1] != gpu_names[0]:
            raise ValueError('model, initialization, data order, environment or schedule mismatch')
        summary = json.loads((directory / 'summary.json').read_text())
        rows = [json.loads(row) for row in (directory / 'metrics.jsonl').read_text().splitlines()]
        train = [r for r in rows if r['phase'] == 'train']
        val = [r for r in rows if r['phase'] == 'val']
        if summary['status'] != 'completed' or [r['step'] for r in train] != list(range(1, cfg['steps'] + 1)):
            raise ValueError('incomplete training run')
        if [r['step'] for r in val] != [0] + [step for step in range(1, cfg['steps'] + 1)
                                            if step % cfg['eval_every'] == 0 or step == cfg['steps']]:
            raise ValueError('incomplete validation log')
        loss = [r['loss'] for r in train]
        smoothed = [sum(loss[max(0, i - 9):i + 1]) / min(i + 1, 10) for i in range(len(loss))]
        color = ('#277C9B', '#BD5946')[index]
        style = ('-', '--')[index]
        label = cfg['variant'].replace('mla-reference-', 'MLA ')
        axes[0].plot([r['step'] for r in train], loss, color=color, alpha=.16, linewidth=.6)
        axes[0].plot([r['step'] for r in train], smoothed, color=color, linestyle=style, label=label)
        axes[1].plot([r['step'] for r in val], [r['loss'] for r in val], color=color,
                     linestyle=style, marker='o', markersize=4,
                     markerfacecolor=color if index == 0 else 'white', label=label)
    if variants != {'mla-reference-naive', 'mla-reference-absorb'}:
        raise ValueError('provide exactly one naive run and one absorb run')
    axes[0].set_title('Train loss: raw steps (faint) and trailing 10-step mean')
    axes[1].set_title(f'Held-out validation: every {cfg["eval_every"]} optimizer steps')
    for axis in axes:
        axis.set_ylabel('Cross-entropy (nats/token)')
        axis.grid(alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend()
        axis.set_xlim(0, cfg['steps'])
    axes[1].set_xlabel('Optimizer step')
    fig.suptitle('DeepSeek-aligned MLA | Seq-Monkey subset\n'
                 f'{cfg["parameters"] / 1e6:.1f}M | {gpu_names[0]} | BF16 AMP | '
                 f'B={cfg["batch"]}, S={cfg["context"]} | dropout={cfg["dropout"]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
