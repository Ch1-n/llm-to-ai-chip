"""Render saved analytical scenarios; no measured hardware data is plotted."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/chapter04/20261101'))
    parser.add_argument('--output', type=Path, default=Path('articles/images/llm-npu-architecture'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((args.data / name).read_text())
    if read('metadata.json')['status'] != 'analytical_estimate_not_hardware_measurement':
        raise ValueError('unexpected data scope')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.savefig(args.output / name, dpi=180)
        plt.close(fig)

    variants = ('dense', 'mla', 'moe', 'hybrid')
    resources = ('matrix', 'dram', 'vector', 'shared_sfu', 'selection', 'dispatch')
    colors = ('#606d99', '#cb654f', '#248995', '#65a271', '#b39a46', '#7d7d7d')
    rows = read('baseline.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, phase in zip(axes, ('prefill', 'decode')):
        selected = [next(r for r in rows if r['request']['variant'] == v and r['request']['phase'] == phase
                         and r['request']['batch'] == 1 and r['request']['context'] == 2048) for v in variants]
        x, width = np.arange(4), .125
        for i, (resource, color) in enumerate(zip(resources, colors)):
            values = [r['resource_ms'][resource] for r in selected]
            ax.bar(x + (i - 2.5) * width, values, width, label=resource, color=color)
        ax.set(xticks=x, xticklabels=variants, yscale='log', ylabel='Resource service load (ms)',
               title=f'{phase.capitalize()} | B=1, context=2048')
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('Hypothetical NPU resource loads, not measured latency\n'
                 'Separate resource bars are not additive; traffic and primitive contracts are explicit')
    save(fig, 'resource-budgets.png')

    grid = read('matrix-bandwidth-grid.json')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, phase in zip(axes, ('prefill', 'decode')):
        values = np.array([[next(r['conditional_lower_bound_ms'] for r in grid
                                if r['request']['phase'] == phase and r['hardware']['matrix_tflops'] == compute
                                and r['hardware']['bandwidth_gbs'] == bw)
                            for bw in (25, 100, 400)] for compute in (5, 20, 80)])
        ax.imshow(values, cmap='YlGnBu', aspect='auto')
        for y in range(3):
            for x in range(3):
                ax.text(x, y, f'{values[y,x]:.3f}', ha='center', va='center',
                        color='white' if values[y,x] > (values.max() + values.min()) / 2 else '#17212b')
        ax.set(xticks=range(3), xticklabels=(25, 100, 400), yticks=range(3), yticklabels=(5, 20, 80),
               xlabel='Assumed DRAM bandwidth (GB/s)', ylabel='Assumed matrix throughput (TFLOP/s)', title=phase.capitalize())
    fig.suptitle('Dense | B=1, context=2048 | conditional resource lower bound (ms)\n'
                 'Fixed vector/SFU rates and SRAM; no achievable schedule is constructed')
    save(fig, 'compute-bandwidth.png')

    sram = read('sram-fusion.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for variant, color in (('dense', '#248995'), ('hybrid', '#cb654f')):
        for mode, style in (('fused', '-'), ('unfused', '--')):
            selected = sorted([r for r in sram if r['status'] != 'infeasible_under_sram_contract'
                               and r['request']['variant'] == variant and r['request']['phase'] == 'prefill'
                               and r['request']['batch'] == 1 and r['request']['ffn_mode'] == mode],
                              key=lambda r: r['hardware']['sram_mib'])
            x = [r['hardware']['sram_mib'] for r in selected]
            axes[0].plot(x, [r['dram_payload_bytes'] / 2**20 for r in selected], style, marker='o',
                         color=color, label=f'{variant} {mode}')
            if mode == 'fused':
                axes[1].plot(x, [r['max_ffn_weight_passes'] for r in selected], marker='o', color=color, label=variant)
    for ax in axes:
        ax.set(xscale='log', yscale='log', xlabel='Assumed SRAM (MiB)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.18)
    axes[0].set(ylabel='Accounted DRAM payload (MiB)', title='Fusion bypasses four FFN intermediates')
    axes[1].set(ylabel='Maximum FFN weight passes per invocation', title='Weights reread unless whole FFN fits')
    fig.suptitle('Prefill | B=1, context=2048 | token-major FFN schedule\n'
                 'Not a universal SRAM-to-speed relationship; infeasible points omitted, not extrapolated')
    save(fig, 'sram-fusion.png')

    sweep = read('sensitivity.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, phase in zip(axes, ('prefill', 'decode')):
        for v, color in zip(variants, colors):
            for axis, style in (('silu_speedup', '-'), ('sfu_speedup', '--')):
                selected = sorted([r for r in sweep if r['axis'] == axis and r['request']['variant'] == v
                                   and r['request']['phase'] == phase and r['request']['batch'] == 1], key=lambda r: r['value'])
                base = selected[0]['conditional_lower_bound_ms']
                ax.plot([r['value'] for r in selected], [base / r['conditional_lower_bound_ms'] for r in selected],
                        style, marker='o', color=color, label=f'{v}: {"SiLU only" if axis == "silu_speedup" else "all SFU"}')
        ax.set(xscale='log', xlabel='Assumed primitive cost reduction factor', ylabel='Resource-bound ratio vs factor 1',
               title=f'{phase.capitalize()} | B=1, context=2048')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('SiLU acceleration vs all SFU work | hypothetical, not measured speedup\n'
                 'SiLU also reduces its vector cost; all-SFU scaling leaves non-SiLU vector work unchanged')
    save(fig, 'activation-sensitivity.png')


if __name__ == '__main__':
    main()
