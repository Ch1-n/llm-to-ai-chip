"""Run bounded experiment stages after installing the desired CUDA backends.

No automatic installation, downloads, SSH access or cloud provisioning occurs.
The output root must be empty. Every child command and return code is retained.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .attention import BACKENDS
from .runtime import environment, fresh_directory, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true', help='tiny configurations; not article performance data')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS,
                        default=['head-loop', 'math', 'sdpa-flash', 'fa2', 'fa3', 'fa4'])
    parser.add_argument('--require-backends', nargs='*', choices=BACKENDS, default=['math'])
    args = parser.parse_args(argv)
    if not set(args.require_backends) <= set(args.backends):
        parser.error('required backends must be included in --backends')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')
    root = fresh_directory(args.output).resolve()
    write_json(root / 'environment.json', environment())
    commands = []
    common = ['--device', args.device, '--dtype', 'bfloat16' if args.device == 'cuda' else 'float32']

    def run(label, operation, arguments):
        command = [sys.executable, '-m', 'tools.chapter02.run', operation, *common,
                   '--output', str(root / label), *map(str, arguments)]
        print('RUN', label, flush=True)
        with (root / f'{label}.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'label': label, 'argv': command, 'returncode': process.returncode})
        write_json(root / 'commands.json', commands)
        if process.returncode:
            raise RuntimeError(f'{label} failed; inspect {root / (label + ".log")}')

    try:
        run('correctness', 'check', ['--backends', *args.backends])
        check = json.loads((root / 'correctness/checks.json').read_text())
        ready = {r['backend'] for r in check['records']
                 if 'backend' in r and r['status'] == 'passed'}
        missing = set(args.require_backends) - ready
        if missing:
            raise RuntimeError(f'required backends unavailable: {sorted(missing)}')
        selected = [b for b in args.backends if b in ready]
        if args.quick:
            # tiny MLA absorbed shapes are not a universal FlashAttention input;
            # only the Dense baseline is compared across the fused backends.
            common_model = ['--preset', 'tiny', '--context', 32, '--warmups', 1, '--repeats', 2]
            run('dense', 'bench', [*common_model, '--variants', 'dense', '--backends', *selected,
                                   '--phases', 'prefill', 'decode', 'train'])
            run('structure', 'bench', [*common_model, '--variants', 'mla-expanded', 'mla-compressed',
                                       'mla-absorbed', 'moe', 'mla-moe', '--backends', 'math',
                                       '--phases', 'prefill', 'decode', 'train'])
            run('profile', 'profile', ['--preset', 'tiny', '--context', 32,
                                       '--variant', 'mla-moe', '--steps', 2, '--warmups', 1])
        else:
            common_model = ['--preset', 'teaching', '--warmups', 5, '--repeats', 20]
            for context in (512, 2048):
                for batch in (1, 8):
                    shape = ['--context', context, '--batch', batch]
                    run(f'dense-s{context}-b{batch}', 'bench', [*common_model, *shape,
                        '--variants', 'dense', '--backends', *selected])
                    run(f'structure-s{context}-b{batch}', 'bench', [*common_model, *shape,
                        '--variants', 'mla-expanded', 'mla-compressed', 'mla-absorbed', 'moe', 'mla-moe',
                        '--backends', 'math'])
            run('training-step', 'bench', [*common_model, '--context', 512, '--batch', 4,
                '--variants', 'dense', 'mla-absorbed', 'moe', 'mla-moe', '--backends', 'math',
                '--phases', 'train', '--repeats', 10])
            run('dense-flash-training-step', 'bench', [*common_model, '--context', 512, '--batch', 4,
                '--variants', 'dense', '--backends', *selected, '--phases', 'train', '--repeats', 10])
            run('attention-forward', 'attention', ['--kv-length', 2048, '--q-lengths', 1, 2048,
                '--backends', *selected])
            run('attention-backward', 'attention', ['--kv-length', 2048, '--q-lengths', 2048,
                '--backends', *selected, '--backward'])
            for variant, backend in (('dense', 'head-loop' if 'head-loop' in ready else 'math'),
                                     ('dense', 'math'), ('mla-absorbed', 'math'), ('moe', 'math')):
                run(f'profile-{variant}-{backend}', 'profile', ['--preset', 'teaching',
                    '--context', 512, '--variant', variant, '--backend', backend, '--steps', 3])
        write_json(root / 'suite.json', {'status': 'completed', 'quick': args.quick,
                   'device': args.device, 'validated_backends': sorted(ready),
                   'unavailable_backends': sorted(set(args.backends) - ready),
                   'note': 'Check per-case status. Missing backends are not silent fallbacks. '
                           'Structure variants are not equal-quality models. No real-text training is launched.'})
        return 0
    except Exception as error:
        write_json(root / 'suite.json', {'status': 'failed', 'reason': str(error)})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
