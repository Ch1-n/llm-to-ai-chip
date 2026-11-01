"""Explicit, bounded CLI for the third article's hybrid-attention experiments."""

import argparse
import gc
import itertools
import json
from pathlib import Path
import sys
import traceback

import torch

from tools.chapter02.runtime import fresh_directory, initialize, write_json
from .config import VARIANTS
from .runtime import Workload, environment, measure


def common(parser):
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    parser.add_argument('--preset', choices=('tiny', 'teaching'), default='teaching')
    parser.add_argument('--output', type=Path, required=True)


def benchmark(args, device, dtype, directory):
    failed, skipped, count = 0, 0, 0
    for variant, phase, batch, context in itertools.product(args.variants, args.phases, args.batches, args.contexts):
        row = {'variant': variant, 'phase': phase, 'batch': batch, 'context': context}
        workload = None
        try:
            if batch > 1 and context > 2048:
                row.update(status='skipped', reason='bounded suite excludes B>1 at context>2048')
                skipped += 1
            else:
                workload = Workload(variant, phase, batch, context, device, dtype, args.preset)
                row.update(workload.metadata)
                row['performance'] = measure(workload, device, args.warmups, args.repeats)
                row['status'] = 'completed'
                count += 1
        except Exception as error:
            row.update(status='failed', reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            failed += 1
        finally:
            if workload is not None:
                del workload
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        with (directory / 'benchmarks.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k in ('variant', 'phase', 'batch', 'context', 'status', 'reason')}), flush=True)
    write_json(directory / 'summary.json', {'completed': count, 'failed': failed, 'skipped': skipped,
                                           'status': 'failed' if failed else 'completed'})
    return int(failed > 0)


def profile(args, device, dtype, directory):
    workload = Workload(args.variant, 'decode', args.batch, args.context, device, dtype, args.preset)
    for _ in range(3):
        workload.prepare()
        out = workload()
        del out
    workload.model.set_profile(True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(args.steps):
            workload.prepare()
            result = workload()
            del result
            profiler.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(directory / 'trace.json'))
    events = profiler.key_averages(group_by_input_shape=True)
    (directory / 'operators.txt').write_text(events.table(sort_by='self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total', row_limit=80))
    write_json(directory / 'profile.json', {**workload.metadata, 'steps': args.steps,
        'status': 'completed', 'scope': 'Profiler overhead excluded from benchmark table; no hardware utilization counters'})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    check = sub.add_parser('check')
    common(check)
    bench = sub.add_parser('bench')
    common(bench)
    bench.add_argument('--variants', nargs='+', choices=(*VARIANTS, 'mha', 'mla'), default=['shared-dense', 'swa', 'hybrid'])
    bench.add_argument('--phases', nargs='+', choices=('prefill', 'decode'), default=['prefill', 'decode'])
    bench.add_argument('--batches', nargs='+', type=int, default=[1])
    bench.add_argument('--contexts', nargs='+', type=int, default=[512, 2048])
    bench.add_argument('--warmups', type=int, default=3)
    bench.add_argument('--repeats', type=int, default=10)
    prof = sub.add_parser('profile')
    common(prof)
    prof.add_argument('--variant', choices=VARIANTS, default='hybrid')
    prof.add_argument('--batch', type=int, default=1)
    prof.add_argument('--context', type=int, default=512)
    prof.add_argument('--steps', type=int, choices=range(1, 6), default=3)
    train = sub.add_parser('train')
    common(train)
    train.add_argument('--variant', choices=('swa', 'hybrid'), default='hybrid')
    train.add_argument('--batch', type=int, default=12)
    train.add_argument('--context', type=int, default=512)
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--steps', type=int, default=300)
    train.add_argument('--eval-every', type=int, default=25)
    train.add_argument('--eval-blocks', type=int, default=192)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--aux-weight', type=float, default=.01)
    args = parser.parse_args(argv)
    for name in ('batch', 'context', 'steps', 'eval_every', 'eval_blocks', 'warmups', 'repeats', 'lr'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if any(value <= 0 for field in ('batches', 'contexts') for value in getattr(args, field, [])):
        parser.error('batch/context lists must be positive')
    if getattr(args, 'aux_weight', 0) < 0:
        parser.error('auxiliary weight must be nonnegative')
    device = initialize(args.device, getattr(torch, args.dtype))
    directory = fresh_directory(args.output)
    write_json(directory / 'environment.json', environment())
    write_json(directory / 'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.operation == 'check':
        from .checks import run
        result = run(device, args.preset == 'teaching')
        write_json(directory / 'checks.json', result)
        return int(result['status'] != 'passed')
    if args.operation == 'bench':
        return benchmark(args, device, getattr(torch, args.dtype), directory)
    if args.operation == 'profile':
        return profile(args, device, getattr(torch, args.dtype), directory)
    from .training import train as train_text
    return train_text(args, device, getattr(torch, args.dtype), directory)


if __name__ == '__main__':
    sys.exit(main())
