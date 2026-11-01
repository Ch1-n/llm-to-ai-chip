"""CLI: environment probe, correctness, controlled benchmarks and profiling."""

import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import traceback

import torch

from .attention import BACKENDS, BackendUnavailable
from .checks import kernel_checks, model_checks, notebook_check
from .config import preset
from .model import VARIANTS, TopKMoE
from .runtime import (Workload, environment, fresh_directory, initialize, measure,
                      synchronize, write_json)


DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


def common(parser):
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=DTYPES, default='float32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)


def model_arguments(parser):
    parser.add_argument('--preset', choices=('tiny', 'teaching'), default='tiny')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--all-logits', action='store_true',
                        help='compute all prompt logits; default only last position in inference')
    parser.add_argument('--aux-weight', type=float, default=.01)


def load_config(args):
    config = preset(args.preset)
    return replace(config, max_context=max(config.max_context, args.context))


def report_moe_load(workload):
    if workload.phase == 'train':
        return None
    modules = [m for m in workload.model.modules() if isinstance(m, TopKMoE)]
    if not modules:
        return None
    for module in modules:
        module.collect_stats = True
    try:
        output = workload()
        return [counts.cpu().tolist() for counts in output.expert_counts]
    finally:
        for module in modules:
            module.collect_stats = False


def benchmark(args, device, dtype, directory):
    config = load_config(args)
    failures, unavailable = 0, 0
    for number, (variant, backend, phase, cache_mode) in enumerate(itertools.product(
            args.variants, args.backends, args.phases, args.cache_modes)):
        # Training never uses a KV cache, so run it once rather than duplicate
        # identical jobs under cat/static labels.
        if phase == 'train' and cache_mode != args.cache_modes[0]:
            continue
        row = {'case': number, 'variant': variant, 'backend': backend,
               'phase': phase, 'cache_mode': cache_mode if phase != 'train' else None}
        workload = None
        try:
            workload = Workload(config, variant, backend, phase, cache_mode,
                                args.batch, args.context, device, dtype,
                                not args.all_logits, args.aux_weight)
            row.update(workload.metadata)
            row['performance'] = measure(workload, device, args.warmups, args.repeats)
            output = workload()
            row['observations'] = workload.observations(output)
            del output
            row['expert_assignments_untimed'] = report_moe_load(workload)
            row['status'] = 'completed'
            tokens = args.batch * (1 if phase == 'decode' else args.context)
            row['performance']['tokens_per_second'] = tokens * 1000 / row['performance']['wall_median_ms']
            row['validation'] = 'finite outputs/gradients only; run check separately before interpreting speed'
        except BackendUnavailable as error:
            row.update(status='unavailable', reason=str(error))
            unavailable += 1
        except Exception as error:
            row.update(status='failed', reason=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exc())
            failures += 1
        finally:
            if workload is not None:
                workload.release()
        with (directory / 'benchmarks.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps({key: row[key] for key in ('case', 'variant', 'backend', 'phase', 'status')},
                         ensure_ascii=False), flush=True)
    write_json(directory / 'summary.json', {'failed': failures, 'unavailable': unavailable,
                                          'gpu_results': device.type == 'cuda',
                                          'status': 'failed' if failures else 'completed_with_unavailable_cases'
                                          if unavailable else 'completed'})
    return 1 if failures or (args.strict and unavailable) else 0


def profile(args, device, dtype, directory):
    config = load_config(args)
    workload = Workload(config, args.variant, args.backend, args.phase, args.cache_mode,
                        args.batch, args.context, device, dtype, not args.all_logits, args.aux_weight)
    try:
        for _ in range(args.warmups):
            output = workload()
            del output
        synchronize(device)
        workload.model.set_profile(True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True, with_stack=args.stacks) as profiler:
            for _ in range(args.steps):
                output = workload()
                del output
                profiler.step()
            synchronize(device)
        profiler.export_chrome_trace(str(directory / 'trace.json'))
        events = profiler.key_averages(group_by_input_shape=True)
        rows = []
        for event in events:
            rows.append({'name': event.key, 'count': event.count, 'input_shapes': event.input_shapes,
                         'device_type': str(event.device_type),
                         'is_user_annotation': getattr(event, 'is_user_annotation', False),
                         'cpu_total_us': event.cpu_time_total, 'cpu_self_us': event.self_cpu_time_total,
                         'device_total_us': getattr(event, 'device_time_total', 0),
                         'device_self_us': getattr(event, 'self_device_time_total', 0),
                         'cpu_memory_bytes': event.cpu_memory_usage,
                         'device_memory_bytes': getattr(event, 'device_memory_usage', 0)})
        (directory / 'operators.txt').write_text(events.table(
            sort_by='self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total', row_limit=100))
        write_json(directory / 'operators.json', rows)
        write_json(directory / 'profile.json', {
            **workload.metadata, 'activities': [str(x) for x in activities],
            'steps': args.steps, 'status': 'completed',
            'note': 'Profiling adds overhead; do not use this trace to report benchmark speed. '
                    'Nested CPU/device totals overlap and cannot be added as disjoint percentages.'})
    finally:
        workload.release()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--output', type=Path)
    check = sub.add_parser('check')
    common(check)
    check.add_argument('--backends', nargs='+', choices=BACKENDS, default=['math', 'head-loop', 'sdpa-math'])
    check.add_argument('--notebook', type=Path)
    check.add_argument('--strict', action='store_true')
    bench = sub.add_parser('bench')
    common(bench)
    model_arguments(bench)
    bench.add_argument('--variants', nargs='+', choices=VARIANTS, default=['dense'])
    bench.add_argument('--backends', nargs='+', choices=BACKENDS, default=['math'])
    bench.add_argument('--phases', nargs='+', choices=('prefill', 'decode', 'train'), default=['prefill', 'decode'])
    bench.add_argument('--cache-modes', nargs='+', choices=('cat', 'static'), default=['cat', 'static'])
    bench.add_argument('--warmups', type=int, default=5)
    bench.add_argument('--repeats', type=int, default=20)
    bench.add_argument('--strict', action='store_true')
    prof = sub.add_parser('profile')
    common(prof)
    model_arguments(prof)
    prof.add_argument('--variant', choices=VARIANTS, default='dense')
    prof.add_argument('--backend', choices=BACKENDS, default='math')
    prof.add_argument('--phase', choices=('prefill', 'decode', 'train'), default='decode')
    prof.add_argument('--cache-mode', choices=('cat', 'static'), default='static')
    prof.add_argument('--steps', type=int, choices=range(1, 6), default=3)
    prof.add_argument('--warmups', type=int, default=3)
    prof.add_argument('--stacks', action='store_true')
    micro = sub.add_parser('attention')
    common(micro)
    micro.add_argument('--backends', nargs='+', choices=BACKENDS, default=['math'])
    micro.add_argument('--batch', type=int, default=1)
    micro.add_argument('--heads', type=int, default=12)
    micro.add_argument('--kv-heads', type=int, default=12)
    micro.add_argument('--kv-length', type=int, default=512)
    micro.add_argument('--q-lengths', nargs='+', type=int, default=[1, 512])
    micro.add_argument('--head-dim', type=int, default=64)
    micro.add_argument('--value-dim', type=int)
    micro.add_argument('--backward', action='store_true')
    micro.add_argument('--warmups', type=int, default=5)
    micro.add_argument('--repeats', type=int, default=20)
    micro.add_argument('--strict', action='store_true')
    train = sub.add_parser('train')
    common(train)
    model_arguments(train)
    train.add_argument('--variant', choices=VARIANTS, default='dense')
    train.add_argument('--backend', choices=BACKENDS, default='math')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--steps', type=int, default=100)
    train.add_argument('--eval-every', type=int, default=25)
    train.add_argument('--eval-blocks', type=int, default=32)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--save-checkpoint', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        result = environment()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            directory = fresh_directory(args.output)
            write_json(directory / 'environment.json', result)
        return 0
    if args.threads < 1:
        parser.error('--threads must be positive')
    if getattr(args, 'aux_weight', 0) < 0:
        parser.error('--aux-weight must be nonnegative')
    if getattr(args, 'warmups', 1) < 1:
        parser.error('--warmups must be positive')
    dtype = DTYPES[args.dtype]
    device = initialize(args.device, dtype, threads=args.threads)
    directory = fresh_directory(args.output)
    write_json(directory / 'environment.json', environment())
    write_json(directory / 'arguments.json', {k: str(v) if isinstance(v, Path) else v
                                            for k, v in vars(args).items()})
    if args.command == 'check':
        records = kernel_checks(device, dtype, args.backends)
        try:
            records.extend(model_checks(device))
            if args.notebook:
                records.append(notebook_check(args.notebook, device))
        except Exception as error:
            records.append({'kind': 'model_checks', 'status': 'failed',
                            'reason': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
        failed = any(r['status'] == 'failed' for r in records)
        unavailable = sum(r['status'] == 'unavailable' for r in records)
        write_json(directory / 'checks.json', {'records': records, 'failed': failed,
                                              'unavailable': unavailable, 'device': str(device)})
        print(f'checks: {len(records)}, failed: {failed}, unavailable: {unavailable}')
        return int(failed or (args.strict and unavailable))
    if args.command == 'bench':
        return benchmark(args, device, dtype, directory)
    if args.command == 'profile':
        return profile(args, device, dtype, directory)
    if args.command == 'attention':
        from .micro import benchmark_attention
        return benchmark_attention(args, device, dtype, directory)
    from .training import train_text
    return train_text(args, device, dtype, directory)


if __name__ == '__main__':
    sys.exit(main())
