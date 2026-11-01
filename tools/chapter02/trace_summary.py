"""Summarize real CUDA activities, excluding nested GPU annotation ranges."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def union_duration(intervals):
    total, left, right = 0., None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (0 if left is None else right - left)


def summarize(trace):
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) >= 0]
    kernels = [e for e in events if e.get('cat') == 'kernel']
    copies = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    runtime = [e for e in events if e.get('cat') == 'cuda_runtime']
    devices = sorted({e.get('args', {}).get('device', e.get('pid')) for e in kernels}, key=str)
    result = {'devices': [], 'runtime_calls': len(runtime),
              'note': 'Kernel coverage is the union of traced kernel intervals divided by the '
                      'first-to-last-kernel span, not hardware utilization. Profiling changes '
                      'timing. GPU annotations and nested operator times are not kernel work.'}
    for device in devices:
        selected = [e for e in kernels if e.get('args', {}).get('device', e.get('pid')) == device]
        intervals = [(e['ts'], e['ts'] + e['dur']) for e in selected]
        begin, end = min(x[0] for x in intervals), max(x[1] for x in intervals)
        active = union_duration(intervals)
        by_name = defaultdict(lambda: {'count': 0, 'duration_us': 0.})
        for event in selected:
            entry = by_name[event['name']]
            entry['count'] += 1
            entry['duration_us'] += event['dur']
        result['devices'].append({
            'device': device, 'kernel_count': len(selected), 'kernel_sum_us': sum(e['dur'] for e in selected),
            'kernel_union_us': active, 'first_to_last_kernel_span_us': end - begin,
            'kernel_coverage': active / (end - begin) if end > begin else None,
            'top_kernels': sorted([{'name': key, **value} for key, value in by_name.items()],
                                  key=lambda x: x['duration_us'], reverse=True)[:30]})
    result['memory_activity_count'] = len(copies)
    result['memory_activity_sum_us'] = sum(e['dur'] for e in copies)
    result['cuda_runtime_by_name'] = {}
    for event in runtime:
        entry = result['cuda_runtime_by_name'].setdefault(event['name'], {'count': 0, 'duration_us': 0.})
        entry['count'] += 1
        entry['duration_us'] += event['dur']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    args.output.write_text(json.dumps(summarize(json.loads(args.trace.read_text())), indent=2) + '\n')


if __name__ == '__main__':
    main()
