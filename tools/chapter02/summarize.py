"""Create a compact, shareable result bundle without copying raw CUDA traces."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .trace_summary import summarize


def collect(root):
    suite = json.loads((root / 'suite.json').read_text())
    if suite['status'] != 'completed':
        raise ValueError('suite is not complete')
    records, micro, profiles = [], [], []
    for path in sorted(root.glob('*/benchmarks.jsonl')):
        for line in path.read_text().splitlines():
            records.append({'run': path.parent.name, **json.loads(line)})
    for path in sorted(root.glob('*/attention.jsonl')):
        for line in path.read_text().splitlines():
            micro.append({'run': path.parent.name, **json.loads(line)})
    if any(r['status'] == 'failed' for r in records + micro):
        raise ValueError('suite contains failed performance cases')
    for path in sorted(root.glob('*/profile.json')):
        metadata = json.loads(path.read_text())
        profiles.append({'run': path.parent.name, 'metadata': metadata,
                         'trace_summary': summarize(json.loads((path.parent / 'trace.json').read_text()))})
    return {'suite': suite, 'environment': json.loads((root / 'environment.json').read_text()),
            'commands': json.loads((root / 'commands.json').read_text()),
            'correctness': json.loads((root / 'correctness/checks.json').read_text()),
            'benchmarks': records, 'attention_microbenchmarks': micro, 'profiles': profiles,
            'counts': {'model_cases': dict(Counter(r['status'] for r in records)),
                       'attention_cases': dict(Counter(r['status'] for r in micro))},
            'interpretation': 'Synthetic-input execution measurements, not text-quality comparisons. '
                              'Different variants are not parameter/quality matched. '
                              'Profile timing is not benchmark timing; CPU and GPU annotations overlap.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--supplement', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--source-archive', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    result = collect(args.root)
    if args.supplement:
        result['supplementary'] = {}
        for item in args.supplement:
            label, path = item.split('=', 1)
            if label in result['supplementary']:
                raise ValueError(f'duplicate supplementary label: {label}')
            result['supplementary'][label] = json.loads(Path(path).read_text())
    if args.source_archive:
        result['source_snapshot'] = {'filename': args.source_archive.name,
            'sha256': hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
            'note': 'Archived measurement sources; compare module hashes with the recorded environment. '
                    'Later diagnostics are separately fingerprinted.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
