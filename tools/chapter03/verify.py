"""Verify downloaded chapter-three training inputs, records, sources and weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    hashes = {s: sha(args.data / f'{s}.bin') for s in ('train', 'val')}
    if hashes != manifest['token_bin_sha256']:
        raise ValueError('local token file hash mismatch')
    if set(manifest['documents']['train']) & set(manifest['documents']['val']):
        raise ValueError('train/validation documents overlap')
    report = {'status': 'passed', 'source_archive_sha256': sha(args.source_archive),
              'token_bin_sha256': hashes, 'runs': {}}
    common, variants = None, set()
    with tarfile.open(args.source_archive) as archive:
        for directory in args.runs:
            meta = json.loads((directory / 'training.json').read_text())
            env = json.loads((directory / 'environment.json').read_text())
            compare = {k: meta[k] for k in ('seed', 'steps', 'batch', 'context', 'eval_every',
                'actual_token_bin_sha256', 'validation_blocks', 'first_epoch_training_order_sha256',
                'optimizer', 'lr_peak', 'precision')}
            compare.update(torch=env['pytorch'], sources=env['source_sha256'])
            if common is None:
                common = compare
            elif compare != common:
                raise ValueError('data/schedule/software mismatch between training runs')
            if meta['actual_token_bin_sha256'] != hashes:
                raise ValueError('training used different data')
            for prefix, group in (('tools/chapter03', env['source_sha256']),
                                  ('tools/chapter02', env['chapter02_source_sha256']),
                                  ('references/deepseek-v4', env['reference_sha256'])):
                for name, expected in group.items():
                    stream = archive.extractfile(prefix + '/' + name)
                    if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                        raise ValueError('source mismatch: ' + prefix + '/' + name)
            rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
            train = [r for r in rows if r['phase'] == 'train']
            val = [r for r in rows if r['phase'] == 'val']
            if [r['step'] for r in train] != list(range(1, meta['steps'] + 1)):
                raise ValueError('missing training records')
            expected = [0] + [s for s in range(1, meta['steps'] + 1) if s % meta['eval_every'] == 0 or s == meta['steps']]
            if [r['step'] for r in val] != expected:
                raise ValueError('missing validation records')
            if not all(math.isfinite(r['loss']) for r in rows):
                raise ValueError('nonfinite loss')
            if not all(math.isfinite(r['gradient_norm_before_clip']) for r in train):
                raise ValueError('nonfinite gradient')
            if meta['variant'] == 'hybrid' and not all(r['indexer_gradient_norm_sum'] > 0 for r in train):
                raise ValueError('indexer did not receive a gradient')
            summary = json.loads((directory / 'summary.json').read_text())
            if summary['status'] != 'completed' or summary['validation_final'] != val[-1]['loss']:
                raise ValueError('summary mismatch')
            checkpoint = torch.load(directory / 'best-model.pt', map_location='cpu', weights_only=True)
            checkpoint_config = json.loads(json.dumps(checkpoint['config']))
            if (checkpoint['variant'] != meta['variant'] or checkpoint_config != meta['config']
                    or checkpoint['validation_loss'] != min(r['loss'] for r in val)
                    or sum(p.numel() for p in checkpoint['model'].values()) != meta['parameters']):
                raise ValueError('checkpoint metadata mismatch')
            if not all(torch.isfinite(p).all().item() for p in checkpoint['model'].values()):
                raise ValueError('nonfinite checkpoint tensor')
            variants.add(meta['variant'])
            report['runs'][meta['variant']] = {'train_records': len(train), 'val_records': len(val),
                'parameters': meta['parameters'], 'validation_final': summary['validation_final'],
                'checkpoint_step': checkpoint['step'], 'checkpoint_sha256': sha(directory / 'best-model.pt'),
                'checkpoint_tensors_finite': True,
                'indexer_gradient_present': meta['variant'] != 'hybrid' or all(r['indexer_gradient_norm_sum'] > 0 for r in train)}
            del checkpoint
    if variants != {'swa', 'hybrid'}:
        raise ValueError('provide one SWA and one Hybrid run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
