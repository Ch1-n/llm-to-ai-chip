"""Verify downloaded MLA training records, data, source snapshot and weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs=2, type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    token_hashes = {split: digest(args.data / f'{split}.bin') for split in ('train', 'val')}
    if token_hashes != manifest['token_bin_sha256']:
        raise ValueError('token data SHA256 mismatch')
    if set(manifest['documents']['train']) & set(manifest['documents']['val']):
        raise ValueError('document hashes overlap between train and validation')
    report = {'status': 'passed', 'actual_token_bin_sha256': token_hashes,
              'source_archive_sha256': digest(args.source_archive), 'runs': {}}
    reference, variants = None, set()
    with tarfile.open(args.source_archive) as archive:
        for directory in args.runs:
            metadata = json.loads((directory / 'training.json').read_text())
            env = json.loads((directory / 'environment.json').read_text())
            selected = {key: metadata[key] for key in ('config', 'steps', 'batch', 'context',
                'eval_every', 'lr_peak', 'warmup_steps', 'seed', 'initial_parameter_sha256',
                'first_epoch_training_order_sha256', 'validation_blocks', 'optimizer', 'precision')}
            selected.update(pytorch=env['pytorch'], source_sha256=env['source_sha256'])
            if reference is None:
                reference = selected
            elif selected != reference:
                raise ValueError('the two training runs are not matched')
            if metadata['actual_token_bin_sha256'] != token_hashes:
                raise ValueError('training used different token data')
            for name, expected in env['source_sha256'].items():
                stream = archive.extractfile(f'tools/chapter02/{name}')
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                    raise ValueError(f'source archive mismatch: {name}')
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            train = [row for row in rows if row['phase'] == 'train']
            val = [row for row in rows if row['phase'] == 'val']
            expected_val = [0] + [step for step in range(1, metadata['steps'] + 1)
                                 if step % metadata['eval_every'] == 0 or step == metadata['steps']]
            if ([row['step'] for row in train] != list(range(1, metadata['steps'] + 1))
                    or [row['step'] for row in val] != expected_val):
                raise ValueError('training or validation records are incomplete')
            if not all(math.isfinite(row['loss']) for row in rows):
                raise ValueError('nonfinite loss in log')
            if not all(math.isfinite(row['gradient_norm_before_clip']) for row in train):
                raise ValueError('nonfinite gradient norm in log')
            summary = json.loads((directory / 'summary.json').read_text())
            if summary['status'] != 'completed' or summary['validation_final'] != val[-1]['loss']:
                raise ValueError('completion summary does not match log')
            weights = directory / 'best-model.pt'
            checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
            if (checkpoint['variant'] != metadata['variant']
                    or checkpoint['config'] != metadata['config']
                    or checkpoint['validation_loss'] != summary['validation_best']
                    or checkpoint['validation_loss'] != min(row['loss'] for row in val)):
                raise ValueError('checkpoint metadata does not match log')
            if not all(torch.isfinite(value).all().item() for value in checkpoint['model'].values()):
                raise ValueError('nonfinite checkpoint tensor')
            if sum(value.numel() for value in checkpoint['model'].values()) != metadata['parameters']:
                raise ValueError('checkpoint parameter count differs from configuration')
            variants.add(metadata['variant'])
            report['runs'][metadata['variant']] = {
                'train_records': len(train), 'validation_records': len(val),
                'checkpoint_step': checkpoint['step'], 'checkpoint_tensors_finite': True,
                'checkpoint_sha256': digest(weights),
                'validation_final': summary['validation_final'],
                'initial_parameter_sha256': metadata['initial_parameter_sha256']}
            del checkpoint
    if variants != {'mla-reference-naive', 'mla-reference-absorb'}:
        raise ValueError('exactly one naive and one absorb training run are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
