#!/usr/bin/env python3
"""A bounded real-text training experiment on the corrected notebook model."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from validate_cuda_notebook import corrected_rope, load_definitions
from chapter02.notebook_adapter import install_attention_backend
from chapter02.runtime import environment, fresh_directory, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--sequence', type=int, default=512)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-batches', type=int, default=16)
    parser.add_argument('--backend', choices=('notebook', 'fa2', 'sdpa-flash'), default='notebook')
    args = parser.parse_args()
    assert min(args.steps, args.batch, args.sequence, args.eval_every, args.eval_batches) > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fresh_directory(args.output)
    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = load_definitions(args.notebook)
    module.precompute_rope_freqs = corrected_rope
    config = module.GPTConfig()
    config.head_size = config.n_embd // config.n_head
    assert args.sequence <= config.block_size
    model = module.GPT(config).cuda()
    install_attention_backend(model, module, args.backend)
    parameter_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        parameter_hash.update(name.encode())
        parameter_hash.update(parameter.detach().cpu().numpy().tobytes())
    initial_parameter_sha256 = parameter_hash.hexdigest()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    token_hashes = {}
    for split in ('train', 'val'):
        with (args.data / f'{split}.bin').open('rb') as stream:
            token_hashes[split] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if token_hashes[split] != manifest['token_bin_sha256'][split]:
            raise ValueError(f'{split} data does not match manifest SHA256')
    if set(manifest['documents']['train']) & set(manifest['documents']['val']):
        raise ValueError('train/validation document hashes overlap')
    arrays = {name: np.memmap(args.data / f'{name}.bin', dtype=np.uint16, mode='r')
              for name in ('train', 'val')}
    block_counts = {name: (len(array) - 1) // args.sequence for name, array in arrays.items()}
    assert block_counts['train'] >= args.batch and block_counts['val'] >= args.batch
    validation_blocks = min(args.eval_batches * args.batch, block_counts['val'])
    validation_ids = np.random.default_rng(2026).choice(block_counts['val'], validation_blocks, replace=False)
    order_generator = np.random.default_rng(1024)
    order = order_generator.permutation(block_counts['train'])
    cursor, epoch = 0, 0

    def batch_from(split, ids):
        starts = np.asarray(ids)[:, None] * args.sequence
        offsets = np.arange(args.sequence + 1)[None]
        packed = np.asarray(arrays[split][starts + offsets], dtype=np.int64)
        return (torch.from_numpy(packed[:, :-1].copy()).cuda(),
                torch.from_numpy(packed[:, 1:].copy()).cuda())

    def evaluate():
        model.eval()
        total_loss, total_tokens = 0., 0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(validation_ids), args.batch):
                x, y = batch_from('val', validation_ids[start:start + args.batch])
                _, loss = model(x, targets=y)
                assert torch.isfinite(loss).item()
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        model.train()
        return total_loss / total_tokens

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    metadata = {'seed': 1024, 'model': asdict(config), 'parameters': sum(p.numel() for p in model.parameters()),
                'backend': args.backend, 'initial_parameter_sha256': initial_parameter_sha256,
                'actual_token_bin_sha256': token_hashes, 'validation_block_ids': validation_ids.tolist(),
                'first_epoch_training_order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                'steps': args.steps, 'batch': args.batch, 'sequence': args.sequence,
                'precision': 'FP32 weights/optimizer; BF16 autocast; no GradScaler',
                'optimizer': 'AdamW lr_peak=3e-4 lr_min=3e-5 weight_decay=0.01 grad_clip=1.0',
                'schedule': '20-step linear warmup then cosine to minimum',
                'validation_blocks': validation_blocks, 'eval_every': args.eval_every,
                'gpu': torch.cuda.get_device_name(), 'pytorch': torch.__version__, 'cuda': torch.version.cuda,
                'notebook_sha256': hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
                'training_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'loader_script_sha256': hashlib.sha256(Path(__file__).with_name('validate_cuda_notebook.py').read_bytes()).hexdigest(),
                'adapter_script_sha256': hashlib.sha256(Path(__file__).with_name('chapter02').joinpath('notebook_adapter.py').read_bytes()).hexdigest(),
                'data_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
                'tokens_available': manifest['tokens'], 'blocks_available': block_counts,
                'training_order': 'Random permutation of packed non-overlapping input blocks; reshuffle on exhaustion.',
                'document_packing': 'Concatenate separately in each split with EOT; attention may cross document boundaries.',
                'interpretation': 'Short real-text learning experiment, not convergence or general capability validation. Fixed held-out validation blocks reused throughout.',
                'backend_comparison': 'Original notebook weights/projections preserved; Attention heads stacked for fused backends. Same seed does not imply identical dropout masks across implementations.'}
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_json(args.output / 'environment.json', environment())
    write_json(args.output / 'arguments.json', {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()})
    rows = []

    def record(row):
        rows.append(row)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    best = evaluate()
    record({'phase': 'val', 'step': 0, 'loss': best, 'tokens': validation_blocks * args.sequence})
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    timings = []
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order = order_generator.permutation(block_counts['train'])
            cursor, epoch = 0, epoch + 1
        ids = order[cursor:cursor + args.batch]
        cursor += args.batch
        torch.cuda.synchronize()
        start = time.perf_counter()
        x, y = batch_from('train', ids)
        if step <= 20:
            lr = 3e-4 * step / 20
        else:
            progress = (step - 20) / max(1, args.steps - 20)
            lr = 3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        assert torch.isfinite(loss).item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        record({'phase': 'train', 'step': step, 'loss': loss.item(), 'lr': lr,
                'grad_norm_before_clip': grad_norm.item(), 'epoch': epoch,
                'tokens_seen': step * args.batch * args.sequence,
                'step_seconds': elapsed, 'tokens_per_second': y.numel() / elapsed})
        if step % args.eval_every == 0 or step == args.steps:
            value = evaluate()
            record({'phase': 'val', 'step': step, 'loss': value,
                    'tokens': validation_blocks * args.sequence})
            if value < best:
                best = value
                torch.save({'step': step, 'config': asdict(config), 'model': model.state_dict(),
                            'val_loss': value, 'data_manifest_sha256': metadata['data_manifest_sha256'],
                            'backend': args.backend,
                            'note': 'Model weights only; no optimizer/RNG state for exact resumption.'},
                           args.output / 'best-model.pt')
    median = statistics.median(timings[10:] or timings)
    result = {'status': 'completed', 'steps': args.steps, 'tokens_seen': args.steps * args.batch * args.sequence,
              'validation_initial': rows[0]['loss'], 'validation_final': rows[-1]['loss'],
              'best_validation': best, 'median_train_step_seconds_after_10': median,
              'median_train_tokens_per_second': args.batch * args.sequence / median,
              'wall_seconds_including_eval_and_checkpoint': time.perf_counter() - start_time,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'train_loss_last_25_mean': statistics.mean([r['loss'] for r in rows if r['phase'] == 'train'][-25:]),
              'epochs_started': epoch + 1}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', json.dumps(result), flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
