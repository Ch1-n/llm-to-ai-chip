"""Optional bounded real-text training using the existing packed Seq-Monkey data."""

from dataclasses import asdict
import hashlib
import json
import math
import statistics
import time

import numpy as np
import torch

from .config import preset
from .model import LanguageModel
from .runtime import synchronize, write_json


def train_text(args, device, dtype, directory):
    cfg = preset(args.preset)
    if not 0 < args.context <= cfg.max_context:
        raise ValueError('training context exceeds configuration')
    if min(args.steps, args.eval_every, args.eval_blocks, args.batch) < 1 or args.lr <= 0:
        raise ValueError('steps, eval settings, batch and lr must be positive')
    manifest_path = args.data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    arrays = {split: np.memmap(args.data / f'{split}.bin', mode='r', dtype=np.uint16)
              for split in ('train', 'val')}
    hashes = {}
    for split, values in arrays.items():
        if len(values) <= args.context or int(values.max()) >= cfg.vocab:
            raise ValueError(f'{split} token data is too short or outside the vocabulary')
        digest = hashlib.sha256()
        with (args.data / f'{split}.bin').open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[split] = digest.hexdigest()
        expected = manifest.get('token_bin_sha256', {}).get(split)
        if expected is not None and hashes[split] != expected:
            raise ValueError(f'{split} token data does not match manifest SHA256')
    documents = manifest.get('documents', {})
    if set(documents.get('train', [])) & set(documents.get('val', [])):
        raise ValueError('train/validation document hashes overlap')
    counts = {split: (len(values) - 1) // args.context for split, values in arrays.items()}
    if counts['train'] < args.batch:
        raise ValueError('not enough full training blocks for one batch')
    rng = np.random.default_rng(1024)
    val_ids = np.random.default_rng(2026).choice(
        counts['val'], min(args.eval_blocks, counts['val']), replace=False)
    order, cursor, epoch = rng.permutation(counts['train']), 0, 0
    model = LanguageModel(cfg, args.variant, args.backend).to(device)
    initial_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        initial_hash.update(name.encode())
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)

    def batch(split, indices):
        offsets = np.asarray(indices)[:, None] * args.context + np.arange(args.context + 1)[None, :]
        data = np.asarray(arrays[split][offsets], dtype=np.int64)
        return (torch.from_numpy(data[:, :-1].copy()).to(device),
                torch.from_numpy(data[:, 1:].copy()).to(device))

    def evaluate():
        model.eval()
        total, tokens = 0., 0
        with torch.inference_mode():
            for start in range(0, len(val_ids), args.batch):
                x, y = batch('val', val_ids[start:start + args.batch])
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    output = model(x, targets=y)
                if not torch.isfinite(output.loss).item():
                    raise AssertionError('nonfinite validation loss')
                total += output.loss.item() * y.numel()
                tokens += y.numel()
        model.train()
        return total / tokens

    def record(row):
        with (directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    metadata = {'config': asdict(cfg), 'variant': args.variant, 'backend': args.backend,
                'steps': args.steps, 'batch': args.batch, 'context': args.context,
                'eval_every': args.eval_every, 'lr_peak': args.lr,
                'warmup_steps': min(20, max(1, args.steps // 10)),
                'initial_parameter_sha256': initial_hash.hexdigest(),
                'first_epoch_training_order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                'data_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'actual_token_bin_sha256': hashes,
                'data_manifest': manifest, 'seed': 1024, 'validation_blocks': val_ids.tolist(),
                'parameters': sum(p.numel() for p in model.parameters()),
                'dropout': cfg.dropout, 'optimizer': 'AdamW weight_decay=.01, grad_clip=1',
                'attention_note': 'reference MLA has upstream FP32 softmax, no attention dropout; '
                                  'configured dropout still applies to block outputs',
                'precision': f'FP32 parameters/optimizer, autocast={dtype}',
                'sampling': 'permuted nonoverlapping input blocks, reshuffle on exhaustion',
                'data_boundary': 'reuse upstream document-level split; attention may cross same-split EOT',
                'throughput_scope': 'model training step only; excludes sampling/H2D, evaluation and saving',
                'interpretation': 'bounded teaching experiment; not equal-quality model comparison or full convergence'}
    write_json(directory / 'training.json', metadata)
    initial = evaluate()
    record({'phase': 'val', 'step': 0, 'loss': initial})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start_wall, timings, train_losses, best = time.perf_counter(), [], [], initial
    warmup = min(20, max(1, args.steps // 10))
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order, cursor, epoch = rng.permutation(counts['train']), 0, epoch + 1
        indices = order[cursor:cursor + args.batch]
        cursor += args.batch
        x, y = batch('train', indices)
        progress = (step - warmup) / max(1, args.steps - warmup)
        lr = args.lr * step / warmup if step <= warmup else args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        synchronize(device)
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            output = model(x, targets=y)
            loss = output.loss + args.aux_weight * output.auxiliary_loss
        if not torch.isfinite(loss).item():
            raise AssertionError('nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        elapsed = time.perf_counter() - begin
        timings.append(elapsed)
        train_losses.append(output.loss.item())
        record({'phase': 'train', 'step': step, 'loss': output.loss.item(),
                'auxiliary_loss': output.auxiliary_loss.item(), 'lr': lr,
                'gradient_norm_before_clip': norm.item(), 'epoch': epoch,
                'tokens_seen': step * args.batch * args.context, 'step_seconds': elapsed,
                'expert_assignments': [v.cpu().tolist() for v in output.expert_counts]})
        if step % args.eval_every == 0 or step == args.steps:
            value = evaluate()
            record({'phase': 'val', 'step': step, 'loss': value})
            if value < best:
                best = value
                if args.save_checkpoint:
                    torch.save({'model': model.state_dict(), 'config': asdict(cfg),
                                'variant': args.variant, 'backend': args.backend,
                                'step': step, 'validation_loss': value,
                                'note': 'weights only, not an exact resumable training checkpoint'},
                               directory / 'best-model.pt')
    result = {'status': 'completed', 'steps': args.steps,
              'tokens_seen': args.steps * args.batch * args.context,
              'validation_initial': initial, 'validation_final': value, 'validation_best': best,
              'train_loss_last_25_mean': statistics.mean(train_losses[-25:]),
              'median_model_step_seconds_after_10': statistics.median(timings[10:] or timings),
              'model_step_seconds': timings,
              'wall_seconds_including_eval_save': time.perf_counter() - start_wall}
    if device.type == 'cuda':
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    write_json(directory / 'summary.json', result)
    return 0
