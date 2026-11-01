from dataclasses import asdict
import hashlib
import json
import math
import statistics
import time

import numpy as np
import torch

from tools.chapter02.runtime import write_json, synchronize
from .config import preset
from .model import LanguageModel


def train(args, device, dtype, directory):
    cfg = preset(args.preset, args.variant)
    if min(args.steps, args.batch, args.context, args.eval_every, args.eval_blocks) < 1 or args.context > cfg.max_context:
        raise ValueError('invalid training dimensions')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    arrays, hashes = {}, {}
    for split in ('train', 'val'):
        file = args.data / f'{split}.bin'
        with file.open('rb') as stream:
            hashes[split] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if hashes[split] != manifest['token_bin_sha256'][split]:
            raise ValueError('token data hash mismatch')
        arrays[split] = np.memmap(file, mode='r', dtype=np.uint16)
        if len(arrays[split]) <= args.context or int(arrays[split].max()) >= cfg.vocab:
            raise ValueError('invalid token range or data length')
    if set(manifest['documents']['train']) & set(manifest['documents']['val']):
        raise ValueError('document split overlap')
    counts = {s: (len(a) - 1) // args.context for s, a in arrays.items()}
    if counts['train'] < args.batch:
        raise ValueError('not enough training blocks')
    val_ids = np.random.default_rng(2026).choice(counts['val'], min(args.eval_blocks, counts['val']), replace=False)
    rng = np.random.default_rng(1024)
    order, cursor, epoch = rng.permutation(counts['train']), 0, 0
    model = LanguageModel(cfg).to(device)
    initial_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        initial_hash.update(name.encode())
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)

    def batch(split, ids):
        offset = np.asarray(ids)[:, None] * args.context + np.arange(args.context + 1)[None]
        data = np.asarray(arrays[split][offset], dtype=np.int64)
        return (torch.from_numpy(data[:, :-1].copy()).to(device),
                torch.from_numpy(data[:, 1:].copy()).to(device))

    def evaluate():
        model.eval()
        total, tokens = 0., 0
        with torch.inference_mode():
            for left in range(0, len(val_ids), args.batch):
                x, y = batch('val', val_ids[left:left + args.batch])
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    loss = model(x, targets=y).loss
                if not torch.isfinite(loss).item():
                    raise ValueError('nonfinite validation loss')
                total += loss.item() * y.numel()
                tokens += y.numel()
        model.train()
        return total / tokens

    def record(row):
        with (directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if row['phase'] == 'val' or row['step'] % 25 == 0:
            print(json.dumps(row, allow_nan=False), flush=True)

    metadata = {'config': asdict(cfg), 'variant': args.variant, 'steps': args.steps,
        'batch': args.batch, 'context': args.context, 'eval_every': args.eval_every,
        'parameters': sum(p.numel() for p in model.parameters()), 'seed': 1024,
        'initial_parameter_sha256': initial_hash.hexdigest(), 'actual_token_bin_sha256': hashes,
        'data_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        'validation_blocks': val_ids.tolist(),
        'first_epoch_training_order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
        'precision': f'FP32 weights/AdamW, autocast={dtype}; compressor/softmax FP32',
        'optimizer': 'AdamW weight_decay=.01 grad_clip=1', 'lr_peak': args.lr,
        'indexer_aux_weight': args.aux_weight,
        'indexer_training': 'teaching KL from detached dense compressed-key attention to indexer distribution; '
                            'not the complete official training recipe',
        'sampling': 'same document-level split and shuffled packed input blocks as chapters 1/2',
        'scope': 'bounded random-initialization training; Dense SwiGLU and ordinary residual, not full DeepSeek V4',
        'timing_scope': 'synchronized model step, excludes sampling/H2D, evaluation and checkpoint saving'}
    write_json(directory / 'training.json', metadata)
    initial = evaluate()
    record({'phase': 'val', 'step': 0, 'loss': initial})
    best, timings, losses = initial, [], []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    wall = time.perf_counter()
    warmup = min(20, max(1, args.steps // 10))
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order, cursor, epoch = rng.permutation(counts['train']), 0, epoch + 1
        x, y = batch('train', order[cursor:cursor + args.batch])
        cursor += args.batch
        progress = (step - warmup) / max(1, args.steps - warmup)
        lr = args.lr * step / warmup if step <= warmup else args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        synchronize(device)
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            result = model(x, targets=y)
            loss = result.loss + args.aux_weight * result.auxiliary_loss
        if not torch.isfinite(loss).item():
            raise ValueError('nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        index_norm = sum(p.grad.float().norm().item() for name, p in model.named_parameters()
                         if '.indexer.' in name and p.grad is not None)
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        timings.append(time.perf_counter() - begin)
        losses.append(result.loss.item())
        record({'phase': 'train', 'step': step, 'loss': result.loss.item(),
            'indexer_auxiliary_loss': result.auxiliary_loss.item(), 'indexer_gradient_norm_sum': index_norm,
            'gradient_norm_before_clip': norm.item(), 'lr': lr, 'epoch': epoch,
            'tokens_seen': step * args.batch * args.context, 'model_step_seconds': timings[-1]})
        if step % args.eval_every == 0 or step == args.steps:
            value = evaluate()
            record({'phase': 'val', 'step': step, 'loss': value})
            if value < best:
                best = value
                torch.save({'model': model.state_dict(), 'config': asdict(cfg), 'variant': args.variant,
                    'step': step, 'validation_loss': value,
                    'note': 'weights only; no optimizer/RNG state for exact resumption'}, directory / 'best-model.pt')
    summary = {'status': 'completed', 'steps': args.steps, 'tokens_seen': args.steps * args.batch * args.context,
        'validation_initial': initial, 'validation_final': value, 'validation_best': best,
        'train_loss_last_25_mean': statistics.mean(losses[-25:]),
        'median_model_step_seconds_after_10': statistics.median(timings[10:] or timings),
        'wall_seconds_including_eval_save': time.perf_counter() - wall,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    write_json(directory / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0
