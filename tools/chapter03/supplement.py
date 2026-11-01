"""Bounded follow-up checks; preserve all earlier measurements and checkpoints."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.chapter02.runtime import (assert_match, errors, fresh_directory, initialize,
                                    percentile, synchronize, write_json)
from .config import Config, preset
from .model import LanguageModel
from .runtime import Workload, cache_stats, environment, measure


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint(path, device):
    value = torch.load(path, map_location='cpu', weights_only=True)
    config = dict(value['config'])
    config['ratios'] = tuple(config['ratios'])
    model = LanguageModel(Config(**config)).to(device).eval()
    model.load_state_dict(value['model'])
    return model, {'variant': value['variant'], 'checkpoint_sha256': sha(path),
                   'trained_context': 512, 'step': value['step'], 'config': config}


def compare(actual, expected, dtype):
    try:
        return assert_match(actual, expected, dtype)
    except AssertionError as error:
        return {**errors(actual, expected), 'passed': False, 'reason': str(error)[:1000]}


def trained_checks(args, device):
    records = []
    for path in args.checkpoints:
        model, meta = checkpoint(path, device)
        for dtype in (torch.float32, torch.bfloat16) if device.type == 'cuda' else (torch.float32,):
            for length in args.contexts:
                if length < 258 or length > model.cfg.max_context:
                    raise ValueError('check context must be 258..max_context')
                torch.manual_seed(2026 + length)
                x = torch.randint(model.cfg.vocab, (1, length), device=device)
                amp = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
                with torch.inference_mode(), amp:
                    # Keep only the final 130 vocabulary vectors after the full pass.
                    full = model(x).logits[:, -130:].clone()
                    changed = x.clone()
                    changed[:, -128:] = (changed[:, -128:] + 7) % model.cfg.vocab
                    future = model(changed).logits[:, -130:-128].clone()
                    causality = compare(future, full[:, :2], dtype)
                    out = model(x[:, :127], capacity=length, last_only=True)
                    for left, right in ((127, 128), (128, length - 129),
                                        (length - 129, length - 128), (length - 128, length)):
                        out = model(x[:, left:right], caches=out.caches, last_only=True)
                    chunked = compare(out.logits, full[:, -1:], dtype)
                    out = model(x[:, :-130], capacity=length, last_only=True)
                    parts = []
                    for pos in range(length - 130, length):
                        out = model(x[:, pos:pos + 1], caches=out.caches, last_only=True)
                        parts.append(out.logits)
                    incremental = compare(torch.cat(parts, 1), full, dtype)
                row = {**meta, 'context': length, 'precision': str(dtype),
                       'precision_note': 'FP32 checkpoint weights; BF16 uses autocast, not native BF16 weights',
                       'causality': causality, 'chunked_final_logits': chunked,
                       'last_130_incremental_logits': incremental,
                       'status': 'passed' if all(v['passed'] for v in (causality, chunked, incremental)) else 'failed'}
                records.append(row)
                print(json.dumps({k: row[k] for k in ('variant', 'context', 'precision', 'status')}), flush=True)
                del full, out, parts, x, changed, future
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(args.output / 'trained-checks.json', {'records': records,
        'status': 'passed' if all(r['status'] == 'passed' for r in records) else 'failed'})
    return all(r['status'] == 'passed' for r in records)


def continuous(args, device):
    records = []
    for variant in args.variants:
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        work = Workload(variant, 'prefill', 1, args.prefix + args.steps, device, dtype, args.preset)
        samples, final_stats = [], None
        model = work.model
        cfg = model.config if work.old else model.cfg
        if args.prefix + args.steps > cfg.max_context:
            raise ValueError('sequence exceeds capacity')

        def prefill():
            kw = {'cache_mode': 'static'} if work.old else {}
            return model(work.tokens[:, :args.prefix], capacity=args.prefix + args.steps,
                         last_only=True, **kw)

        with torch.inference_mode():
            warm = prefill()
            for pos in range(args.prefix, args.prefix + min(8, args.steps)):
                warm = model(work.tokens[:, pos:pos + 1], caches=warm.caches, last_only=True)
            del warm
            for repeat in range(args.repeats):
                out = prefill()
                for pos in range(args.prefix, args.prefix + args.steps):
                    synchronize(device)
                    start = time.perf_counter()
                    out = model(work.tokens[:, pos:pos + 1], caches=out.caches, last_only=True)
                    synchronize(device)
                    elapsed = (time.perf_counter() - start) * 1000
                    samples.append({'repeat': repeat, 'context': pos + 1, 'ms': elapsed,
                                    'closes_4': (pos + 1) % 4 == 0,
                                    'closes_128': (pos + 1) % 128 == 0})
                if not torch.isfinite(out.logits).all().item():
                    raise ValueError('nonfinite decode logits')
                final_stats = cache_stats(out.caches)
                del out
        values = [r['ms'] for r in samples]
        row = {'variant': variant, 'prefix': args.prefix, 'steps_per_repeat': args.steps,
               'repeats': args.repeats, 'samples': samples, 'mean_ms': statistics.mean(values),
               'median_ms': statistics.median(values), 'p95_ms': percentile(values, .95),
               'p99_ms': percentile(values, .99), 'tokens_per_second': 1000 / statistics.mean(values),
               'final_cache': final_stats, 'weights': f'random seed 1024, native {dtype}',
               'scope': 'continuous cache advancement with pre-generated random tokens, B1; '
                        'not sampled text generation; per-token synchronization included, prefix excluded',
               'status': 'completed'}
        records.append(row)
        print(json.dumps({k: row[k] for k in ('variant', 'mean_ms', 'p95_ms')}), flush=True)
        del work, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(args.output / 'continuous.json', {'records': records})


def ablation(args, device):
    records = []
    for variant, topk in [('shared-dense', 8), ('swa', 8), ('csa', 8), ('hca', 8),
                          ('hybrid', 1), ('hybrid', 8), ('hybrid', 32)]:
        for length in args.contexts:
            for phase in ('prefill', 'decode'):
                dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
                work = Workload(variant, phase, 1, length, device, dtype, args.preset)
                config = replace(work.model.cfg, index_topk=topk)
                work.model.cfg = config
                for block in work.model.blocks:
                    block.attention.cfg = config
                    if block.attention.indexer is not None:
                        block.attention.indexer.cfg = config
                # Top-k does not change tensor shapes or random initialization.
                # Rebuild a decode prefix after changing selection policy.
                if phase == 'decode':
                    from .runtime import clone_cache_positions
                    with torch.inference_mode():
                        work.prefix = clone_cache_positions(work.model(work.tokens[:, :-1],
                            capacity=length, last_only=True).caches)
                row = {**work.metadata, 'config': asdict(config), 'index_topk': topk,
                       'performance': measure(work, device, 3, 10), 'status': 'completed',
                       'scope': 'random weights, no quality matching; top-k variants share initialization; '
                                'single-path layer schedules change parameter sets'}
                records.append(row)
                print(json.dumps({'variant': variant, 'topk': topk, 'context': length,
                                  'phase': phase, 'ms': row['performance']['wall_median_ms']}), flush=True)
                del work
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    write_json(args.output / 'ablation.json', {'records': records})


def history_eval(args, device):
    manifest = json.loads((args.data / 'manifest.json').read_text())
    digest = sha(args.data / 'val.bin')
    if digest != manifest['token_bin_sha256']['val']:
        raise ValueError('validation data hash mismatch')
    data = np.memmap(args.data / 'val.bin', mode='r', dtype=np.uint16)
    maximum = max(args.contexts)
    available = (len(data) - 1) // maximum
    if available < args.samples:
        raise ValueError('not enough nonoverlapping validation samples')
    blocks = np.random.default_rng(2026).choice(available, args.samples, replace=False)
    records = []
    for path in args.checkpoints:
        model, meta = checkpoint(path, device)
        for block in blocks:
            end = (int(block) + 1) * maximum
            targets = torch.tensor(np.asarray(data[end - args.suffix + 1:end + 1], dtype=np.int64), device=device)[None]
            for length in args.contexts:
                tokens = torch.tensor(np.asarray(data[end - length:end], dtype=np.int64), device=device)[None]
                amp = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                with torch.inference_mode(), amp:
                    prefix = model(tokens[:, :-args.suffix], capacity=length, last_only=True)
                    output = model(tokens[:, -args.suffix:], caches=prefix.caches)
                    nll = F.cross_entropy(output.logits.float().reshape(-1, model.cfg.vocab),
                                          targets.reshape(-1), reduction='none')
                row = {**meta, 'context': length, 'validation_block': int(block),
                       'end_token_offset': end, 'suffix_tokens': args.suffix,
                       'mean_nll': nll.mean().item(), 'per_token_nll': nll.cpu().tolist(),
                       'status': 'completed'}
                if not torch.isfinite(nll).all().item():
                    raise ValueError('nonfinite suffix loss')
                records.append(row)
                del prefix, output, tokens, nll
            print(json.dumps({'variant': meta['variant'], 'validation_block': int(block)}), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summaries = [{'variant': v, 'context': length,
                  'mean_nll': statistics.mean(r['mean_nll'] for r in records if r['variant'] == v and r['context'] == length)}
                 for v in sorted({r['variant'] for r in records}) for length in args.contexts]
    write_json(args.output / 'history-eval.json', {'records': records, 'summary': summaries,
        'val_bin_sha256': digest, 'scope': 'same target suffix, different retained prefix; packed held-out text may cross EOT; '
        'RoPE restarts at zero for each cropped input; checkpoints trained at context 512 for only 300 steps; '
        'not a long-context benchmark or evidence of retrieval capability'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('check', 'continuous', 'ablation', 'history'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--preset', default='teaching', choices=('teaching', 'tiny'))
    parser.add_argument('--checkpoints', nargs='+', type=Path, default=[])
    parser.add_argument('--contexts', nargs='+', type=int, default=[2048, 8192])
    parser.add_argument('--variants', nargs='+', default=['mha', 'mla', 'shared-dense', 'swa', 'hybrid'])
    parser.add_argument('--prefix', type=int, default=512)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--suffix', type=int, default=128)
    args = parser.parse_args()
    if min(args.contexts + [args.prefix, args.steps, args.repeats, args.samples, args.suffix]) < 1:
        parser.error('dimensions must be positive')
    if args.operation in ('check', 'history') and not args.checkpoints:
        parser.error('checkpoints required')
    if args.operation == 'history' and (args.data is None or min(args.contexts) <= args.suffix):
        parser.error('data required and context must exceed suffix')
    fresh_directory(args.output)
    device = initialize(args.device, torch.float32)
    write_json(args.output / 'environment.json', environment())
    write_json(args.output / 'arguments.json', {k: str(v) if isinstance(v, Path) else
        [str(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v
        for k, v in vars(args).items()})
    result = {'check': trained_checks, 'continuous': continuous, 'ablation': ablation,
              'history': history_eval}[args.operation](args, device)
    if result is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
