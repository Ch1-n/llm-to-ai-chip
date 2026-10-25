#!/usr/bin/env python3
"""Audit the user's notebook definitions without running its dataset/training cells.

The notebook is supplied separately and is not distributed in this repository.
RoPE is repaired in memory only, explicitly recorded in the JSON report.
"""

import argparse
import ast
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_definitions(path):
    module = types.ModuleType('audited_notebook')
    sys.modules[module.__name__] = module
    exec('import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
         'import math\nfrom dataclasses import dataclass', module.__dict__)
    wanted = {'GPTConfig', 'precompute_rope_freqs', 'rotate_half', 'apply_rope',
              'SingleHeadAttention', 'MultiHeadAttention', 'FeedForward', 'Block', 'GPT'}
    found = set()
    for cell in json.loads(path.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        parsed = ast.parse(''.join(cell['source']))
        nodes = [node for node in parsed.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook-definitions>', 'exec'),
                 module.__dict__)
            found.update(node.name for node in nodes)
    assert found == wanted, wanted - found
    return module


def corrected_rope(dimension, length, theta=10000.0):
    assert dimension % 2 == 0
    frequency = theta ** (-torch.arange(0, dimension, 2).float() / dimension)
    angle = torch.outer(torch.arange(length).float(), frequency)
    return angle.cos().repeat_interleave(2, -1), angle.sin().repeat_interleave(2, -1)


def errors(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm() / expected.norm().clamp_min(1e-12)).item()}


def compare(actual, expected, atol=3e-5, rtol=3e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {**errors(actual, expected), 'atol': atol, 'rtol': rtol, 'passed': True}


def correctness(module):
    result = {}
    basis = torch.zeros(1, 1, 64, device='cuda')
    basis[..., 0] = 1
    cos, sin = module.precompute_rope_freqs(64, 16)
    rotated = module.apply_rope(basis, cos[1:2].cuda(), sin[1:2].cuda())
    original_norm_error = abs(rotated.norm().item() - 1)
    result['original_rope'] = {'norm_error': original_norm_error,
                               'bug_reproduced': original_norm_error > 1e-3}
    module.precompute_rope_freqs = corrected_rope
    cos, sin = corrected_rope(64, 16)
    rotated = module.apply_rope(basis, cos[1:2].cuda(), sin[1:2].cuda())
    expected = torch.zeros_like(basis)
    expected[..., 0] = torch.cos(torch.tensor(1.)).item()
    expected[..., 1] = torch.sin(torch.tensor(1.)).item()
    result['repaired_rope_basis'] = compare(rotated, expected, 1e-6, 1e-5)
    result['repaired_rope_norm'] = compare(rotated.norm(), basis.norm(), 1e-6, 1e-5)
    stale = module.GPTConfig(n_embd=24, n_head=3)
    result['config_derived_head_size'] = {'actual': stale.head_size, 'expected': 8,
                                        'bug_reproduced': stale.head_size != 8}
    config = module.GPTConfig(block_size=32, n_layer=3, n_head=3, n_embd=24,
                              head_size=8, vocab_size=97, dropout=0.1)
    torch.manual_seed(1024)
    model = module.GPT(config).cuda().eval()
    ids = torch.randint(97, (2, 32), device='cuda')
    with torch.inference_mode():
        reference, _, reference_cache = model(ids, use_cache=True)
        for chunks in ([1] * 32, [5, 1, 7, 19], [32]):
            cache, offset, parts = None, 0, []
            for length in chunks:
                value, _, cache = model(ids[:, offset:offset + length],
                                        past_key_values=cache, use_cache=True)
                offset += length
                for layer in cache:
                    for k, v in layer:
                        assert k.shape == v.shape == (2, offset, 8)
                parts.append(value)
            result['cached_chunks_' + '_'.join(map(str, chunks))] = compare(
                torch.cat(parts, 1), reference)
            for layer, full_layer in zip(cache, reference_cache):
                for head, full_head in zip(layer, full_layer):
                    for tensor, full_tensor in zip(head, full_head):
                        compare(tensor, full_tensor)
        changed = ids.clone()
        changed[:, 16:] = (changed[:, 16:] + 11) % 97
        changed_logits, _ = model(changed)
        result['causality'] = compare(changed_logits[:, :16], reference[:, :16])
        try:
            model(ids[:, :1], past_key_values=cache, use_cache=True)
        except ValueError:
            result['context_overflow_rejected'] = True
        else:
            raise AssertionError('context overflow not rejected')
    observations = []
    for length, count in ((4, 0), (4, 1), (4, 3), (31, 3), (32, 1), (34, 3)):
        prompt = torch.arange(length, device='cuda')[None] % 97
        calls = []
        hook = model.register_forward_pre_hook(lambda _, inputs: calls.append(inputs[0].shape[1]))
        model.train()
        try:
            output = model.generate(prompt, count)
        finally:
            hook.remove()
        assert model.training
        observations.append({'prompt': length, 'requested_new_tokens': count,
                             'output_length': output.shape[1], 'forward_lengths': calls,
                             'prefix_preserved': output.shape[1] >= length and
                             torch.equal(output[:, :length], prompt)})
    result['original_generate_observations'] = observations
    return result


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def benchmark(call, repeats, warmups=3):
    for _ in range(warmups):
        output = call()
        del output
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, events = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        output = call()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        events.append(begin.elapsed_time(end))
        del output
    return {'repeats': repeats, 'warmups': warmups,
            'wall_median_ms': statistics.median(wall),
            'wall_p10_ms': percentile(wall, 0.1), 'wall_p90_ms': percentile(wall, 0.9),
            'cuda_event_median_ms': statistics.median(events),
            'baseline_allocated_bytes': base,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated() - base,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def full_model_runs(module, repeats):
    records, precision, training = [], [], []
    reference = None
    for name, dtype in (('float32', torch.float32), ('bfloat16', torch.bfloat16),
                        ('float16', torch.float16)):
        torch.manual_seed(1024)
        config = module.GPTConfig(dropout=0.0)
        model = module.GPT(config).cuda().to(dtype).eval()
        assert sum(p.numel() for p in model.parameters()) == 119690496
        ids = (torch.arange(512, device='cuda')[None] * 17 + 9) % config.vocab_size
        with torch.inference_mode():
            full, _ = model(ids)
            first, _, cache = model(ids[:, :511], use_cache=True)
            last, _, cache = model(ids[:, 511:], past_key_values=cache, use_cache=True)
            comparison = errors(last, full[:, -1:])
            if dtype == torch.float32:
                comparison = compare(last, full[:, -1:])
            else:
                # Same-dtype cache/full paths differ in reduction shape; allow low-precision rounding.
                comparison = compare(last, full[:, -1:], atol=0.08 if dtype == torch.bfloat16 else 0.01,
                                     rtol=0.08 if dtype == torch.bfloat16 else 0.01)
            sample = full[:, -1:].float().clone()
            if reference is None:
                reference = sample
            precision.append({'dtype': name, 'cached_vs_full': comparison,
                              'vs_fp32': errors(sample, reference),
                              'top1_same_as_fp32': torch.equal(sample.argmax(-1), reference.argmax(-1))})
            del full, first, last, cache, sample
            for batch in (1, 12):
                for length in (128, 512):
                    prompt = ids[:, :length].expand(batch, -1).contiguous()
                    call = lambda: model(prompt, use_cache=True)
                    metrics = benchmark(call, repeats)
                    records.append({'dtype': name, 'phase': 'prefill', 'batch': batch,
                                    'query_tokens': length, 'visible_kv_tokens': length,
                                    'input_tokens_per_s': batch * length * 1000 / metrics['wall_median_ms'],
                                    **metrics})
                    print('BENCH', name, 'prefill', batch, length, metrics['wall_median_ms'], flush=True)
                for visible in (129, 512):
                    prompt = ids[:, :visible - 1].expand(batch, -1).contiguous()
                    _, _, past = model(prompt, use_cache=True)
                    token = ids[:, visible - 1:visible].expand(batch, -1).contiguous()
                    metrics = benchmark(lambda: model(token, past_key_values=past, use_cache=True), repeats)
                    records.append({'dtype': name, 'phase': 'decode', 'batch': batch,
                                    'query_tokens': 1, 'visible_kv_tokens': visible,
                                    'aggregate_tokens_per_s': batch * 1000 / metrics['wall_median_ms'],
                                    **metrics})
                    print('BENCH', name, 'decode', batch, visible, metrics['wall_median_ms'], flush=True)
                    del past
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Exercise the original dropout and a real optimizer, but synthetic data, not a convergence study.
    for name, dtype in (('float32', torch.float32), ('bfloat16', torch.bfloat16),
                        ('float16', torch.float16)):
        torch.manual_seed(1024)
        model = module.GPT(module.GPTConfig()).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
        losses = []
        x = torch.randint(50257, (12, 512), device='cuda')
        y = torch.roll(x, -1, 1)
        torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(x, targets=y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            print('TRAIN', name, step, losses[-1], flush=True)
        torch.cuda.synchronize()
        training.append({'dtype': name, 'mode': 'FP32 weights; AMP for low precision',
                         'batch': 12, 'sequence': 512, 'steps': 3, 'losses': losses,
                         'finite_gradients': True,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        del model, optimizer, scaler, loss, x, y
        gc.collect()
        torch.cuda.empty_cache()
    return records, precision, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert args.repeats > 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = load_definitions(args.notebook)
    environment = {'python': platform.python_version(), 'pytorch': torch.__version__,
                   'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                   'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                   'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                     '--format=csv,noheader'], text=True).strip(),
                   'tf32': False, 'cpu_threads': 4}
    report = {'environment': environment,
              'notebook_sha256': hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
              'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Random weights/tokens. Correctness and smoke tests, not language quality or convergence.',
              'patches': ['Adjacent-pair RoPE uses repeat_interleave, in memory only.',
                          'Small test explicitly sets head_size; original config unchanged.'],
              'benchmark_method': 'PyTorch eager, separate Python loop per head, explicit scores, dynamic torch.cat cache. No compile/SDPA/FlashAttention. Prefill computes all-position logits. Decode reuses a fixed prefix and measures one new position. Inference weights cast to dtype, not autocast. Wall time synchronizes CUDA; event time also includes launch gaps. Cache construction excluded from Decode timing.',
              'correctness': correctness(module)}
    print('CORRECTNESS', json.dumps(report['correctness']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    records, precision, training = full_model_runs(module, args.repeats)
    report.update(benchmarks=records, precision_checks=precision, training_smoke=training,
                  status='completed')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('COMPLETED', args.output, flush=True)


if __name__ == '__main__':
    main()
