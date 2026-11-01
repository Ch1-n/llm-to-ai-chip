"""Compare the trainable MLA component with pinned, unmodified upstream classes."""

import argparse
import ast
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import types
from typing import Literal, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .attention import causal_mask
from .cache import KVCache
from .checks import assert_match
from .config import preset
from .mla_reference import MLAReference
from .runtime import environment, fresh_directory, initialize, write_json


SOURCE_SHA256 = 'fdb7995bc234badbf0011b5af8cf391be94343abc51305428cb9a04452116f02'
SOURCE_COMMIT = 'b15f0dbbbe6a4bc403306175698439ef380f5fb5'
SOURCE_PATH = Path(__file__).resolve().parents[2] / 'references/deepseek-v3/model.py'


def load_upstream(path=SOURCE_PATH, implementation='naive'):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError('upstream source differs from pinned SHA256')
    wanted = {'ModelArgs', 'linear', 'Linear', 'ColumnParallelLinear', 'RowParallelLinear',
              'RMSNorm', 'precompute_freqs_cis', 'apply_rotary_emb', 'MLA'}
    parsed = ast.parse(content.decode())
    nodes = [n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('required upstream definitions are missing')
    module = types.ModuleType(f'pinned_deepseek_mla_{implementation}')
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
                           Literal=Literal, Optional=Optional, Tuple=Tuple,
                           world_size=1, rank=0, block_size=128, gemm_impl='bf16',
                           attn_impl=implementation)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    module.Linear.dtype = torch.float32
    return module


def upstream_args(module, cfg, batch=2):
    return module.ModelArgs(max_batch_size=batch, max_seq_len=cfg.max_context,
        original_seq_len=cfg.max_context, dim=cfg.width, n_heads=cfg.heads,
        q_lora_rank=cfg.q_rank, kv_lora_rank=cfg.kv_rank,
        qk_nope_head_dim=cfg.nope_dim, qk_rope_head_dim=cfg.rope_dim,
        v_head_dim=cfg.value_dim, rope_theta=cfg.rope_theta, rope_factor=1.)


def mask_for(query, total, device, dtype):
    mask = torch.zeros(query, total, device=device, dtype=dtype)
    return mask.masked_fill(~causal_mask(query, total, device), float('-inf'))


def initialize_weights(model):
    for name, parameter in model.named_parameters():
        if 'norm.weight' in name:
            nn.init.ones_(parameter)
        else:
            nn.init.normal_(parameter, std=.02)


def inference_case(cfg, path, device, dtype, chunks, cache_mode):
    module = load_upstream(implementation=path)
    original = module.MLA(upstream_args(module, cfg)).to(device=device, dtype=dtype).eval()
    candidate = MLAReference(cfg, path).to(device=device, dtype=dtype).eval()
    initialize_weights(candidate)
    original.load_state_dict(candidate.state_dict(), strict=True)
    freqs = module.precompute_freqs_cis(upstream_args(module, cfg)).to(device)
    x = torch.randn(2, sum(chunks), cfg.width, device=device, dtype=dtype)
    cache, offset, results = KVCache(cache_mode, cfg.max_context), 0, []
    expected, observed = [], []
    with torch.inference_mode():
        for length in chunks:
            part = x[:, offset:offset + length]
            expected.append(original(part, offset, freqs[offset:offset + length],
                                     mask_for(length, offset + length, device, dtype)))
            observed.append(candidate(part, offset, cache))
            results.append(assert_match(observed[-1], expected[-1], dtype))
            offset += length
            if path == 'naive':
                for field in ('k', 'v'):
                    actual = cache.storage[field][:, :, :offset].transpose(1, 2)
                    upstream = getattr(original, f'{field}_cache')[:, :offset]
                    assert_match(actual, upstream, dtype)
            else:
                assert_match(cache.storage['latent'][:, 0, :offset], original.kv_cache[:, :offset], dtype)
                assert_match(cache.storage['rope'][:, 0, :offset], original.pe_cache[:, :offset], dtype)
        assert_match(torch.cat(observed, 1), candidate(x), dtype)
    return {'kind': 'upstream_inference_output_and_cache', 'path': path,
            'dtype': str(dtype), 'chunks': list(chunks), 'cache_mode': cache_mode,
            'status': 'passed', 'chunk_errors': results, 'cache_logical_bytes': cache.logical_bytes()}


def gradient_case(cfg, path, device, amp=False):
    module = load_upstream(implementation=path)
    original = module.MLA(upstream_args(module, cfg)).to(device).train()
    candidate = MLAReference(cfg, path).to(device).train()
    initialize_weights(candidate)
    original.load_state_dict(candidate.state_dict(), strict=True)
    freqs = module.precompute_freqs_cis(upstream_args(module, cfg)).to(device)
    x = torch.randn(2, 13, cfg.width, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
        expected = original(x, 0, freqs[:13], mask_for(13, 13, device, x.dtype))
        observed = candidate(y)
    expected.float().square().mean().backward()
    observed.float().square().mean().backward()
    precision = torch.bfloat16 if amp else torch.float32
    output = assert_match(observed, expected, precision)
    gradient = assert_match(y.grad, x.grad, precision)
    params = {}
    upstream = dict(original.named_parameters())
    for name, parameter in candidate.named_parameters():
        params[name] = assert_match(parameter.grad, upstream[name].grad, precision)
    return {'kind': 'upstream_output_input_parameter_gradients', 'path': path,
            'precision': 'bf16_amp' if amp else 'fp32', 'status': 'passed',
            'output': output, 'input_gradient': gradient, 'parameter_gradients': params}


def equivalence_case(cfg, device, amp=False):
    a, b = (MLAReference(cfg, path).to(device) for path in ('naive', 'absorb'))
    initialize_weights(a)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 13, cfg.width, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
        left, right = a(x), b(y)
    left.float().square().mean().backward()
    right.float().square().mean().backward()
    precision = torch.bfloat16 if amp else torch.float32
    output = assert_match(right, left, precision)
    gradient = assert_match(y.grad, x.grad, precision)
    params = [assert_match(q.grad, p.grad, precision) for p, q in zip(a.parameters(), b.parameters())]
    return {'kind': 'naive_absorb_equivalence_output_input_parameter_gradients',
            'precision': 'bf16_amp' if amp else 'fp32', 'status': 'passed',
            'output': output, 'input_gradient': gradient, 'parameter_gradients': params}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--preset', choices=('audit', 'teaching'), default='audit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    device = initialize(args.device, torch.float32)
    directory = fresh_directory(args.output)
    write_json(directory / 'environment.json', environment())
    cfg = (replace(preset('teaching'), max_context=64) if args.preset == 'teaching'
           else replace(preset('tiny'), width=128, heads=2, q_rank=64, kv_rank=64,
                        nope_dim=32, rope_dim=32, value_dim=64, max_context=64))
    records = []
    for path in ('naive', 'absorb'):
        for dtype in (torch.float32, torch.bfloat16) if device.type == 'cuda' else (torch.float32,):
            for chunks in ((17,), (5, 1, 11), (1,) * 17):
                for cache_mode in ('cat', 'static'):
                    try:
                        records.append(inference_case(cfg, path, device, dtype, chunks, cache_mode))
                    except Exception as error:
                        records.append({'path': path, 'dtype': str(dtype), 'chunks': list(chunks),
                                        'cache_mode': cache_mode, 'status': 'failed', 'reason': str(error)})
        for amp in (False, True) if device.type == 'cuda' else (False,):
            try:
                records.append(gradient_case(cfg, path, device, amp))
            except Exception as error:
                records.append({'path': path, 'amp': amp, 'status': 'failed', 'reason': str(error)})
    for amp in (False, True) if device.type == 'cuda' else (False,):
        try:
            records.append(equivalence_case(cfg, device, amp))
        except Exception as error:
            records.append({'kind': 'naive_absorb_equivalence', 'amp': amp,
                            'status': 'failed', 'reason': str(error)})
    result = {'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_SHA256,
              'config': asdict(cfg),
              'scope': 'single-device unquantized reference classes; no FP8/TP/YaRN/FlashMLA',
              'records': records, 'failed': any(r['status'] != 'passed' for r in records)}
    write_json(directory / 'checks.json', result)
    print(json.dumps({'checks': len(records), 'failed': result['failed']}))
    return int(result['failed'])


if __name__ == '__main__':
    sys.exit(main())
