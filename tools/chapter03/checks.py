from dataclasses import asdict
import json

import torch

from tools.chapter02.runtime import assert_match
from .attention import Attention, Cache, gathered_attention
from .config import preset
from .model import LanguageModel
from .reference import arguments, dense_oracle, load


def official_check(cfg, ratio, device):
    module = load()
    original = module.Attention(cfg.ratios.index(ratio), arguments(module, cfg)).float().to(device).eval()
    ours = Attention(cfg, ratio).to(device).eval()
    ours.apply(LanguageModel._init)
    original.load_state_dict(ours.state_dict())
    x = torch.randn(1, 137, cfg.width, device=device)
    cache = Cache(cfg.max_context)
    errors = []
    with torch.inference_mode(), torch.device(device):
        errors.append(assert_match(ours(x[:, :129], cache)[0], original(x[:, :129], 0), torch.float32))
        for pos in range(129, 137):
            errors.append(assert_match(ours(x[:, pos:pos + 1], cache)[0],
                                       original(x[:, pos:pos + 1], pos), torch.float32))
        if ratio:
            assert_match(cache.main.history[:, :cache.main.count],
                         original.kv_cache[:, cfg.window:cfg.window + cache.main.count], torch.float32)
            if ratio == 4:
                assert_match(cache.index.history[:, :cache.index.count],
                             original.indexer.kv_cache[:, :cache.index.count], torch.float32)
    return {'kind': 'original_v4_unquantized_attention_output_cache', 'ratio': ratio,
            'precision': 'float32', 'status': 'passed', 'comparisons': errors,
            'reference_adapters': ['quantization disabled', 'orthogonal Hadamard omitted',
                                   'stable tied-score ordering', 'Transformers dense eager core']}


def core_check(device):
    q = torch.randn(2, 7, 4, 16, device=device, requires_grad=True)
    kv = torch.randn(2, 17, 16, device=device, requires_grad=True)
    sink = torch.randn(4, device=device, requires_grad=True)
    ids = torch.tensor([[[0, 2, -1], [1, 3, -1], [0, 4, 8], [2, 5, 6], [3, 8, 9],
                         [4, 5, 11], [6, 10, 16]]] * 2, device=device)
    a, b = gathered_attention(q, kv, ids, sink, .25), dense_oracle(q, kv, sink, ids, .25)
    grads_a = torch.autograd.grad(a.square().sum(), (q, kv, sink), retain_graph=True)
    grads_b = torch.autograd.grad(b.square().sum(), (q, kv, sink))
    return {'kind': 'gathered_core_vs_transformers_output_and_gradients', 'status': 'passed',
            'output': assert_match(a, b, torch.float32),
            'gradients': [assert_match(x, y, torch.float32) for x, y in zip(grads_a, grads_b)]}


def model_check(cfg, device, dtype, variant):
    model = LanguageModel(cfg).to(device=device, dtype=dtype).eval()
    tokens = torch.randint(cfg.vocab, (1, 137), device=device)
    with torch.inference_mode():
        full = model(tokens).logits
        result = model(tokens[:, :3], capacity=137)
        outputs, offset = [result.logits], 3
        for size in (1, 123, 1, 1, 8):
            result = model(tokens[:, offset:offset + size], caches=result.caches)
            outputs.append(result.logits)
            offset += size
        comparison = assert_match(torch.cat(outputs, 1), full, dtype)
        changed = tokens.clone()
        changed[:, 126:] = (changed[:, 126:] + 7) % cfg.vocab
        causal = assert_match(model(changed).logits[:, :126], full[:, :126], dtype)
    return {'kind': 'full_model_chunked_cache_and_causality', 'variant': variant,
            'dtype': str(dtype), 'status': 'passed', 'cache_error': comparison, 'causality': causal}


def run(device, teaching=False):
    audit = preset('tiny')
    rows = []

    def attempt(label, function):
        try:
            row = function()
        except Exception as error:
            row = {**label, 'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k in ('kind', 'status', 'ratio', 'dtype', 'variant', 'reason')}), flush=True)

    for ratio in (0, 4, 128):
        attempt({'kind': 'official', 'ratio': ratio}, lambda r=ratio: official_check(audit, r, device))
    attempt({'kind': 'core'}, lambda: core_check(device))
    for variant in ('swa', 'hybrid'):
        for dtype in ((torch.float32, torch.bfloat16) if device.type == 'cuda' else (torch.float32,)):
            cfg = preset('teaching' if teaching else 'tiny', variant)
            attempt({'kind': 'model', 'variant': variant, 'dtype': str(dtype)},
                    lambda c=cfg, d=dtype, v=variant: model_check(c, device, d, v))
    return {'status': 'passed' if all(r['status'] == 'passed' for r in rows) else 'failed',
            'records': rows, 'model_preset': 'teaching' if teaching else 'tiny',
            'scope': 'unquantized V4-derived attention; no complete pretrained model equivalence'}
