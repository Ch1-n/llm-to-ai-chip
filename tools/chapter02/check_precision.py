"""Additional native-precision, full-model cache and backend comparisons."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch

from .config import preset
from .model import LanguageModel, MLA, VARIANTS
from .runtime import assert_match, environment, errors, initialize, write_json


def check(device, dtype, variants=VARIANTS):
    cfg = replace(preset('teaching'), max_context=128)
    torch.manual_seed(2026)
    tokens = torch.randint(cfg.vocab, (1, 33), device=device)
    rows = []

    def compare(actual, expected, **labels):
        try:
            row = {**labels, **assert_match(actual, expected, dtype), 'status': 'passed'}
        except AssertionError as error:
            row = {**labels, **errors(actual, expected), 'status': 'not_close', 'reason': str(error)}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'reason'}), flush=True)

    for variant in variants:
        torch.manual_seed(1024)
        model = LanguageModel(cfg, variant).to(device=device, dtype=dtype).eval()
        with torch.inference_mode():
            full = model(tokens).logits
            for mode in ('cat', 'static'):
                cache, offset, parts = None, 0, []
                for length in (17, 1, 15):
                    result = model(tokens[:, offset:offset + length], caches=cache,
                                   cache_mode=mode, capacity=33)
                    cache, offset = result.caches, offset + length
                    parts.append(result.logits)
                compare(torch.cat(parts, 1), full, kind='native_precision_cached_full',
                        variant=variant, cache=mode)
                del result, cache, parts
            if variant == 'dense':
                for backend in ('head-loop', 'sdpa-math', 'sdpa-flash', 'fa2'):
                    if (device.type == 'cpu' or dtype == torch.float32) and backend in ('sdpa-flash', 'fa2'):
                        continue
                    for block in model.blocks:
                        block.attention.backend = backend
                    result = model(tokens)
                    compare(result.logits, full, kind='native_precision_dense_backend', backend=backend)
                    del result
            if variant == 'mla-expanded':
                for path in ('compressed', 'absorbed'):
                    for module in model.modules():
                        if isinstance(module, MLA):
                            module.path = path
                    result = model(tokens)
                    compare(result.logits, full, kind='native_precision_mla_paths', path=path)
                    del result
        del model, full
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    dtype = getattr(torch, args.dtype)
    device = initialize(args.device, dtype)
    rows = check(device, dtype, args.variants)
    status = 'not_close' if any(row['status'] != 'passed' for row in rows) else 'passed'
    write_json(args.output, {'environment': environment(), 'records': rows, 'status': status,
                            'note': 'Teaching parameter sizes with a 33-token correctness input; '
                                    'not long-context validation or a text-quality evaluation.'})
    print(json.dumps({'checks': len(rows), 'status': status}))
    return int(status != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
