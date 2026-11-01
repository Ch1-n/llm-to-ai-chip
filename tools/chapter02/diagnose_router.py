"""Measure native-precision full/chunked routing sensitivity, without changing tolerances."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch

from .config import preset
from .model import LanguageModel, TopKMoE
from .runtime import environment, errors, initialize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists')
    device = initialize('cuda', torch.bfloat16)
    torch.manual_seed(2026)
    tokens = torch.randint(50257, (1, 33), device=device)
    records = []
    for fp32 in (False, True):
        for variant in ('moe', 'mla-moe'):
            cfg = replace(preset('teaching'), router_float32=fp32)
            torch.manual_seed(1024)
            model = LanguageModel(cfg, variant).to(device=device, dtype=torch.bfloat16).eval()
            routed = {}

            def hook(name):
                def capture(module, inputs):
                    flat = inputs[0].reshape(-1, cfg.width)
                    probabilities, _, indices = module.route(flat)
                    sorted_probability = probabilities.sort(-1, descending=True).values
                    margin = sorted_probability[:, cfg.top_k - 1] - sorted_probability[:, cfg.top_k]
                    routed.setdefault(name, []).append((indices.sort(-1).values.cpu(), margin.cpu()))
                return capture

            handles = [m.register_forward_pre_hook(hook(name))
                       for name, m in model.named_modules() if isinstance(m, TopKMoE)]
            try:
                with torch.inference_mode():
                    full = model(tokens).logits
                    full_routes = {name: entries[0] for name, entries in routed.items()}
                    routed.clear()
                    caches, start, parts = None, 0, []
                    for count in (17, 1, 15):
                        output = model(tokens[:, start:start + count], caches=caches,
                                       cache_mode='static', capacity=33)
                        caches, start = output.caches, start + count
                        parts.append(output.logits)
                    layers = []
                    for name, entries in routed.items():
                        indices = torch.cat([entry[0] for entry in entries])
                        expected, margin = full_routes[name]
                        changed = (indices != expected).any(-1)
                        layers.append({'layer': name, 'changed_tokens': changed.sum().item(),
                                       'changed_positions': changed.nonzero().flatten().tolist(),
                                       'full_route_margin_at_changes': margin[changed].tolist()})
                    record = {'variant': variant, 'router_projection_fp32': fp32,
                              'logits': errors(torch.cat(parts, 1), full), 'layers': layers}
                    # Replay full-pass expert IDs only to isolate the source of
                    # the error. This oracle is never used for speed or quality claims.
                    originals = {}
                    for name, module in model.named_modules():
                        if not isinstance(module, TopKMoE):
                            continue
                        originals[name] = module.route

                        def forced(flat, name=name, route=module.route):
                            probability, _, _ = route(flat)
                            index = full_routes[name][0][start:start + flat.shape[0]].to(device)
                            selected = probability.gather(-1, index)
                            weight = (selected / selected.sum(-1, keepdim=True)).to(flat.dtype)
                            return probability, weight, index

                        module.route = forced
                    caches, start, parts = None, 0, []
                    for count in (17, 1, 15):
                        output = model(tokens[:, start:start + count], caches=caches,
                                       cache_mode='static', capacity=33)
                        caches, start = output.caches, start + count
                        parts.append(output.logits)
                    record['diagnostic_full_route_replay_logits'] = errors(torch.cat(parts, 1), full)
                    record['replay_note'] = 'Oracle expert IDs from full forward; diagnostic only, not executable inference optimization.'
                    for name, module in model.named_modules():
                        if name in originals:
                            module.route = originals[name]
                    records.append(record)
                    print(json.dumps(record), flush=True)
            finally:
                for handle in handles:
                    handle.remove()
            del model, full, output, caches, parts
            torch.cuda.empty_cache()
    write_json(args.output, {'environment': environment(), 'records': records,
                            'note': 'FP32 router projection reduces one source of rounding, '
                                    'but cannot guarantee identical routing for near-tied inputs.'})


if __name__ == '__main__':
    main()
