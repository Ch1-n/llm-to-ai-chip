"""Record trained MoE routing differences without forcing expert decisions."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import LanguageModel, TopKMoE
from .runtime import assert_match, environment, errors, fresh_directory, initialize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fresh_directory(args.output)
    device = initialize('cuda', torch.float32)
    value = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    with (args.data / 'val.bin').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['token_bin_sha256']['val']:
        raise ValueError('validation hash mismatch')
    array = np.memmap(args.data / 'val.bin', mode='r', dtype=np.uint16)
    x = torch.tensor(np.asarray(array[:256], dtype=np.int64), device=device)[None]
    records = []
    for name, dtype, amp_enabled, fp32_router in (
        ('fp32', torch.float32, False, False), ('native-bf16', torch.bfloat16, False, False),
        ('bf16-autocast', torch.float32, True, False),
        ('native-bf16-fp32-router', torch.bfloat16, False, True)):
        cfg = replace(ModelConfig(**value['config']), router_float32=fp32_router)
        model = LanguageModel(cfg, value['variant'], 'math').to(device=device, dtype=dtype).eval()
        model.load_state_dict(value['model'])
        captured = {}

        def hook(label):
            def capture(module, inputs):
                flat = inputs[0].reshape(-1, cfg.width)
                prob, _, ids = module.route(flat)
                sorted_prob = prob.sort(-1, descending=True).values
                margin = sorted_prob[:, cfg.top_k - 1] - sorted_prob[:, cfg.top_k]
                captured.setdefault(label, []).append((ids.sort(-1).values.cpu(), margin.cpu()))
            return capture

        handles = [module.register_forward_pre_hook(hook(label)) for label, module in model.named_modules()
                   if isinstance(module, TopKMoE)]
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with torch.inference_mode(), amp:
            full = model(x).logits
            expected = {label: entries[0] for label, entries in captured.items()}
            captured.clear()
            caches, parts, left = None, [], 0
            for size in (127, 1, 128):
                output = model(x[:, left:left + size], caches=caches, cache_mode='static', capacity=256)
                caches = output.caches
                parts.append(output.logits)
                left += size
            actual = torch.cat(parts, 1)
            try:
                comparison = assert_match(actual, full, torch.bfloat16 if amp_enabled else dtype)
            except AssertionError as error:
                comparison = {**errors(actual, full), 'passed': False, 'reason': str(error)[:800]}
            layers = []
            for label, entries in captured.items():
                ids = torch.cat([e[0] for e in entries])
                changed = (ids != expected[label][0]).any(-1)
                layers.append({'layer': label, 'changed_tokens': changed.sum().item(),
                               'positions': changed.nonzero().flatten().tolist(),
                               'full_margin_at_changes': expected[label][1][changed].float().tolist()})
        for handle in handles:
            handle.remove()
        records.append({'precision': name, 'logits': comparison, 'layers': layers})
        print(json.dumps({'precision': name, 'passed': comparison['passed'],
                          'route_changes': sum(r['changed_tokens'] for r in layers)}), flush=True)
        del model, full, output, caches, actual, parts
        torch.cuda.empty_cache()
    with args.checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(args.output / 'trained-moe-check.json', {'records': records,
        'checkpoint_sha256': checkpoint_sha, 'val_bin_sha256': digest,
        'scope': 'one held-out packed text of 256 tokens, trained checkpoint; compare full vs 127/1/128; '
                 'no forced routes, no blanket low-precision equivalence claim'})
    write_json(args.output / 'environment.json', environment())


if __name__ == '__main__':
    main()
