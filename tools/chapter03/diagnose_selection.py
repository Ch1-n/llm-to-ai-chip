"""Observe trained full/cached Indexer choices without forcing any selection."""

import argparse
import json
from pathlib import Path

import torch

from tools.chapter02.runtime import fresh_directory, initialize, write_json
from .runtime import environment
from .supplement import checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, default=2048)
    args = parser.parse_args()
    fresh_directory(args.output)
    device = initialize('cuda', torch.float32)
    model, meta = checkpoint(args.checkpoint, device)
    captured = {}
    for number, block in enumerate(model.blocks):
        attention = block.attention
        if attention.indexer is None:
            continue
        original = attention.forward

        def tracked(x, cache=None, collect=False, original=original, attention=attention, number=number):
            value = original(x, cache, collect=True)
            stats = attention.last_stats
            local = stats['pool'].shape[1] - stats['compressed_entries']
            ids = stats['indices'][..., -model.cfg.index_topk:]
            ids = torch.where(ids >= local, ids - local, -1).sort(-1).values
            captured.setdefault(number, []).append(ids.cpu())
            return value

        attention.forward = tracked
    torch.manual_seed(2026 + args.context)
    x = torch.randint(model.cfg.vocab, (1, args.context), device=device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = model(x).logits[:, -130:].float().cpu()
        expected = {i: entries[0][:, -130:] for i, entries in captured.items()}
        captured.clear()
        output = model(x[:, :-130], capacity=args.context, last_only=True)
        captured.clear()
        pieces = []
        for pos in range(args.context - 130, args.context):
            output = model(x[:, pos:pos + 1], caches=output.caches, last_only=True)
            pieces.append(output.logits.float().cpu())
        actual = torch.cat(pieces, 1)
    layers = []
    for i, entries in captured.items():
        ids = torch.cat(entries, 1)
        changed = (ids != expected[i]).any(-1)[0]
        layers.append({'layer': i, 'changed_tokens': changed.sum().item(),
                       'positions_zero_based': (changed.nonzero().flatten() + args.context - 130).tolist(),
                       'full_ids_at_changes': expected[i][0, changed].tolist(),
                       'cached_ids_at_changes': ids[0, changed].tolist()})
    delta = (actual - full).abs()
    write_json(args.output / 'selection-diagnostic.json', {**meta, 'context': args.context,
        'layers': layers, 'max_abs_by_last_130_tokens': delta.amax(-1)[0].tolist(),
        'out_of_original_elementwise_tolerance_count': (delta > .08 + .08 * full.abs()).sum().item(),
        'scope': 'same trained checkpoint/input as failed BF16 AMP check; no forced selections; '
                 'correlation with route changes does not isolate all causes'})
    write_json(args.output / 'environment.json', environment())
    print(json.dumps(layers), flush=True)


if __name__ == '__main__':
    main()
