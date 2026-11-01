from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import statistics
import time

import torch

from tools.chapter02.runtime import environment as old_environment, percentile
from tools.chapter02.model import LanguageModel as OldModel
from tools.chapter02.config import preset as old_preset
from .config import preset
from .model import LanguageModel


def environment():
    result = old_environment()
    result['chapter02_source_sha256'] = result.pop('source_sha256')
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob('*.py'))}
    refs = Path(__file__).resolve().parents[2] / 'references/deepseek-v4'
    result['reference_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(refs.glob('*')) if p.is_file()}
    result['reference_scope'] = 'V4 Flash unquantized attention only; no full V4 weights/MoE/mHC/MTP/Muon'
    return result


def cache_stats(caches):
    if not caches:
        return {'logical_bytes': 0, 'allocated_storage_bytes': 0, 'layers': []}
    if hasattr(caches[0], 'accounting'):
        rows = [c.accounting() for c in caches]
        return {'logical_bytes': sum(r['logical_bytes'] for r in rows),
                'allocated_storage_bytes': sum(r['allocated_storage_bytes'] for r in rows), 'layers': rows}
    return {'logical_bytes': sum(c.logical_bytes() for c in caches),
            'allocated_storage_bytes': sum(c.allocated_bytes() for c in caches), 'layers': []}


def clone_cache_positions(caches):
    # History rows before the prefix boundary never change during a one-token decode.
    # Tail/local arrays are replaced, not modified in-place; no history data copy is needed.
    result = []
    for cache in caches:
        item = copy.copy(cache)
        if hasattr(item, 'main'):
            item.main, item.index = copy.copy(cache.main), copy.copy(cache.index)
        result.append(item)
    return result


class Workload:
    def __init__(self, variant, phase, batch, context, device, dtype, preset_name='teaching'):
        torch.manual_seed(1024)
        self.old = variant in ('mha', 'mla')
        self.phase, self.context = phase, context
        if self.old:
            cfg = old_preset(preset_name)
            self.model = OldModel(cfg, 'dense' if variant == 'mha' else 'mla-reference-absorb', 'math')
        else:
            cfg = preset(preset_name, variant)
            self.model = LanguageModel(cfg)
        if not 2 <= context <= cfg.max_context:
            raise ValueError('context outside configuration')
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.tokens = torch.randint(cfg.vocab, (batch, context), device=device)
        self.prefix = self.caches = None
        with torch.inference_mode():
            if phase == 'decode':
                if self.old:
                    output = self.model(self.tokens[:, :-1], cache_mode='static', capacity=context, last_only=True)
                else:
                    output = self.model(self.tokens[:, :-1], capacity=context, last_only=True)
                self.prefix = clone_cache_positions(output.caches)
        self.metadata = {'variant': variant, 'phase': phase, 'batch': batch, 'context': context,
            'config': asdict(cfg), 'dtype': str(dtype), 'parameters': sum(p.numel() for p in self.model.parameters()),
            'lm_head': 'last position only', 'weights': 'random seed=1024; no quality-matched architectures',
            'implementation': 'eager PyTorch math; hybrid uses query tiles and stable argsort, not FlashMLA',
            'decode_scope': 'prefix build and cache-position restoration excluded for all variants',
            'prefill_scope': 'includes fresh cache allocation and population',
            'precision': 'native parameter dtype; compression and softmax accumulate in FP32'}

    def prepare(self):
        if self.phase == 'decode':
            self.caches = clone_cache_positions(self.prefix)

    def __call__(self):
        with torch.inference_mode():
            if self.phase == 'decode':
                return self.model(self.tokens[:, -1:], caches=self.caches, last_only=True)
            if self.old:
                return self.model(self.tokens, cache_mode='static', capacity=self.context, last_only=True)
            return self.model(self.tokens, capacity=self.context, last_only=True)


def measure(workload, device, warmups=3, repeats=10):
    for _ in range(warmups):
        workload.prepare()
        result = workload()
        del result
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples, events = [], []
    for _ in range(repeats):
        workload.prepare()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        result = workload()
        if device.type == 'cuda':
            end.record()
            end.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        if device.type == 'cuda':
            events.append(begin.elapsed_time(end))
        stats = cache_stats(result.caches)
        if not torch.isfinite(result.logits).all().item():
            raise ValueError('nonfinite benchmark output')
        del result
    return {'wall_ms': samples, 'wall_median_ms': statistics.median(samples),
            'wall_p10_ms': percentile(samples, .1), 'wall_p90_ms': percentile(samples, .9),
            'cuda_event_ms': events,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'warmups': warmups, 'repeats': repeats, 'cache': stats,
            'timing_note': 'synchronized wall time; CUDA Events include launch gaps, not pure kernel work'}
