"""Measurement and provenance helpers; profiling is never used for speed numbers."""

from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from .model import LanguageModel


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'use a new empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def environment():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'pytorch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'packages': {},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob('*.py'))
                                if not p.name.startswith('._')}}
    for package in ('torch', 'numpy', 'flash-attn', 'flash-attn-3', 'flash-attn-4',
                    'nvidia-cutlass-dsl'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    if torch.cuda.is_available():
        result['gpus'] = [
            {'index': i, 'name': torch.cuda.get_device_name(i),
             'compute_capability': torch.cuda.get_device_capability(i),
             'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
        try:
            result['nvidia_smi'] = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired) as error:
            result['nvidia_smi_error'] = str(error)
    return result


def initialize(device, dtype, seed=1024, threads=4):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('this harness validates CPU or CUDA, not MPS')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        if device.index is None:
            device = torch.device('cuda', 0)
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 is not supported')
    elif dtype != torch.float32:
        raise ValueError('use float32 for the CPU CLI smoke tests')
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    index = int(position)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)


def measure(call, device, warmups, repeats):
    if warmups < 1 or repeats < 1:
        raise ValueError('warmups and repeats must be positive')
    for _ in range(warmups):
        value = call()
        del value
    synchronize(device)
    gc.collect()
    base = None
    if device.type == 'cuda':
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall, gpu = [], []
    for _ in range(repeats):
        if device.type == 'cuda':
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        synchronize(device)
        start = time.perf_counter()
        if device.type == 'cuda':
            begin.record()
        value = call()
        if device.type == 'cuda':
            end.record()
        synchronize(device)
        wall.append((time.perf_counter() - start) * 1000)
        if device.type == 'cuda':
            gpu.append(begin.elapsed_time(end))
        del value
    result = {'warmups': warmups, 'repeats': repeats, 'wall_ms': wall,
              'wall_median_ms': statistics.median(wall),
              'wall_p10_ms': percentile(wall, .1), 'wall_p90_ms': percentile(wall, .9),
              'cuda_event_ms': gpu, 'cuda_event_median_ms': statistics.median(gpu) if gpu else None,
              'baseline_allocated_bytes': base,
              'timing_note': 'Synchronized wall time includes host launch/allocation overhead. '
                             'CUDA Event spans can include idle launch gaps; not pure kernel time.'}
    if device.type == 'cuda':
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) - base,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def errors(actual, expected):
    delta = actual.detach().double() - expected.detach().double()
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm() / expected.detach().double().norm().clamp_min(1e-12)).item()}


def assert_match(actual, expected, dtype):
    atol, rtol = {torch.float32: (3e-5, 3e-4), torch.float16: (.015, .015),
                  torch.bfloat16: (.08, .08)}[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    result = errors(actual, expected)
    relative_limit = {torch.float32: 1e-3, torch.float16: .03, torch.bfloat16: .1}[dtype]
    if result['relative_l2'] > relative_limit:
        raise AssertionError(f'relative L2 {result["relative_l2"]} exceeds {relative_limit}')
    return {**result, 'atol': atol, 'rtol': rtol,
            'relative_l2_limit': relative_limit, 'passed': True}


class Workload:
    """One fixed-shape workload. Decode always starts at the same prefix length."""

    def __init__(self, config, variant, backend, phase, cache_mode, batch, context,
                 device, dtype, last_only=True, auxiliary_weight=0.01):
        if context < 2 or context > config.max_context or batch < 1:
            raise ValueError('need batch >= 1 and 2 <= context <= max_context')
        self.device, self.dtype, self.phase = device, dtype, phase
        self.last_only, self.auxiliary_weight = last_only, auxiliary_weight
        self.cache_mode, self.caches = cache_mode, None
        torch.manual_seed(1024)
        self.model = LanguageModel(config, variant, backend).to(device)
        self.tokens = torch.randint(config.vocab, (batch, context + 1), device=device)
        self.input = self.tokens[:, :context].contiguous()
        self.targets = self.tokens[:, 1:context + 1].contiguous()
        self.optimizer, self.scaler = None, None
        if phase == 'train':
            self.model.train()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=.01)
            self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
        else:
            self.model.to(dtype).eval()
            if phase == 'decode':
                self.input = self.tokens[:, context - 1:context].contiguous()
                with torch.inference_mode():
                    self.caches = self.model(self.tokens[:, :context - 1], cache_mode=cache_mode,
                                             capacity=context, last_only=True).caches
        self.metadata = {'config': asdict(config), 'variant': variant, 'backend': backend,
                         'phase': phase, 'cache_mode': cache_mode if phase != 'train' else None,
                         'batch': batch, 'context': context, 'dtype': str(dtype),
                         'parameters': sum(p.numel() for p in self.model.parameters()),
                         'lm_head': 'all positions' if phase == 'train' or not last_only else 'last position',
                         'inputs': 'fixed synthetic token IDs, not text quality evaluation',
                         'training_precision': 'FP32 parameters/AdamW; selected autocast dtype',
                         'train_timing': 'zero_grad + forward + CE/aux loss + backward + AdamW step',
                         'decode_timing': 'prefix construction excluded; visible KV length includes current token',
                         'prefill_timing': 'includes allocating and writing a fresh cache',
                         'optimization': 'eager PyTorch, no compile/CUDA Graph; MoE uses expert-loop gather/index_add'}
        if variant in ('moe', 'mla-moe'):
            self.metadata['moe'] = {
                'routed_experts': config.experts, 'top_k': config.top_k,
                'shared_experts': config.shared_experts,
                'routed_ffn_weight_parameters_total': config.layers * config.experts * 3 * config.width * config.expert_hidden,
                'routed_ffn_weight_parameters_per_token': config.layers * config.top_k * 3 * config.width * config.expert_hidden,
                'router_parameters': config.layers * config.width * config.experts,
                'note': 'These FFN-only counts exclude attention, embedding/head and shared experts. '
                        'Variants are not parameter-matched or quality-matched.'}

    def __call__(self):
        if self.phase == 'train':
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                output = self.model(self.input, targets=self.targets)
                loss = output.loss + self.auxiliary_weight * output.auxiliary_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return output
        with torch.inference_mode():
            if self.phase == 'decode':
                for cache in self.caches:
                    cache.rewind(self.metadata['context'] - 1)
                return self.model(self.input, caches=self.caches, last_only=self.last_only)
            return self.model(self.input, cache_mode=self.cache_mode,
                              capacity=self.metadata['context'], last_only=self.last_only)

    def observations(self, output):
        if not torch.isfinite(output.logits).all().item():
            raise AssertionError('nonfinite logits')
        result = {'logits_shape': list(output.logits.shape),
                  'logits_finite': True,
                  'cache_logical_bytes': sum(c.logical_bytes() for c in output.caches or []),
                  'cache_allocated_bytes': sum(c.allocated_bytes() for c in output.caches or [])}
        if output.loss is not None:
            if not torch.isfinite(output.loss).item():
                raise AssertionError('nonfinite loss')
            for parameter in self.model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise AssertionError('nonfinite gradient')
            result.update(loss=output.loss.item(), auxiliary_loss=output.auxiliary_loss.item(),
                          expert_assignments=[counts.cpu().tolist() for counts in output.expert_counts])
        return result

    def release(self):
        self.model, self.optimizer, self.scaler, self.caches = None, None, None, None
        self.tokens, self.input, self.targets = None, None, None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
