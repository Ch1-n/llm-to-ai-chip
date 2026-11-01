"""Explicit attention dispatch with lower-right causal alignment for cached queries."""

from contextlib import nullcontext
import importlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BACKENDS = ('head-loop', 'math', 'sdpa-math', 'sdpa-flash', 'fa2', 'fa3', 'fa4')


class BackendUnavailable(RuntimeError):
    """A requested backend cannot run; do not silently benchmark a fallback."""


def scope(name, enabled=False):
    return torch.profiler.record_function(name) if enabled else nullcontext()


def causal_mask(q_len, kv_len, device):
    if not 0 < q_len <= kv_len:
        raise ValueError('expected 0 < q_len <= kv_len')
    queries = torch.arange(q_len, device=device) + kv_len - q_len
    return torch.arange(kv_len, device=device)[None, :] <= queries[:, None]


def import_flash(name):
    candidates = {'fa2': ('flash_attn',),
                  'fa3': ('flash_attn_3.flash_attn_interface', 'flash_attn_interface'),
                  'fa4': ('flash_attn.cute',)}[name]
    failures = []
    for module in candidates:
        try:
            return importlib.import_module(module).flash_attn_func
        except (ImportError, AttributeError, OSError) as error:
            failures.append(f'{module}: {error}')
    raise BackendUnavailable('; '.join(failures))


def attention(q, k, v, backend='math', dropout=0.0, profile=False, scale=None):
    """Q/K/V are [batch, heads, tokens, channels]; V channels may differ.

    The queries represent the final q_len positions of the key sequence. Boolean
    SDPA masks use True for visible positions, unlike masked_fill's mask.
    """
    if backend not in BACKENDS:
        raise ValueError(f'unknown backend: {backend}')
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError('Q/K/V must be rank four')
    if (q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]):
        raise ValueError('incompatible Q/K/V shapes')
    if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
        raise ValueError('Q/K/V need the same precision and device')
    if not 0 < q.shape[2] <= k.shape[2]:
        raise ValueError('invalid query/key sequence lengths')
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if backend in ('fa2', 'sdpa-flash') and q.shape[-1] != v.shape[-1]:
        raise BackendUnavailable(f'{backend}: unequal QK/V dimensions are not enabled in this harness')
    if backend in ('fa2', 'fa3', 'fa4'):
        if q.device.type != 'cuda' or q.dtype not in (torch.float16, torch.bfloat16):
            raise BackendUnavailable(f'{backend} requires CUDA FP16/BF16 in this experiment')
        if backend == 'fa3' and torch.cuda.get_device_capability(q.device)[0] != 9:
            raise BackendUnavailable('official FA3 requires Hopper SM90')
        if backend in ('fa3', 'fa4') and dropout:
            raise BackendUnavailable('FA3/FA4 dropout is not enabled in this harness')
        function = import_flash(backend)
        kwargs = {'causal': True, 'softmax_scale': scale}
        if backend == 'fa2':
            kwargs['dropout_p'] = dropout
        with scope(f'attention.{backend}', profile):
            result = function(q.transpose(1, 2).contiguous(),
                              k.transpose(1, 2).contiguous(),
                              v.transpose(1, 2).contiguous(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        return result.transpose(1, 2)

    if k.shape[1] != q.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k, v = (x.repeat_interleave(repeats, dim=1) for x in (k, v))
    square = q.shape[2] == k.shape[2]
    mask = None if square else causal_mask(q.shape[2], k.shape[2], q.device)
    if backend.startswith('sdpa-'):
        if backend == 'sdpa-flash' and q.device.type != 'cuda':
            raise BackendUnavailable('sdpa-flash requires CUDA')
        if backend == 'sdpa-flash' and q.dtype not in (torch.float16, torch.bfloat16):
            raise BackendUnavailable('sdpa-flash requires FP16/BF16 in this harness')
        if backend == 'sdpa-flash' and not square and q.shape[2] > 1:
            raise BackendUnavailable('sdpa-flash cached multi-token mask is not enabled in this harness')
        selected = SDPBackend.MATH if backend == 'sdpa-math' else SDPBackend.FLASH_ATTENTION
        with sdpa_kernel(selected), scope(f'attention.{backend}', profile):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None if q.shape[2] == 1 else mask, dropout_p=dropout,
                is_causal=square, scale=scale)

    mask = causal_mask(q.shape[2], k.shape[2], q.device) if square else mask

    def eager(a, b, c):
        with scope('attention.qk', profile):
            scores = (a * scale) @ b.transpose(-2, -1)
        with scope('attention.softmax', profile):
            scores = scores.masked_fill(~mask, float('-inf'))
            probability = scores.softmax(dim=-1)
            probability = F.dropout(probability, p=dropout, training=dropout > 0)
        with scope('attention.pv', profile):
            return probability @ c

    if backend == 'head-loop':
        return torch.cat([eager(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1])
                          for h in range(q.shape[1])], dim=1)
    return eager(q, k, v)
