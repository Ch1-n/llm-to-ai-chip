"""Unquantized V4 attention equations with bounded local/state caches.

Adapted from DeepSeek-V4's MIT-licensed inference reference, revision
60d8d70770c6776ff598c94bb586a859a38244f1. See references/deepseek-v4/LICENSE.
No FP8/FP4, Hadamard rotation, YaRN or optimized sparse kernel is used.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def scope(name, enabled):
    return torch.profiler.record_function(name) if enabled else nullcontext()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)


def rotary(x, positions, theta, rope_dim, inverse=False):
    tail = x[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    pairs = torch.view_as_complex(tail.contiguous())
    freq = theta ** (-torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim)
    angles = positions.float()[:, None] * freq[None]
    rotations = torch.polar(torch.ones_like(angles), -angles if inverse else angles)
    if x.ndim == 4:
        rotations = rotations[None, :, None]
    else:
        rotations = rotations[None]
    rotated = torch.view_as_real(pairs * rotations).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-rope_dim], rotated), -1)


@dataclass
class CompressionState:
    capacity: int
    history: torch.Tensor | None = None
    count: int = 0
    tail_values: torch.Tensor | None = None
    tail_scores: torch.Tensor | None = None
    previous_values: torch.Tensor | None = None
    previous_scores: torch.Tensor | None = None


@dataclass
class Cache:
    capacity: int
    length: int = 0
    raw: torch.Tensor | None = None
    main: CompressionState | None = None
    index: CompressionState | None = None

    def tensors(self):
        if self.raw is not None:
            yield 'raw', self.raw, self.raw.numel() * self.raw.element_size()
        for label, state in (('compressed', self.main), ('indexer', self.index)):
            if state is None:
                continue
            for name in ('history', 'tail_values', 'tail_scores', 'previous_values', 'previous_scores'):
                value = getattr(state, name)
                if value is not None:
                    size = value.numel() * value.element_size()
                    if name == 'history':
                        size = value.shape[0] * state.count * value.shape[-1] * value.element_size()
                    yield label + '.' + name, value, size

    def accounting(self):
        fields, storage = {}, {}
        for name, tensor, logical in self.tensors():
            fields[name] = logical
            s = tensor.untyped_storage()
            storage[(str(tensor.device), s.data_ptr())] = s.nbytes()
        return {'logical_bytes': sum(fields.values()), 'allocated_storage_bytes': sum(storage.values()),
                'fields': fields, 'length': self.length}


class Compressor(nn.Module):
    def __init__(self, cfg, ratio, dim):
        super().__init__()
        self.cfg, self.ratio, self.dim = cfg, ratio, dim
        self.overlap = ratio == 4
        channels = dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(cfg.width, channels, bias=False)
        self.wgate = nn.Linear(cfg.width, channels, bias=False)
        self.ape = nn.Parameter(torch.zeros(ratio, channels))
        self.norm = RMSNorm(dim, cfg.eps)

    def forward(self, x, start, dtype, state=None):
        ratio, dim = self.ratio, self.dim
        # Both projections and the per-channel pooling weights are computed in FP32.
        with torch.autocast(x.device.type, enabled=False):
            values = F.linear(x.float(), self.wkv.weight.float())
            scores = F.linear(x.float(), self.wgate.weight.float())
            count = 0 if state is None else state.count
            if state is not None and state.tail_values is not None:
                values = torch.cat((state.tail_values, values), 1)
                scores = torch.cat((state.tail_scores, scores), 1)
            blocks = values.shape[1] // ratio
            cutoff = blocks * ratio
            if blocks:
                v = values[:, :cutoff].unflatten(1, (blocks, ratio))
                s = scores[:, :cutoff].unflatten(1, (blocks, ratio)) + self.ape.float()
                if self.overlap:
                    if state is not None and state.previous_values is not None:
                        first_v, first_s = state.previous_values, state.previous_scores
                    else:
                        first_v = v.new_zeros(x.shape[0], 1, ratio, dim)
                        first_s = s.new_full((x.shape[0], 1, ratio, dim), float('-inf'))
                    previous_v = torch.cat((first_v, v[:, :-1, :, :dim]), 1)
                    previous_s = torch.cat((first_s, s[:, :-1, :, :dim]), 1)
                    pooled_v = torch.cat((previous_v, v[..., dim:]), 2)
                    pooled_s = torch.cat((previous_s, s[..., dim:]), 2)
                else:
                    pooled_v, pooled_s = v, s
                compressed = self.norm((pooled_v * pooled_s.softmax(2)).sum(2).to(dtype))
                positions = (torch.arange(blocks, device=x.device) + count) * ratio
                compressed = rotary(compressed, positions, self.cfg.compress_rope_theta, self.cfg.rope_dim)
            else:
                compressed = x.new_empty(x.shape[0], 0, dim, dtype=dtype)
            if state is None:
                return compressed
            if torch.is_grad_enabled():
                raise ValueError('compression state is inference-only')
            if count != start // ratio or count + blocks > state.capacity:
                raise ValueError('inconsistent compression position/capacity')
            if state.history is None:
                state.history = compressed.new_empty(x.shape[0], state.capacity, dim)
            if blocks:
                state.history[:, count:count + blocks].copy_(compressed)
                if self.overlap:
                    state.previous_values = v[:, -1:, :, :dim].clone()
                    state.previous_scores = s[:, -1:, :, :dim].clone()
            state.tail_values = values[:, cutoff:].clone()
            state.tail_scores = scores[:, cutoff:].clone()
            state.count += blocks
            return state.history[:, :state.count]


class Indexer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wq_b = nn.Linear(cfg.q_rank, cfg.index_heads * cfg.index_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.width, cfg.index_heads, bias=False)
        self.compressor = Compressor(cfg, 4, cfg.index_dim)

    def queries(self, x, qr, positions):
        cfg = self.cfg
        q = self.wq_b(qr).unflatten(-1, (cfg.index_heads, cfg.index_dim))
        q = rotary(q, positions, cfg.compress_rope_theta, cfg.rope_dim)
        weights = self.weights_proj(x) * (cfg.index_dim ** -.5 * cfg.index_heads ** -.5)
        return q, weights

    def scores(self, q, weights, keys, visible):
        score = torch.einsum('bqhd,bnd->bqhn', q, keys)
        score = (score.relu() * weights.unsqueeze(-1)).sum(2).float()
        return score.masked_fill(~visible[None], float('-inf'))


def gathered_attention(q, pool, indices, sink, scale):
    """Indices [B,Q,K], Q [B,Q,H,D]. Invalid slots never enter the softmax."""
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = pool[batch, indices.clamp_min(0)]
    scores = torch.einsum('bqhd,bqkd->bqhk', q, selected).float() * scale
    scores = scores.masked_fill(indices[:, :, None] < 0, float('-inf'))
    sink_logits = sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sink_logits), -1).softmax(-1)[..., :-1].to(q.dtype)
    return torch.einsum('bqhk,bqkd->bqhd', probabilities, selected)


def dense_shared_attention(q, pool, positions, raw_start, sink, scale):
    scores = torch.einsum('bqhd,btd->bqht', q, pool).float() * scale
    visible = torch.arange(raw_start, raw_start + pool.shape[1], device=q.device)[None] <= positions[:, None]
    scores = scores.masked_fill(~visible[None, :, None], float('-inf'))
    sinks = sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
    p = torch.cat((scores, sinks), -1).softmax(-1)[..., :-1].to(q.dtype)
    return torch.einsum('bqht,btd->bqhd', p, pool)


class Attention(nn.Module):
    def __init__(self, cfg, ratio):
        super().__init__()
        self.cfg, self.ratio, self.profile = cfg, ratio, False
        self.attn_sink = nn.Parameter(torch.zeros(cfg.heads))
        self.wq_a = nn.Linear(cfg.width, cfg.q_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_rank, cfg.eps)
        self.wq_b = nn.Linear(cfg.q_rank, cfg.heads * cfg.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.width, cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.wo_a = nn.Linear(cfg.heads * cfg.head_dim // cfg.o_groups, cfg.o_groups * cfg.o_rank, bias=False)
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_rank, cfg.width, bias=False)
        if ratio:
            self.compressor = Compressor(cfg, ratio, cfg.head_dim)
        self.indexer = Indexer(cfg) if ratio == 4 else None
        self.last_stats = {}

    def forward(self, x, cache=None, collect=False):
        cfg, ratio = self.cfg, self.ratio
        start = 0 if cache is None else cache.length
        if start + x.shape[1] > cfg.max_context:
            raise ValueError('context capacity exceeded')
        if cache is not None and start + x.shape[1] > cache.capacity:
            raise ValueError('cache capacity exceeded')
        if cache is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError('hybrid cache requires eval and no_grad/inference_mode')
        positions = torch.arange(start, start + x.shape[1], device=x.device)
        theta = cfg.compress_rope_theta if ratio else cfg.rope_theta
        with scope('hybrid.projections_rope', self.profile):
            qr = self.q_norm(self.wq_a(x))
            q = self.wq_b(qr).unflatten(-1, (cfg.heads, cfg.head_dim))
            q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.eps)).to(q.dtype)
            q = rotary(q, positions, theta, cfg.rope_dim)
            raw = rotary(self.kv_norm(self.wkv(x)), positions, theta, cfg.rope_dim)
        with scope('hybrid.cache_local', self.profile):
            old = None if cache is None else cache.raw
            raw_start = start - (0 if old is None else old.shape[1])
            local_pool = raw if old is None else torch.cat((old, raw), 1)
        compressed = index_keys = None
        if ratio:
            with scope('hybrid.compress_kv', self.profile):
                if cache is not None and cache.main is None:
                    cache.main = CompressionState(cache.capacity // ratio)
                compressed = self.compressor(x, start, raw.dtype, None if cache is None else cache.main)
            if self.indexer is not None:
                with scope('hybrid.indexer_compress', self.profile):
                    if cache is not None and cache.index is None:
                        cache.index = CompressionState(cache.capacity // ratio)
                    index_keys = self.indexer.compressor(x, start, raw.dtype,
                                                        None if cache is None else cache.index)
                with scope('hybrid.indexer_queries', self.profile):
                    iq, iw = self.indexer.queries(x, qr, positions)
        pool = local_pool if compressed is None else torch.cat((local_pool, compressed), 1)
        parts, aux_sum, aux_count, selections = [], x.new_zeros((), dtype=torch.float32), 0, []
        window = min(cfg.window, start + x.shape[1])
        for left in range(0, x.shape[1], cfg.query_tile):
            right = min(left + cfg.query_tile, x.shape[1])
            pos = positions[left:right]
            if not ratio and cfg.window == cfg.max_context:
                with scope('hybrid.dense_shared_core', self.profile):
                    parts.append(dense_shared_attention(q[:, left:right], pool, pos, raw_start,
                                                        self.attn_sink, cfg.head_dim ** -.5))
                continue
            with scope('hybrid.local_indices', self.profile):
                idx = pos[:, None] - torch.arange(window - 1, -1, -1, device=x.device)[None]
                idx = torch.where(idx < raw_start, -1, idx - raw_start)
                idx = idx[None].expand(x.shape[0], -1, -1)
            if ratio and compressed.shape[1]:
                visible = (torch.arange(compressed.shape[1], device=x.device)[None] + 1) * ratio <= pos[:, None] + 1
                if self.indexer is not None:
                    with scope('hybrid.indexer_score_topk', self.profile):
                        scores = self.indexer.scores(iq[:, left:right], iw[:, left:right], index_keys, visible)
                        # Stable ties prefer older block IDs, independent of future masked slots.
                        chosen = scores.argsort(dim=-1, descending=True, stable=True)[..., :cfg.index_topk]
                        valid = visible[None].expand(x.shape[0], -1, -1).gather(-1, chosen)
                        selected = torch.where(valid, chosen + local_pool.shape[1], -1)
                    if self.training:
                        # A teaching distillation objective gives the discrete indexer a gradient.
                        # It is not a reproduction of DeepSeek's full indexer-training recipe.
                        with scope('hybrid.indexer_auxiliary_training', self.profile):
                            teacher_logits = torch.einsum('bqhd,bnd->bqhn', q[:, left:right], compressed).float() * cfg.head_dim ** -.5
                            allowed = visible.any(-1)
                            if allowed.any():
                                teacher_logits = teacher_logits[:, allowed].masked_fill(~visible[allowed][None, :, None], float('-inf'))
                                target = teacher_logits.softmax(-1).mean(2).detach()
                                student = scores[:, allowed].log_softmax(-1)
                                term = torch.where(visible[allowed][None], target * (target.clamp_min(1e-30).log() - student), 0.)
                                aux_sum = aux_sum + term.sum()
                                aux_count += x.shape[0] * int(allowed.sum())
                else:
                    selected = torch.arange(compressed.shape[1], device=x.device)[None, None].expand(x.shape[0], len(pos), -1)
                    selected = torch.where(visible[None], selected + local_pool.shape[1], -1)
                idx = torch.cat((idx, selected), -1)
            if collect:
                selections.append(idx.detach())
            with scope('hybrid.gather_attention_sink', self.profile):
                parts.append(gathered_attention(q[:, left:right], pool, idx, self.attn_sink, cfg.head_dim ** -.5))
        with scope('hybrid.inverse_rope_output', self.profile):
            out = rotary(torch.cat(parts, 1), positions, theta, cfg.rope_dim, inverse=True)
            out = out.unflatten(-2, (cfg.o_groups, cfg.heads // cfg.o_groups)).flatten(-2)
            weight = self.wo_a.weight.view(cfg.o_groups, cfg.o_rank, -1)
            out = torch.einsum('bsgd,grd->bsgr', out, weight).flatten(-2)
            out = self.wo_b(out)
        if cache is not None:
            cache.raw = local_pool[:, -cfg.window:].clone()
            cache.length = start + x.shape[1]
        self.last_stats = {'ratio': ratio, 'compressed_entries': 0 if compressed is None else compressed.shape[1],
                           'local_history_entries': min(cfg.window, start + x.shape[1])}
        if collect:
            self.last_stats['indices'] = torch.cat(selections, 1)
            self.last_stats['pool'] = pool.detach()
        return out, aux_sum / max(aux_count, 1)
