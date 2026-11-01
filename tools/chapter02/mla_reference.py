"""Single-device, unquantized MLA paths aligned with DeepSeek-V3's reference.

Adapted from DeepSeek-V3 inference/model.py, commit
b15f0dbbbe6a4bc403306175698439ef380f5fb5 (MIT, Copyright (c) 2023 DeepSeek).
See references/deepseek-v3/LICENSE-CODE. This is not FlashMLA or a V3 replica.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .attention import BackendUnavailable, causal_mask, scope


def rotary(x, start, theta):
    # Match the upstream complex FP32 rotation, casting back only at the end.
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    frequencies = 1. / theta ** (torch.arange(0, x.shape[-1], 2, device=x.device).float()
                                 / x.shape[-1])
    positions = torch.arange(start, start + x.shape[1], device=x.device).float()
    angles = torch.outer(positions, frequencies)
    rotations = torch.polar(torch.ones_like(angles), angles)[None, :, None]
    return torch.view_as_real(pairs * rotations).flatten(-2).to(x.dtype)


class MLAReference(nn.Module):
    """Same packed projections in both paths; cache mutation is inference-only."""

    def __init__(self, config, path='absorb', backend='math'):
        super().__init__()
        if path not in ('naive', 'absorb'):
            raise ValueError(path)
        if backend != 'math':
            raise BackendUnavailable('MLA reference uses explicit einsum, not a fused Attention backend')
        self.config, self.path, self.profile = config, path, False
        self.wq_a = nn.Linear(config.width, config.q_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_rank, eps=1e-6)
        self.wq_b = nn.Linear(config.q_rank, config.heads * (config.nope_dim + config.rope_dim), bias=False)
        self.wkv_a = nn.Linear(config.width, config.kv_rank + config.rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(config.kv_rank, eps=1e-6)
        self.wkv_b = nn.Linear(config.kv_rank, config.heads * (config.nope_dim + config.value_dim), bias=False)
        self.wo = nn.Linear(config.heads * config.value_dim, config.width, bias=False)

    def forward(self, x, start=0, cache=None):
        cfg = self.config
        with scope('mla.reference.projections_rope', self.profile):
            q = self.wq_b(self.q_norm(self.wq_a(x)))
            q = q.unflatten(-1, (cfg.heads, cfg.nope_dim + cfg.rope_dim))
            q_nope, q_pe = q.split((cfg.nope_dim, cfg.rope_dim), -1)
            q_pe = rotary(q_pe, start, cfg.rope_theta)
            c, k_pe = self.wkv_a(x).split((cfg.kv_rank, cfg.rope_dim), -1)
            c = self.kv_norm(c)
            k_pe = rotary(k_pe.unsqueeze(2), start, cfg.rope_theta)
        if self.path == 'naive':
            with scope('mla.reference.expand_new_kv', self.profile):
                kv = self.wkv_b(c).unflatten(-1, (cfg.heads, cfg.nope_dim + cfg.value_dim))
                k_nope, v = kv.split((cfg.nope_dim, cfg.value_dim), -1)
                k = torch.cat((k_nope, k_pe.expand(-1, -1, cfg.heads, -1)), -1)
                q = torch.cat((q_nope, q_pe), -1)
            if cache is not None:
                stored = cache.append(k=k.transpose(1, 2), v=v.transpose(1, 2))
                k, v = stored['k'].transpose(1, 2), stored['v'].transpose(1, 2)
            with scope('mla.reference.qk', self.profile):
                scores = torch.einsum('bshd,bthd->bsht', q, k)
        else:
            with scope('mla.reference.absorb_qk', self.profile):
                weight = self.wkv_b.weight.view(cfg.heads, cfg.nope_dim + cfg.value_dim, cfg.kv_rank)
                q_c = torch.einsum('bshd,hdc->bshc', q_nope, weight[:, :cfg.nope_dim])
            if cache is not None:
                stored = cache.append(latent=c.unsqueeze(1), rope=k_pe.transpose(1, 2))
                c, k_pe = stored['latent'].squeeze(1), stored['rope'].transpose(1, 2)
            with scope('mla.reference.qk', self.profile):
                scores = (torch.einsum('bshc,btc->bsht', q_c, c)
                          + torch.einsum('bshr,btr->bsht', q_pe, k_pe.squeeze(2)))
        with scope('mla.reference.fp32_softmax', self.profile):
            scores = scores * (cfg.nope_dim + cfg.rope_dim) ** -.5
            mask = causal_mask(x.shape[1], scores.shape[-1], x.device)
            scores = scores.masked_fill(~mask[:, None, :], float('-inf'))
            probabilities = scores.softmax(-1, dtype=torch.float32).to(x.dtype)
        with scope('mla.reference.pv', self.profile):
            if self.path == 'naive':
                output = torch.einsum('bsht,bthd->bshd', probabilities, v)
            else:
                output = torch.einsum('bsht,btc->bshc', probabilities, c)
                output = torch.einsum('bshc,hdc->bshd', output, weight[:, -cfg.value_dim:])
        with scope('mla.reference.output_projection', self.profile):
            return F.dropout(self.wo(output.flatten(-2)), cfg.dropout, self.training)
