"""Readable Dense/MLA/MoE models using standard PyTorch operators.

MLA is a small mathematical demonstrator with decoupled RoPE, not a faithful
DeepSeek checkpoint implementation. All three MLA paths share parameter names.
"""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .attention import attention, scope
from .cache import KVCache
from .mla_reference import MLAReference


VARIANTS = ('dense', 'mla-expanded', 'mla-compressed', 'mla-absorbed', 'moe', 'mla-moe',
            'mla-reference-naive', 'mla-reference-absorb')


def rope(x, start, theta):
    positions = torch.arange(start, start + x.shape[2], device=x.device).float()
    frequencies = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float()
                            / x.shape[-1])
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().repeat_interleave(2, -1).to(x.dtype)
    sin = angles.sin().repeat_interleave(2, -1).to(x.dtype)
    pairs = x.unflatten(-1, (-1, 2))
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class MHA(nn.Module):
    def __init__(self, config, backend):
        super().__init__()
        self.config, self.backend, self.profile = config, backend, False
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.proj = nn.Linear(config.width, config.width)

    def project(self, layer, x):
        heads, dim = self.config.heads, self.config.width // self.config.heads
        if self.backend == 'head-loop':
            return torch.stack([F.linear(x, layer.weight[h * dim:(h + 1) * dim],
                                         layer.bias[h * dim:(h + 1) * dim])
                                for h in range(heads)], dim=1)
        return layer(x).unflatten(-1, (heads, dim)).transpose(1, 2)

    def forward(self, x, start=0, cache=None):
        cfg = self.config
        with scope('mha.projections', self.profile):
            q, k, v = (self.project(layer, x) for layer in (self.query, self.key, self.value))
        with scope('mha.rope', self.profile):
            q, k = (rope(t, start, cfg.rope_theta) for t in (q, k))
        if cache is not None:
            with scope(f'cache.{cache.mode}', self.profile):
                values = cache.append(k=k, v=v)
                k, v = values['k'], values['v']
        output = attention(q, k, v, self.backend,
                           cfg.dropout if self.training else 0.0, self.profile)
        with scope('mha.output_projection', self.profile):
            output = self.proj(output.transpose(1, 2).flatten(-2))
            return F.dropout(output, p=cfg.dropout, training=self.training)


class MLA(nn.Module):
    def __init__(self, config, backend, path='absorbed'):
        super().__init__()
        self.config, self.backend, self.path, self.profile = config, backend, path, False
        if path not in ('expanded', 'compressed', 'absorbed'):
            raise ValueError(path)
        self.q_down = nn.Linear(config.width, config.q_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_rank, eps=1e-6)
        self.q_up = nn.Linear(config.q_rank, config.heads * (config.nope_dim + config.rope_dim),
                              bias=False)
        self.kv_down = nn.Linear(config.width, config.kv_rank + config.rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(config.kv_rank, eps=1e-6)
        self.k_up = nn.Linear(config.kv_rank, config.heads * config.nope_dim, bias=False)
        self.v_up = nn.Linear(config.kv_rank, config.heads * config.value_dim, bias=False)
        self.out = nn.Linear(config.heads * config.value_dim, config.width, bias=False)

    def forward(self, x, start=0, cache=None):
        cfg = self.config
        with scope('mla.down_projections', self.profile):
            q = self.q_up(self.q_norm(self.q_down(x)))
            q = q.unflatten(-1, (cfg.heads, cfg.nope_dim + cfg.rope_dim)).transpose(1, 2)
            q_nope, q_rope = q.split((cfg.nope_dim, cfg.rope_dim), dim=-1)
            latent, k_rope = self.kv_down(x).split((cfg.kv_rank, cfg.rope_dim), dim=-1)
            latent = self.kv_norm(latent).unsqueeze(1)
            k_rope = rope(k_rope.unsqueeze(1), start, cfg.rope_theta)
            q_rope = rope(q_rope, start, cfg.rope_theta)
        if self.path == 'expanded':
            with scope('mla.expand_new_kv', self.profile):
                c = latent.squeeze(1)
                k_nope = self.k_up(c).unflatten(-1, (cfg.heads, cfg.nope_dim)).transpose(1, 2)
                v = self.v_up(c).unflatten(-1, (cfg.heads, cfg.value_dim)).transpose(1, 2)
                k = torch.cat((k_nope, k_rope.expand(-1, cfg.heads, -1, -1)), dim=-1)
            if cache is not None:
                with scope(f'cache.{cache.mode}', self.profile):
                    values = cache.append(k=k, v=v)
                    k, v = values['k'], values['v']
            q = torch.cat((q_nope, q_rope), dim=-1)
        else:
            if cache is not None:
                with scope(f'cache.{cache.mode}', self.profile):
                    values = cache.append(latent=latent, rope=k_rope)
                    latent, k_rope = values['latent'], values['rope']
            if self.path == 'compressed':
                with scope('mla.reconstruct_history', self.profile):
                    c = latent.squeeze(1)
                    k_nope = self.k_up(c).unflatten(-1, (cfg.heads, cfg.nope_dim)).transpose(1, 2)
                    v = self.v_up(c).unflatten(-1, (cfg.heads, cfg.value_dim)).transpose(1, 2)
                    k = torch.cat((k_nope, k_rope.expand(-1, cfg.heads, -1, -1)), dim=-1)
                    q = torch.cat((q_nope, q_rope), dim=-1)
            else:
                with scope('mla.absorb_key_projection', self.profile):
                    key_weight = self.k_up.weight.view(cfg.heads, cfg.nope_dim, cfg.kv_rank)
                    q_latent = torch.einsum('bhtd,hdc->bhtc', q_nope, key_weight)
                    q = torch.cat((q_latent, q_rope), dim=-1)
                    k = torch.cat((latent, k_rope), dim=-1)
                    v = latent
        output = attention(q, k, v, self.backend,
                           cfg.dropout if self.training else 0.0, self.profile,
                           scale=(cfg.nope_dim + cfg.rope_dim) ** -0.5)
        if self.path == 'absorbed':
            with scope('mla.value_projection_after_attention', self.profile):
                weight = self.v_up.weight.view(cfg.heads, cfg.value_dim, cfg.kv_rank)
                output = torch.einsum('bhtc,hvc->bhtv', output, weight)
        with scope('mla.output_projection', self.profile):
            output = self.out(output.transpose(1, 2).flatten(-2))
            return F.dropout(output, p=cfg.dropout, training=self.training)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopKMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.profile, self.collect_stats = config, False, False
        self.router = nn.Linear(config.width, config.experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.width, config.expert_hidden)
                                     for _ in range(config.experts)])
        self.shared = SwiGLU(config.width, config.expert_hidden) if config.shared_experts else None

    def route(self, flat):
        if self.config.router_float32:
            # Casting already-rounded BF16 logits is too late to protect a
            # near-tied Top-k decision. Perform the projection itself in FP32.
            with torch.autocast(flat.device.type, enabled=False):
                logits = F.linear(flat.float(), self.router.weight.float())
        else:
            logits = self.router(flat)
        probabilities = logits.float().softmax(-1)
        selected, indices = probabilities.topk(self.config.top_k, dim=-1)
        weights = (selected / selected.sum(-1, keepdim=True)).to(flat.dtype)
        return probabilities, weights, indices

    def forward(self, x):
        cfg = self.config
        flat = x.reshape(-1, cfg.width)
        with scope('moe.router_topk', self.profile):
            probabilities, weights, indices = self.route(flat)
        output = torch.zeros_like(flat)
        with scope('moe.dispatch_experts_combine', self.profile):
            for expert_id, expert in enumerate(self.experts):
                tokens, slots = torch.where(indices == expert_id)
                # This readable nonzero/gather implementation is deliberately not
                # a fused/grouped-GEMM MoE kernel and may synchronize on CUDA.
                with scope(f'moe.expert.{expert_id}', self.profile):
                    value = expert(flat[tokens]) * weights[tokens, slots, None]
                    output = output.index_add(0, tokens, value)
        if self.shared is not None:
            with scope('moe.shared_expert', self.profile):
                output = output + self.shared(flat)
        counts, auxiliary = None, flat.new_zeros(())
        if self.training or self.collect_stats:
            counts = torch.bincount(indices.flatten(), minlength=cfg.experts)
            fractions = counts.to(probabilities.dtype) / (flat.shape[0] * cfg.top_k)
            auxiliary = cfg.experts * (fractions.detach() * probabilities.mean(0)).sum()
        return output.reshape_as(x), auxiliary, None if counts is None else counts.detach()


class Block(nn.Module):
    def __init__(self, config, variant, backend):
        super().__init__()
        self.config, self.profile = config, False
        path = variant.removeprefix('mla-') if variant.startswith('mla-') else None
        if variant == 'mla-moe':
            path = 'absorbed'
        if variant.startswith('mla-reference-'):
            self.attention = MLAReference(config, variant.removeprefix('mla-reference-'), backend)
        else:
            self.attention = MHA(config, backend) if path is None else MLA(config, backend, path)
        self.ffn = TopKMoE(config) if variant in ('moe', 'mla-moe') else SwiGLU(config.width, config.hidden)
        self.norm1 = nn.RMSNorm(config.width, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.width, eps=1e-6)

    def forward(self, x, start=0, cache=None):
        with scope('block.norm_attention_residual', self.profile):
            x = x + self.attention(self.norm1(x), start, cache)
        with scope('block.norm_ffn_residual', self.profile):
            value = self.ffn(self.norm2(x))
            if isinstance(value, tuple):
                ffn, auxiliary, counts = value
            else:
                ffn, auxiliary, counts = value, x.new_zeros(()), None
            x = x + F.dropout(ffn, self.config.dropout, self.training)
        return x, auxiliary, counts


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    caches: list | None
    auxiliary_loss: torch.Tensor
    expert_counts: list


class LanguageModel(nn.Module):
    def __init__(self, config, variant='dense', backend='math'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.config, self.variant, self.profile = config, variant, False
        self.embedding = nn.Embedding(config.vocab, config.width)
        self.blocks = nn.ModuleList([Block(config, variant, backend) for _ in range(config.layers)])
        self.final_norm = nn.RMSNorm(config.width, eps=1e-6)
        self.head = nn.Linear(config.width, config.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def set_profile(self, enabled):
        for module in self.modules():
            if hasattr(module, 'profile'):
                module.profile = enabled

    def forward(self, tokens, targets=None, caches=None, cache_mode=None,
                capacity=None, last_only=False):
        if tokens.ndim != 2 or tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError('tokens must be nonempty [batch, sequence]')
        if targets is not None and (targets.shape != tokens.shape or last_only):
            raise ValueError('targets must match tokens, and need all-position logits')
        if (caches is not None or cache_mode is not None) and (self.training or torch.is_grad_enabled()):
            raise ValueError('KV cache requires eval plus no_grad/inference_mode')
        start = 0
        if caches is not None:
            if len(caches) != self.config.layers or len({c.length for c in caches}) != 1:
                raise ValueError('inconsistent layer caches')
            start = caches[0].length
        if start + tokens.shape[1] > self.config.max_context:
            raise ValueError('model context exceeded')
        if caches is None and cache_mode is not None:
            caches = [KVCache(cache_mode, capacity) for _ in self.blocks]
        with scope('model.embedding', self.profile):
            x = self.embedding(tokens)
        auxiliary, counts = x.new_zeros(()), []
        for index, block in enumerate(self.blocks):
            with scope(f'model.layer.{index}', self.profile):
                x, extra, load = block(x, start, None if caches is None else caches[index])
                auxiliary = auxiliary + extra
                if load is not None:
                    counts.append(load)
        with scope('model.final_norm_lm_head', self.profile):
            logits = self.head(self.final_norm(x[:, -1:] if last_only else x))
        loss = None
        if targets is not None:
            with scope('model.cross_entropy', self.profile):
                loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab), targets.reshape(-1))
        return ModelOutput(logits, loss, caches, auxiliary / self.config.layers, counts)
