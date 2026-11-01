from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tools.chapter02.model import SwiGLU
from .attention import Attention, Cache, RMSNorm


@dataclass
class Output:
    logits: torch.Tensor
    loss: torch.Tensor | None
    auxiliary_loss: torch.Tensor
    caches: list | None


class Block(nn.Module):
    def __init__(self, cfg, ratio):
        super().__init__()
        self.attention = Attention(cfg, ratio)
        self.norm1, self.norm2 = RMSNorm(cfg.width, cfg.eps), RMSNorm(cfg.width, cfg.eps)
        self.ffn = SwiGLU(cfg.width, cfg.hidden)

    def forward(self, x, cache=None):
        value, aux = self.attention(self.norm1(x), cache)
        x = x + value
        return x + self.ffn(self.norm2(x)), aux


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, r) for r in cfg.ratios])
        self.final_norm = RMSNorm(cfg.width, cfg.eps)
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def set_profile(self, value):
        for block in self.blocks:
            block.attention.profile = value

    def forward(self, tokens, targets=None, caches=None, capacity=None, last_only=False):
        if tokens.ndim != 2 or tokens.shape[1] == 0 or (targets is not None and targets.shape != tokens.shape):
            raise ValueError('invalid tokens/targets')
        if targets is not None and (caches is not None or capacity is not None or last_only):
            raise ValueError('training requires all logits without KV cache')
        if caches is None and capacity is not None:
            if self.training or torch.is_grad_enabled() or capacity > self.cfg.max_context:
                raise ValueError('invalid cache context')
            caches = [Cache(capacity) for _ in self.blocks]
        if caches is not None and (len(caches) != len(self.blocks) or len({c.length for c in caches}) != 1):
            raise ValueError('inconsistent layer cache positions')
        x, losses = self.embedding(tokens), []
        for index, block in enumerate(self.blocks):
            x, aux = block(x, None if caches is None else caches[index])
            losses.append(aux)
        logits = self.head(self.final_norm(x[:, -1:] if last_only else x))
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab), targets.reshape(-1))
        return Output(logits, loss, torch.stack(losses).mean(), caches)
