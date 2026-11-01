"""Pinned original V4 classes, with quantization disabled and a dense core oracle.

Only selected AST definitions execute. The TileLang sparse kernel is replaced
with Transformers' eager Attention over the exact selected key positions.
"""

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys
import types
from typing import Literal, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


BASE = Path(__file__).resolve().parents[2] / 'references/deepseek-v4'
SOURCE_SHA = 'ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71'
HF_SHA = '0375649d727a9fbea1756a500336af7d9ca66469e63c16bfabded25c85e5cebd'


def definitions(path, digest, wanted, namespace):
    text = path.read_bytes()
    if hashlib.sha256(text).hexdigest() != digest:
        raise ValueError('reference source SHA256 mismatch')
    parsed = ast.parse(text.decode())
    nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes} != set(wanted):
        raise ValueError('reference definitions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def dense_oracle(q, kv, sink, indices, scale):
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    if not hasattr(dense_oracle, 'function'):
        definitions(BASE / 'transformers-modeling.py', HF_SHA,
                    ('repeat_kv', 'eager_attention_forward'), namespace)
        dense_oracle.function = namespace['eager_attention_forward']
    b, s, h, _ = q.shape
    n = kv.shape[1]
    safe = torch.where(indices >= 0, indices.long(), n)
    mask = q.new_full((b, s, n + 1), float('-inf'))
    mask.scatter_(-1, safe, 0.)
    module = types.SimpleNamespace(num_key_value_groups=h, sinks=sink, training=False)
    return dense_oracle.function(module, q.transpose(1, 2), kv[:, None], kv[:, None],
                                 mask[:, None, :, :n], scale)[0]


def load():
    module = types.ModuleType('pinned_unquantized_deepseek_v4')
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
        lru_cache=lru_cache, Tuple=Tuple, Optional=Optional, Literal=Literal,
        world_size=1, rank=0, block_size=128, fp4_block_size=32,
        default_dtype=torch.float32, scale_fmt=None, scale_dtype=torch.float32,
        act_quant=lambda x, *args: x, fp4_act_quant=lambda x, *args: x,
        rotate_activation=lambda x: x, sparse_attn=dense_oracle)
    definitions(BASE / 'model.py', SOURCE_SHA,
        ('ModelArgs', 'linear', 'Linear', 'ColumnParallelLinear', 'RowParallelLinear',
         'RMSNorm', 'precompute_freqs_cis', 'apply_rotary_emb', 'get_window_topk_idxs',
         'get_compress_topk_idxs', 'Compressor', 'Indexer', 'Attention'), module.__dict__)
    native_forward = module.Indexer.forward

    def stable_indexer(self, x, qr, start_pos, offset):
        # Execute the original state updates; only canonicalize tied score selection.
        native_forward(self, x, qr, start_pos, offset)
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        module.apply_rotary_emb(q[..., -self.rope_head_dim:], self.freqs_cis[start_pos:start_pos + x.shape[1]])
        keys = self.kv_cache[:x.shape[0], :(start_pos + x.shape[1]) // self.compress_ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -.5)
        scores = (torch.einsum('bshd,btd->bsht', q, keys).relu() * weights.unsqueeze(-1)).sum(2)
        positions = torch.arange(start_pos, start_pos + x.shape[1], device=x.device)
        visible = (torch.arange(keys.shape[1], device=x.device)[None] + 1) * self.compress_ratio <= positions[:, None] + 1
        scores = scores.masked_fill(~visible[None], float('-inf'))
        ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :self.index_topk]
        valid = visible[None].expand(x.shape[0], -1, -1).gather(-1, ids)
        return torch.where(valid, ids + offset, -1)

    module.Indexer.forward = stable_indexer
    return module


def arguments(module, cfg, batch=1):
    return module.ModelArgs(max_batch_size=batch, max_seq_len=cfg.max_context,
        dim=cfg.width, n_layers=cfg.layers, n_heads=cfg.heads, q_lora_rank=cfg.q_rank,
        head_dim=cfg.head_dim, rope_head_dim=cfg.rope_dim, norm_eps=cfg.eps,
        o_groups=cfg.o_groups, o_lora_rank=cfg.o_rank, window_size=cfg.window,
        compress_ratios=cfg.ratios, compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=0, rope_theta=cfg.rope_theta, index_n_heads=cfg.index_heads,
        index_head_dim=cfg.index_dim, index_topk=cfg.index_topk)
