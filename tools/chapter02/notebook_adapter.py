"""Swap only Attention execution in an externally supplied Chapter 1 notebook."""

from types import MethodType

import torch

from .attention import attention


def install_attention_backend(model, module, backend):
    """Keep the notebook's parameter names, initialization, projections and RoPE.

    The existing head projections are stacked for one multi-head Attention call.
    No weights are replaced; cached K/V retain the notebook's per-head layout.
    """
    if backend == 'notebook':
        return
    if backend not in ('math', 'sdpa-math', 'sdpa-flash', 'fa2'):
        raise ValueError(f'unsupported notebook training backend: {backend}')

    def forward(self, x, cos, sin, past_key_values=None, start_pos=0):
        queries, keys, values, cache = [], [], [], []
        if past_key_values is not None:
            if len(past_key_values) != len(self.heads):
                raise ValueError('cache head count does not match model')
            if any(k.shape[1] != start_pos or v.shape[1] != start_pos
                   for k, v in past_key_values):
                raise ValueError('cache length does not match start position')
        for index, head in enumerate(self.heads):
            k, v, q = head.key(x), head.value(x), head.query(x)
            q, k = module.apply_rope(q, cos, sin), module.apply_rope(k, cos, sin)
            if past_key_values is not None:
                old_k, old_v = past_key_values[index]
                k, v = torch.cat((old_k, k), 1), torch.cat((old_v, v), 1)
            queries.append(q)
            keys.append(k)
            values.append(v)
            cache.append((k, v))
        result = attention(torch.stack(queries, 1), torch.stack(keys, 1),
                           torch.stack(values, 1), backend,
                           self.heads[0].dropout.p if self.training else 0.0)
        result = result.transpose(1, 2).flatten(-2)
        return self.dropout(self.proj(result)), cache

    for layer in model.modules():
        if isinstance(layer, module.MultiHeadAttention):
            layer.forward = MethodType(forward, layer)
