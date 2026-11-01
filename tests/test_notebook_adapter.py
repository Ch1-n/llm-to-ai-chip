"""Check the adapter contract without distributing the external notebook."""

import copy
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter02.attention import attention
    from tools.chapter02.notebook_adapter import install_attention_backend

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.query, self.key, self.value = (nn.Linear(8, 4) for _ in range(3))
            self.dropout = nn.Dropout(.1)

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([Head(), Head()])
            self.proj, self.dropout = nn.Linear(8, 8), nn.Dropout(.1)

        def forward(self, x, cos, sin, past_key_values=None, start_pos=0):
            outputs, caches = [], []
            for index, head in enumerate(self.heads):
                q, k, v = head.query(x) * cos, head.key(x) * cos, head.value(x)
                if past_key_values is not None:
                    old_k, old_v = past_key_values[index]
                    k, v = torch.cat((old_k, k), 1), torch.cat((old_v, v), 1)
                outputs.append(attention(q[:, None], k[:, None], v[:, None], 'math',
                                         head.dropout.p if self.training else 0.)[:, 0])
                caches.append((k, v))
            return self.dropout(self.proj(torch.cat(outputs, -1))), caches


@unittest.skipIf(torch is None, 'requires optional PyTorch')
class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        torch.set_num_threads(1)
        self.module = SimpleNamespace(MultiHeadAttention=MultiHeadAttention,
                                      apply_rope=lambda x, cos, sin: x * cos)

    def test_no_parameter_changes_and_gradients(self):
        original = MultiHeadAttention().double().eval()
        adapted = copy.deepcopy(original)
        before = {name: id(p) for name, p in adapted.named_parameters()}
        install_attention_backend(adapted, self.module, 'sdpa-math')
        self.assertEqual(before, {name: id(p) for name, p in adapted.named_parameters()})
        for a, b in zip(original.parameters(), adapted.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        cos, sin = torch.ones(7, 4, dtype=torch.float64), torch.zeros(7, 4, dtype=torch.float64)
        a, _ = original(x, cos, sin)
        b, _ = adapted(y, cos, sin)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)
        for p, q in zip(original.parameters(), adapted.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-10)

    def test_chunked_cache_layout_and_causality(self):
        model = MultiHeadAttention().double().eval()
        install_attention_backend(model, self.module, 'sdpa-math')
        x = torch.randn(2, 7, 8, dtype=torch.float64)
        cos, sin = torch.ones(7, 4), torch.zeros(7, 4)
        full, _ = model(x, cos, sin)
        first, cache = model(x[:, :3], cos[:3], sin[:3])
        last, cache = model(x[:, 3:], cos[3:], sin[3:], cache, 3)
        torch.testing.assert_close(torch.cat((first, last), 1), full)
        self.assertTrue(all(k.shape == v.shape == (2, 7, 4) for k, v in cache))
        changed = x.clone()
        changed[:, 4:] += 10
        torch.testing.assert_close(model(changed, cos, sin)[0][:, :4], full[:, :4])
        with self.assertRaises(ValueError):
            model(x[:, :1], cos[:1], sin[:1], cache, 6)


if __name__ == '__main__':
    unittest.main()
