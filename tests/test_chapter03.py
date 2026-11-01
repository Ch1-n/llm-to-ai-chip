"""Synthetic mathematical fixtures only; these are not language-training results."""

from dataclasses import replace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter03.attention import Attention, Cache, gathered_attention
    from tools.chapter03.config import Config, preset
    from tools.chapter03.model import LanguageModel
    from tools.chapter03.reference import arguments, dense_oracle, load


@unittest.skipIf(torch is None, 'requires optional torch')
class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        self.cfg = preset('tiny')

    def attention(self, ratio):
        a = Attention(self.cfg, ratio)
        a.apply(LanguageModel._init)
        return a

    def test_configuration_guards(self):
        for values in ({'heads': 11}, {'rope_dim': 3}, {'index_topk': 0}, {'ratios': (0,)}):
            with self.assertRaises(ValueError):
                Config(**values)

    def test_gathered_core_vs_transformers_output_and_gradients(self):
        q = torch.randn(2, 5, 4, 8, requires_grad=True)
        kv = torch.randn(2, 9, 8, requires_grad=True)
        sink = torch.randn(4, requires_grad=True)
        index = torch.tensor([[[0, 1, -1], [1, 3, -1], [0, 4, 7], [3, 5, -1], [1, 7, 8]]] * 2)
        a = gathered_attention(q, kv, index, sink, 8 ** -.5)
        b = dense_oracle(q, kv, sink, index, 8 ** -.5)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        left = torch.autograd.grad(a.square().sum(), (q, kv, sink), retain_graph=True)
        right = torch.autograd.grad(b.square().sum(), (q, kv, sink))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)

    def test_original_official_prefill_and_decode(self):
        for ratio in (0, 4, 128):
            module = load()
            cfg = replace(self.cfg, ratios=(ratio,) * self.cfg.layers)
            official = module.Attention(0, arguments(module, cfg)).float().eval()
            ours = self.attention(ratio).eval()
            official.load_state_dict(ours.state_dict(), strict=True)
            x = torch.randn(1, 137, cfg.width)
            cache = Cache(cfg.max_context)
            with torch.inference_mode():
                a = official(x[:, :129], 0)
                b = ours(x[:, :129], cache)[0]
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                for position in range(129, 137):
                    a = official(x[:, position:position + 1], position)
                    b = ours(x[:, position:position + 1], cache)[0]
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)

    def test_partial_blocks_and_arbitrary_chunked_prefill(self):
        x = torch.randn(1, 137, self.cfg.width)
        for ratio in (0, 4, 128):
            model = self.attention(ratio).eval()
            with torch.inference_mode():
                full = model(x)[0]
                for chunks in ((3, 1, 5, 118, 10), (127, 1, 1, 8), (137,)):
                    cache, offset, out = Cache(137), 0, []
                    for size in chunks:
                        out.append(model(x[:, offset:offset + size], cache)[0])
                        offset += size
                    torch.testing.assert_close(torch.cat(out, 1), full, atol=3e-6, rtol=3e-5)
                    self.assertLessEqual(cache.raw.shape[1], self.cfg.window)
                    if ratio:
                        self.assertEqual(cache.main.count, 137 // ratio)
                        self.assertEqual(cache.main.tail_values.shape[1], 137 % ratio)
                    with self.assertRaises(ValueError):
                        model(x[:, :1], cache)

    def test_causality(self):
        x = torch.randn(1, 137, self.cfg.width)
        y = x.clone()
        y[:, 126:] += 100
        for ratio in (0, 4, 128):
            model = self.attention(ratio).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(x)[0][:, :126], model(y)[0][:, :126])

    def test_exact_indexer_ties_are_prefix_invariant(self):
        model = self.attention(4).eval()
        with torch.no_grad():
            model.indexer.weights_proj.weight.zero_()
        x = torch.randn(1, 37, self.cfg.width)
        with torch.inference_mode():
            full = model(x, collect=True)[0]
            chosen = model.last_stats['indices'][0, 8:, -2:] - 37
            torch.testing.assert_close(chosen, torch.tensor([0, 1]).expand_as(chosen))
            cache, out, offset = Cache(37), [], 0
            for size in (3, 4, 7, 23):
                out.append(model(x[:, offset:offset + size], cache)[0])
                offset += size
            torch.testing.assert_close(torch.cat(out, 1), full)

    def test_full_model_cache_and_real_indexer_gradients(self):
        model = LanguageModel(self.cfg)
        tokens = torch.randint(self.cfg.vocab, (2, 137))
        model.eval()
        with torch.inference_mode():
            full = model(tokens).logits
            first = model(tokens[:, :127], capacity=137)
            last = model(tokens[:, 127:], caches=first.caches)
            torch.testing.assert_close(torch.cat((first.logits, last.logits), 1), full, atol=4e-6, rtol=4e-5)
        model.train()
        result = model(tokens, targets=tokens.roll(-1, dims=1))
        (result.loss + .01 * result.auxiliary_loss).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        indexer = model.blocks[1].attention.indexer
        self.assertGreater(indexer.wq_b.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(indexer.compressor.wgate.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
