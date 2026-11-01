"""CPU mathematical checks; CUDA kernel validation is a separate remote step."""

from dataclasses import replace
import unittest

try:
    import torch
except ImportError:
    torch = None

from tools.chapter02.config import ModelConfig, preset

if torch is not None:
    from tools.chapter02.attention import BackendUnavailable, attention, causal_mask
    from tools.chapter02.cache import KVCache
    from tools.chapter02.model import LanguageModel, MLA, MHA, TopKMoE, VARIANTS, rope


class ConfigTests(unittest.TestCase):
    def test_reject_invalid_configuration(self):
        for values in ({'width': 31}, {'rope_dim': 3}, {'top_k': 9}, {'kv_rank': 0},
                       {'dropout': 1}, {'shared_experts': 2}):
            with self.assertRaises(ValueError):
                ModelConfig(**values)


@unittest.skipIf(torch is None, 'PyTorch is optional for the article estimate tests')
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        torch.set_num_threads(1)
        self.cfg = preset('tiny')

    def test_lower_right_mask(self):
        expected = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        torch.testing.assert_close(causal_mask(2, 4, 'cpu'), expected)
        self.assertTrue(causal_mask(1, 4, 'cpu').all())

    def test_rope_independent_rotation(self):
        x = torch.zeros(1, 1, 1, 8)
        x[..., 0] = 1
        y = rope(x, 1, 10000.)
        self.assertAlmostEqual(y[0, 0, 0, 0].item(), torch.cos(torch.tensor(1.)).item())
        self.assertAlmostEqual(y[0, 0, 0, 1].item(), torch.sin(torch.tensor(1.)).item())
        random = torch.randn(2, 3, 7, 8)
        torch.testing.assert_close(rope(random, 3, 10000.).norm(dim=-1), random.norm(dim=-1))

    def test_attention_backends_and_gradients(self):
        for q_len, k_len in ((7, 7), (1, 7), (3, 7)):
            values = [torch.randn(2, 4, q_len, 8, dtype=torch.float64),
                      torch.randn(2, 2, k_len, 8, dtype=torch.float64),
                      torch.randn(2, 2, k_len, 6, dtype=torch.float64)]
            reference = None
            for backend in ('math', 'head-loop', 'sdpa-math'):
                tensors = [x.clone().requires_grad_() for x in values]
                result = attention(*tensors, backend=backend)
                result.square().sum().backward()
                record = [result.detach()] + [x.grad for x in tensors]
                if reference is None:
                    reference = record
                else:
                    for a, b in zip(record, reference):
                        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)

    def test_backend_no_cpu_fallback(self):
        x = torch.randn(1, 2, 4, 8)
        for backend in ('sdpa-flash', 'fa2', 'fa3', 'fa4'):
            with self.assertRaises(BackendUnavailable):
                attention(x, x, x, backend)

    def test_mha_head_projection_equivalence(self):
        reference = MHA(self.cfg, 'math').double()
        candidate = MHA(self.cfg, 'head-loop').double()
        candidate.load_state_dict(reference.state_dict())
        x = torch.randn(2, 7, self.cfg.width, dtype=torch.float64)
        a, b = reference(x), candidate(x)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        a.square().sum().backward()
        b.square().sum().backward()
        for p, q in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)

    def test_mla_three_paths_forward_and_backward(self):
        models = [MLA(self.cfg, 'math', path).double() for path in
                  ('expanded', 'compressed', 'absorbed')]
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
        xs = [torch.randn(2, 7, self.cfg.width, dtype=torch.float64).requires_grad_()]
        xs.extend(xs[0].detach().clone().requires_grad_() for _ in range(2))
        outputs = [model(x) for model, x in zip(models, xs)]
        for out in outputs:
            out.square().sum().backward()
        for index in (1, 2):
            torch.testing.assert_close(outputs[0], outputs[index], atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(xs[0].grad, xs[index].grad, atol=1e-10, rtol=1e-9)
            for a, b in zip(models[0].parameters(), models[index].parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9)

    def test_full_chunked_and_token_cached_models(self):
        tokens = torch.randint(self.cfg.vocab, (2, 9))
        for variant in VARIANTS:
            model = LanguageModel(self.cfg, variant).eval()
            with torch.inference_mode():
                full = model(tokens).logits
                for mode in ('cat', 'static'):
                    for chunks in ((3, 1, 5), (1,) * 9):
                        caches, offset, parts = None, 0, []
                        for length in chunks:
                            result = model(tokens[:, offset:offset + length], caches=caches,
                                           cache_mode=mode, capacity=12)
                            caches = result.caches
                            parts.append(result.logits)
                            offset += length
                        torch.testing.assert_close(torch.cat(parts, 1), full, atol=3e-6, rtol=3e-5)
                        self.assertTrue(all(cache.length == 9 for cache in caches))

    def test_causality_all_variants(self):
        tokens = torch.randint(self.cfg.vocab, (2, 9))
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 13) % self.cfg.vocab
        for variant in VARIANTS:
            model = LanguageModel(self.cfg, variant).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(tokens).logits[:, :5],
                                           model(changed).logits[:, :5])

    def test_mla_cache_bytes(self):
        cfg = self.cfg
        tokens = torch.randint(cfg.vocab, (2, 9))
        for variant in ('mla-expanded', 'mla-compressed', 'mla-absorbed'):
            model = LanguageModel(cfg, variant).eval()
            with torch.inference_mode():
                result = model(tokens, cache_mode='static', capacity=12)
            width = (cfg.heads * (cfg.nope_dim + cfg.rope_dim + cfg.value_dim)
                     if variant == 'mla-expanded' else cfg.kv_rank + cfg.rope_dim)
            self.assertEqual(sum(c.logical_bytes() for c in result.caches),
                             2 * 9 * cfg.layers * width * 4)
            self.assertEqual(sum(c.allocated_bytes() for c in result.caches),
                             2 * 12 * cfg.layers * width * 4)

    def test_cache_rewind_and_transaction_validation(self):
        for mode in ('static', 'cat'):
            cache = KVCache(mode, 8)
            with torch.inference_mode():
                cache.append(k=torch.ones(1, 2, 4, 8), v=torch.ones(1, 2, 4, 8))
                before = cache.length
                with self.assertRaises(ValueError):
                    cache.append(k=torch.ones(1, 2, 1, 8), v=torch.ones(1, 3, 1, 8))
                self.assertEqual(cache.length, before)
                cache.rewind(3)
                fields = cache.append(k=torch.full((1, 2, 1, 8), 7.),
                                      v=torch.full((1, 2, 1, 8), 8.))
                self.assertTrue((fields['k'][:, :, -1] == 7).all())
                self.assertTrue((fields['k'][:, :, :3] == 1).all())

    def test_moe_matches_all_experts_reference_and_gradients(self):
        model = TopKMoE(replace(self.cfg, shared_experts=1, router_float32=True)).double()
        other = TopKMoE(replace(self.cfg, shared_experts=1, router_float32=True)).double()
        other.load_state_dict(model.state_dict())
        x = torch.randn(2, 5, self.cfg.width, dtype=torch.float64).requires_grad_()
        y = x.detach().clone().requires_grad_()
        actual, aux, counts = model(x)
        flat = y.flatten(0, 1)
        probability = torch.nn.functional.linear(flat.float(), other.router.weight.float()).softmax(-1)
        selected, index = probability.topk(self.cfg.top_k, -1)
        weights = (selected / selected.sum(-1, keepdim=True)).to(flat.dtype)
        every = torch.stack([expert(flat) for expert in other.experts], dim=1)
        gathered = every.gather(1, index[..., None].expand(-1, -1, self.cfg.width))
        expected = ((gathered * weights[..., None]).sum(1) + other.shared(flat)).reshape_as(y)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        self.assertEqual(counts.sum().item(), 2 * 5 * self.cfg.top_k)
        self.assertTrue(torch.isfinite(aux))
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-8)
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-7, rtol=3e-5)

    def test_optimizer_step_all_variants(self):
        tokens = torch.randint(self.cfg.vocab, (2, 10))
        for variant in VARIANTS:
            model = LanguageModel(self.cfg, variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            result = model(tokens[:, :-1], targets=tokens[:, 1:])
            loss = result.loss + 0.01 * result.auxiliary_loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()

    def test_context_training_cache_and_last_only_guards(self):
        model = LanguageModel(replace(self.cfg, max_context=4)).eval()
        with torch.inference_mode():
            with self.assertRaises(ValueError):
                model(torch.ones(1, 5, dtype=torch.long))
            x = torch.ones(1, 4, dtype=torch.long)
            torch.testing.assert_close(model(x, last_only=True).logits, model(x).logits[:, -1:])
        with self.assertRaises(ValueError):
            model(torch.ones(1, 1, dtype=torch.long), cache_mode='cat')

    def test_teaching_dense_parameter_count_matches_chapter_one(self):
        with torch.device('meta'):
            model = LanguageModel(preset('teaching'))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 119690496)


if __name__ == '__main__':
    unittest.main()
