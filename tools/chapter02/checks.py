"""Device-side correctness checks run before collecting performance results."""

from dataclasses import replace
import math
import torch

from .attention import BackendUnavailable, attention
from .config import preset
from .model import LanguageModel, MLA, VARIANTS
from .runtime import assert_match


def kernel_checks(device, dtype, backends):
    records = []
    for backend in backends:
        for q_len, k_len in ((17, 17), (1, 17), (5, 17)):
            for q_heads, kv_heads in ((4, 4), (4, 2)):
                row = {'kind': 'attention_output_and_qkv_gradients', 'backend': backend,
                       'q_length': q_len, 'kv_length': k_len,
                       'query_heads': q_heads, 'kv_heads': kv_heads}
                try:
                    torch.manual_seed(2026)
                    tensors = [torch.randn(2, q_heads, q_len, 64, device=device, dtype=dtype),
                               torch.randn(2, kv_heads, k_len, 64, device=device, dtype=dtype),
                               torch.randn(2, kv_heads, k_len, 64, device=device, dtype=dtype)]
                    inputs = [x.requires_grad_() for x in tensors]
                    truth = [x.detach().double().requires_grad_() for x in inputs]
                    expected = attention(*truth, 'math')
                    probe = torch.randn_like(expected) / math.sqrt(expected.numel())
                    expected.backward(probe)
                    actual = attention(*inputs, backend)
                    actual.backward(probe.to(dtype))
                    row['output'] = assert_match(actual, expected, dtype)
                    row['gradients'] = [assert_match(a.grad, b.grad, dtype)
                                        for a, b in zip(inputs, truth)]
                    row['status'] = 'passed'
                except BackendUnavailable as error:
                    row.update(status='unavailable', reason=str(error))
                except Exception as error:
                    row.update(status='failed', reason=f'{type(error).__name__}: {error}')
                records.append(row)
    return records


def model_checks(device):
    cfg = preset('tiny')
    records = []
    torch.manual_seed(2026)
    ids = torch.randint(cfg.vocab, (2, 13), device=device)
    for variant in VARIANTS:
        model = LanguageModel(cfg, variant).to(device).eval()
        with torch.inference_mode():
            expected = model(ids).logits
            for cache_mode in ('cat', 'static'):
                outputs, caches, offset = [], None, 0
                for length in (3, 1, 9):
                    output = model(ids[:, offset:offset + length], caches=caches,
                                   cache_mode=cache_mode, capacity=16)
                    outputs.append(output.logits)
                    offset += length
                    caches = output.caches
                match = assert_match(torch.cat(outputs, 1), expected, torch.float32)
                records.append({'kind': 'full_vs_chunked', 'variant': variant,
                                'cache_mode': cache_mode, 'status': 'passed', **match})
                changed = ids.clone()
                changed[:, 7:] = (changed[:, 7:] + 11) % cfg.vocab
                assert_match(model(changed).logits[:, :7], expected[:, :7], torch.float32)
        model.train()
        output = model(ids[:, :-1], targets=ids[:, 1:])
        (output.loss + .01 * output.auxiliary_loss).backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise AssertionError(f'nonfinite {variant} gradient')
        records.append({'kind': 'causality_and_training_backward', 'variant': variant,
                        'status': 'passed'})
        del model

    # This checks the algebraic transformation at nontrivial head dimensions,
    # independently of whether an optional fused kernel accepts these shapes.
    cfg = replace(cfg, width=128, heads=2, nope_dim=32, rope_dim=32,
                  value_dim=64, kv_rank=64, q_rank=64)
    reference = MLA(cfg, 'math', 'expanded').to(device)
    x = torch.randn(2, 13, cfg.width, device=device, requires_grad=True)
    expected = reference(x)
    expected.square().mean().backward()
    for path in ('compressed', 'absorbed'):
        other = MLA(cfg, 'math', path).to(device)
        other.load_state_dict(reference.state_dict())
        y = x.detach().clone().requires_grad_()
        actual = other(y)
        actual.square().mean().backward()
        match = assert_match(actual, expected, torch.float32)
        assert_match(y.grad, x.grad, torch.float32)
        for a, b in zip(other.parameters(), reference.parameters()):
            assert_match(a.grad, b.grad, torch.float32)
        records.append({'kind': 'mla_absorption_output_input_parameter_gradients',
                        'path': path, 'status': 'passed', **match})
    return records


def notebook_check(path, device):
    """Optional exact weight mapping to chapter one's corrected local notebook."""
    from tools.validate_cuda_notebook import corrected_rope, load_definitions
    module = load_definitions(path)
    module.precompute_rope_freqs = corrected_rope
    cfg = preset('tiny')
    original_cfg = module.GPTConfig(n_embd=cfg.width, n_head=cfg.heads,
                                    head_size=cfg.width // cfg.heads, n_layer=cfg.layers,
                                    block_size=cfg.max_context, vocab_size=cfg.vocab, dropout=0.)
    old = module.GPT(original_cfg).to(device).eval()
    # The notebook rounds FFN width up to a multiple of 256.
    cfg = replace(cfg, hidden=old.blocks[0].ffn.gate_proj.out_features)
    new = LanguageModel(cfg).to(device).eval()
    with torch.no_grad():
        new.embedding.weight.copy_(old.token_embedding_table.weight)
        new.final_norm.load_state_dict(old.ln_final.state_dict())
        new.head.load_state_dict(old.lm_head.state_dict())
        for a, b in zip(new.blocks, old.blocks):
            a.norm1.load_state_dict(b.ln1.state_dict())
            a.norm2.load_state_dict(b.ln2.state_dict())
            for name in ('query', 'key', 'value'):
                target = getattr(a.attention, name)
                target.weight.copy_(torch.cat([getattr(h, name).weight for h in b.att.heads]))
                target.bias.copy_(torch.cat([getattr(h, name).bias for h in b.att.heads]))
            a.attention.proj.load_state_dict(b.att.proj.state_dict())
            for first, second in (('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')):
                getattr(a.ffn, first).load_state_dict(getattr(b.ffn, second).state_dict())
    ids = torch.randint(cfg.vocab, (2, 13), device=device)
    with torch.inference_mode():
        expected, _ = old(ids)
        match = assert_match(new(ids).logits, expected, torch.float32)
    return {'kind': 'chapter_one_notebook_weight_mapping', 'status': 'passed', **match,
            'note': 'RoPE fixed in memory; source notebook left unchanged'}
