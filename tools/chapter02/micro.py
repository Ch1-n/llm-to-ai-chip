"""Attention-only timing isolates kernel effects from projection and LM-head work."""

import json
import torch

from .attention import attention, BackendUnavailable
from .runtime import assert_match, measure, write_json


def benchmark_attention(args, device, dtype, directory):
    if min(args.batch, args.heads, args.kv_heads, args.kv_length, args.head_dim) < 1:
        raise ValueError('attention dimensions must be positive')
    value_dim = args.value_dim or args.head_dim
    failed, unavailable = 0, 0
    for backend in args.backends:
        for q_length in args.q_lengths:
            row = {'backend': backend, 'batch': args.batch, 'q_length': q_length,
                   'kv_length': args.kv_length, 'head_dim': args.head_dim,
                   'value_dim': value_dim, 'heads': args.heads, 'kv_heads': args.kv_heads,
                   'phase': 'forward_backward' if args.backward else 'forward'}
            try:
                if not 0 < q_length <= args.kv_length:
                    raise ValueError('need 0 < q_length <= kv_length')
                torch.manual_seed(1024)
                tensors = [torch.randn(args.batch, args.heads, q_length, args.head_dim, device=device, dtype=dtype),
                           torch.randn(args.batch, args.kv_heads, args.kv_length, args.head_dim, device=device, dtype=dtype),
                           torch.randn(args.batch, args.kv_heads, args.kv_length, value_dim, device=device, dtype=dtype)]
                for tensor in tensors:
                    tensor.requires_grad_(args.backward)

                def call():
                    if args.backward:
                        for tensor in tensors:
                            tensor.grad = None
                        output = attention(*tensors, backend)
                        output.float().square().mean().backward()
                        return output
                    with torch.inference_mode():
                        return attention(*tensors, backend)

                row['performance'] = measure(call, device, args.warmups, args.repeats)
                # A bounded prefix validates outputs without allocating a huge
                # reference score matrix for the long-context performance case.
                with torch.no_grad():
                    sample = [t[:, :, :min(17, t.shape[2])] for t in tensors]
                    actual = attention(*sample, backend)
                    expected = attention(*[t.double() for t in sample], 'math')
                    row['bounded_output_check'] = assert_match(actual, expected, dtype)
                row['status'] = 'completed'
            except BackendUnavailable as error:
                row.update(status='unavailable', reason=str(error))
                unavailable += 1
            except Exception as error:
                row.update(status='failed', reason=f'{type(error).__name__}: {error}')
                failed += 1
            with (directory / 'attention.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({k: row[k] for k in ('backend', 'q_length', 'status')}), flush=True)
    write_json(directory / 'summary.json', {'failed': failed, 'unavailable': unavailable,
                                          'scope': 'attention only; no projection, KV append, LM head or optimizer'})
    return int(failed > 0 or (args.strict and unavailable > 0))
