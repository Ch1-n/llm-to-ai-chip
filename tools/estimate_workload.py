#!/usr/bin/env python3
"""Analytical dense-decoder estimates, not hardware measurements."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    layers: int = 6
    hidden: int = 768
    query_heads: int = 12
    intermediate: int = 2048
    vocabulary: int = 50257
    batch: int = 1
    prefill_tokens: int = 512
    decode_visible_tokens: int = 512
    bytes_per_element: int = 2

    def validate(self):
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError('all configuration values must be positive')
        if self.hidden % self.query_heads:
            raise ValueError('hidden must be divisible by query_heads')


def kv_bytes(layers, batch, tokens, kv_heads, head_dim, element_bytes):
    dimensions = (layers, batch, tokens, kv_heads, head_dim, element_bytes)
    if any(value <= 0 for value in dimensions):
        raise ValueError('cache dimensions must be positive')
    return 2 * layers * batch * tokens * kv_heads * head_dim * element_bytes


def block_flops(config, query_tokens, visible_tokens):
    config.validate()
    if query_tokens <= 0 or visible_tokens < query_tokens:
        raise ValueError('invalid query/visible lengths')
    b, h, i = config.batch, config.hidden, config.intermediate
    return {
        'qkv_projection': 6 * b * query_tokens * h * h,
        'attention_qk_and_pv': 4 * b * query_tokens * visible_tokens * h,
        'output_projection': 2 * b * query_tokens * h * h,
        'swiglu_three_projections': 6 * b * query_tokens * h * i,
    }


def estimate(config):
    config.validate()
    h, i, v, l = config.hidden, config.intermediate, config.vocabulary, config.layers
    projection_parameters_per_block = 4 * h * h + 3 * h * i
    # Q/K/V and attention output have biases; FFN and LM head do not.
    bias_parameters = l * 4 * h
    norm_parameters = (2 * l + 1) * h
    parameters = l * projection_parameters_per_block + 2 * v * h + bias_parameters + norm_parameters
    prefill = block_flops(config, config.prefill_tokens, config.prefill_tokens)
    decode = block_flops(config, 1, config.decode_visible_tokens)
    head_one_position = 2 * config.batch * h * v
    cache = kv_bytes(l, config.batch, config.decode_visible_tokens, config.query_heads,
                     h // config.query_heads, config.bytes_per_element)
    return {
        'config': asdict(config),
        'head_dim': h // config.query_heads,
        'assumptions': {
            'status': 'analytical_estimate_not_measurement',
            'flops_per_multiply_add': 2,
            'attention': 'dense MHA; full score products including entries later masked',
            'omitted_flops': ['bias addition', 'RMSNorm', 'RoPE', 'Softmax', 'SiLU', 'elementwise product', 'residual addition', 'sampling'],
            'parameters': 'untied embedding/LM head; QKV and attention-output biases; no FFN biases; RMSNorm gains',
            'storage_precision': '2 bytes is an accounting assumption, not an executed FP16/BF16 benchmark',
            'ideal_weight_read': 'each decoder/LM-head projection weight fetched once per decode batch; no persistent on-chip weight retention; embedding is a row gather',
            'cache_excludes': ['allocator padding', 'page metadata', 'temporary buffers', 'activation memory', 'quantization scales'],
        },
        'parameters': parameters,
        'weight_capacity_bytes': parameters * config.bytes_per_element,
        'prefill_per_block_flops': prefill,
        'decode_per_block_flops': decode,
        'prefill_decoder_flops': sum(prefill.values()) * l,
        'decode_decoder_flops': sum(decode.values()) * l,
        'lm_head_one_position_flops': head_one_position,
        'lm_head_all_prefill_positions_flops': head_one_position * config.prefill_tokens,
        'decode_decoder_and_lm_head_flops': sum(decode.values()) * l + head_one_position,
        'kv_cache_capacity_bytes': cache,
        'kv_new_token_append_bytes': 2 * l * config.batch * h * config.bytes_per_element,
        'ideal_decode_projection_weight_read_bytes':
            (l * projection_parameters_per_block + v * h) * config.bytes_per_element,
    }


def make_report(config):
    report = estimate(config)
    report['cache_capacity_examples'] = [
        {'layers': 32, 'batch': 1, 'query_heads': 32, 'head_dim': 128,
         'kv_heads': kv_heads, 'tokens': tokens, 'bytes_per_element': 2,
         'cache_bytes': kv_bytes(32, 1, tokens, kv_heads, 128, 2),
         'scope': 'independent hypothetical configuration, not the 512-token teaching model'}
        for tokens in (8192, 16384, 32768)
        for kv_heads in (32, 8)
    ]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/llm-inference')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = make_report(Config())
    (args.output / 'estimates.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'block-flops.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('operation', 'prefill_flops', 'decode_flops'))
        for operation, count in report['prefill_per_block_flops'].items():
            writer.writerow((operation, count, report['decode_per_block_flops'][operation]))
    print(f"parameters={report['parameters']:,}")
    print(f"prefill_decoder_GFLOPs={report['prefill_decoder_flops'] / 1e9:.6f}")
    print(f"decode_decoder_and_lm_head_MFLOPs={report['decode_decoder_and_lm_head_flops'] / 1e6:.6f}")
    print(f"MHA_KV_cache_MiB={report['kv_cache_capacity_bytes'] / 2**20:.6f}")
    print(f"ideal_decode_projection_weight_read_MiB={report['ideal_decode_projection_weight_read_bytes'] / 2**20:.6f}")


if __name__ == '__main__':
    main()
