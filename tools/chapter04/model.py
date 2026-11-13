"""Operation contracts derived from the teaching models, not executed kernels."""

from dataclasses import asdict, dataclass, field
from functools import lru_cache

from tools.chapter02.config import ModelConfig
from tools.chapter03.config import Config as HybridConfig


VARIANTS = ('dense', 'mla', 'moe', 'hybrid')


def ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class Request:
    variant: str = 'dense'
    phase: str = 'decode'
    batch: int = 1
    context: int = 2048
    element_bytes: int = 2
    route_policy: str = 'balanced'
    ffn_mode: str = 'fused'
    token_tile: int = 128
    hidden_tile: int = 128

    def __post_init__(self):
        if self.variant not in VARIANTS or self.phase not in ('prefill', 'decode'):
            raise ValueError('unsupported variant/phase')
        if any(type(v) is not int or v < 1 for v in (self.batch, self.context,
            self.element_bytes, self.token_tile, self.hidden_tile)):
            raise ValueError('dimensions must be positive integers')
        if self.context > ModelConfig().max_context:
            raise ValueError('context exceeds the teaching model limit')
        if self.route_policy not in ('balanced', 'concentrated') or self.ffn_mode not in ('fused', 'unfused'):
            raise ValueError('unsupported execution contract')

    @property
    def queries(self):
        return self.context if self.phase == 'prefill' else 1


@dataclass
class Op:
    name: str
    parents: list
    matrix_flops: int = 0
    matrix_fp32_flops: int = 0
    vector_ops: int = 0
    exp: int = 0
    reciprocal: int = 0
    rsqrt: int = 0
    selection_items: int = 0
    weight_bytes: int = 0
    state_read_bytes: int = 0
    state_write_bytes: int = 0
    activation_bytes: int = 0
    gemms: list = field(default_factory=list)
    logical_shape: dict = field(default_factory=dict)
    activation: bool = False
    commands: int = 1

    @property
    def dram_bytes(self):
        return self.weight_bytes + self.state_read_bytes + self.state_write_bytes + self.activation_bytes


def route_load(tokens, experts, k, policy):
    if tokens < 0 or not 1 <= k <= experts:
        raise ValueError('invalid routing dimensions')
    if policy == 'concentrated':
        return [tokens] * k + [0] * (experts - k)
    if policy != 'balanced':
        raise ValueError(policy)
    # Assignments are consecutive expert IDs; no token selects the same expert twice.
    n, r = divmod(tokens * k, experts)
    return [n + (i < r) for i in range(experts)]


def ffn_plan(rows, width, hidden, sram, element_bytes=2, token_tile=128, hidden_tile=128):
    """Token-major fused FFN; no weight residency across calls or experts."""
    if min(rows, width, hidden, sram, element_bytes, token_tile, hidden_tile) < 1:
        raise ValueError('invalid FFN plan dimensions')
    weights = 3 * width * hidden * element_bytes

    def activation(m, n):
        # One input tile, gate/up buffers and FP32 output accumulation; in-place SiLU/multiply.
        return m * width * element_bytes + 2 * m * n * element_bytes + 4 * m * width

    def sizes(limit):
        result = [limit]
        while result[-1] > 1:
            result.append(max(1, result[-1] // 2))
        return result

    candidates = []
    for m in sizes(min(rows, token_tile)):
        for n in sizes(min(hidden, hidden_tile)):
            retained = weights + activation(m, n) <= sram
            weight_buffer = weights if retained else 3 * width * n * element_bytes
            scratch = weight_buffer + activation(m, n)
            if scratch <= sram:
                candidates.append((1 if retained else ceildiv(rows, m), -m, -n, scratch, retained))
    if not candidates:
        raise ValueError('SRAM cannot hold the minimum FFN tile')
    passes, neg_m, neg_n, scratch, retained = min(candidates)
    m, n = -neg_m, -neg_n
    blocks = ceildiv(rows, m)
    return {'rows': rows, 'token_tile': m, 'hidden_tile': n, 'token_blocks': blocks,
            'hidden_blocks': ceildiv(hidden, n), 'scratch_bytes': scratch,
            'all_ffn_weights_retained_within_call': retained,
            'weight_passes': passes,
            'fused_intermediate_dram_bytes': 0,
            # gate/up outputs, SiLU output and gated product: write once/read once each.
            'unfused_intermediate_dram_bytes': 8 * rows * hidden * element_bytes,
            'boundary_activation_bytes': 2 * rows * width * element_bytes,
            'scope': 'no double buffering/banking/ECC; exclusive FFN SRAM allocation; '
                     'enumerate halved tile sizes, minimize weight passes then maximize M/N; '
                     'not a latency-optimal scheduler; unfused comparison keeps the same weight-pass schedule'}


def hybrid_layer_state(cfg, length, ratio, batch=1, element_bytes=2):
    result = {'local': batch * min(cfg.window, length) * cfg.head_dim * element_bytes,
              'main_history': 0, 'index_history': 0, 'compression_fp32': 0}
    if ratio:
        count, tail = divmod(length, ratio)
        overlap = ratio == 4
        result['main_history'] = batch * count * cfg.head_dim * element_bytes
        for dim in ([cfg.head_dim, cfg.index_dim] if overlap else [cfg.head_dim]):
            channels = dim * (2 if overlap else 1)
            result['compression_fp32'] += 2 * batch * tail * channels * 4
            if overlap and count:
                result['compression_fp32'] += 2 * batch * ratio * dim * 4
        if overlap:
            result['index_history'] = batch * count * cfg.index_dim * element_bytes
    return result


@lru_cache(maxsize=256)
def attention_counts(phase, length, window=0, ratio=0, topk=0):
    positions = range(1, length + 1) if phase == 'prefill' else (length,)
    local = compressed = index = 0
    for pos in positions:
        local += min(window or pos, pos)
        visible = pos // ratio if ratio else 0
        compressed += min(topk, visible) if ratio == 4 else visible
        index += visible if ratio == 4 else 0
    return {'local_pairs': local, 'compressed_pairs': compressed, 'index_candidates': index,
            'pairs': local + compressed}


def parameter_count(variant, cfg=None, hybrid=None):
    c, h = cfg or ModelConfig(), hybrid or HybridConfig()
    d, n, e = c.width, c.hidden, c.experts
    if variant in ('dense', 'moe'):
        ffn = 3 * d * (n if variant == 'dense' else e * c.expert_hidden) + (d * e if variant == 'moe' else 0)
        return 2 * c.vocab * d + c.width + c.layers * (4 * d * d + 4 * d + 2 * d + ffn)
    if variant == 'mla':
        attention = (d * c.q_rank + c.q_rank * c.heads * (c.nope_dim + c.rope_dim)
                     + d * (c.kv_rank + c.rope_dim)
                     + c.kv_rank * c.heads * (c.nope_dim + c.value_dim)
                     + c.heads * c.value_dim * d + c.q_rank + c.kv_rank)
        return 2 * c.vocab * d + d + c.layers * (attention + 2 * d + 3 * d * n)
    if variant != 'hybrid':
        raise ValueError(variant)
    d = h.width
    total = 2 * h.vocab * d + d
    for ratio in h.ratios:
        total += (d * h.q_rank + h.q_rank * h.heads * h.head_dim + d * h.head_dim
                  + h.o_rank * h.heads * h.head_dim + h.o_groups * h.o_rank * d
                  + h.heads + h.q_rank + h.head_dim + 2 * d + 3 * d * h.hidden)
        if ratio:
            channels = h.head_dim * (2 if ratio == 4 else 1)
            total += 2 * d * channels + ratio * channels + h.head_dim
        if ratio == 4:
            total += (h.q_rank * h.index_heads * h.index_dim + d * h.index_heads
                      + 4 * d * h.index_dim + 8 * h.index_dim + h.index_dim)
    return total


def build(request, sram_bytes, cfg=None, hybrid=None):
    c, h = cfg or ModelConfig(), hybrid or HybridConfig()
    r, b, q, t, z = request, request.batch, request.queries, request.context, request.element_bytes
    d, heads, layers, hidden = (h.width, h.heads, h.layers, h.hidden) if r.variant == 'hybrid' else (c.width, c.heads, c.layers, c.hidden)
    rows, ops, plans, capacities = b * q, [], [], []
    reserve = 0
    if r.variant == 'hybrid':
        for ratio in h.ratios:
            if ratio:
                for dim in ([h.head_dim, h.index_dim] if ratio == 4 else [h.head_dim]):
                    slots = min(t, ratio)
                    reserve += 2 * b * slots * dim * (2 if ratio == 4 else 1) * 4
                    if ratio == 4 and t >= ratio:
                        reserve += 2 * b * ratio * dim * 4
    ffn_sram = sram_bytes - reserve
    if ffn_sram < 1:
        raise ValueError('persistent compressor reservation exhausts SRAM; spilling is not modeled')

    def add(name, parents, **kwargs):
        ops.append(Op(name, list(parents), **kwargs))
        return name

    def linear(name, parents, m, n, k, groups=1, fp32=False, passes=1):
        return add(name, parents, **{('matrix_fp32_flops' if fp32 else 'matrix_flops'): 2 * m * n * k * groups},
                   weight_bytes=groups * n * k * z * passes, gemms=[{'m': m, 'n': n, 'k': k, 'groups': groups, 'fp32': fp32}])

    def norm(name, parent, count, width):
        return add(name, [parent], vector_ops=count * 4 * width, rsqrt=count,
                   weight_bytes=width * z, logical_shape={'rows': count, 'width': width})

    def softmax(name, parents, elements, row_count):
        return add(name, parents, exp=elements, reciprocal=row_count,
                   vector_ops=4 * elements - 2 * row_count,
                   logical_shape={'rows': row_count, 'total_entries_including_sink_or_padding': elements})

    previous = add('embedding', [], activation_bytes=rows * d * z)
    for layer in range(layers):
        p = f'layer{layer}.'
        residual = previous
        x = norm(p + 'norm_attention', previous, rows, d)
        if r.variant in ('dense', 'moe'):
            query = linear(p + 'qkv_projection', [x], rows, 3 * d, d)
            ops[-1].vector_ops = 3 * rows * d
            ops[-1].weight_bytes += 3 * d * z
            pairs = attention_counts(r.phase, t)['pairs']
            cap = 2 * b * t * d * z
            read = 2 * b * max(0, t - 1) * d * z if r.phase == 'decode' else 0
            write = cap if r.phase == 'prefill' else 2 * b * d * z
            ready = add(p + 'rope_cache', [query], vector_ops=6 * rows * d,
                        state_read_bytes=read, state_write_bytes=write)
            score = add(p + 'qk', [ready], matrix_flops=2 * b * pairs * d, vector_ops=b * heads * pairs,
                        logical_shape={'batch': b, 'heads': heads, 'queries': q, 'max_keys': t, 'head_dim': d // heads, 'causal_pairs_per_head': pairs})
            prob = softmax(p + 'softmax', [score], b * heads * pairs, rows * heads)
            value = add(p + 'pv', [prob, ready], matrix_flops=2 * b * pairs * d,
                        logical_shape={'batch': b, 'heads': heads, 'queries': q, 'max_keys': t,
                                       'value_dim': d // heads, 'causal_pairs_per_head': pairs})
            attn = linear(p + 'output_projection', [value], rows, d, d)
            ops[-1].vector_ops = rows * d
            ops[-1].weight_bytes += d * z
            capacities.append({'layer': layer, 'total': cap, 'kv': cap})
        elif r.variant == 'mla':
            a = linear(p + 'q_down', [x], rows, c.q_rank, d)
            a = norm(p + 'q_norm', a, rows, c.q_rank)
            query = linear(p + 'q_up', [a], rows, heads * (c.nope_dim + c.rope_dim), c.q_rank)
            latent = linear(p + 'kv_down', [x], rows, c.kv_rank + c.rope_dim, d)
            latent = norm(p + 'kv_norm', latent, rows, c.kv_rank)
            # Separate RoPE key, never counted as part of the latent dimension.
            cap = b * t * (c.kv_rank + c.rope_dim) * z
            state = add(p + 'rope_cache', [query, latent],
                        vector_ops=3 * rows * (heads + 1) * c.rope_dim,
                        state_read_bytes=b * max(0, t - 1) * (c.kv_rank + c.rope_dim) * z if r.phase == 'decode' else 0,
                        state_write_bytes=cap if r.phase == 'prefill' else b * (c.kv_rank + c.rope_dim) * z)
            absorbed = linear(p + 'absorb_query', [state], rows, c.kv_rank, c.nope_dim, heads)
            pairs = attention_counts(r.phase, t)['pairs']
            score = add(p + 'qk', [absorbed, state], matrix_flops=2 * b * heads * pairs * (c.kv_rank + c.rope_dim),
                        vector_ops=b * heads * pairs, logical_shape={'batch': b, 'heads': heads,
                        'queries': q, 'max_keys': t, 'dot_channels': c.kv_rank + c.rope_dim, 'causal_pairs_per_head': pairs})
            prob = softmax(p + 'softmax', [score], b * heads * pairs, rows * heads)
            val = add(p + 'pv_latent', [prob, state], matrix_flops=2 * b * heads * pairs * c.kv_rank,
                      logical_shape={'batch': b, 'heads': heads, 'queries': q, 'max_keys': t,
                                     'value_channels': c.kv_rank, 'causal_pairs_per_head': pairs})
            val = linear(p + 'value_up', [val], rows, c.value_dim, c.kv_rank, heads)
            attn = linear(p + 'output_projection', [val], rows, d, heads * c.value_dim)
            capacities.append({'layer': layer, 'total': cap, 'latent_and_rope': cap})
        else:
            ratio = h.ratios[layer]
            qr = linear(p + 'q_down', [x], rows, h.q_rank, d)
            qr = norm(p + 'q_norm', qr, rows, h.q_rank)
            query = linear(p + 'q_up', [qr], rows, heads * h.head_dim, h.q_rank)
            query = add(p + 'head_norm_rope', [query], vector_ops=rows * heads * (3 * h.head_dim + 3 * h.rope_dim), rsqrt=rows * heads)
            raw = linear(p + 'shared_kv', [x], rows, h.head_dim, d)
            raw = norm(p + 'kv_norm', raw, rows, h.head_dim)
            state = hybrid_layer_state(h, t, ratio, b, z)
            history = hybrid_layer_state(h, t - 1, ratio, b, z)
            write = sum(state.values()) - state['compression_fp32'] if r.phase == 'prefill' else b * h.head_dim * z
            read = 0
            if r.phase == 'decode':
                read += b * min(h.window - 1, t - 1) * h.head_dim * z
                if ratio:
                    selected = min(h.index_topk, t // ratio) if ratio == 4 else t // ratio
                    read += b * selected * h.head_dim * z + history['index_history']
                    # Compression state is reserved in SRAM between tokens in this contract.
            ready = add(p + 'rope_cache', [raw, query], vector_ops=3 * rows * h.rope_dim,
                        state_read_bytes=read, state_write_bytes=write)
            compressed_ready = ready
            if ratio:
                channels = h.head_dim * (2 if ratio == 4 else 1)
                projected = linear(p + 'compress_projection', [x], rows, 2 * channels, d, fp32=True)
                blocks = t // ratio if r.phase == 'prefill' else int(t % ratio == 0)
                slots = ratio * (2 if ratio == 4 else 1)
                elements = b * blocks * slots * h.head_dim
                pooled = softmax(p + 'compress_gate', [projected], elements, b * blocks * h.head_dim)
                pooled = add(p + 'compress_pool', [pooled],
                             vector_ops=2 * elements + b * blocks * (4 * h.head_dim + 3 * h.rope_dim),
                             rsqrt=b * blocks, weight_bytes=(ratio * channels + h.head_dim) * z * bool(blocks))
                compressed_ready = add(p + 'compressed_ready', [pooled, ready], commands=0)
                if r.phase == 'decode':
                    # Completed vectors are written before selection; FP32 unfinished state stays on chip.
                    ops[-1].state_write_bytes = b * blocks * h.head_dim * z
                if ratio == 4:
                    ip = linear(p + 'index_compress_projection', [x], rows, 4 * h.index_dim, d, fp32=True)
                    ie = b * blocks * 8 * h.index_dim
                    ip = softmax(p + 'index_compress_gate', [ip], ie, b * blocks * h.index_dim)
                    ip = add(p + 'index_compress_pool', [ip],
                             vector_ops=2 * ie + b * blocks * (4 * h.index_dim + 3 * h.rope_dim), rsqrt=b * blocks,
                             weight_bytes=(8 * h.index_dim + h.index_dim) * z * bool(blocks),
                             state_write_bytes=b * blocks * h.index_dim * z if r.phase == 'decode' else 0)
                    iq = linear(p + 'index_query', [qr], rows, h.index_heads * h.index_dim, h.q_rank)
                    iw = linear(p + 'index_head_weights', [x], rows, h.index_heads, d)
                    candidates = attention_counts(r.phase, t, h.window, ratio, h.index_topk)['index_candidates']
                    scored = add(p + 'index_scores', [iq, iw, ip, ready],
                                 matrix_flops=2 * b * candidates * h.index_heads * h.index_dim,
                                 vector_ops=4 * b * candidates * h.index_heads + 3 * rows * h.index_heads * h.rope_dim)
                    chosen = add(p + 'index_selection', [scored], selection_items=b * candidates)
                    compressed_ready = add(p + 'gather_ready', [chosen, compressed_ready], commands=0)
            counts = attention_counts(r.phase, t, h.window, ratio, h.index_topk)
            selected = min(h.index_topk, t // ratio) if ratio == 4 else t // ratio if ratio else 0
            shape = {'batch': b, 'heads': heads, 'queries': q, 'head_dim': h.head_dim,
                     'max_selected_keys': min(h.window, t) + selected,
                     'valid_pairs_per_head': counts['pairs'], 'ratio': ratio}
            score = add(p + 'qk', [compressed_ready], matrix_flops=2 * b * heads * counts['pairs'] * h.head_dim,
                        vector_ops=b * heads * counts['pairs'], logical_shape=shape)
            prob = softmax(p + 'softmax_sink', [score], b * heads * counts['pairs'] + rows * heads, rows * heads)
            val = add(p + 'pv', [prob, compressed_ready], matrix_flops=2 * b * heads * counts['pairs'] * h.head_dim,
                      vector_ops=3 * rows * heads * h.rope_dim, logical_shape=shape)
            val = linear(p + 'group_output', [val], rows, h.o_rank, heads * h.head_dim // h.o_groups, h.o_groups)
            attn = linear(p + 'output_projection', [val], rows, d, h.o_groups * h.o_rank)
            capacities.append({'layer': layer, 'ratio': ratio, **state, 'total': sum(state.values())})
        residual2 = add(p + 'attention_residual', [residual, attn], vector_ops=rows * d)
        ffn_input = norm(p + 'norm_ffn', residual2, rows, d)
        counts = route_load(rows, c.experts, c.top_k, r.route_policy) if r.variant == 'moe' else [rows]
        if r.variant == 'moe':
            router = linear(p + 'router_projection', [ffn_input], rows, c.experts, d)
            router = softmax(p + 'router_softmax', [router], rows * c.experts, rows)
            router = add(p + 'router_topk', [router], selection_items=rows * c.experts,
                         reciprocal=rows, vector_ops=rows * c.top_k)
            ffn_input = add(p + 'dispatch', [router, ffn_input], activation_bytes=rows * c.top_k * d * z)
        ends = []
        for expert, count in enumerate(counts):
            if not count:
                continue
            n = c.expert_hidden if r.variant == 'moe' else hidden
            plan = ffn_plan(count, d, n, ffn_sram, z, r.token_tile, r.hidden_tile)
            plans.append({'layer': layer, 'expert': expert if r.variant == 'moe' else None, **plan})
            f = p + (f'expert{expert}.' if r.variant == 'moe' else 'ffn.')
            gate = linear(f + 'gate', [ffn_input], count, n, d, passes=plan['weight_passes'])
            ops[-1].gemms[0].update(m_tile=plan['token_tile'], n_tile=plan['hidden_tile'])
            up = linear(f + 'up', [ffn_input], count, n, d, passes=plan['weight_passes'])
            ops[-1].gemms[0].update(m_tile=plan['token_tile'], n_tile=plan['hidden_tile'])
            active = add(f + 'silu', [gate], exp=count * n, reciprocal=count * n,
                         vector_ops=3 * count * n, activation=True, commands=int(r.ffn_mode == 'unfused'),
                         logical_shape={'rows': count, 'width': n})
            product = add(f + 'multiply', [active, up], vector_ops=count * n, commands=int(r.ffn_mode == 'unfused'))
            end = linear(f + 'down', [product], count, d, n, passes=plan['weight_passes'])
            ops[-1].gemms[0].update(m_tile=plan['token_tile'], k_tile=plan['hidden_tile'])
            ops[-1].activation_bytes = plan['boundary_activation_bytes']
            if r.ffn_mode == 'unfused':
                ops[-1].activation_bytes += plan['unfused_intermediate_dram_bytes']
            ends.append(end)
        if r.variant == 'moe':
            combine = add(p + 'combine', ends, vector_ops=2 * rows * c.top_k * d,
                          activation_bytes=rows * c.top_k * d * z)
            ends = [combine]
        previous = add(p + 'ffn_residual', [residual2, *ends], vector_ops=rows * d)
    previous = norm('final_norm', previous, b, d)
    linear('lm_head_last_position', [previous], b, c.vocab, d)
    return {'request': asdict(r), 'parameters': parameter_count(r.variant, c, h),
            'weight_capacity_bytes': parameter_count(r.variant, c, h) * z,
            'logical_state_bytes': sum(v['total'] for v in capacities), 'layer_states': capacities,
            'ops': ops, 'ffn_plans': plans, 'compressor_sram_reservation_bytes': reserve,
            'precision_contract': 'BF16-like 2-byte storage by default; compressor GEMMs budgeted at FP32 rate; '
                                  'FP32 compression state and FFN accumulation; not a numerical accuracy test'}
