"""Resource/dependency bounds and explicit analytical budgets, not scheduling simulation."""

from dataclasses import asdict, dataclass
import math

from .model import ceildiv


@dataclass(frozen=True)
class Hardware:
    matrix_tflops: float = 20.
    fp32_matrix_tflops: float = 5.
    bandwidth_gbs: float = 100.
    vector_gops: float = 64.
    exp_gops: float = 4.
    reciprocal_gops: float = 4.
    rsqrt_gops: float = 1.
    selection_gitems: float = 1.
    sram_mib: float = 4.
    dispatch_us: float = .2
    silu_speedup: float = 1.
    sfu_speedup: float = 1.
    linear_tile_m: int = 16
    linear_tile_n: int = 16
    linear_tile_k: int = 16

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (name != 'dispatch_us' and value == 0):
                raise ValueError('hardware rates/dimensions must be finite and positive')
        for name in ('linear_tile_m', 'linear_tile_n', 'linear_tile_k'):
            if type(getattr(self, name)) is not int:
                raise ValueError('linear tile dimensions must be integers')

    @property
    def sram_bytes(self):
        return int(self.sram_mib * 2**20)


def budget(workload, hardware):
    hw = hardware
    sums = dict(matrix=0., dram=0., vector=0., shared_sfu=0., selection=0., dispatch=0.)
    completion, details = {}, []
    totals = dict(matrix_flops=0, matrix_fp32_flops=0, vector_ops=0, exp=0,
                  reciprocal=0, rsqrt=0, selection_items=0, weight_bytes=0,
                  state_read_bytes=0, state_write_bytes=0, activation_bytes=0)
    issued_total, serial, silu_serial = 0, 0., 0.
    for op in workload['ops']:
        if any(parent not in completion for parent in op.parents):
            raise ValueError('operation DAG must be topologically ordered')
        for key in totals:
            totals[key] += getattr(op, key)
        padding, padding32 = 0, 0
        for gemm in op.gemms:
            issued = 2 * gemm['groups']
            for axis, tile in (('m', hw.linear_tile_m), ('n', hw.linear_tile_n), ('k', hw.linear_tile_k)):
                size, software_tile = gemm[axis], gemm.get(axis + '_tile', gemm[axis])
                whole, tail = divmod(size, software_tile)
                issued *= whole * ceildiv(software_tile, tile) * tile + ceildiv(tail, tile) * tile
            useful = 2 * gemm['groups'] * gemm['m'] * gemm['n'] * gemm['k']
            if gemm['fp32']:
                padding32 += issued - useful
            else:
                padding += issued - useful
        issued_total += op.matrix_flops + op.matrix_fp32_flops + padding + padding32
        # Non-matrix vector/SFU throughput shares the SiLU speedup in that stage only.
        factor = hw.silu_speedup if op.activation else 1.
        resources = {
            'matrix': (op.matrix_flops + padding) / (hw.matrix_tflops * 1e12)
                      + (op.matrix_fp32_flops + padding32) / (hw.fp32_matrix_tflops * 1e12),
            'dram': op.dram_bytes / (hw.bandwidth_gbs * 1e9),
            'vector': op.vector_ops / (hw.vector_gops * 1e9 * factor),
            'shared_sfu': (op.exp / hw.exp_gops + op.reciprocal / hw.reciprocal_gops
                           + op.rsqrt / hw.rsqrt_gops) / (1e9 * hw.sfu_speedup * factor),
            'selection': op.selection_items / (hw.selection_gitems * 1e9),
            'dispatch': op.commands * hw.dispatch_us / 1e6}
        for name, value in resources.items():
            sums[name] += value
        service = max(value for name, value in resources.items() if name != 'dispatch')
        # Dispatch is a separate precedence node, followed by overlapping service resources.
        duration = service + resources['dispatch']
        completion[op.name] = max((completion[p] for p in op.parents), default=0.) + duration
        serial += duration
        if op.activation:
            silu_serial += duration
        details.append({'name': op.name, 'own_budget_ms': duration * 1e3,
                        'dram_bytes': op.dram_bytes, 'resources_ms': {k: v * 1e3 for k, v in resources.items()}})
    payload = sum(totals[k] for k in ('weight_bytes', 'state_read_bytes', 'state_write_bytes', 'activation_bytes'))
    critical = max(completion.values(), default=0.)
    bound = max(sums.values())
    return {'hardware': asdict(hw), 'request': workload['request'], 'totals': totals,
            'silu_elements': sum(op.exp for op in workload['ops'] if op.activation),
            'parameters': workload['parameters'], 'weight_capacity_bytes': workload['weight_capacity_bytes'],
            'logical_state_bytes': workload['logical_state_bytes'], 'dram_payload_bytes': payload,
            'useful_matrix_flops_per_dram_byte': (totals['matrix_flops'] + totals['matrix_fp32_flops']) / payload,
            'issued_linear_padding_plus_useful_attention_flops': issued_total,
            'resource_ms': {k: v * 1e3 for k, v in sums.items()},
            'whole_stage_barrier_path_ms': critical * 1e3,
            'conditional_lower_bound_ms': bound * 1e3,
            'serial_analytical_budget_ms': serial * 1e3,
            'silu_fraction_of_serial_budget': silu_serial / serial,
            'largest_aggregate_resource': max(sums, key=sums.get),
            'max_ffn_scratch_bytes': max(p['scratch_bytes'] for p in workload['ffn_plans']),
            'compressor_sram_reservation_bytes': workload['compressor_sram_reservation_bytes'],
            'max_ffn_weight_passes': max(p['weight_passes'] for p in workload['ffn_plans']),
            'stage_budgets': details,
            'status': 'analytical_scenario_not_hardware_measurement',
            'timing_scope': 'conditional on traffic, tile, primitive-throughput and dependency contracts; '
                            'resource lower bound is not an achievable schedule; whole-stage barrier path '
                            'disallows producer/consumer tile streaming and is not a lower bound; '
                            'serial budget is not an upper bound'}
