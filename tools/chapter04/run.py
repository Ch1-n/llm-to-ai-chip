"""CPU-only analytical experiments for chapter four; no measured NPU timings."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import platform

from .hardware import Hardware, budget
from .model import Request, VARIANTS, build
from tools.chapter02.config import ModelConfig
from tools.chapter03.config import Config as HybridConfig


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def evaluate(request, hw):
    work = build(request, hw.sram_bytes)
    return work, budget(work, hw)


def compact(report):
    return {key: value for key, value in report.items() if key != 'stage_budgets'}


def csv_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/chapter04/20261101'))
    parser.add_argument('--contexts', type=int, nargs='+', default=[512, 2048, 8192])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--hardware', type=Path, help='JSON fields matching the Hardware dataclass')
    parser.add_argument('--token-tile', type=int, default=128)
    parser.add_argument('--hidden-tile', type=int, default=128)
    parser.add_argument('--ffn-mode', choices=('fused', 'unfused'), default='fused')
    parser.add_argument('--route-policy', choices=('balanced', 'concentrated'), default='balanced')
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('use a new empty output directory')
    # Validate the whole request grid before writing any result.
    contract = {'token_tile': args.token_tile, 'hidden_tile': args.hidden_tile,
                'ffn_mode': args.ffn_mode, 'route_policy': args.route_policy}
    requests = [Request(v, phase, b, t, **contract) for v, phase, b, t in
                itertools.product(VARIANTS, ('prefill', 'decode'), args.batches, args.contexts)]
    hw = Hardware(**json.loads(args.hardware.read_text())) if args.hardware else Hardware()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, inventories = [], []
    for request in requests:
        work, report = evaluate(request, hw)
        baseline.append(compact(report))
        if request.batch == 1 and request.context == args.contexts[len(args.contexts) // 2]:
            inventories.append({**report, 'ops': [asdict(op) for op in work['ops']],
                                'ffn_plans': work['ffn_plans'], 'layer_states': work['layer_states']})
    write(args.output / 'baseline.json', baseline)
    write(args.output / 'operator-inventory.json', inventories)
    csv_rows(args.output / 'baseline.csv', [
        {**r['request'], 'logical_state_bytes': r['logical_state_bytes'],
         'matrix_flops': r['totals']['matrix_flops'], 'matrix_fp32_flops': r['totals']['matrix_fp32_flops'],
         'exp_elements': r['totals']['exp'], 'silu_elements': r['silu_elements'],
         'dram_payload_bytes': r['dram_payload_bytes'], 'conditional_lower_bound_ms': r['conditional_lower_bound_ms'],
         'whole_stage_barrier_path_ms': r['whole_stage_barrier_path_ms'],
         'serial_analytical_budget_ms': r['serial_analytical_budget_ms'],
         'largest_aggregate_resource': r['largest_aggregate_resource']}
        for r in baseline])
    axes = {'matrix_tflops': [5., 20., 80.], 'bandwidth_gbs': [25., 100., 400.],
            'sram_mib': [.125, 1., 4., 16.], 'silu_speedup': [1., 4., 16.],
            'sfu_speedup': [1., 4., 16.], 'dispatch_us': [0., .2, 2.],
            'linear_tile_m': [1, 16, 64]}
    sweep = []
    for variant, phase, batch in itertools.product(VARIANTS, ('prefill', 'decode'), (1, 16)):
        request = Request(variant, phase, batch, 2048, **contract)
        for axis, values in axes.items():
            for value in values:
                hardware = replace(hw, **{axis: value})
                try:
                    _, report = evaluate(request, hardware)
                    row = {**compact(report), 'axis': axis, 'value': value}
                except ValueError as error:
                    row = {'axis': axis, 'value': value, 'request': asdict(request),
                           'hardware': asdict(hardware), 'status': 'infeasible_under_sram_contract', 'reason': str(error)}
                sweep.append(row)
        for mode, route in itertools.product(('fused', 'unfused'), ('balanced', 'concentrated') if variant == 'moe' else ('balanced',)):
            _, report = evaluate(replace(request, ffn_mode=mode, route_policy=route), hw)
            sweep.append({**compact(report), 'axis': 'ffn_route_contract', 'value': mode + '/' + route})
    write(args.output / 'sensitivity.json', sweep)
    grid = []
    for phase, matrix, bandwidth in itertools.product(('prefill', 'decode'), (5., 20., 80.), (25., 100., 400.)):
        _, report = evaluate(Request('dense', phase, 1, 2048, **contract), replace(hw, matrix_tflops=matrix, bandwidth_gbs=bandwidth))
        grid.append(compact(report))
    write(args.output / 'matrix-bandwidth-grid.json', grid)
    sram_fusion = []
    for variant, phase, batch, capacity, mode in itertools.product(
        ('dense', 'hybrid'), ('prefill', 'decode'), (1, 16), (.125, 1., 4., 16.), ('fused', 'unfused')):
        request = replace(Request(variant, phase, batch, 2048, **contract), ffn_mode=mode)
        hardware = replace(hw, sram_mib=capacity)
        try:
            _, report = evaluate(request, hardware)
            sram_fusion.append(compact(report))
        except ValueError as error:
            sram_fusion.append({'request': asdict(request), 'hardware': asdict(hardware),
                                'status': 'infeasible_under_sram_contract', 'reason': str(error)})
    write(args.output / 'sram-fusion.json', sram_fusion)
    metadata = {'status': 'analytical_estimate_not_hardware_measurement', 'python': platform.python_version(),
                'hardware': asdict(hw), 'hardware_origin': 'hypothetical, not a product specification or GPU calibration',
                'model_origin': 'teaching configs in tools/chapter02 and tools/chapter03; not full DeepSeek',
                'model_configs': {'chapter02': asdict(ModelConfig()), 'hybrid': asdict(HybridConfig())},
                'execution_contract': contract,
                'baseline_cases': len(baseline), 'sensitivity_cases': len(sweep), 'matrix_bandwidth_cases': len(grid),
                'sram_fusion_cases': len(sram_fusion),
                'infeasible_sram_fusion_cases': sum(r['status'] == 'infeasible_under_sram_contract' for r in sram_fusion),
                'infeasible_sram_cases': sum(r['status'] == 'infeasible_under_sram_contract' for r in sweep),
                'source_sha256': {str(p.resolve().relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                    for directory in (Path(__file__).parent, Path('tools/chapter02'), Path('tools/chapter03'))
                    for p in sorted(directory.glob('*.py'))},
                'contracts': [
                    'Prefill starts empty; Decode context includes the current token; last-position LM head only.',
                    '2 FLOPs per multiply-add; Attention counts useful causal entries, not full masked square products.',
                    'Linear GEMMs have padded tile dimensions; Attention/Indexer padding is not modeled.',
                    'Matrix FP32 and BF16-equivalent work shares one array at separate assumed rates.',
                    'Vector arithmetic includes explicit reduction-operation equivalents; no measured utilization.',
                    'RoPE sin/cos tables assumed precomputed; primitive vector counts are an explicit arithmetic budget, not ISA instruction counts.',
                    'Exp/reciprocal/rsqrt share one SFU budget; selection counts scanned candidates, not sorting comparisons.',
                    'No projection-weight residency across requests; each non-FFN matrix weight read once per call.',
                    'Attention assumes IO-aware on-chip scores; no score/probability DRAM spill or query-tile KV rereads.',
                    'Other non-FFN intermediate tensors assumed reused on chip; residual spills and their feasibility are not modeled.',
                    'QKV is one target logical command in Dense/MoE; this is not the original PyTorch launch structure.',
                    'Decode reads requested state payload once per batch item; MLA latent reused between QK/PV.',
                    'Hybrid selected main vectors budgeted as DRAM reads even if just produced; actual selected IDs unknown.',
                    'FP32 compressor buffers reserve SRAM, including a full pending block; no spilling of those buffers.',
                    'SRAM feasibility check covers compressor reservation and one exclusive FFN tile only, not the whole NPU.',
                    'FFN token-major schedule rereads all weights per token tile unless full FFN weights fit.',
                    'Unfused FFN uses the same weight passes, plus four intermediate tensors written/read once.',
                    'MoE routing is deterministic balanced/concentrated synthetic allocation, not measured router decisions.',
                    'Dispatch is one scalar queue, may overlap device resources; commands count logical operations, not CUDA kernels.',
                    'SiLU speedup hypothetically accelerates its vector/SFU work, without changing semantics or testing approximation error.',
                    'Resource lower bound is conditional, not predicted latency; no achievable schedule is constructed.',
                    'Whole-stage dependency barriers and serial budgets exclude tile streaming; neither is a latency guarantee.',
                    'No banking, contention, DMA startup, cache misses, tensor-core efficiency calibration, area or power model.']}
    write(args.output / 'metadata.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('status', 'baseline_cases', 'sensitivity_cases', 'infeasible_sram_cases')}))
    return 0


if __name__ == '__main__':
    main()
