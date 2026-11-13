from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.chapter04.hardware import Hardware, budget
from tools.chapter04.model import (Op, Request, attention_counts, build, ffn_plan,
                                  hybrid_layer_state, parameter_count, route_load)
from tools.chapter03.config import Config as HybridConfig
from tools.chapter04.run import main


class WorkloadTests(unittest.TestCase):
    def test_parameters_match_existing_models(self):
        self.assertEqual({v: parameter_count(v) for v in ('dense', 'mla', 'moe', 'hybrid')},
                         dict(dense=119690496, mla=111858432, moe=148038912, hybrid=109310216))

    def test_dense_decode_matches_chapter_one_matrix_count(self):
        work = build(Request(context=512), Hardware().sram_bytes)
        self.assertEqual(sum(op.matrix_flops for op in work['ops']), 171566592)
        self.assertEqual(work['logical_state_bytes'], 9 * 2**20)
        self.assertEqual(sum(op.state_write_bytes for op in work['ops']), 18 * 2**10)

    def test_existing_8192_state_accounting(self):
        expected = {'dense': 144., 'mla': 15., 'moe': 144., 'hybrid': .865234375}
        for v, mib in expected.items():
            self.assertEqual(build(Request(v, context=8192), Hardware().sram_bytes)['logical_state_bytes'], mib * 2**20)

    def test_compressor_incomplete_and_completed_states(self):
        cfg = HybridConfig()
        self.assertEqual(sum(hybrid_layer_state(cfg, 3, 4).values()), 4992)
        self.assertEqual(sum(hybrid_layer_state(cfg, 4, 4).values()), 3776)
        self.assertEqual(hybrid_layer_state(cfg, 129, 128)['main_history'], 128)
        self.assertEqual(hybrid_layer_state(cfg, 129, 128)['compression_fp32'], 512)

    def test_causal_pairs_and_closed_blocks(self):
        self.assertEqual(attention_counts('prefill', 5)['pairs'], 15)
        self.assertEqual(attention_counts('prefill', 5, 2)['pairs'], 9)
        self.assertEqual(attention_counts('decode', 127, 8, 128)['compressed_pairs'], 0)
        self.assertEqual(attention_counts('decode', 128, 8, 128)['compressed_pairs'], 1)
        self.assertEqual(attention_counts('decode', 127, 8, 4, 2)['pairs'], 10)

    def test_routing_preserves_top_k_and_load_extremes(self):
        for tokens in (1, 3, 16, 512):
            balanced = route_load(tokens, 8, 2, 'balanced')
            self.assertEqual(sum(balanced), 2 * tokens)
            self.assertLessEqual(max(balanced) - min(balanced), 1)
            self.assertEqual(route_load(tokens, 8, 2, 'concentrated'), [tokens, tokens, 0, 0, 0, 0, 0, 0])

    def test_silu_counts_match_activated_ffn(self):
        hw = Hardware()
        for v, hidden in (('dense', 2048), ('mla', 2048), ('moe', 2 * 512), ('hybrid', 2048)):
            r = budget(build(Request(v, 'prefill', 4, 32), hw.sram_bytes), hw)
            self.assertEqual(r['silu_elements'], 6 * 4 * 32 * hidden)

    def test_ffn_plan_capacity_and_retention(self):
        small = ffn_plan(512, 768, 2048, 2**20)
        large = ffn_plan(512, 768, 2048, 16 * 2**20)
        self.assertLessEqual(small['scratch_bytes'], 2**20)
        self.assertGreater(small['weight_passes'], large['weight_passes'])
        self.assertTrue(large['all_ffn_weights_retained_within_call'])
        self.assertEqual(large['weight_passes'], 1)
        self.assertEqual(small['unfused_intermediate_dram_bytes'], 8 * 512 * 2048 * 2)
        self.assertGreaterEqual(ffn_plan(2048, 768, 2048, 128 * 1024)['token_tile'], 8)
        with self.assertRaises(ValueError):
            ffn_plan(1, 768, 2048, 1)

    def test_no_silent_compressor_spill(self):
        with self.assertRaisesRegex(ValueError, 'compressor reservation'):
            build(Request('hybrid', batch=16), int(.125 * 2**20))

    def test_prefill_keeps_fp32_compression_state_on_chip(self):
        work = build(Request('hybrid', 'prefill', context=513), Hardware().sram_bytes)
        fp32 = sum(layer['compression_fp32'] for layer in work['layer_states'])
        self.assertGreater(fp32, 0)
        self.assertEqual(sum(op.state_write_bytes for op in work['ops']), work['logical_state_bytes'] - fp32)

    def test_fusion_changes_traffic_not_math(self):
        hw = Hardware()
        a = budget(build(Request(phase='prefill'), hw.sram_bytes), hw)
        b = budget(build(Request(phase='prefill', ffn_mode='unfused'), hw.sram_bytes), hw)
        for key in ('matrix_flops', 'vector_ops', 'exp', 'reciprocal', 'weight_bytes'):
            self.assertEqual(a['totals'][key], b['totals'][key])
        self.assertEqual(b['dram_payload_bytes'] - a['dram_payload_bytes'], 6 * 8 * 2048 * 2048 * 2)

    def test_tile_padding_and_resource_floor(self):
        hw = Hardware()
        work = build(Request(), hw.sram_bytes)
        a = budget(work, hw)
        b = budget(work, replace(hw, linear_tile_m=1))
        self.assertGreater(a['issued_linear_padding_plus_useful_attention_flops'], b['issued_linear_padding_plus_useful_attention_flops'])
        self.assertGreaterEqual(a['conditional_lower_bound_ms'], max(a['resource_ms'].values()))
        self.assertLessEqual(a['conditional_lower_bound_ms'], a['serial_analytical_budget_ms'])

    def test_more_bandwidth_cannot_worsen_fixed_contract(self):
        hw = Hardware()
        work = build(Request(), hw.sram_bytes)
        slow = budget(work, hw)
        fast = budget(work, replace(hw, bandwidth_gbs=400))
        self.assertEqual(slow['totals'], fast['totals'])
        self.assertGreaterEqual(slow['conditional_lower_bound_ms'], fast['conditional_lower_bound_ms'])

    def test_silu_acceleration_does_not_speed_other_sfu_work(self):
        hw = Hardware()
        work = build(Request(phase='prefill'), hw.sram_bytes)
        a, b = budget(work, hw), budget(work, replace(hw, silu_speedup=16))
        self.assertEqual(a['totals'], b['totals'])
        self.assertEqual(a['resource_ms']['dram'], b['resource_ms']['dram'])
        non_silu = sum((op.exp / hw.exp_gops + op.reciprocal / hw.reciprocal_gops + op.rsqrt / hw.rsqrt_gops)
                      / 1e6 for op in work['ops'] if not op.activation)
        self.assertGreaterEqual(b['resource_ms']['shared_sfu'], non_silu)
        self.assertGreater(a['resource_ms']['shared_sfu'], b['resource_ms']['shared_sfu'])

    def test_invalid_inputs(self):
        for kwargs in ({'batch': 0}, {'context': 8193}, {'variant': 'other'}, {'batch': True}):
            with self.assertRaises(ValueError):
                Request(**kwargs)
        for kwargs in ({'matrix_tflops': 0}, {'bandwidth_gbs': math.nan}, {'dispatch_us': -1}, {'linear_tile_m': 1.5}):
            with self.assertRaises(ValueError):
                Hardware(**kwargs)

    def test_cli_writes_explicit_assumptions_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(temp) / 'report'
            self.assertEqual(main(['--contexts', '32', '--batches', '1', '--output', str(out)]), 0)
            meta = json.loads((out / 'metadata.json').read_text())
            self.assertEqual(meta['baseline_cases'], 8)
            self.assertEqual(meta['status'], 'analytical_estimate_not_hardware_measurement')
            self.assertGreater(meta['infeasible_sram_cases'], 0)
            self.assertEqual(len(json.loads((out / 'baseline.json').read_text())), 8)
            with self.assertRaises(SystemExit):
                main(['--output', str(out)])
