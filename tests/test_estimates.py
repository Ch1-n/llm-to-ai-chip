import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from estimate_workload import Config, block_flops, estimate, kv_bytes


class EstimateTests(unittest.TestCase):
    def test_dense_block_matches_matrix_shapes(self):
        c = Config()
        result = block_flops(c, 512, 512)
        self.assertEqual(result['qkv_projection'], 1811939328)
        self.assertEqual(result['attention_qk_and_pv'], 805306368)
        self.assertEqual(result['output_projection'], 603979776)
        self.assertEqual(result['swiglu_three_projections'], 4831838208)
        self.assertEqual(sum(result.values()), 8053063680)

    def test_decoder_and_head_accounting(self):
        result = estimate(Config())
        self.assertEqual(result['parameters'], 119690496)
        self.assertEqual(result['prefill_decoder_flops'], 48318382080)
        self.assertEqual(result['decode_decoder_and_lm_head_flops'], 171566592)
        self.assertEqual(result['lm_head_all_prefill_positions_flops'], 39523713024)

    def test_cache_capacity_and_append_are_different(self):
        result = estimate(Config())
        self.assertEqual(result['kv_cache_capacity_bytes'], 9 * 2**20)
        self.assertEqual(result['kv_new_token_append_bytes'], 18 * 2**10)

    def test_gqa_capacity_example(self):
        mha = kv_bytes(32, 1, 8192, 32, 128, 2)
        gqa = kv_bytes(32, 1, 8192, 8, 128, 2)
        self.assertEqual(mha, 4 * 2**30)
        self.assertEqual(gqa, 2**30)
        self.assertEqual(mha, 4 * gqa)

    def test_batch_scaling(self):
        one = estimate(Config())
        twelve = estimate(Config(batch=12))
        self.assertEqual(one['parameters'], twelve['parameters'])
        self.assertEqual(twelve['kv_cache_capacity_bytes'], 12 * one['kv_cache_capacity_bytes'])
        self.assertEqual(twelve['prefill_decoder_flops'], 12 * one['prefill_decoder_flops'])
        self.assertEqual(twelve['ideal_decode_projection_weight_read_bytes'], one['ideal_decode_projection_weight_read_bytes'])

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            estimate(Config(hidden=769))
        with self.assertRaises(ValueError):
            kv_bytes(32, 1, 0, 8, 128, 2)


if __name__ == '__main__':
    unittest.main()
