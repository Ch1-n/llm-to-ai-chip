"""Independent checks against the pinned upstream MLA definitions."""

from dataclasses import replace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter02.check_mla_reference import (equivalence_case, gradient_case,
                                                    inference_case)
    from tools.chapter02.config import preset
    from tools.chapter02.mla_reference import MLAReference
    from tools.chapter02.attention import BackendUnavailable


@unittest.skipIf(torch is None, 'requires optional torch')
class MLAReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        torch.set_num_threads(1)
        self.cfg = replace(preset('tiny'), max_context=32)

    def test_upstream_prefill_and_chunked_cache(self):
        for path in ('naive', 'absorb'):
            for mode in ('cat', 'static'):
                result = inference_case(self.cfg, path, torch.device('cpu'), torch.float32,
                                        (3, 1, 5), mode)
                self.assertEqual(result['status'], 'passed')

    def test_upstream_input_and_all_parameter_gradients(self):
        for path in ('naive', 'absorb'):
            self.assertEqual(gradient_case(self.cfg, path, torch.device('cpu'))['status'], 'passed')

    def test_naive_absorb_equivalence(self):
        self.assertEqual(equivalence_case(self.cfg, torch.device('cpu'))['status'], 'passed')

    def test_no_silent_flash_fallback(self):
        with self.assertRaises(BackendUnavailable):
            MLAReference(self.cfg, backend='fa2')


if __name__ == '__main__':
    unittest.main()
