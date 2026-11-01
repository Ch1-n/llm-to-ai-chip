import unittest

from tools.chapter02.trace_summary import summarize, union_duration


class TraceSummaryTests(unittest.TestCase):
    def test_overlap_union(self):
        self.assertEqual(union_duration([(1, 4), (2, 5), (7, 8)]), 5)
        self.assertEqual(union_duration([]), 0)

    def test_exclude_gpu_annotations(self):
        result = summarize({'traceEvents': [
            {'ph': 'X', 'cat': 'gpu_user_annotation', 'name': 'block', 'ts': 0, 'dur': 100},
            {'ph': 'X', 'cat': 'kernel', 'name': 'gemm', 'ts': 1, 'dur': 3, 'args': {'device': 0}},
            {'ph': 'X', 'cat': 'kernel', 'name': 'gemm', 'ts': 3, 'dur': 3, 'args': {'device': 0}},
            {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaLaunchKernel', 'ts': 0, 'dur': 1}]})
        row = result['devices'][0]
        self.assertEqual(row['kernel_count'], 2)
        self.assertEqual(row['kernel_sum_us'], 6)
        self.assertEqual(row['kernel_union_us'], 5)
        self.assertEqual(row['kernel_coverage'], 1)
