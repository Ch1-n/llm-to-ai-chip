"""Small CLI integration tests, using artificial tokens rather than corpus data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import numpy as np
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter02.run import main


@unittest.skipIf(torch is None, 'requires optional torch and numpy')
class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_command(self, operation, arguments=()):
        destination = self.root / operation
        with redirect_stdout(io.StringIO()):
            code = main([operation, '--output', str(destination), '--threads', '1', *arguments])
        self.assertEqual(code, 0)
        return destination

    def test_model_benchmark_all_phases(self):
        directory = self.run_command('bench', ['--context', '8', '--warmups', '1', '--repeats', '1',
            '--variants', 'dense', 'mla-absorbed', 'moe', 'mla-moe',
            '--phases', 'prefill', 'decode', 'train', '--strict'])
        rows = [json.loads(line) for line in (directory / 'benchmarks.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r['status'] == 'completed' for r in rows))
        self.assertTrue(all(r['performance']['cuda_event_median_ms'] is None for r in rows))
        for row in rows:
            if row['phase'] == 'decode':
                self.assertEqual(row['observations']['logits_shape'][1], 1)

    def test_attention_forward_and_backward(self):
        for backward in (False, True):
            name = 'attention'
            destination = self.root / f'micro-{backward}'
            with redirect_stdout(io.StringIO()):
                result = main([name, '--output', str(destination), '--threads', '1',
                    '--kv-length', '9', '--q-lengths', '1', '9', '--head-dim', '8',
                    '--heads', '4', '--kv-heads', '2', '--backends', 'math', 'head-loop', 'sdpa-math',
                    '--warmups', '1', '--repeats', '1', '--strict', *(['--backward'] if backward else [])])
            self.assertEqual(result, 0)

    def test_profile_outputs(self):
        directory = self.run_command('profile', ['--variant', 'mla-moe', '--context', '8',
                                                '--warmups', '1', '--steps', '1'])
        self.assertTrue((directory / 'trace.json').is_file())
        operators = json.loads((directory / 'operators.json').read_text())
        self.assertTrue(any(row['name'] == 'mla.absorb_key_projection' for row in operators))
        self.assertTrue(any(row['name'] == 'moe.dispatch_experts_combine' for row in operators))

    def test_text_training_pipeline_with_artificial_fixture(self):
        data = self.root / 'data'
        data.mkdir()
        (data / 'manifest.json').write_text(json.dumps({'source': 'artificial unit-test fixture, not Seq-Monkey'}))
        for split in ('train', 'val'):
            (np.arange(257, dtype=np.uint16) % 97).tofile(data / f'{split}.bin')
        for variant in ('dense', 'mla-absorbed', 'moe', 'mla-moe'):
            destination = self.root / f'train-{variant}'
            with redirect_stdout(io.StringIO()):
                result = main(['train', '--output', str(destination), '--threads', '1',
                    '--data', str(data), '--context', '8', '--batch', '2', '--steps', '3',
                    '--eval-every', '2', '--eval-blocks', '4', '--variant', variant])
            self.assertEqual(result, 0)
            summary = json.loads((destination / 'summary.json').read_text())
            self.assertEqual(summary['tokens_seen'], 48)
            self.assertEqual(summary['status'], 'completed')

    def test_unavailable_not_silently_replaced(self):
        destination = self.root / 'unavailable'
        with redirect_stdout(io.StringIO()):
            code = main(['bench', '--output', str(destination), '--backends', 'fa3',
                         '--context', '8', '--warmups', '1', '--repeats', '1', '--strict'])
        self.assertEqual(code, 1)
        rows = [json.loads(line) for line in (destination / 'benchmarks.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['status'] == 'unavailable' for row in rows))
        self.assertTrue(all('performance' not in row for row in rows))


if __name__ == '__main__':
    unittest.main()
