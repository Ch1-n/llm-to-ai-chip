from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter03.run import main


@unittest.skipIf(torch is None, 'requires optional numpy and torch')
class HybridCLITests(unittest.TestCase):
    def test_small_benchmark(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['bench', '--preset', 'tiny', '--output', temp,
                '--contexts', '17', '--variants', 'shared-dense', 'swa', 'hybrid',
                '--warmups', '1', '--repeats', '1']), 0)
            rows = [json.loads(line) for line in (Path(temp) / 'benchmarks.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['status'] == 'completed' for r in rows))

    def test_training_with_explicit_artificial_fixture(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            hashes = {}
            for split in ('train', 'val'):
                p = data / f'{split}.bin'
                (np.arange(1200, dtype=np.uint16) % 97).tofile(p)
                hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()
            (data / 'manifest.json').write_text(json.dumps({'token_bin_sha256': hashes,
                'documents': {'train': ['artificial-train'], 'val': ['artificial-val']}}))
            self.assertEqual(main(['train', '--preset', 'tiny', '--output', str(root / 'run'),
                '--data', str(data), '--context', '137', '--batch', '1', '--steps', '2',
                '--eval-every', '1', '--eval-blocks', '2']), 0)
            summary = json.loads((root / 'run/summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
