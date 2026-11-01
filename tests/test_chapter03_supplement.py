from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.chapter03.config import preset
    from tools.chapter03.model import LanguageModel
    from tools.chapter03.supplement import main


@unittest.skipIf(torch is None, 'optional torch and numpy')
class SupplementTests(unittest.TestCase):
    def call(self, args):
        with patch.object(sys, 'argv', ['supplement'] + args), redirect_stdout(io.StringIO()):
            main()

    def test_continuous_advances_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            self.call(['continuous', '--device', 'cpu', '--preset', 'tiny', '--variants',
                       'swa', 'hybrid', '--prefix', '127', '--steps', '5', '--repeats', '2', '--output', temp])
            rows = json.loads((Path(temp) / 'continuous.json').read_text())['records']
            for row in rows:
                self.assertEqual([r['context'] for r in row['samples']], list(range(128, 133)) * 2)
                self.assertTrue(all(r['length'] == 132 for r in row['final_cache']['layers']))

    def test_checkpoint_checks_and_paired_suffix(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = preset('tiny', 'hybrid')
            torch.manual_seed(1024)
            path = root / 'fixture.pt'
            torch.save({'model': LanguageModel(cfg).state_dict(), 'config': asdict(cfg),
                        'variant': 'hybrid', 'step': 0}, path)
            self.call(['check', '--device', 'cpu', '--checkpoints', str(path),
                       '--contexts', '258', '--output', str(root / 'check')])
            self.assertEqual(json.loads((root / 'check/trained-checks.json').read_text())['status'], 'passed')
            values = np.arange(1600, dtype=np.uint16) % 97
            values.tofile(root / 'val.bin')
            (root / 'manifest.json').write_text(json.dumps({'token_bin_sha256': {
                'val': hashlib.sha256((root / 'val.bin').read_bytes()).hexdigest()}}))
            self.call(['history', '--device', 'cpu', '--checkpoints', str(path), '--data', temp,
                       '--contexts', '129', '258', '--suffix', '8', '--samples', '2',
                       '--output', str(root / 'history')])
            result = json.loads((root / 'history/history-eval.json').read_text())
            self.assertEqual(len(result['records']), 4)
            self.assertEqual(result['records'][0]['end_token_offset'], result['records'][1]['end_token_offset'])
