import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unicodedata
import unittest


source = Path(__file__).resolve().parents[1] / 'tools/prepare_seq_monkey.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'document_key')
namespace = {'hashlib': hashlib, 'unicodedata': unicodedata}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
document_key = namespace['document_key']


class PreparationTests(unittest.TestCase):
    def test_normalized_duplicates_have_same_split(self):
        _, one = document_key('caf\u00e9 hello\r\nworld')
        _, two = document_key('cafe\u0301 hello world')
        self.assertEqual(one, two)
        self.assertEqual(int(one[:16], 16) % 20, int(two[:16], 16) % 20)

    def test_different_documents_remain_distinct(self):
        self.assertNotEqual(document_key('first text')[1], document_key('second text')[1])

    def test_binary_json_lines_from_streaming_tar(self):
        output = io.BytesIO()
        payload = b'{"text":"first"}\n{"text":"second"}\n'
        with tarfile.open(fileobj=output, mode='w:bz2') as archive:
            member = tarfile.TarInfo('data.jsonl')
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        output.seek(0)
        with tarfile.open(fileobj=output, mode='r|bz2') as archive:
            member = next(member for member in archive if member.isfile())
            with archive.extractfile(member) as lines:
                self.assertEqual([json.loads(line)['text'] for line in lines], ['first', 'second'])


if __name__ == '__main__':
    unittest.main()
