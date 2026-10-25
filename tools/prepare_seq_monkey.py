#!/usr/bin/env python3
"""Stream a bounded Seq-Monkey subset; split whole documents before packing."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import unicodedata

import numpy as np
import requests
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
from tiktoken_ext.openai_public import r50k_pat_str

SOURCE = 'https://modelscope.cn/api/v1/datasets/ddzhu123/seq-monkey/repo'
ARCHIVE = 'mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2'
REVISION = '63aa0850487f8ea5f99abe57f47797ba83c090b5'


def document_key(text):
    text = unicodedata.normalize('NFC', text.replace('\r\n', '\n')).strip()
    # Ignore whitespace differences for deduplication and the document-level split.
    key = hashlib.sha256(''.join(text.split()).encode()).hexdigest()
    return text, key


def load_tokenizer(directory):
    expected = {'vocab.bpe': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
                'encoder.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'}
    for name, digest in expected.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    ranks = data_gym_to_mergeable_bpe_ranks(str(directory / 'vocab.bpe'),
                                         str(directory / 'encoder.json'))
    return tiktoken.Encoding(name='gpt2', pat_str=r50k_pat_str,
                            mergeable_ranks=ranks, special_tokens={'<|endoftext|>': 50256},
                            explicit_n_vocab=50257)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--train-tokens', type=int, default=4_000_000)
    parser.add_argument('--val-tokens', type=int, default=200_000)
    parser.add_argument('--min-documents', type=int, default=2000)
    parser.add_argument('--max-documents', type=int, default=20000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'manifest.json').exists(), 'Use a new output directory'
    encoding = load_tokenizer(args.tokenizer_dir)
    counts = {'train': 0, 'val': 0}
    docs = {'train': [], 'val': []}
    seen, skipped, truncated, scanned = set(), 0, 0, 0
    files = {name: (args.output / f'{name}.bin').open('wb') for name in counts}
    saved = (args.output / 'selected-documents.jsonl').open('w')
    params = {'Revision': REVISION, 'FilePath': ARCHIVE}
    with requests.get(SOURCE, params=params, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode='r|bz2') as archive:
            member = next(member for member in archive if member.isfile() and member.name.endswith('.jsonl'))
            with archive.extractfile(member) as lines:
                for line in lines:
                    scanned += 1
                    row = json.loads(line)
                    text = row.get('text')
                    if not isinstance(text, str):
                        skipped += 1
                        continue
                    text, key = document_key(text)
                    if key in seen or len(text) < 100:
                        skipped += 1
                        continue
                    seen.add(key)
                    split = 'val' if int(key[:16], 16) % 20 == 0 else 'train'
                    tokens = encoding.encode_ordinary(text)
                    if len(tokens) < 32:
                        skipped += 1
                        continue
                    if len(tokens) > 8192:
                        truncated += 1
                    tokens = tokens[:8192] + [50256]
                    files[split].write(np.asarray(tokens, dtype=np.uint16).tobytes())
                    saved.write(json.dumps({'split': split, 'sha256': key, 'text': text}, ensure_ascii=False) + '\n')
                    counts[split] += len(tokens)
                    docs[split].append(key)
                    total_docs = len(docs['train']) + len(docs['val'])
                    if total_docs % 200 == 0:
                        print('PROGRESS', total_docs, counts, flush=True)
                    if total_docs >= args.min_documents and all(
                            counts[name] >= getattr(args, f'{name}_tokens') for name in counts):
                        break
                    if total_docs >= args.max_documents:
                        break
    saved.close()
    for file in files.values():
        file.close()
    assert not set(docs['train']) & set(docs['val'])
    assert counts['train'] >= args.train_tokens and counts['val'] >= args.val_tokens
    manifest = {'source_repository': 'https://github.com/mobvoi/seq-monkey-data',
                'mirror': 'https://modelscope.cn/datasets/ddzhu123/seq-monkey',
                'mirror_revision_requested': REVISION,
                'source_archive': ARCHIVE, 'source_archive_bytes': 11356716280,
                'source_archive_sha256_from_mirror_metadata':
                    'c4a0488a28be6fc4fc4ae2fd15fd3ad54507828ee5f40e0b09ae2742f96fdf27',
                'full_archive_integrity_verified': False,
                'scope': 'First qualifying documents in a streamed third-party mirror; not random corpus-wide sampling. Full archive was not downloaded or verified against official MD5.',
                'license_declared_by_original': 'Apache-2.0',
                'split': 'NFC text, whitespace-insensitive SHA256, first 16 hex digits modulo 20; 0=val, else=train. Exact normalized duplicates removed before split. Near-duplicates not detected.',
                'tokenizer': 'GPT-2; vocab 50257; original tokenizer files verified by SHA256',
                'document_token_limit': 8192, 'eot': 50256, 'scanned_lines': scanned,
                'skipped': skipped, 'truncated_documents': truncated,
                'tokens': counts, 'documents': docs,
                'token_bin_sha256': {name: hashlib.sha256((args.output / f'{name}.bin').read_bytes()).hexdigest()
                                     for name in counts},
                'prepare_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETE', counts, {name: len(keys) for name, keys in docs.items()}, flush=True)


if __name__ == '__main__':
    main()
