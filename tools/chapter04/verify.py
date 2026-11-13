"""Check saved analytical records, source fingerprints and earlier cache counts."""

import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/chapter04/20261101'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    reject = lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON constant: ' + value))
    read = lambda name: json.loads((args.data / name).read_text(), parse_constant=reject)
    meta = read('metadata.json')
    require(meta['status'] == 'analytical_estimate_not_hardware_measurement', 'incorrect data scope')
    for name, expected in meta['source_sha256'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'source changed: ' + name)
    baseline, sensitivity, grid, sram = [read(n) for n in
        ('baseline.json', 'sensitivity.json', 'matrix-bandwidth-grid.json', 'sram-fusion.json')]
    for rows, key in ((baseline, 'baseline_cases'), (sensitivity, 'sensitivity_cases'),
                      (grid, 'matrix_bandwidth_cases'), (sram, 'sram_fusion_cases')):
        require(len(rows) == meta[key], 'record count mismatch: ' + key)
        for row in rows:
            if row['status'] == 'infeasible_under_sram_contract':
                require(bool(row['reason']), 'missing infeasibility reason')
                continue
            require(row['status'] == 'analytical_scenario_not_hardware_measurement', 'incorrect row scope')
            require(all(n >= 0 for n in row['totals'].values()), 'negative work')
            expected_bytes = sum(row['totals'][k] for k in ('weight_bytes', 'state_read_bytes', 'state_write_bytes', 'activation_bytes'))
            require(row['dram_payload_bytes'] == expected_bytes, 'traffic accounting mismatch')
            require(abs(row['conditional_lower_bound_ms'] - max(row['resource_ms'].values())) < 1e-9,
                    'resource bound mismatch')
            require(row['conditional_lower_bound_ms'] <= row['serial_analytical_budget_ms'] + 1e-9,
                    'resource bound exceeds serial budget')
            allocated = row['max_ffn_scratch_bytes'] + row['compressor_sram_reservation_bytes']
            require(allocated <= row['hardware']['sram_mib'] * 2**20, 'FFN/compressor SRAM overcommit')
    old = [json.loads(s) for s in (root / 'data/chapter03/20261101/benchmarks.jsonl').read_text().splitlines()]
    comparisons = []
    for variant, earlier in (('dense', 'mha'), ('mla', 'mla'), ('hybrid', 'hybrid')):
        for row in baseline:
            r = row['request']
            if r['variant'] == variant and r['batch'] == 1:
                reference = next(o for o in old if o['variant'] == earlier and o['phase'] == r['phase']
                                 and o['context'] == r['context'] and o['batch'] == r['batch'])
                require(row['logical_state_bytes'] == reference['performance']['cache']['logical_bytes'],
                        'cache differs from chapter three')
                comparisons.append({k: r[k] for k in ('variant', 'phase', 'context', 'batch')})
    expected = {'dense': 119690496, 'mla': 111858432, 'moe': 148038912, 'hybrid': 109310216}
    require(all(r['parameters'] == expected[r['request']['variant']] for r in baseline), 'parameter count mismatch')
    result = {'verification_status': 'passed', 'scope': 'analytical record integrity, not hardware or numerical-model validation',
              'source_fingerprints': len(meta['source_sha256']), 'baseline_cases': len(baseline),
              'sensitivity_cases': len(sensitivity), 'matrix_bandwidth_cases': len(grid),
              'sram_fusion_cases': len(sram), 'earlier_cache_comparisons': len(comparisons),
              'infeasible_cases_retained': sum(r['status'] == 'infeasible_under_sram_contract' for r in sensitivity + sram),
              'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.iterdir())
                              if p.is_file() and p.name != 'verification.json'}}
    (args.data / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
