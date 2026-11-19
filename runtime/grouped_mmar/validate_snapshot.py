#!/usr/bin/env python3
"""CPU-only validator for 31's read-only service snapshot checklist."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEX64 = re.compile(r'^[0-9a-f]{64}$')


def flatten(doc, prefix=''):
    out = {}
    if isinstance(doc, dict):
        for key, value in doc.items():
            out.update(flatten(value, f'{prefix}.{key}' if prefix else key))
    else:
        out[prefix] = doc
    return out


def validate(doc, contract):
    flat = flatten(doc)
    leaf = {path.split('.')[-1]: value for path, value in flat.items()}
    missing, invalid, expected_bad = [], [], []
    for group, keys in contract['required'].items():
        for key in keys:
            path = f'{group}.{key}'
            if path not in flat or flat[path] in (None, '', []):
                missing.append(path)
    for key in contract.get('sha_fields', []):
        value = leaf.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not HEX64.match(value):
            invalid.append(key)
    for key, expected in contract.get('expected', {}).items():
        value = leaf.get(key)
        if value is None:
            continue
        if len(expected) == 64 and value != expected:
            expected_bad.append({'field': key, 'expected': expected, 'actual': value})
        elif len(expected) < 64 and not str(value).startswith(expected):
            expected_bad.append({'field': key, 'expected_prefix': expected, 'actual': value})
    return {'schema': 'grouped-mmar-snapshot-check/1', 'missing': missing,
            'bad_sha256': invalid, 'expected_mismatch': expected_bad,
            'ok': not (missing or invalid or expected_bad)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--contract', type=Path,
                   default=Path(__file__).with_name('snapshot_contract.json'))
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    contract = json.loads(a.contract.read_text())
    result = validate(json.loads(a.snapshot.read_text()), contract)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
