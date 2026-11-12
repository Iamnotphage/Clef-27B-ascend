#!/usr/bin/env python3
"""Rebuild compact evidence from an existing full analysis summary.

python scripts/compact_systemone_profile.py \
    --summary /absolute/analysis-agent/summary.json \
    --output /absolute/new-evidence/compact-evidence.json --max-bytes 200000

Uses the frozen parser's definitions. It reads existing exports and their index;
it never parses a large trace, imports Torch, sends requests, or runs NPU work.
Existing submitted compact files remain intact when a separate output is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_systemone_profile import compact_evidence, file_sha
except ImportError:
    from analyze_systemone_profile import compact_evidence, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=0,
                        help='Optional delivery-size check; zero disables the limit')
    args = parser.parse_args()
    if args.max_bytes < 0 or args.summary.resolve() == args.output.resolve():
        parser.error('Use a separate output and a nonnegative size limit')
    summary = json.loads(args.summary.read_text())
    compact = compact_evidence(summary, args.summary.resolve().parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
    size = args.output.stat().st_size
    passed = compact['passed'] and (not args.max_bytes or size <= args.max_bytes)
    print(json.dumps({'passed': passed, 'output': str(args.output.resolve()),
                      'bytes': size, 'sha256': file_sha(args.output),
                      'source_summary_sha256': file_sha(args.summary),
                      'source_index_scope': 'Existing analysis directory; original submitted compact remains unchanged unless explicitly selected as output'}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
