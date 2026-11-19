#!/usr/bin/env python3
"""Verify a staged copy of the final-acceptance combo against its frozen pins.

Local mode checks every file listed in `prep_file_shas` under the stage root.
Remote mode (`--ssh-host clef`) reads each `remote_files` entry with one
`sha256sum` call and compares; it is read-only and never touches the service.

Exit code is nonzero on any missing or mismatched file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

COMBO = 'runtime/final_acceptance_combo.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_local(stage: Path, mapping: dict) -> list[dict]:
    result = []
    for rel, expected in sorted(mapping.items()):
        path = stage / rel
        actual = sha(path) if path.is_file() else None
        result.append({'path': rel, 'expected': expected, 'actual': actual,
                       'ok': actual == expected})
    return result


def parse_sha256sum(text: str) -> dict:
    found = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            found[parts[1].lstrip('*')] = parts[0]
    return found


def check_remote(host: str, entries: list[dict]) -> list[dict]:
    paths = [entry['path'] for entry in entries]
    if not paths:
        return []
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'sha256sum', '--', *paths],
                          capture_output=True, text=True, timeout=120)
    found = parse_sha256sum(proc.stdout)
    result = []
    for entry in entries:
        actual = found.get(entry['path'])
        result.append({'path': entry['path'], 'expected': entry['sha256'], 'actual': actual,
                       'ok': actual == entry['sha256']})
    return result


def summarize(rows: list[dict]) -> dict:
    bad = [row for row in rows if not row['ok']]
    return {'checked': len(rows), 'failed': len(bad), 'ok': not bad, 'mismatches': bad}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--ssh-host')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    stage = a.stage.resolve()
    combo = json.loads((stage / COMBO).read_text())
    report = {'schema': 'final-acceptance-combo-check/1', 'stage': str(stage)}
    report['local'] = summarize(check_local(stage, combo.get('prep_file_shas', {})))
    if a.ssh_host:
        report['remote'] = summarize(check_remote(a.ssh_host, combo.get('remote_files', [])))
    report['ok'] = report['local']['ok'] and report.get('remote', {'ok': True})['ok']
    print(json.dumps(report, ensure_ascii=False, indent=2 if a.json else 1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
