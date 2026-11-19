#!/usr/bin/env python3
"""Local CPU evidence for the aa3e official entry.

Runs the real-installer entry tests on mkdtemp trees plus the launcher
regressions.  Inputs are explicit local paths; no network, NPU, service or
production tree is touched.  The 197863d5 original preimage is 32's read-only
fixture and is never copied into the submission tree.

  python3 runtime/run_official_entry_evidence.py \
    --original <197863d5 copy>/clef_joint_schema.py \
    --out runtime/official_entry_cpu_evidence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = {
    'Clef-27B/inference.py': None,
    'runtime/apply_joint_fused_attention.py':
        'b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406',
    'runtime/joint_fused_attention/clef_joint_schema.py':
        'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503',
    'runtime/apply_joint_single_field_patch_fused.py':
        '36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7',
    'runtime/apply_joint_service_check.py':
        '1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c',
    'runtime/joint_service_check/apply_joint_single_field_patch.py':
        '36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7',
    'runtime/joint_single_field/clef_joint_schema.py':
        'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863',
    'runtime/apply_joint_single_field_patch.py':
        '76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9',
}
ORIGINAL_SHA = '197863d5019f8b5326421f7616fbda13b030c27023248bf130628af7722bf4be'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run_suite(module: str, env_extra: dict) -> dict:
    env = dict(os.environ, PYTHONPATH=str(HERE), **env_extra)
    proc = subprocess.run([sys.executable, '-m', 'unittest', '-v', module],
                          cwd=str(HERE), capture_output=True, text=True, env=env, timeout=900)
    output = proc.stdout + proc.stderr
    ran = re.search(r'Ran (\d+) tests', output)
    skipped = re.search(r'skipped=(\d+)', output)
    return {'module': module, 'ok': proc.returncode == 0, 'returncode': proc.returncode,
            'ran': int(ran.group(1)) if ran else None,
            'skipped': int(skipped.group(1)) if skipped else 0,
            'tail': output.strip().splitlines()[-10:]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True,
                   help='32 read-only copy of the 197863d5 joint schema preimage')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    require(a.original.is_file(), f'original preimage missing: {a.original}')
    require(sha(a.original) == ORIGINAL_SHA,
            f'original preimage SHA mismatch: {sha(a.original)}')
    objects = {}
    for rel, expected in ASSETS.items():
        path = REPO / rel
        require(path.is_file(), f'asset missing: {rel}')
        actual = sha(path)
        require(expected is None or actual == expected, f'asset SHA mismatch: {rel}')
        objects[rel] = actual
    evidence = {
        'schema': 'official-entry-aa3e-cpu-evidence/1',
        'no_network_no_npu': True,
        'fixture': {'original_preimage_sha256': sha(a.original),
                    'note': '32 read-only copy; not part of the submission tree'},
        'objects': objects,
        'suites': [
            run_suite('test_joint_fused_entry',
                      {'CLEF_ORIGINAL_JOINT_SCHEMA': str(a.original.resolve())}),
            run_suite('test_inference_launch', {}),
        ],
    }
    evidence['ok'] = all(s['ok'] for s in evidence['suites'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': evidence['ok'], 'objects': objects}, ensure_ascii=False, indent=2))
    for suite in evidence['suites']:
        print(f"{suite['module']}: ok={suite['ok']} ran={suite['ran']} skipped={suite['skipped']}")
    return 0 if evidence['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
