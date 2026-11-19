#!/usr/bin/env python3
"""Build local mkdtemp fixtures and run the typed-phase CPU evidence suites.

Every input is explicit; the harness never defaults to a production or compute
path, never uses ssh/scp/sftp, and never imports torch/NPU.  Frozen sources are
read read-only and copied into one mkdtemp root; all tests then work only on
those copies.  The frozen f810 manifest (0338f87b...) is extracted from a local
git object and SHA-checked before use.

  python3 runtime/run_typed_phase_evidence.py \
    --bundle-source <ebca769>/runtime --code-source <cc7-stage> \
    --f810-source <f810 head> --guard-source <guard preimage> \
    --clean-source <clean-env-expected.frozen.json> --out <evidence.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PINS = {
    'joint_fused_service_ab.py': '127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648',
    'joint_fused_source_control.py': 'a1c502a3a709b836d31dc994a156f702d6a826ce5f2854beeec9ac50701f7299',
    'joint_fused_attention_window.py': '700cf10cdc3fda18d1ae7f113158342e1ef6ec114569e52b15b134949bfe8473',
    'apply_joint_fused_attention.py': 'b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406',
    'apply_joint_service_check.py': '1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c',
}
DIR_PINS = {
    'joint_fused_attention/clef_joint_schema.py':
        'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503',
    'joint_service_check/apply_joint_single_field_patch.py':
        '36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7',
    'd0/d0_run.py': '7cdd2cfee7ef73c1403b620fdedd6c78882e33eaac466bf4f44885a166e9d11e',
    'd0/d0_restore.py': '4c4afeae9b8511c6a1c85a01a2a4d3be2c9eaf0a379ca44f5120dfa361474c84',
    'd0/d0_launch.py': '779542c625e20b789ee74eaf04f58be6f0b24c5e5e558012737ba79e2ca07fc0',
    'd0/d0_gate.py': '5dae02c2fa3e32e2a6500b831ad7029f368d9405bd2ea3df0ca4100d92ab1543',
    'd0/d0_executor.py': '1c91e0c4e35defcd315fc633e30432188336fa402bdf531208ebaa40008ff8f4',
    'd0/d0_trace_parse.py': '13413bf0ee1eff07a6bdd4be584f7dda74f981a6b48bbc099e5b6fc532c0fb5f',
}
BASE_MANIFEST_OBJECT = ('a44c766be6fb54fdd010c92867f2fb065509c1d2',
                        '0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2')
CANDIDATE_SHA = '31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4'
F810 = 'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863'
GUARD_PRE = '76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9'
CLEAN = 'fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230'
SUPERVISOR = '172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876'
PHASE = '888a8d7fac79bd243fe894c25324aca0f498e311cab91be77df1dc2cfdc0de44'
RUNNER = '7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def copy_checked(source: Path, target: Path, expected: str, label: str) -> str:
    require(source.is_file(), f'{label} missing: {source}')
    actual = sha(source)
    require(actual == expected, f'{label} SHA mismatch: {actual}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return actual


def build_fixtures(a) -> dict:
    sources = [Path(a.bundle_source), Path(a.code_source), Path(a.owner_source),
               Path(a.f810_source), Path(a.guard_source), Path(a.clean_source),
               Path(a.candidate_manifest)]
    for source in sources:
        require(source.exists(), f'missing explicit source: {source}')
        require(not str(source.resolve()).startswith('/workspace'),
                f'refusing production path as source: {source}')
    root = Path(tempfile.mkdtemp(prefix='clef-typed-evidence-')).resolve()
    bundle = root / 'bundle'
    for name, digest in PINS.items():
        copy_checked(Path(a.bundle_source) / name, bundle / 'runtime' / name, digest, name)
    for name, digest in DIR_PINS.items():
        copy_checked(Path(a.bundle_source) / name, bundle / 'runtime' / name, digest, name)
    owner = root / 'owner'
    copy_checked(Path(a.owner_source) / 'final_acceptance_supervisor.py',
                 owner / 'final_acceptance_supervisor.py', SUPERVISOR, 'supervisor')
    copy_checked(Path(a.owner_source) / 'final_service_phase.py',
                 owner / 'final_service_phase.py', PHASE, 'phase module')
    code = root / 'code'
    copy_checked(Path(a.code_source) / 'runtime/final_acceptance.py',
                 code / 'runtime/final_acceptance.py', RUNNER, 'acceptance runner')
    f810 = root / 'f810.py'
    guard = root / 'guard.py'
    clean = root / 'clean.json'
    candidate = root / 'candidate-manifest.json'
    copy_checked(Path(a.f810_source), f810, F810, 'f810 head')
    copy_checked(Path(a.guard_source), guard, GUARD_PRE, 'guard preimage')
    copy_checked(Path(a.clean_source), clean, CLEAN, 'clean expected')
    copy_checked(Path(a.candidate_manifest), candidate, CANDIDATE_SHA, 'candidate manifest')
    obj, digest = BASE_MANIFEST_OBJECT
    base = root / 'base-manifest.json'
    blob = subprocess.run(['git', '-C', str(REPO), 'cat-file', '-p', obj],
                          capture_output=True, check=True).stdout
    base.write_bytes(blob)
    require(sha(base) == digest, 'frozen base manifest SHA mismatch')
    return {'root': root, 'bundle': bundle, 'owner': owner, 'code': code, 'f810': f810,
            'guard': guard, 'clean': clean, 'candidate': candidate, 'base': base,
            'sources': {str(k): str(v) for k, v in vars(a).items() if isinstance(v, str)}}


def run_suite(name: str, module: str, env_extra: dict, fixtures: dict) -> dict:
    env = dict(os.environ, PYTHONPATH=str(HERE), **env_extra)
    cmd = [sys.executable, '-m', 'unittest', '-v', module]
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=env,
                          timeout=900)
    output = proc.stdout + proc.stderr
    ran = 'Ran ' + output.split('Ran ', 1)[1].split(' tests', 1)[0] + ' tests' \
        if 'Ran ' in output else 'unknown'
    skipped = re.search(r'skipped=(\d+)', output)
    ok = proc.returncode == 0
    return {'name': name, 'module': module, 'ok': ok, 'returncode': proc.returncode,
            'ran': ran, 'skipped': int(skipped.group(1)) if skipped else 0,
            'tail': output.strip().splitlines()[-12:]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-source', type=Path, required=True,
                   help='ebca769 runtime dir with the five pinned files plus postimage/d0 dirs')
    p.add_argument('--owner-source', type=Path, required=True,
                   help='dir with final_acceptance_supervisor.py + final_service_phase.py')
    p.add_argument('--code-source', type=Path, required=True,
                   help='cc7 stage dir containing runtime/final_acceptance.py')
    p.add_argument('--f810-source', type=Path, required=True)
    p.add_argument('--guard-source', type=Path, required=True)
    p.add_argument('--clean-source', type=Path, required=True)
    p.add_argument('--candidate-manifest', type=Path, default=HERE / 'final_acceptance_candidate_manifest.json')
    p.add_argument('--records-dir', type=Path, required=True,
                   help='32 read-only typed-fixtures dir (manifest.json + SHA256SUMS + 16 records)')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    require(a.records_dir.is_dir(), f'records dir missing: {a.records_dir}')
    require((a.records_dir / 'SHA256SUMS').is_file(), 'records SHA256SUMS missing')
    for line in (a.records_dir / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        require(sha(a.records_dir / name.lstrip('*')) == digest, f'record SHA mismatch: {name}')
    fixtures = build_fixtures(a)
    env = {
        'CLEF_BASE_MANIFEST': str(fixtures['base']),
        'CLEF_REAL_BUNDLE': str(fixtures['bundle']),
        'CLEF_REAL_OWNER': str(fixtures['owner']),
        'CLEF_REAL_CODE': str(fixtures['code']),
        'CLEF_REAL_AB': str(fixtures['bundle'] / 'runtime'),
        'CLEF_REAL_REPO': str(fixtures['bundle']),
        'CLEF_REAL_MANIFEST': str(fixtures['base']),
        'CLEF_REAL_CANDIDATE_MANIFEST': str(fixtures['candidate']),
        'CLEF_REAL_F810': str(fixtures['f810']),
        'CLEF_REAL_GUARD': str(fixtures['guard']),
        'CLEF_REAL_CLEAN_EXPECT': str(fixtures['clean']),
        'CLEF_REAL_PYTHON': a.python,
        'CLEF_TYPED_BUNDLE': str(fixtures['bundle']),
        'CLEF_TYPED_OWNER': str(fixtures['owner']),
        'CLEF_TYPED_CODE': str(fixtures['code']),
        'CLEF_TYPED_MANIFEST': str(fixtures['base']),
        'CLEF_TYPED_CANDIDATE': str(fixtures['candidate']),
        'CLEF_TYPED_F810': str(fixtures['f810']),
        'CLEF_TYPED_GUARD': str(fixtures['guard']),
        'CLEF_TYPED_CLEAN': str(fixtures['clean']),
        'CLEF_TYPED_PYTHON': a.python,
        'CLEF_TYPED_FIXTURE_ROOT': str(fixtures['root']),
        'CLEF_REAL_RECORDS': str(a.records_dir.resolve()),
    }
    suites = [
        run_suite('frozen a1ca unit', 'test_candidate_service_phase', env, fixtures),
        run_suite('candidate manifest', 'test_candidate_manifest', env, fixtures),
        run_suite('typed unit', 'test_candidate_phase_typed', env, fixtures),
        run_suite('frozen a1ca real-method integration',
                  'test_candidate_service_integration', env, fixtures),
        run_suite('typed real-method integration',
                  'test_candidate_phase_typed_integration', env, fixtures),
    ]
    typed_suite = next(s for s in suites if s['module'] == 'test_candidate_phase_typed')
    require(typed_suite['skipped'] == 0,
            f"typed suite still skips {typed_suite['skipped']} tests; real records must light all")
    report = {
        'schema': 'candidate-typed-phase-cpu-evidence/1',
        'ok': all(s['ok'] for s in suites),
        'fixture_root_schema': 'mkdtemp; all test work confined under it',
        'no_network_no_npu': True,
        'pending': {},
        'objects': {
            'a1ca_entry': sha(HERE / 'candidate_service_phase.py'),
            'typed_entry': sha(HERE / 'candidate_service_phase_typed.py'),
            'candidate_manifest': sha(HERE / 'final_acceptance_candidate_manifest.json'),
            'base_manifest': sha(fixtures['base']),
            'f810_head': sha(fixtures['f810']),
            'guard_preimage': sha(fixtures['guard']),
            'clean_expected': sha(fixtures['clean']),
            'supervisor': sha(fixtures['owner'] / 'final_acceptance_supervisor.py'),
            'phase_module': sha(fixtures['owner'] / 'final_service_phase.py'),
            'acceptance_runner': sha(fixtures['code'] / 'runtime/final_acceptance.py'),
        },
        'real_records': {
            'dir_name': a.records_dir.name,
            'sha256sums_sha256': sha(a.records_dir / 'SHA256SUMS'),
            'manifest_sha256': sha(a.records_dir / 'manifest.json'),
            'files_verified': len([
                line for line in (a.records_dir / 'SHA256SUMS').read_text().splitlines() if line.strip()]),
        },
        'suites': suites,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('ok', 'objects')}, ensure_ascii=False, indent=2))
    for suite in suites:
        print(f"{suite['name']}: ok={suite['ok']} {suite['ran']}")
    shutil.rmtree(fixtures['root'], ignore_errors=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
