#!/usr/bin/env python3
"""Freeze the exact owner27 5c window; print-only unless --execute is supplied.

Requires worker20 scheduling and worker23 review before owner27 execution.
The original 5c wrapper owns stop, timeout, process cleanup and recovery.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from window_common import (OWNER, WRAPPER, WRAPPER_SHA, REPO, STOP, STOP_SHA,
                           TREE, PREIMAGE_SHA, PYTHON, REL, CELL, PARAMETERS,
                           N1_CELL, check_sha, frozen)


def plan(a):
    bundle = Path(__file__).resolve().parent
    identity = frozen(bundle, a.expect_payload_sha)
    manifest = json.loads((bundle / 'payload.json').read_text())
    check_sha(WRAPPER, WRAPPER_SHA)
    check_sha(STOP, STOP_SHA)
    run = a.run_dir.resolve()
    if run.exists() or not run.is_relative_to(Path('/workspace/yangchen/logs')):
        raise ValueError('Window requires a new directory under /workspace/yangchen/logs')
    cmd = [PYTHON, str(WRAPPER), '--run-dir', str(run), '--repo', str(REPO),
           '--tree', str(TREE), '--python', PYTHON, '--device', '0', '--port', '18143',
           '--stop-script', str(STOP), '--expect-stop-sha', STOP_SHA,
           '--stop-timeout', '120', '--probe-timeout', '240', '--wait-timeout', '900',
           '--preimage-sha', PREIMAGE_SHA, '--candidate-kind', 'external-vendor',
           '--vendor-installer-rel', REL + 'window_hook.py',
           '--vendor-installer-source', str(bundle / 'window_hook.py'),
           '--vendor-installer-sha', manifest['files']['window_hook.py'],
           '--vendor-apply-flag=--install', '--vendor-revert-flag=--revert',
           '--vendor-extra-arg=--run-dir=' + str(run),
           '--vendor-extra-arg=--tree=' + str(TREE),
           '--vendor-extra-arg=--expect-payload-sha=' + a.expect_payload_sha,
           '--probe-rel', REL + 'window_probe.py', '--probe-source', str(bundle / 'window_probe.py'),
           '--probe-sha', manifest['files']['window_probe.py'],
           '--probe-schema', 'n1-probe/1', '--env-flag=', '--client-rel=',
           '--candidate-opp-prepend=', '--candidate-lib-prepend=',
           '--child-env', 'ASCEND_RT_VISIBLE_DEVICES=0,1',
           '--child-env', 'CLEF_GROUPED29_EXECUTION_OWNER=' + a.owner,
           '--child-env', 'CLEF_GROUPED29_PAYLOAD_SHA=' + a.expect_payload_sha]
    for name, digest in {**manifest['files'], 'payload.json': a.expect_payload_sha}.items():
        if name in ('window_hook.py', 'window_probe.py'):
            continue
        cmd += ['--payload', f'{bundle / name}:{REL + name}:{digest}']
    return {'schema': 'grouped29-supervised-plan/1', 'owner': a.owner,
            'activation_required_from': 'clef-27b-ascend-20', 'argv': cmd, 'shell': shlex.join(cmd),
            'wrapper_sha256': WRAPPER_SHA, 'payload_sha256': a.expect_payload_sha,
            'adapter_sha256': identity['adapter']['sha256'], 'cell': CELL, 'parameters': PARAMETERS,
            'n1_cell': str(N1_CELL), 'compatibility_schema': 'n1-probe/1 for 5c gate_passed check only',
            'boundary': 'Grouped GEMM+AllReduce followed by native AddRMSNorm',
            'installation': 'validation-only hook, no vendor and no system mutations',
            'timing': {'round_orders': ['AB', 'BA', 'AB', 'BA'], 'reps_per_round': 10,
                       'raw_per_rank': 80, 'warmup_per_arm': 2, 'profile_after_gate': True},
            'rank_processes': '--run-path --start-method=spawn; same supervised probe PGID'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', required=True, choices=[OWNER])
    p.add_argument('--expect-payload-sha', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    result = plan(a)
    print(json.dumps(result, indent=2), flush=True)
    return subprocess.call(result['argv']) if a.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
