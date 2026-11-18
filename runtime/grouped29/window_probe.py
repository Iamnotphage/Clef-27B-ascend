#!/usr/bin/env python3
"""Owner27 first-cell adapter for the frozen 5c --tree/--device/--output API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from window_common import (OWNER, N1_CELL, check_head, check_window_owner, frozen,
                           staged_paths, sha256, CELL, PARAMETERS)


def command(stage, run, python):
    return [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            '--nproc-per-node=2', '--max-restarts=0', '--run-path', '--start-method=spawn',
            str(stage / 'run_probe.py'), '--owner', OWNER,
            '--identity', str(stage / 'identity.json'), '--adapter', str(stage / 'clef_grouped29.so'),
            '--gate', str(stage / 'fixed_gate.py'), '--output', str(run / 'probe/cell'),
            '--rows', '967', '--seed', '20261118', '--n1-cell', str(N1_CELL), '--timing', '--profile']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # n1-probe/1 is the existing 5c status contract, not an operator identity.
    report = {'schema': 'n1-probe/1', 'protocol': 'grouped29-window/1', 'gate_passed': False,
              'cell': CELL, 'parameters': PARAMETERS, 'error': None}
    try:
        stage, run = staged_paths(__file__, a.output.resolve().parent.parent)
        if a.device != 0 or a.output.resolve() != run / 'probe/head-probe.json':
            raise ValueError('Requires fixed supervised probe output and device0')
        check_head(a.tree)
        check_window_owner()
        identity = frozen(stage, os.environ['CLEF_GROUPED29_PAYLOAD_SHA'])
        env = dict(os.environ)
        env['PYTHONPATH'] = str(stage) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['CLEF_GROUPED29_PROBE_PGID'] = str(os.getpgrp())
        cmd = command(stage, run, sys.executable)
        report.update(argv=cmd, pgid=os.getpgrp(), identity_sha256=sha256(stage / 'identity.json'),
                      adapter_sha256=identity['adapter']['sha256'])
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        # MultiprocessContext/spawn inherits the window probe PGID. The outer
        # 5c wrapper owns all bounds and cleanup; no nested process group here.
        rc = subprocess.call(cmd, env=env, start_new_session=False)
        report['child_rc'] = rc
        if rc != 0:
            raise RuntimeError(f'Grouped29 two-rank probe failed: {rc}')
        from aggregate import aggregate
        report['aggregate'] = aggregate(run / 'probe/cell', stage, identity, os.getpgrp())
        report['gate_passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
