"""CPU-only control-flow tests for the Grouped MMAR probe package.

No torch/torch_npu import, no device, no installer.  Device behavior is
validated by 31 under the frozen manifest; these tests only cover the plan,
path guards, release ordering, timeout watchdog and metadata checklist.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_probe
import validate_snapshot


def plan_args(root, arm='A0', extra=()):
    return ['--arm', arm, '--temp-root', str(root), '--out', str(root / 'out'), *extra]


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='grouped-mmar-cpu-'))

    def test_first_matrix_and_flags(self):
        for arm, flags in (('A8', 0), ('A0', 0), ('H1', run_probe.KEEP_GROUP),
                           ('H2', run_probe.KEEP_WORKSPACE)):
            a = run_probe.parse_args(plan_args(self.root, arm))
            plan = run_probe.build_plan(a)
            self.assertEqual(plan['flags'], flags)
            self.assertEqual(plan['first_matrix'], list(run_probe.FIRST_MATRIX))
            self.assertFalse(plan['no_service_lifecycle'] is False)
            self.assertTrue(plan['no_f810_revert'])
            self.assertIn('sync does not return', ' '.join(plan['failure_branches']))
        a = run_probe.parse_args(plan_args(self.root, 'A12'))
        plan = run_probe.build_plan(a)
        self.assertTrue(plan['external'])
        self.assertEqual(plan['source_sha256'],
                         '6c1a90aba84501a901c413eaa5391889d60beff48b766bed9865ab5bf26eeb9f')
        self.assertEqual(plan['argv'], ['2', '1', '0,1'])

    def test_matrix_contains_a12_and_keepalive_arms(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_probe.main(plan_args(self.root, extra=('--matrix',))), 0)
        matrix = json.loads(output.getvalue())
        self.assertEqual(matrix['first_matrix'], ['A8', 'A12', 'A0', 'H1', 'H2'])

    def test_plan_mode_does_not_import_torch(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_probe.main(plan_args(self.root, 'H1')), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['status'], 'plan_not_executed')
        self.assertNotIn('torch', sys.modules)

    def test_relative_paths_rejected(self):
        a = run_probe.parse_args(['--arm', 'A0', '--temp-root', 'relative', '--out', 'out'])
        with self.assertRaises(ValueError):
            run_probe.build_plan(a)

    def test_a13_plan_is_native_mmar_control(self):
        a = run_probe.parse_args(plan_args(self.root, 'A13'))
        plan = run_probe.build_plan(a)
        self.assertTrue(plan['native'])
        self.assertEqual(plan['rows'], 967)
        self.assertEqual(plan['k'], 8704)
        self.assertEqual(plan['n'], 5120)
        self.assertEqual(plan['flags'], 0)

    def test_a14_plan_is_native_reduce_scatter_control(self):
        a = run_probe.parse_args(plan_args(self.root, 'A14'))
        plan = run_probe.build_plan(a)
        self.assertTrue(plan['native'])
        self.assertEqual(plan['rows'], 128)
        self.assertEqual(plan['flags'], 0)

    def test_weight_requires_sha(self):
        weight = self.root / 'w.bin'
        weight.write_bytes(b'x')
        a = run_probe.parse_args(plan_args(self.root, extra=('--weight-bin', str(weight))))
        with self.assertRaises(ValueError):
            run_probe.build_plan(a)


class ReleaseOrderingTest(unittest.TestCase):
    def test_release_only_after_sync_and_only_with_keep(self):
        calls = []
        run_probe.sync_and_release(lambda: calls.append('sync'), lambda i: calls.append(('release', i)),
                                   7, True)
        self.assertEqual(calls, ['sync', ('release', 7)])
        calls.clear()
        run_probe.sync_and_release(lambda: calls.append('sync'), lambda i: calls.append(('release', i)),
                                   0, True)
        self.assertEqual(calls, ['sync'])
        calls.clear()
        run_probe.sync_and_release(lambda: calls.append('sync'), lambda i: calls.append(('release', i)),
                                   7, False)
        self.assertEqual(calls, ['sync'])

    def test_sync_failure_skips_release(self):
        calls = []

        def boom():
            raise RuntimeError('sync failed')

        with self.assertRaises(RuntimeError):
            run_probe.sync_and_release(boom, lambda i: calls.append(('release', i)), 7, True)
        self.assertEqual(calls, [])


class WatchdogTest(unittest.TestCase):
    def test_watchdog_exits_124_and_writes_marker(self):
        root = Path(tempfile.mkdtemp(prefix='grouped-mmar-wd-'))
        out = root / 'wd'
        proc = subprocess.run(
            [sys.executable, str(Path(run_probe.__file__)), '--watchdog-selftest', '0.2',
             '--temp-root', str(root), '--out', str(out)],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 124, proc.stderr)
        marker = json.loads((out / 'timeout.json').read_text())
        self.assertEqual(marker['schema'], 'grouped-mmar-probe-timeout/1')


class SnapshotTest(unittest.TestCase):
    def contract(self):
        return json.loads((Path(run_probe.__file__).with_name('snapshot_contract.json')).read_text())

    def valid_doc(self):
        contract = self.contract()
        doc = {}
        for group, keys in contract['required'].items():
            doc[group] = {key: ('a' * 64 if key.endswith('_sha256') else 1) for key in keys}
        for key, expected in contract['expected'].items():
            group = next(g for g, keys in contract['required'].items() if key in keys)
            doc[group][key] = (expected if len(expected) == 64
                               else expected + '0' * (64 - len(expected)))
        return doc

    def test_valid_and_mismatch(self):
        contract = self.contract()
        doc = self.valid_doc()
        result = validate_snapshot.validate(doc, contract)
        self.assertTrue(result['ok'], result)
        doc['files']['source_aa3e_sha256'] = '0' * 64
        result = validate_snapshot.validate(doc, contract)
        self.assertFalse(result['ok'])
        self.assertTrue(result['expected_mismatch'])

    def test_missing_field(self):
        contract = self.contract()
        doc = self.valid_doc()
        doc['service'].pop('argv')
        result = validate_snapshot.validate(doc, contract)
        self.assertIn('service.argv', result['missing'])


class FrozenAssetsTest(unittest.TestCase):
    def test_gate_bytes_unchanged(self):
        path = Path(run_probe.__file__).with_name('fixed_gate.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '041a66aa81c52d2a53bd825cf406323d7185759242669dff64b3551f86eba849')


if __name__ == '__main__':
    unittest.main()
