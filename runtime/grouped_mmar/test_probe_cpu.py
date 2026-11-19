"""CPU-only control-flow tests for the Grouped MMAR probe package.

No torch/torch_npu import, no device, no installer.  Device behavior is
validated by 31 under the frozen manifest; these tests only cover the plan,
path guards, release ordering, timeout watchdog and metadata checklist.
"""
import builtins
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import symtable
import sys
import tempfile
import unittest
from unittest import mock

import run_probe
import validate_snapshot


def f821_violations(path):
    """Equivalent F821 check via symtable (ruff/pyflakes are not installed)."""
    source = Path(path).read_text()
    top = symtable.symtable(source, str(path), 'exec')
    module_names = {s.get_name() for s in top.get_symbols()
                    if s.is_assigned() or s.is_imported() or s.is_namespace()}
    allowed = (set(dir(builtins)) | module_names |
               {'__name__', '__file__', '__doc__', '__builtins__'})
    bad = []

    def walk(table, enclosing):
        local = {s.get_name() for s in table.get_symbols()
                 if s.is_assigned() or s.is_parameter() or s.is_imported()}
        scope = enclosing | local
        for sym in table.get_symbols():
            if sym.is_referenced():
                name = sym.get_name()
                if (sym.is_assigned() or sym.is_parameter() or sym.is_imported()
                        or sym.is_free()):
                    continue
                if name in scope or name in allowed:
                    continue
                bad.append((table.get_name(), name))
        for child in table.get_children():
            walk(child, scope)

    walk(top, set())
    return bad


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
        self.assertEqual(plan['flags'], run_probe.KEEP_ALL)

    def test_h4k_h5k_keep_all_discriminators(self):
        for arm, flags, names in (('H4K', 24, ['keep_all', 'direct_launch']),
                                  ('H5K', 40, ['keep_all', 'refetch_stream'])):
            a = run_probe.parse_args(plan_args(self.root, arm))
            plan = run_probe.build_plan(a)
            self.assertTrue(plan['native'])
            self.assertEqual(plan['rows'], 128)
            self.assertEqual(plan['flags'], flags)
            self.assertEqual(plan['flag_names'], names)
            self.assertNotIn('memset_workspace', plan['flag_names'])

    def test_a15_padding_row_blocks_and_bias_free_plan(self):
        self.assertEqual(run_probe.padded_rows(967, 2), 968)
        self.assertEqual(run_probe.padded_rows(1595, 2), 1596)
        self.assertEqual(run_probe.padded_rows(307, 2), 308)
        self.assertEqual(run_probe.padded_rows(968, 2), 968)
        self.assertEqual(run_probe.row_block(0, 967, 2), (0, 484))
        self.assertEqual(run_probe.row_block(1, 967, 2), (484, 968))
        with self.assertRaises(ValueError):
            run_probe.padded_rows(0, 2)
        with self.assertRaises(ValueError):
            run_probe.row_block(2, 967, 2)
        a = run_probe.parse_args(plan_args(self.root, 'A15'))
        plan = run_probe.build_plan(a)
        self.assertTrue(plan['native'])
        self.assertEqual(plan['flags'], run_probe.KEEP_ALL)
        self.assertEqual(plan['flag_names'], ['keep_all'])
        self.assertEqual(plan['rows'], 967)
        self.assertEqual(plan['bias'], 'nullptr')
        source = Path(run_probe.__file__).read_text()
        self.assertIn('a15_flow(', source)
        self.assertIn('all_gather_into_tensor', source)
        self.assertIn("'diagnostic_only_not_an_official_gate'", source)

    def test_arm_output_keeps_a14_and_crops_a15(self):
        self.assertEqual(run_probe.arm_output('A14', 'y'), 'y')
        self.assertEqual(run_probe.arm_output('A11', 'y'), 'y')
        self.assertEqual(run_probe.arm_output('A15', 'y', [0, 1, 2, 3], 2), [0, 1])
        # A14 keeps the MRS output; A15 output and its reference both cover all rows.
        y = list(range(8))
        full = list(range(16))
        self.assertEqual(len(run_probe.arm_output('A14', y)), 8)
        self.assertEqual(len(run_probe.arm_output('A15', y, full, 16)), 16)

    def test_out_dir_shared_by_two_ranks(self):
        import threading
        out = self.root / 'shared-out'
        barrier = threading.Barrier(8)
        errors = []

        def worker():
            barrier.wait()
            try:
                run_probe.prepare_out(out)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertTrue(out.is_dir())

    def test_weight_requires_sha(self):
        weight = self.root / 'w.bin'
        weight.write_bytes(b'x')
        a = run_probe.parse_args(plan_args(self.root, extra=('--weight-bin', str(weight))))
        with self.assertRaises(ValueError):
            run_probe.build_plan(a)


class F821Test(unittest.TestCase):
    def test_probe_and_window_have_no_undefined_names(self):
        for name in ('run_probe.py', 'supervised_probe_window.py'):
            self.assertEqual(f821_violations(Path(__file__).with_name(name)), [], name)

    def test_checker_catches_an_undefined_name(self):
        root = Path(tempfile.mkdtemp(prefix='grouped-mmar-f821-'))
        bad = root / 'bad.py'
        bad.write_text('def f(a):\n    return k + a\n')
        self.assertIn(('f', 'k'), f821_violations(bad))

    def test_checker_catches_the_old_a15_pad_line(self):
        root = Path(tempfile.mkdtemp(prefix='grouped-mmar-f821-old-'))
        old = root / 'old_run_probe.py'
        old.write_text(
            'import torch\n'
            'def execute(a, rows, m_pad):\n'
            '    x_mrs = torch.ones((m_pad, k))\n'
            '    return x_mrs\n')
        self.assertIn(('execute', 'k'), f821_violations(old))


class A15RecordTest(unittest.TestCase):
    def test_missing_reference_is_fail_closed(self):
        record, error = run_probe.a15_record(
            [1.0], None, 1, 1, diff=lambda o, r: {}, equal=lambda o, r: False)
        self.assertIsNone(record)
        self.assertEqual(error, 'missing_equivalence_record')

    def test_wrong_shape_is_fail_closed(self):
        record, error = run_probe.a15_record(
            [[0.0] * 512], [[0.0] * 512], 127, 512,
            diff=lambda o, r: {'max_abs': 0.0, 'mean_abs': 0.0},
            equal=lambda o, r: True)
        self.assertIsNone(record)
        self.assertEqual(error, 'invalid_equivalence_shape')

    def test_record_contains_equivalence_and_diagnostic(self):
        output = [[1.0] * 512 for _ in range(127)]
        reference = [[2.0] * 512 for _ in range(127)]
        record, error = run_probe.a15_record(
            output, reference, 127, 512,
            diff=lambda o, r: {'max_abs': 0.01, 'mean_abs': 1e-5},
            equal=lambda o, r: False)
        self.assertIsNone(error)
        self.assertEqual(record['reduced'], {'max_abs': 0.01, 'mean_abs': 1e-5,
                                             'bit_equal': False})
        self.assertEqual(record['equivalence']['shape'], [127, 512])
        self.assertTrue(record['diagnostic_gate']['within_bounds'])
        self.assertEqual(record['diagnostic_gate']['status'],
                         'diagnostic_only_not_an_official_gate')
        record, error = run_probe.a15_record(
            output, reference, 127, 512,
            diff=lambda o, r: {'max_abs': 2.0, 'mean_abs': 5e-5},
            equal=lambda o, r: False)
        self.assertIsNone(error)
        self.assertFalse(record['diagnostic_gate']['within_bounds'])


class A15FlowTest(unittest.TestCase):
    def test_a15_rows127_pad_gather_crop_with_stubs(self):
        rows, k, n, world, rank = 127, 256, 512, 2, 1
        x = [[float(i)] * k for i in range(rows)]
        w = [[0.0] * n for _ in range(k)]
        calls = {}

        def ones(shape):
            m, width = shape
            self.assertEqual(width, k)
            return [[1.0] * width for _ in range(m)]

        def empty(shape, like):
            m, width = shape
            self.assertEqual(width, n)
            self.assertEqual(len(like[0]), n)
            return [[9.0] * width for _ in range(m)]

        def run_mrs(xm, wm, world_, flags):
            calls['mrs'] = (len(xm), len(xm[0]), flags)
            calls['pad_row'] = xm[rows]
            calls['orig_ok'] = xm[:rows] == x
            self.assertEqual(wm, w)
            self.assertEqual(world_, world)
            return [[5.0] * n for _ in range((rows + 1) // world)], 123, 7

        def sync():
            calls['sync'] = calls.get('sync', 0) + 1

        def all_gather(out, y):
            calls['gather'] = (len(out), len(y))
            self.assertEqual(len(y), (rows + 1) // world)
            for offset, row in enumerate(y):
                out[64 + offset] = row

        def matmul(xx, ww):
            calls['matmul_len'] = len(xx)
            self.assertEqual(ww, w)
            return [[1.0] * n for _ in range(len(xx))]

        def all_reduce(tensor):
            calls['all_reduce'] = len(tensor)
            for row in tensor:
                row[0] += 1.0
            return None  # mirrors torch.distributed.all_reduce's in-place None

        output, reference, info = run_probe.a15_flow(
            rows, k, n, world, rank, x, w, ones=ones, empty=empty,
            run_mrs=run_mrs, sync=sync, all_gather=all_gather,
            final_sync=lambda: None, matmul=matmul, all_reduce=all_reduce,
            flags=run_probe.KEEP_ALL)
        self.assertEqual(calls['mrs'], (128, k, run_probe.KEEP_ALL))
        self.assertEqual(calls['pad_row'], [1.0] * k)
        self.assertTrue(calls['orig_ok'])
        self.assertEqual(calls['sync'], 1)
        self.assertEqual(calls['gather'], (128, 64))
        self.assertEqual(calls['matmul_len'], rows)
        self.assertEqual(calls['all_reduce'], rows)
        self.assertEqual(len(output), rows)
        self.assertEqual(len(reference), rows)
        # Result/equivalence shape: output and reference both cover the original rows.
        self.assertIsNotNone(reference)  # the in-place all_reduce must not null it
        self.assertEqual(reference[0][0], 2.0)
        max_abs = max(abs(out - ref) for out_row, ref_row in zip(output, reference)
                      for out, ref in zip(out_row, ref_row))
        self.assertEqual(max_abs, 8.0)
        self.assertEqual(output[0][0], 9.0)   # rank0 block placeholder
        self.assertEqual(output[126][0], 5.0)  # last real row from rank1 block
        self.assertEqual(info['pad'], {'rows': rows, 'padded_rows': 128,
                                       'pad_value': 1.0})
        self.assertEqual(info['row_block'], {'rank': rank, 'start': 64, 'end': 128,
                                             'padded_end': 128})
        self.assertEqual(info['workspace_bytes'], 123)
        self.assertEqual(info['run_id'], 7)


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

    def test_call_release_records_error_without_aborting(self):
        self.assertEqual(run_probe.call_release(lambda i: 0, 7, True), (0, None))
        self.assertEqual(run_probe.call_release(lambda i: 0, 0, True), (0, None))
        self.assertEqual(run_probe.call_release(lambda i: 0, 7, False), (0, None))

        def boom(_):
            raise NotImplementedError('no fallback for zero-tensor op')

        rc, error = run_probe.call_release(boom, 7, True)
        self.assertIsNone(rc)
        self.assertTrue(error.startswith('NotImplementedError: no fallback'))
        # The numeric gate is api/sync/finite/reference and must not depend on release;
        # harness_pass additionally requires clean cleanup, and pass/exit follow numeric.
        source = Path(run_probe.__file__).read_text()
        numeric = source.split('numeric_pass = bool(', 1)[1].split('\n', 1)[0]
        self.assertNotIn('release', numeric)
        harness = source.split("meta['harness_pass'] = bool(", 1)[1].split('\n', 1)[0]
        self.assertIn('harness_release_ok', harness)
        self.assertIn("meta['pass'] = numeric_pass", source)
        # Teardown covers both ranks on success and failure paths, without a new collective.
        self.assertIn('finally:', source)
        self.assertIn('dist.is_initialized()', source)
        self.assertIn("meta['harness_cleanup_error']", source)


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
