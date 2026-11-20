"""CPU-only control-flow tests for the full supervised probe window.

External boundaries (d0_stop, /proc, npu-smi, HTTP) are fakes except in the
real foreground fixture, which starts a temporary HTTP + manifest service and
verifies the window returns while that service is still alive.  The window's
recovery gate, source checks, branch gate and cleanup are real.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import supervised_probe_window as win


class FakeProcess:
    def __init__(self, pid=5252):
        self.pid = pid
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode


def table(pid, extra=(), pgid=None):
    pgid = pid if pgid is None else pgid
    rows = {pid: {'pid': pid, 'ppid': 1, 'pgid': pgid, 'state': 'S',
                  'starttime': '1', 'boot_id': 'b', 'cmdline': ['vllm']}}
    for child in extra:
        rows[child] = {'pid': child, 'ppid': pid, 'pgid': pgid, 'state': 'S',
                       'starttime': '2', 'boot_id': 'b', 'cmdline': ['worker']}
    return rows


def maps(adapter='/x/clef_mmar_probe.so', binary=None, with_fwk=False):
    libs = [{'path': adapter, 'sha256': 'a', 'soname': None},
            {'path': '/x/libhccl.so', 'sha256': 'b', 'soname': 'libhccl.so'},
            {'path': '/x/libopapi_math.so', 'sha256': 'c', 'soname': 'libopapi_math.so'}]
    if with_fwk:
        libs.append({'path': '/x/libhccl_fwk.so', 'sha256': 'd', 'soname': 'libhccl_fwk.so'})
    cmd = [binary, '2', '1', '0,1'] if binary else ['probe']
    return [{'pid': 7001, 'cmdline': cmd, 'libs': libs}]


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='grouped-mmar-window-'))
        self.source = self.root / 'clef_joint_schema.py'
        self.source.write_text('aa3e source\n')
        self.sha = win.sha256(self.source)
        self.identity = self.root / 'id.json'
        self.identity.write_text(json.dumps({'pid': 4242, 'starttime': '1', 'boot_id': 'b',
                                             'pgid': 4242, 'ppid': 1}))
        self.text = self.root / 'text.json'; self.text.write_text('{"q":"x"}')
        self.image = self.root / 'image.json'; self.image.write_text('{"q":"y"}')
        self.restore_output = self.root / 'restore'
        self.run_dir = self.root / 'run'
        self.stop_script = self.root / 'stop.py'
        self.stop_script.write_text('#!/usr/bin/env python3\n')
        self.stop_sha = win.sha256(self.stop_script)

    def args(self, execute=False, extra=()):
        return ['--run-dir', str(self.run_dir), '--root-identity', str(self.identity),
                '--child-argv', json.dumps([sys.executable, '-c', 'print(1)']),
                '--source-target', str(self.source), '--expected-source-sha', self.sha,
                '--official-start', str(self.root / 'start.sh'),
                '--activate', str(self.root / 'activate.sh'),
                '--service-root', str(self.root), '--model-dir', str(self.root / 'model'),
                '--restore-output', str(self.restore_output),
                '--text-body', str(self.text), '--image-body', str(self.image),
                '--stop-script', str(self.stop_script), '--expect-stop-sha', self.stop_sha,
                '--arm', 'A0', '--adapter', '/x/clef_mmar_probe.so', *extra] + (
                    ['--execute'] if execute else [])

    def test_plan_only_validates_and_prints(self):
        self.assertEqual(win.main(self.args()), 0)
        self.assertFalse(self.run_dir.exists())

    def test_settle_timeout_plan_default_and_explicit(self):
        import contextlib
        import io
        for extra, expected in (((), 30.0), (('--settle-timeout', '300'), 300.0)):
            with self.subTest(expected=expected):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(win.main(self.args(extra=extra)), 0)
                self.assertEqual(json.loads(output.getvalue())['settle_timeout'], expected)
                self.assertFalse(self.run_dir.exists())

    def test_settle_timeout_invalid_before_stop_or_child(self):
        import contextlib
        import io
        for value in ('nan', 'NaN', 'inf', '+inf', '-inf', '1e309',
                      '0', '-0', '-1', 'bad', ''):
            with self.subTest(value=value), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(win, 'execute') as execute, \
                    mock.patch.object(win, 'run') as stop, \
                    mock.patch.object(win.subprocess, 'Popen') as child:
                with self.assertRaises(SystemExit) as error:
                    win.main(self.args(execute=True, extra=('--settle-timeout=' + value,)))
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()
                stop.assert_not_called()
                child.assert_not_called()
                self.assertFalse(self.run_dir.exists())

    def test_a13_arm_builds_torchrun_child(self):
        import contextlib
        import io
        argv = self.args(extra=('--arm', 'A13'))
        index = argv.index('--child-argv')
        del argv[index:index + 2]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(win.main(argv), 0)
        plan = json.loads(output.getvalue())
        self.assertIn('A13', plan['child_argv'])
        self.assertIn('--adapter', plan['child_argv'])

    def test_a14_arm_builds_torchrun_child(self):
        import contextlib
        import io
        argv = self.args(extra=('--arm', 'A14'))
        index = argv.index('--child-argv')
        del argv[index:index + 2]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(win.main(argv), 0)
        plan = json.loads(output.getvalue())
        self.assertIn('A14', plan['child_argv'])
        self.assertIn('--adapter', plan['child_argv'])

    def test_h4k_h5k_a15_arms_build_torchrun_child(self):
        import contextlib
        import io
        for arm in ('H4K', 'H5K', 'A15'):
            argv = self.args(extra=('--arm', arm))
            index = argv.index('--child-argv')
            del argv[index:index + 2]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(win.main(argv), 0)
            plan = json.loads(output.getvalue())
            self.assertIn(arm, plan['child_argv'])
            self.assertIn('--adapter', plan['child_argv'])

    def test_begin_restore_blocked_when_subreaper_not_restored(self):
        result = {}
        with mock.patch.object(win, 'disable_subreaper', return_value=False), \
                mock.patch.object(win, 'start_restore') as start_mock:
            self.assertFalse(win.begin_restore({}, Path('/tmp'), result))
        start_mock.assert_not_called()
        self.assertTrue(result['restore_blocked'])
        self.assertIn('blocked_subreaper_restore', result['error'])

    def test_restore_subreaper_restores_original(self):
        with mock.patch.object(win, 'set_subreaper', return_value=True) as set_mock:
            self.assertTrue(win.restore_subreaper(1))
            set_mock.assert_called_once_with(True)
            set_mock.reset_mock()
            self.assertTrue(win.restore_subreaper(0))
            set_mock.assert_called_once_with(False)
        self.assertFalse(win.restore_subreaper(None))

    def test_restored_root_waits_for_wrapper_exit(self):
        wrapper = {'pid': 7, 'ppid': 8, 'pgid': 7, 'state': 'S',
                   'starttime': '1', 'boot_id': 'b', 'cmdline': []}
        root = {'pid': 7, 'ppid': 1, 'pgid': 7, 'state': 'S',
                'starttime': '1', 'boot_id': 'b', 'cmdline': []}
        with mock.patch.object(win, 'scan', side_effect=[{7: wrapper}, {7: root}]):
            live = win.wait_restored_root(7, timeout=1, interval=0.01)
        self.assertEqual(live['ppid'], 1)

    def test_restored_root_timeout_returns_last(self):
        wrapper = {'pid': 7, 'ppid': 8, 'pgid': 7, 'state': 'S',
                   'starttime': '1', 'boot_id': 'b', 'cmdline': []}
        with mock.patch.object(win, 'scan', return_value={7: wrapper}):
            live = win.wait_restored_root(7, timeout=0.03, interval=0.01)
        self.assertEqual(live['ppid'], 8)

    def test_plan_only_does_not_require_subreaper(self):
        import contextlib
        import io
        with mock.patch.object(win.sys, 'platform', 'linux'), \
                mock.patch.object(win, 'enable_subreaper') as sub_mock:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = win.main(self.args())
        self.assertEqual(rc, 0)
        sub_mock.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['status'], 'plan_not_executed')

    def test_linux_subreaper_failure_blocks_before_stop(self):
        calls = []
        with mock.patch.object(win.sys, 'platform', 'linux'), \
                mock.patch.object(win, 'enable_subreaper', return_value=False), \
                mock.patch.object(win, 'scan', return_value=table(4242, (4243,))), \
                mock.patch.object(win, 'proc_identity', side_effect=lambda pid: {
                    'pid': pid, 'ppid': 1, 'pgid': pid, 'sid': pid, 'state': 'S',
                    'starttime': '1', 'boot_id': 'b', 'cmdline': ['root']}), \
                mock.patch.object(win, 'port_open', return_value=0), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, 'run', side_effect=lambda *a, **k: calls.append(a)):
            rc = win.main(self.args(execute=True))
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertIn('subreaper unavailable', result['error'])

    def test_missing_stop_script_rejected(self):
        argv = self.args()
        argv[argv.index('--stop-script') + 1] = str(self.root / 'missing.py')
        with self.assertRaises(ValueError):
            win.main(argv)

    def test_missing_probe_body_rejected(self):
        argv = self.args()
        argv[argv.index('--text-body') + 1] = str(self.root / 'missing.json')
        with self.assertRaises(ValueError):
            win.main(argv)

    def test_restore_output_must_be_fresh(self):
        self.restore_output.mkdir()
        with self.assertRaises(ValueError):
            win.main(self.args())

    def full_success(self, branch='aa3e-noop', stop_rc=0):
        stopped = {'value': False}

        def fake_restore(plan, run_dir, result):
            self.restore_output.mkdir(parents=True, exist_ok=True)
            (self.restore_output / 'launch-manifest.json').write_text(json.dumps(
                {'pid': 6262, 'output_dir': str(self.restore_output.resolve())}))
            (self.restore_output / 'joint-patch-record.json').write_text(
                json.dumps({'branch': branch}))
            result['restore'] = {'started': True, 'waited': False}
            return None

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
                return {'argv': argv, 'rc': stop_rc, 'log': str(log)}
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            return table(4242, (4243,)) if not stopped['value'] else table(6262, (6263,))

        with mock.patch.object(win, 'scan', side_effect=fake_scan), \
                mock.patch.object(win, 'port_open', side_effect=[0, 111, 111]), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, 'run', side_effect=fake_run), \
                mock.patch.object(win, 'start_restore', side_effect=fake_restore), \
                mock.patch.object(win, 'npu_evidence', return_value=[]), \
                mock.patch.object(win, 'wait_health', return_value={'status': 'ready'}), \
                mock.patch.object(win, 'env_evidence', return_value={'match': True}), \
                mock.patch.object(win, 'read_root_env', return_value={}), \
                mock.patch.object(win, 'joint_env', return_value={'ok': True}), \
                mock.patch.object(win, 'http_json', return_value={'status': 'ready', 'config': 's1'}), \
                mock.patch.object(win, 'post_json', return_value=(200, {'ok': True})), \
                mock.patch.object(win, 'proc_identity', side_effect=lambda pid: {
                    'pid': pid, 'ppid': 1, 'pgid': pid, 'state': 'S',
                    'starttime': '1', 'boot_id': 'b', 'cmdline': ['child']}), \
                mock.patch.object(win, 'maps_evidence', return_value=maps()), \
                mock.patch.object(win.subprocess, 'Popen', return_value=FakeProcess()):
            return win.main(self.args(execute=True))

    def test_full_success_path(self):
        self.assertEqual(self.full_success(), 0)
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertTrue(result['ok'])
        self.assertEqual(result['restore_branch'], 'aa3e-noop')
        self.assertEqual(result['source_sha256_after_restore'], self.sha)
        self.assertEqual(result['steps']['stopped']['port'], 111)
        self.assertEqual(result['recovery_gate']['reason'], 'restore')

    class TimeoutProcess(FakeProcess):
        def __init__(self):
            super().__init__(pid=5252)
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired('torchrun', timeout)
            self.returncode = 0
            return 0

    def timeout_settle_case(self, clear_rounds=2, persistent=False, settle_timeout=None,
                           settle_override=None):
        ranks = [
            {'pid': 5253, 'ppid': 5252, 'pgid': 5253, 'sid': 5253,
             'starttime': '2', 'boot_id': 'b', 'cmdline': ['rank0']},
            {'pid': 5254, 'ppid': 5252, 'pgid': 5254, 'sid': 5254,
             'starttime': '2', 'boot_id': 'b', 'cmdline': ['rank1']},
        ]
        alive = {e['pid']: {'pid': e['pid'], 'ppid': 5252, 'pgid': e['pid'],
                            'sid': e['pid'], 'state': 'S', 'starttime': '2',
                            'boot_id': 'b', 'cmdline': e['cmdline']}
                 for e in ranks}

        class FakeRegistry:
            def __init__(self, entries):
                self.entries = entries

            def start(self):
                return self

            def stop(self):
                pass

            def record_now(self):
                pass

            def snapshot(self):
                return [dict(entry) for entry in self.entries]

        class Clock:
            def __init__(self):
                self.t = 0.0

            def monotonic(self):
                return self.t

            def sleep(self, seconds):
                self.t += seconds

        clock = Clock()
        tables = [alive] * clear_rounds + [{}, {}]
        seq = iter(tables)
        real_settle = win.settle_attributed

        def table_reader():
            if persistent:
                return alive
            try:
                return next(seq)
            except StopIteration:
                return {}

        def fake_settle(entries, child_pid, **kwargs):
            if settle_override is not None:
                return settle_override
            return real_settle(entries, child_pid, clock=clock.monotonic,
                               sleep=clock.sleep, table_reader=table_reader, **kwargs)

        started = []
        stopped = {'value': False}

        def fake_restore(plan, run_dir, result):
            self.restore_output.mkdir(parents=True, exist_ok=True)
            (self.restore_output / 'launch-manifest.json').write_text(json.dumps(
                {'pid': 6262, 'output_dir': str(self.restore_output.resolve())}))
            (self.restore_output / 'joint-patch-record.json').write_text(
                json.dumps({'branch': 'aa3e-noop'}))
            started.append(True)
            result['restore'] = {'started': True, 'waited': False}

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            return table(4242, (4243,)) if not stopped['value'] else table(6262, (6263,))

        def fake_proc(pid):
            return {'pid': pid, 'ppid': 1, 'pgid': pid, 'state': 'S',
                    'starttime': '2', 'boot_id': 'b', 'cmdline': ['child']}

        with ExitStack() as stack:
            def patch(target, name, **kwargs):
                return stack.enter_context(mock.patch.object(target, name, **kwargs))

            patch(win, 'DescendantRegistry', return_value=FakeRegistry(ranks))
            patch(win, '_scan_direct', return_value=alive)
            patch(win, '_read_proc', side_effect=lambda pid: alive.get(pid))
            patch(win, '_signal_targets')
            patch(win.time, 'sleep')
            settle = patch(win, 'settle_attributed', side_effect=fake_settle)
            begin = patch(win, 'begin_restore', wraps=win.begin_restore)
            patch(win, 'scan', side_effect=fake_scan)
            patch(win, 'port_open', side_effect=[0, 111, 111])
            patch(win, 'pending_clients', return_value=[])
            patch(win, 'run', side_effect=fake_run)
            patch(win, 'start_restore', side_effect=fake_restore)
            patch(win, 'npu_evidence', return_value=[])
            patch(win, 'wait_health', return_value={'status': 'ready'})
            patch(win, 'env_evidence', return_value={'match': True})
            patch(win, 'read_root_env', return_value={})
            patch(win, 'joint_env', return_value={'ok': True})
            patch(win, 'http_json', return_value={'status': 'ready', 'config': 's1'})
            patch(win, 'post_json', return_value=(200, {'ok': True}))
            patch(win, 'proc_identity', side_effect=fake_proc)
            patch(win, 'maps_evidence', return_value=maps())
            patch(win.subprocess, 'Popen', return_value=self.TimeoutProcess())
            if persistent and settle_timeout is None:
                settle_timeout = 2.0
            extra = () if settle_timeout is None else ('--settle-timeout', str(settle_timeout))
            code = win.main(self.args(execute=True, extra=extra))
            settle.assert_called_once()
            self.assertEqual(settle.call_args.kwargs['deadline'],
                             30.0 if settle_timeout is None else settle_timeout)
            self.assertEqual(begin.call_count, len(started))
            self.assertLessEqual(begin.call_count, 1)
        result = json.loads((self.run_dir / 'window.json').read_text())
        return code, result, started

    def test_timeout_parent_exits_then_delayed_clear_allows_one_restore(self):
        code, result, started = self.timeout_settle_case(clear_rounds=2)
        child = result['steps']['child']
        self.assertTrue(child['timed_out'])
        self.assertTrue(child['settle']['clean'])
        self.assertEqual(child['survivors'], [])
        self.assertEqual(result['recovery_gate']['reason'], 'restore')
        self.assertEqual(len(started), 1)
        self.assertFalse(result['ok'])  # timeout fact is preserved
        self.assertEqual(child['settle']['deadline'], 30.0)
        self.assertEqual(child['settle']['min_empty'], 2)
        self.assertEqual(child['settle']['empty_rounds'], 2)

    def test_settle_timeout_300_reaches_real_settle_and_restores_once(self):
        code, result, started = self.timeout_settle_case(clear_rounds=520, settle_timeout=300)
        settle = result['steps']['child']['settle']
        self.assertEqual(result['plan']['settle_timeout'], 300.0)
        self.assertEqual(settle['deadline'], 300.0)
        self.assertEqual(settle['elapsed'], 260.5)
        self.assertEqual(settle['elapsed_ns'], 260_500_000_000)
        self.assertEqual(settle['empty_rounds'], 2)
        self.assertTrue(settle['clean'])
        self.assertFalse(settle['deadline_hit'])
        self.assertEqual(len(started), 1)
        self.assertEqual(code, 1)  # child timeout remains a failed probe

    def test_settle_budget_not_reset_across_exception_and_finally(self):
        code, result, started = self.timeout_settle_case(persistent=True, settle_timeout=300)
        settle = result['steps']['child']['settle']
        self.assertEqual(settle['deadline'], 300.0)
        self.assertEqual(settle['elapsed'], 300.0)
        self.assertEqual(settle['elapsed_ns'], 300_000_000_000)
        self.assertEqual(len(settle['rounds']), 601)
        self.assertTrue(settle['deadline_hit'])
        self.assertFalse(settle['clean'])
        self.assertIn('child group survivors', result['steps']['child']['error'])
        self.assertEqual(started, [])
        self.assertEqual(code, 1)

    def test_timeout_persistent_residual_is_bounded_and_blocks_restore(self):
        code, result, started = self.timeout_settle_case(persistent=True)
        child = result['steps']['child']
        self.assertTrue(child['settle']['deadline_hit'])
        self.assertGreaterEqual(len(child['settle']['rounds']), 2)
        self.assertLessEqual(child['settle']['rounds'][-1]['t'], 2.0)
        self.assertEqual(result['recovery_gate']['reason'], 'blocked_settle_deadline')
        self.assertEqual(started, [])
        self.assertFalse(result['ok'])

    def test_settle_unclean_empty_result_blocks_restore(self):
        outcome = {'clean': False, 'deadline_hit': False, 'residual': [],
                   'table': {}, 'states': {}, 'rounds': [{'t': 0, 'residual': []}],
                   'empty_rounds': 1, 'deadline': 30.0, 'elapsed': 0.0, 'min_empty': 2}
        code, result, started = self.timeout_settle_case(settle_override=outcome)
        self.assertEqual(result['recovery_gate']['reason'], 'blocked_settle_unclean')
        self.assertEqual(result['recovery_gate']['settle'], outcome)
        self.assertEqual(result['restore']['settle'], outcome)
        self.assertEqual(started, [])
        self.assertEqual(code, 1)

    def test_settle_deadline_even_with_clean_empty_result_blocks_restore(self):
        outcome = {'clean': True, 'deadline_hit': True, 'residual': [],
                   'table': {}, 'states': {}, 'rounds': [{'t': 30, 'residual': []}],
                   'empty_rounds': 2, 'deadline': 30.0, 'elapsed': 30.0, 'min_empty': 2}
        code, result, started = self.timeout_settle_case(settle_override=outcome)
        self.assertEqual(result['recovery_gate']['reason'], 'blocked_settle_deadline')
        self.assertEqual(result['steps']['child']['settle'], outcome)
        self.assertEqual(started, [])
        self.assertEqual(code, 1)

    def test_settle_one_empty_at_deadline_blocks_restore(self):
        code, result, started = self.timeout_settle_case(clear_rounds=2, settle_timeout=1)
        settle = result['steps']['child']['settle']
        self.assertEqual(settle['empty_rounds'], 1)
        self.assertEqual(settle['residual'], [])
        self.assertEqual(settle['elapsed'], 1.0)
        self.assertTrue(settle['deadline_hit'])
        self.assertFalse(settle['clean'])
        self.assertEqual(result['recovery_gate']['reason'], 'blocked_settle_deadline')
        self.assertEqual(started, [])
        self.assertEqual(code, 1)

    def test_settle_second_empty_on_deadline_blocks_restore(self):
        code, result, started = self.timeout_settle_case(clear_rounds=0, settle_timeout=0.5)
        settle = result['steps']['child']['settle']
        self.assertEqual(settle['empty_rounds'], 2)
        self.assertEqual(settle['residual'], [])
        self.assertEqual(settle['elapsed_ns'], 500_000_000)
        self.assertFalse(settle['clean'])
        self.assertTrue(settle['deadline_hit'])
        self.assertEqual(result['recovery_gate']['reason'], 'blocked_settle_deadline')
        self.assertEqual(started, [])
        self.assertEqual(code, 1)

    def test_f810_to_fused_marks_isolation_failure(self):
        self.assertEqual(self.full_success('f810->fused'), 1)
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertTrue(result['isolation_failed'])
        self.assertFalse(result['ok'])

    def test_stop_failure_with_old_root_gone_still_restores(self):
        self.assertEqual(self.full_success(stop_rc=1), 1)
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertTrue(result['restore'].get('started'))
        self.assertFalse(result['ok'])

    def blocked_case(self, port_sequence, pre_table, post_table, health=None):
        stopped = {'value': False}
        started = []

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            return post_table if stopped['value'] else pre_table

        def fake_health(*a, **k):
            if health is None:
                raise RuntimeError('no health')
            return health

        with mock.patch.object(win, 'scan', side_effect=fake_scan), \
                mock.patch.object(win, 'port_open', side_effect=port_sequence), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, 'run', side_effect=fake_run), \
                mock.patch.object(win, 'start_restore', side_effect=lambda *a: started.append(a)), \
                mock.patch.object(win, 'http_json', side_effect=fake_health), \
                mock.patch.object(win.subprocess, 'Popen', return_value=FakeProcess()):
            code = win.main(self.args(execute=True))
        result = json.loads((self.run_dir / 'window.json').read_text())
        return code, result, started

    def test_live_service_blocks_second_start(self):
        code, result, started = self.blocked_case([0, 0, 0], table(4242, (4243,)),
                                                  table(4242, (4243,)))
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'old_root_alive_partial')
        self.assertFalse(result['ok'])

    def test_live_healthy_service_is_skipped_not_restarted(self):
        code, result, started = self.blocked_case(
            [0, 0, 0], table(4242, (4243,)), table(4242, (4243,)),
            health={'status': 'ready'})
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'skipped_service_alive_healthy')
        self.assertEqual(result['restore']['health'], {'status': 'ready'})

    def test_live_service_with_port_111_is_blocked(self):
        code, result, started = self.blocked_case([0, 0, 111], table(4242, (4243,)),
                                                  table(4242, (4243,)))
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'old_root_alive_partial')

    def test_port_open_without_root_blocks(self):
        code, result, started = self.blocked_case([0, 111, 0], table(4242, (4243,)), {})
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'blocked_port_0')

    def test_old_pgid_remnant_blocks(self):
        remnant = {4343: {'pid': 4343, 'ppid': 1, 'pgid': 4242, 'state': 'S',
                          'starttime': '2', 'boot_id': 'b', 'cmdline': ['old']}}
        code, result, started = self.blocked_case([0, 111, 111], table(4242, (4243,)), remnant)
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'blocked_old_pgid_members')

    def test_runner_residual_blocks(self):
        started = []
        stopped = {'value': False}

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            return table(4242, (4243,)) if not stopped['value'] else {}

        calls = {'n': 0}
        def fake_pending(*a, **k):
            calls['n'] += 1
            return [] if calls['n'] == 1 else [777]
        with mock.patch.object(win, 'scan', side_effect=fake_scan), \
                mock.patch.object(win, 'port_open', side_effect=[0, 111, 111]), \
                mock.patch.object(win, 'pending_clients', side_effect=fake_pending), \
                mock.patch.object(win, 'run', side_effect=fake_run), \
                mock.patch.object(win, 'start_restore', side_effect=lambda *a: started.append(a)), \
                mock.patch.object(win.subprocess, 'Popen', return_value=FakeProcess()):
            code = win.main(self.args(execute=True))
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertEqual(code, 1)
        self.assertEqual(started, [])
        self.assertEqual(result['restore']['reason'], 'blocked_runner_residual')

    def test_a12_evidence_rejects_error_lines(self):
        out = self.root / 'a12.out'; err = self.root / 'a12.err'
        out.write_text('rank 0 hcom name is x\n[INFO] mc2 costtime = 1, deviceid = 0\n'
                       '[INFO] mc2 costtime = 2, deviceid = 1\n')
        err.write_text('')
        self.assertTrue(win.a12_evidence(out, err, 0)['ok'])
        err.write_text('[ERROR] acl interface return err x:1, retcode: 507033\n')
        self.assertFalse(win.a12_evidence(out, err, 0)['ok'])

    def test_maps_per_pid_and_coverage(self):
        entries = maps(binary='/x/a12_grouped_mmar')
        self.assertTrue(win.required_map_ok('A12', None, entries))
        coverage = win.map_coverage(entries)
        self.assertTrue(coverage['libhccl'])
        self.assertTrue(coverage['opapi_math'])
        self.assertFalse(coverage['hccl_fwk'])
        self.assertEqual(win.binary_processes(entries), [7001])
        self.assertTrue(win.required_map_ok('A0', '/x/clef_mmar_probe.so', entries))
        self.assertFalse(win.required_map_ok(
            'A12', None, [{'pid': 1, 'cmdline': ['x'],
                           'libs': [{'path': '/x/libhccl.so'}]}]))

    def test_npu_rows_parser(self):
        text = ('Process id in container\n'
                '| 7 0 | 123 | VLLMWorker_TP | 1 | 111 |\n'
                '| 7 1 | 124 | VLLMWorker_TP | 1 | 112 |\n')
        rows = win.npu_rows(text)
        self.assertEqual([r['chip'] for r in rows], [0, 1])
        self.assertEqual([r['pid'] for r in rows], [111, 112])

    def test_joint_env_requires_exact_keys(self):
        live = {'CLEF_JOINT_FUSED_ATTN': b'1',
                'CLEF_JOINT_ALLOWED_POSTIMAGE_SHA': win.AA3E.encode(),
                'CLEF_JOINT_SINGLE_FIELD': b'1'}
        self.assertTrue(win.joint_env(live)['ok'])
        live['CLEF_JOINT_SMALLQ'] = b'1'
        self.assertFalse(win.joint_env(live)['ok'])

    def test_cleanup_protected_member_is_not_signalled(self):
        calls = []
        with mock.patch.object(win, 'scan', return_value=table(10, (11,))), \
                mock.patch.object(win.os, 'killpg', side_effect=lambda *a: calls.append(a)):
            outcome = win.cleanup_child(10, table(10, (11,)), {11})
        self.assertFalse(outcome['killed'])
        self.assertEqual(calls, [])

    def test_restore_argv_is_clean(self):
        a = win.argparse.Namespace(service_root='/root', activate='/act', official_start='/start',
                                   model_dir='/model', restore_output='/out')
        argv = win.build_restore_argv(a)
        self.assertEqual(argv[:3], ['env', '-i', f'PATH={win.CLEAN_PATH}'])
        self.assertIn('setsid', argv)
        self.assertIn('/start', argv[-1])

    def test_real_foreground_http_restore_keeps_service_alive(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        service = self.root / 'service.py'
        service.write_text(f'''import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--port', type=int)
a = p.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
with open(os.path.join(a.output_dir, 'launch-manifest.json'), 'w') as f:
    json.dump({{'pid': os.getpid(), 'output_dir': os.path.abspath(a.output_dir)}}, f)
with open(os.path.join(a.output_dir, 'joint-patch-record.json'), 'w') as f:
    json.dump({{'branch': 'aa3e-noop'}}, f)
class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = json.dumps({{'status': 'ready', 'model': 'clef', 'revision': 'x'}}
                          if self.path == '/health' else {{'config': 's1'}}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = json.dumps({{'ok': True}}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
HTTPServer(('127.0.0.1', a.port), H).serve_forever()
''')
        start = self.root / 'start.sh'
        start.write_text(f'''#!/bin/bash
out=""
port=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) out="$2"; shift 2;;
    --port) port="$2"; shift 2;;
    --model-dir) shift 2;;
    *) shift;;
  esac
done
exec "{sys.executable}" "{service}" --output-dir "$out" --port "$port"
''')
        start.chmod(0o755)
        (self.root / 'activate.sh').write_text('#!/bin/bash\ntrue\n')
        (self.root / 'activate.sh').chmod(0o755)
        (self.root / 'model').mkdir()
        stopped = {'value': False}

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            if not stopped['value']:
                return table(4242, (4243,))
            manifest = self.restore_output / 'launch-manifest.json'
            if manifest.is_file():
                pid = json.loads(manifest.read_text())['pid']
                return table(pid, (pid + 1,))
            return {}

        restore_argv = ['/bin/bash', str(start), '--output-dir', str(self.restore_output),
                        '--port', str(port)]
        with mock.patch.object(win, 'build_restore_argv', return_value=restore_argv), \
                mock.patch.object(win, 'scan', side_effect=fake_scan), \
                mock.patch.object(win, 'port_open', side_effect=[0, 111, 111]), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, 'run', side_effect=fake_run), \
                mock.patch.object(win, 'npu_evidence', return_value=[]), \
                mock.patch.object(win, 'env_evidence', return_value={'match': True}), \
                mock.patch.object(win, 'read_root_env', return_value={}), \
                mock.patch.object(win, 'joint_env', return_value={'ok': True}), \
                mock.patch.object(win, 'proc_identity', side_effect=lambda pid: {
                    'pid': pid, 'ppid': 1, 'pgid': pid, 'state': 'S',
                    'starttime': '1', 'boot_id': 'b', 'cmdline': ['child']}), \
                mock.patch.object(win, 'maps_evidence', return_value=maps()):
            rc = win.main(self.args(execute=True, extra=('--restore-timeout', '5', '--port', str(port))))
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertIn('restore_checks', result, result.get('error'))
        pid = result['restore_checks']['pid']
        try:
            self.assertEqual(rc, 0, result.get('error'))
            self.assertTrue(result['ok'])
            os.kill(pid, 0)
        finally:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class SettleAttributedTest(unittest.TestCase):
    class Clock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            return self.t

        def sleep(self, seconds):
            self.t += seconds

    def entry(self, pid, start='10'):
        return {'pid': pid, 'ppid': 1, 'pgid': pid, 'sid': pid,
                'starttime': start, 'boot_id': 'b'}

    def row(self, pid, state='S', start='10'):
        return {'pid': pid, 'ppid': 1, 'pgid': pid, 'sid': pid, 'state': state,
                'starttime': start, 'boot_id': 'b', 'cmdline': []}

    def seq(self, tables):
        it = iter(tables)
        calls = {'n': 0}

        def reader():
            calls['n'] += 1
            return next(it)

        reader.calls = calls
        return reader

    def test_exit_delay_clears_after_two_empty_rounds(self):
        clock = self.Clock()
        reader = self.seq([{5253: self.row(5253)}, {5253: self.row(5253)}, {}, {}])
        out = win.settle_attributed([self.entry(5253)], 5252, clock=clock.monotonic,
                                    sleep=clock.sleep, table_reader=reader,
                                    deadline=30.0, interval=1.0)
        self.assertTrue(out['clean'])
        self.assertFalse(out['deadline_hit'])
        self.assertEqual(len(out['rounds']), 4)
        self.assertEqual(out['rounds'][0]['residual'], [5253])
        self.assertEqual(out['rounds'][0]['states']['5253']['state'], 'S')
        self.assertEqual(out['rounds'][0]['states']['5253']['starttime'], '10')
        self.assertEqual(out['residual'], [])
        self.assertEqual(reader.calls['n'], 4)
        self.assertEqual(out['elapsed'], 3.0)

    def test_second_empty_at_or_after_deadline_is_not_clean(self):
        for read_delay in (0.0, 0.25):
            with self.subTest(read_delay=read_delay):
                clock = self.Clock()
                reads = []

                def reader():
                    if reads:
                        clock.t += read_delay
                    reads.append(clock.t)
                    return {}

                out = win.settle_attributed([], 5252, clock=clock.monotonic,
                                           sleep=clock.sleep, table_reader=reader,
                                           deadline=0.5, interval=0.5)
                self.assertEqual(out['empty_rounds'], 2)
                self.assertEqual(out['residual'], [])
                self.assertEqual(out['elapsed'], 0.5 + read_delay)
                self.assertTrue(out['deadline_hit'])
                self.assertFalse(out['clean'])
                self.assertEqual(len(reads), 2)

    def test_short_budget_caps_sleep_and_still_requires_two_empty_rounds(self):
        clock = self.Clock()
        reader = self.seq([{}, {}])
        out = win.settle_attributed([], 5252, clock=clock.monotonic,
                                    sleep=clock.sleep, table_reader=reader,
                                    deadline=0.1)
        self.assertEqual(out['elapsed'], 0.1)
        self.assertEqual(out['min_empty'], 2)
        self.assertEqual(out['empty_rounds'], 2)
        self.assertTrue(out['deadline_hit'])
        self.assertFalse(out['clean'])

    def test_persistent_non_z_hits_deadline_and_fails_closed(self):
        clock = self.Clock()
        alive = {5253: self.row(5253)}
        reader = self.seq([alive] * 10)
        out = win.settle_attributed([self.entry(5253)], 5252, clock=clock.monotonic,
                                    sleep=clock.sleep, table_reader=reader,
                                    deadline=3.0, interval=1.0)
        self.assertFalse(out['clean'])
        self.assertTrue(out['deadline_hit'])
        self.assertEqual(out['residual'], [5253])
        self.assertEqual(out['states']['5253']['state'], 'S')
        self.assertGreaterEqual(len(out['rounds']), 4)

    def test_zombie_not_residual_and_reaped(self):
        clock = self.Clock()
        zombie = {5253: self.row(5253, state='Z')}
        reader = self.seq([zombie, zombie])
        with mock.patch.object(win, '_reap') as reap:
            out = win.settle_attributed([self.entry(5253)], 5252, clock=clock.monotonic,
                                        sleep=clock.sleep, table_reader=reader, reap=True)
        self.assertTrue(out['clean'])
        self.assertEqual(out['residual'], [])
        self.assertTrue(reap.called)

    def test_single_snapshot_per_round_never_calls_read_proc(self):
        clock = self.Clock()
        reader = self.seq([{5253: self.row(5253)}, {}, {}])
        with mock.patch.object(win, '_read_proc',
                               side_effect=AssertionError('double read')):
            out = win.settle_attributed([self.entry(5253)], 5252,
                                        clock=clock.monotonic, sleep=clock.sleep,
                                        table_reader=reader)
        self.assertTrue(out['clean'])
        self.assertEqual(reader.calls['n'], 3)

    def test_missing_state_is_unknown(self):
        clock = self.Clock()
        bare = self.row(5253)
        bare['state'] = None
        reader = self.seq([{5253: bare}, {}, {}])
        out = win.settle_attributed([self.entry(5253)], 5252, clock=clock.monotonic,
                                    sleep=clock.sleep, table_reader=reader)
        self.assertEqual(out['rounds'][0]['states']['5253']['state'], 'unknown')

    def test_baseline_not_in_entries_is_not_residual(self):
        clock = self.Clock()
        table = {5253: self.row(5253), 4242: self.row(4242)}
        reader = self.seq([table] * 10)
        out = win.settle_attributed([self.entry(5253)], 5252, clock=clock.monotonic,
                                    sleep=clock.sleep, table_reader=reader,
                                    deadline=1.0, interval=0.5)
        self.assertIn(5253, out['residual'])
        self.assertNotIn(4242, out['residual'])
        self.assertNotIn('4242', out['states'])


class AttributedTargetsTest(unittest.TestCase):
    def test_pid_reuse_and_starttime_drift_rejected(self):
        entry = {'pid': 77, 'starttime': '10', 'boot_id': 'b', 'pgid': 77, 'sid': 77}
        table = {77: {'pid': 77, 'ppid': 1, 'pgid': 77, 'sid': 77, 'state': 'S',
                      'starttime': '11', 'boot_id': 'b', 'cmdline': []}}
        targets = win._attributed_targets([entry], table=table, reader=table.get)
        self.assertEqual(targets['verified'], [])
        self.assertTrue(targets['drift'])
        self.assertEqual(targets['groups'], {})
        self.assertEqual(targets['singles'], set())

        class FakeRegistry:
            def snapshot(self):
                return [entry]

        self.assertEqual(win.attributed_survivors(FakeRegistry(), table=table,
                                                  reader=table.get), [77])
        self.assertTrue(win.cleanup_blocks_restore({'reason': 'protected_overlap'}))
        self.assertFalse(win.cleanup_blocks_restore({'reason': None}))

    def test_gate_without_settle_uses_real_proc_attribution(self):
        entry = {'pid': 777, 'ppid': 1, 'pgid': 777, 'sid': 777,
                 'starttime': '10', 'boot_id': 'b'}
        row = {'pid': 777, 'ppid': 1, 'pgid': 777, 'sid': 777, 'state': 'S',
               'starttime': '10', 'boot_id': 'b', 'cmdline': []}

        class FakeRegistry:
            def snapshot(self):
                return [dict(entry)]

        args = types.SimpleNamespace(port=1, expected_source_sha='x')
        saved = {'pid': 2 ** 22, 'starttime': '1', 'boot_id': 'x', 'pgid': 2 ** 22}
        with mock.patch.object(win, 'scan', return_value={}), \
                mock.patch.object(win, 'port_open', return_value=111), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, '_scan_direct', return_value={777: row}), \
                mock.patch.object(win, '_read_proc',
                                  side_effect=lambda pid: row if pid == 777 else None):
            allow, reason, context = win.recovery_gate(
                args, {}, saved, True, 9000, 'x', FakeRegistry())
        self.assertFalse(allow)
        self.assertEqual(reason, 'blocked_probe_residual')
        self.assertIn(777, context['attributed_survivors'])

    def test_gate_with_settle_table_uses_snapshot_without_real_read(self):
        entry = {'pid': 777, 'ppid': 1, 'pgid': 777, 'sid': 777,
                 'starttime': '10', 'boot_id': 'b'}

        class FakeRegistry:
            def snapshot(self):
                return [dict(entry)]

        args = types.SimpleNamespace(port=1, expected_source_sha='x')
        saved = {'pid': 2 ** 22, 'starttime': '1', 'boot_id': 'x', 'pgid': 2 ** 22}
        with mock.patch.object(win, 'scan', return_value={}), \
                mock.patch.object(win, 'port_open', return_value=111), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, '_scan_direct',
                                  side_effect=AssertionError('must not read')), \
                mock.patch.object(win, '_read_proc',
                                  side_effect=AssertionError('must not read')):
            allow, reason, context = win.recovery_gate(
                args, {}, saved, True, 9000, 'x', FakeRegistry(),
                table={}, settle={'clean': True})
        self.assertTrue(allow)
        self.assertEqual(reason, 'restore')

    def test_unknown_group_member_is_not_group_killed(self):
        entry = {'pid': 88, 'starttime': '10', 'boot_id': 'b', 'pgid': 88, 'sid': 88}
        table = {
            88: {'pid': 88, 'ppid': 1, 'pgid': 88, 'sid': 88, 'state': 'S',
                 'starttime': '10', 'boot_id': 'b', 'cmdline': []},
            99: {'pid': 99, 'ppid': 1, 'pgid': 88, 'sid': 88, 'state': 'S',
                 'starttime': '12', 'boot_id': 'b', 'cmdline': []},
        }
        targets = win._attributed_targets([entry], table=table, reader=table.get)
        self.assertEqual(targets['unknown'], [99])
        self.assertEqual(targets['groups'], {})
        self.assertEqual(targets['singles'], {88})

    def test_dead_leader_group_member_is_attributed(self):
        entry = {'pid': 70, 'starttime': '10', 'boot_id': 'b', 'pgid': 70, 'sid': 70}
        table = {71: {'pid': 71, 'ppid': 1, 'pgid': 70, 'sid': 70, 'state': 'S',
                      'starttime': '11', 'boot_id': 'b', 'cmdline': []}}
        targets = win._attributed_targets([entry], table=table, reader=table.get)
        self.assertEqual(targets['unknown'], [71])
        self.assertEqual(targets['groups'], {})

    def test_reparent_does_not_break_registered_identity(self):
        entry = {'pid': 66, 'ppid': 5, 'starttime': '10', 'boot_id': 'b',
                 'pgid': 66, 'sid': 66}
        table = {66: {'pid': 66, 'ppid': 1, 'pgid': 66, 'sid': 66, 'state': 'S',
                      'starttime': '10', 'boot_id': 'b', 'cmdline': []}}
        targets = win._attributed_targets([entry], table=table, reader=table.get)
        self.assertEqual([row['pid'] for row in targets['verified']], [66])
        self.assertEqual(targets['groups'], {66: {66}})


@unittest.skipUnless(Path('/proc').is_dir() and sys.platform.startswith('linux'),
                     'Linux /proc required')
class AttributedProcessTest(unittest.TestCase):
    def test_orphan_after_immediate_parent_exit_is_registered_and_cleaned(self):
        self.assertTrue(win.enable_subreaper())
        code = (
            'import subprocess, sys\n'
            'w = subprocess.Popen([sys.executable, "-c", '
            '"import time; time.sleep(60)"], start_new_session=True)\n'
            'print(w.pid, flush=True)\n'
            'sys.exit(0)\n')
        token = 'test-orphan-token'
        env = {**os.environ, win.OWNER_ENV: token}
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)  # deliberately no token
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  env=env, start_new_session=True, text=True)
        worker = None
        try:
            worker = int(parent.stdout.readline().strip())
            parent.wait(timeout=10)
            root = Path(tempfile.mkdtemp(prefix='grouped-mmar-registry-'))
            registry = win.DescendantRegistry(parent.pid, root / 'descendants.json', set(),
                                              token=token).start()
            try:
                deadline = time.time() + 5
                snapshot = []
                while time.time() < deadline:
                    snapshot = registry.snapshot()
                    if any(entry['pid'] == worker for entry in snapshot):
                        break
                    time.sleep(0.1)
                self.assertTrue(any(entry['pid'] == worker for entry in snapshot))
                self.assertFalse(any(entry['pid'] == unrelated.pid for entry in snapshot))
                entry = next(e for e in snapshot if e['pid'] == worker)
                self.assertEqual(entry['ppid'], os.getpid())
                self.assertIn(worker, win.attributed_survivors(registry))
                args = types.SimpleNamespace(port=1, expected_source_sha='x')
                with mock.patch.object(win, 'scan', return_value={}), \
                        mock.patch.object(win, 'port_open', return_value=111), \
                        mock.patch.object(win, 'pending_clients', return_value=[]):
                    allow, reason, context = win.recovery_gate(
                        args, {}, {'pid': 2 ** 22, 'starttime': '1', 'boot_id': 'x',
                                   'pgid': 2 ** 22}, True, parent.pid, 'x', registry)
                self.assertFalse(allow)
                self.assertEqual(reason, 'blocked_probe_residual')
                self.assertIn(worker, context['attributed_survivors'])
                win.cleanup_attributed(parent.pid, None, registry, set(), reap=True)
                self.assertEqual(win.attributed_survivors(registry), [])
                self.assertFalse(Path(f'/proc/{worker}').exists())
                self.assertIsNotNone(win._read_proc(unrelated.pid))  # no arbitrary kill
            finally:
                registry.stop()
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if worker is not None:
                try:
                    os.killpg(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            unrelated.kill()
            unrelated.wait(timeout=5)

    def test_stop_before_cleanup_records_orphan_without_waiting(self):
        self.assertTrue(win.enable_subreaper())
        code = (
            'import subprocess, sys\n'
            'w = subprocess.Popen([sys.executable, "-c", '
            '"import time; time.sleep(60)"], start_new_session=True)\n'
            'print(w.pid, flush=True)\n'
            'sys.exit(0)\n')
        token = 'test-race-token'
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  env={**os.environ, win.OWNER_ENV: token},
                                  start_new_session=True, text=True)
        worker = None
        try:
            worker = int(parent.stdout.readline().strip())
            parent.wait(timeout=10)
            root = Path(tempfile.mkdtemp(prefix='grouped-mmar-registry-race-'))
            registry = win.DescendantRegistry(parent.pid, root / 'descendants.json', set(),
                                              token=token).start()
            try:
                registry.stop()  # synchronous final record, no wait for the poller
                self.assertTrue(any(entry['pid'] == worker for entry in registry.snapshot()))
                win.cleanup_attributed(parent.pid, None, registry, set(), reap=True)
                self.assertFalse(Path(f'/proc/{worker}').exists())
            finally:
                registry.stop()
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if worker is not None:
                try:
                    os.killpg(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_adopted_intermediate_and_child_are_recorded(self):
        self.assertTrue(win.enable_subreaper())
        intermediate = ('import subprocess, sys, time\n'
                        'subprocess.Popen([sys.executable, "-c", '
                        '"import time; time.sleep(60)"], start_new_session=True)\n'
                        'time.sleep(60)\n')
        parent_code = ('import subprocess, sys\n'
                       f'p = subprocess.Popen([sys.executable, "-c", {intermediate!r}], '
                       'start_new_session=True)\n'
                       'print(p.pid, flush=True)\n'
                       'sys.exit(0)\n')
        token = 'test-chain-token'
        parent = subprocess.Popen([sys.executable, '-c', parent_code],
                                  stdout=subprocess.PIPE,
                                  env={**os.environ, win.OWNER_ENV: token},
                                  start_new_session=True, text=True)
        root = Path(tempfile.mkdtemp(prefix='grouped-mmar-registry-chain-'))
        registry = None
        try:
            intermediate_pid = int(parent.stdout.readline().strip())
            parent.wait(timeout=10)
            deadline = time.time() + 5
            grandchild = None
            while time.time() < deadline and grandchild is None:
                for row in win._scan_direct().values():
                    if row['ppid'] == intermediate_pid:
                        grandchild = row['pid']
                        break
                if grandchild is None:
                    time.sleep(0.05)
            self.assertIsNotNone(grandchild)
            registry = win.DescendantRegistry(parent.pid, root / 'descendants.json', set(),
                                              token=token).start()
            registry.stop()
            snapshot = registry.snapshot()
            self.assertGreaterEqual(len(snapshot), 2)
            win.cleanup_attributed(parent.pid, None, registry, set(), reap=True)
            self.assertEqual(win.attributed_survivors(registry), [])
            for entry in snapshot:
                self.assertFalse(Path(f"/proc/{entry['pid']}").exists())
        finally:
            if registry is not None:
                registry.stop()
                for entry in registry.snapshot():
                    try:
                        os.killpg(entry['pid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)

    def test_daemon_reparents_to_window_only_while_subreaper(self):
        self.assertTrue(win.enable_subreaper())
        code = (
            'import os, sys, time\n'
            'if os.fork() == 0:\n'
            '    os.setsid()\n'
            '    if os.fork() == 0:\n'
            '        print(os.getpid(), flush=True)\n'
            '        time.sleep(60)\n'
            '        os._exit(0)\n'
            '    os._exit(0)\n'
            'sys.exit(0)\n')

        def spawn():
            proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                    text=True, start_new_session=True)
            pid = int(proc.stdout.readline().strip())
            proc.wait(timeout=10)
            return pid

        def kill_exact(pid):
            # The printed pid is the grandchild; its PGID is the setsid child's
            # pid, so killpg(pid) can miss it. Kill only this exact pid.
            live = win._read_proc(pid)
            if live is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            win._reap([{'pid': pid}])

        first = spawn()
        try:
            deadline = time.time() + 5
            live = None
            while time.time() < deadline:
                live = win._read_proc(first)
                if live:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(live)
            self.assertEqual(live['ppid'], os.getpid())
        finally:
            kill_exact(first)

        self.assertTrue(win.disable_subreaper())
        second = spawn()
        try:
            deadline = time.time() + 5
            live = None
            while time.time() < deadline:
                live = win._read_proc(second)
                if live and live['ppid'] != os.getpid():
                    break
                time.sleep(0.05)
            self.assertIsNotNone(live)
            self.assertNotEqual(live['ppid'], os.getpid())
        finally:
            kill_exact(second)
            win.enable_subreaper()

    def test_fake_restore_reaches_pid1_after_disable(self):
        self.assertTrue(win.enable_subreaper())
        root = Path(tempfile.mkdtemp(prefix='grouped-mmar-fake-restore-'))
        run_dir = root / 'run'
        run_dir.mkdir()
        restore_output = root / 'restore'
        restore_output.mkdir()
        manifest = restore_output / 'launch-manifest.json'
        script = root / 'fake_restore.py'
        script.write_text(
            'import json, os, sys, time\n'
            'manifest = sys.argv[1]\n'
            'if os.fork() == 0:\n'
            '    os.setsid()\n'
            '    with open(manifest, "w") as handle:\n'
            '        json.dump({"pid": os.getpid()}, handle)\n'
            '    time.sleep(60)\n'
            '    os._exit(0)\n'
            'os._exit(0)\n')
        plan = {'restore_argv': [sys.executable, str(script), str(manifest)],
                'restore_output': str(restore_output)}
        result = {}
        calls = []
        original_start_restore = win.start_restore

        def counted(*args, **kwargs):
            calls.append(1)
            return original_start_restore(*args, **kwargs)

        pid = None
        original_subreaper = None
        try:
            self.assertTrue(win.set_subreaper(True))
            original_subreaper = win.get_subreaper()
            self.assertEqual(original_subreaper, 1)
            with mock.patch.object(win, 'start_restore', side_effect=counted):
                self.assertTrue(win.begin_restore(plan, run_dir, result))
            self.assertTrue(result['subreaper_cleared_for_restore'])
            self.assertEqual(win.get_subreaper(), 0)
            self.assertEqual(len(calls), 1)
            deadline = time.time() + 5
            while time.time() < deadline and not manifest.is_file():
                time.sleep(0.05)
            self.assertTrue(manifest.is_file())
            pid = int(json.loads(manifest.read_text())['pid'])
            live = win.wait_restored_root(pid, timeout=5, interval=0.05)
            self.assertIsNotNone(live)
            self.assertEqual(live['ppid'], 1)
            self.assertEqual(live['pgid'], pid)
            self.assertTrue(win.restore_subreaper(original_subreaper))
            self.assertEqual(win.get_subreaper(), 1)
        finally:
            if pid is None and manifest.is_file():
                try:
                    pid = int(json.loads(manifest.read_text())['pid'])
                except (OSError, ValueError, KeyError):
                    pid = None
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if original_subreaper is not None:
                win.restore_subreaper(original_subreaper)
            else:
                win.enable_subreaper()

    def test_protected_identity_is_not_signaled(self):
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        try:
            live = win._read_proc(sleeper.pid)
            entry = {key: live[key] for key in ('pid', 'starttime', 'boot_id', 'pgid', 'sid')}
            outcome = win.cleanup_attributed(0, entry, None, {sleeper.pid})
            self.assertFalse(outcome['killed'])
            self.assertEqual(outcome['reason'], 'protected_overlap')
            self.assertTrue(win.cleanup_blocks_restore(outcome))
            self.assertIsNotNone(win._read_proc(sleeper.pid))
        finally:
            sleeper.kill()
            sleeper.wait(timeout=5)

    def test_drift_identity_is_not_signaled(self):
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        try:
            live = win._read_proc(sleeper.pid)
            entry = {key: live[key] for key in ('pid', 'boot_id', 'pgid', 'sid')}
            entry['starttime'] = '0'  # wrong identity
            outcome = win.cleanup_attributed(0, entry, None, set())
            self.assertTrue(outcome['drift'])
            self.assertNotIn(sleeper.pid, outcome['singles'])
            self.assertNotIn(sleeper.pid, outcome['groups'])
            self.assertIsNotNone(win._read_proc(sleeper.pid))
        finally:
            sleeper.kill()
            sleeper.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
