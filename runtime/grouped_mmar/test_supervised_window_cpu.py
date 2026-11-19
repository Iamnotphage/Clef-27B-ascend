"""CPU-only control-flow tests for the full supervised probe window.

External boundaries (d0_stop, /proc, npu-smi, HTTP) are fakes except in the
real foreground fixture, which starts a temporary HTTP + manifest service and
verifies the window returns while that service is still alive.  The window's
recovery gate, source checks, branch gate and cleanup are real.
"""
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
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  start_new_session=True, text=True)
        worker = None
        try:
            worker = int(parent.stdout.readline().strip())
            parent.wait(timeout=10)
            root = Path(tempfile.mkdtemp(prefix='grouped-mmar-registry-'))
            registry = win.DescendantRegistry(parent.pid, root / 'descendants.json', set()).start()
            try:
                deadline = time.time() + 5
                snapshot = []
                while time.time() < deadline:
                    snapshot = registry.snapshot()
                    if any(entry['pid'] == worker for entry in snapshot):
                        break
                    time.sleep(0.1)
                self.assertTrue(any(entry['pid'] == worker for entry in snapshot))
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

    def test_protected_identity_is_not_signaled(self):
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        try:
            live = win._read_proc(sleeper.pid)
            entry = {key: live[key] for key in ('pid', 'starttime', 'boot_id', 'pgid', 'sid')}
            outcome = win.cleanup_attributed(0, entry, None, {sleeper.pid})
            self.assertFalse(outcome['killed'])
            self.assertEqual(outcome['reason'], 'protected_overlap')
            self.assertIsNotNone(win._read_proc(sleeper.pid))
        finally:
            sleeper.kill()
            sleeper.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
