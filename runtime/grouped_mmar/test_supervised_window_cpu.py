"""CPU-only control-flow tests for the full supervised probe window.

External boundaries (d0_stop, official start.sh, HTTP, /proc, npu-smi) are
fakes; the window's own ordering, source checks, branch gate and cleanup are
real.  No torch/NPU/service is touched.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import supervised_probe_window as win


class FakeProcess:
    def __init__(self, pid=5252):
        self.pid = pid
        self.returncode = None
        self.timed_out = False

    def wait(self, timeout=None):
        if self.timed_out and timeout is not None and timeout == 240.0:
            self.timed_out = False
            raise subprocess.TimeoutExpired('child', timeout)
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode


def table(pid, extra=()):
    rows = {pid: {'pid': pid, 'ppid': 1, 'pgid': pid, 'state': 'S',
                  'starttime': '1', 'boot_id': 'b', 'cmdline': ['vllm']}}
    for child in extra:
        rows[child] = {'pid': child, 'ppid': pid, 'pgid': pid, 'state': 'S',
                       'starttime': '2', 'boot_id': 'b', 'cmdline': ['worker']}
    return rows


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
        self.restore_output.mkdir()
        (self.restore_output / 'launch-manifest.json').write_text(json.dumps({'pid': 6262}))
        (self.restore_output / 'joint-patch-record.json').write_text(
            json.dumps({'branch': 'aa3e-noop'}))
        self.run_dir = self.root / 'run'

    def args(self, execute=False, extra=()):
        return ['--run-dir', str(self.run_dir), '--root-identity', str(self.identity),
                '--child-argv', json.dumps([sys.executable, '-c', 'print(1)']),
                '--source-target', str(self.source), '--expected-source-sha', self.sha,
                '--official-start', str(self.root / 'start.sh'),
                '--activate', str(self.root / 'activate.sh'),
                '--service-root', str(self.root), '--model-dir', str(self.root / 'model'),
                '--restore-output', str(self.restore_output),
                '--text-body', str(self.text), '--image-body', str(self.image),
                '--stop-script', str(self.root / 'stop.py'), *extra] + (
                    ['--execute'] if execute else [])

    def test_plan_only_validates_and_prints(self):
        self.assertEqual(win.main(self.args()), 0)
        self.assertFalse(self.run_dir.exists())

    def test_missing_probe_body_rejected(self):
        argv = self.args()
        argv[argv.index('--text-body') + 1] = str(self.root / 'missing.json')
        with self.assertRaises(ValueError):
            win.main(argv)

    def full_success(self, branch='aa3e-noop'):
        (self.restore_output / 'joint-patch-record.json').write_text(
            json.dumps({'branch': branch}))
        stopped = {'value': False}

        def fake_run(argv, timeout, log):
            if 'stop.py' in ' '.join(argv):
                stopped['value'] = True
            return {'argv': argv, 'rc': 0, 'log': str(log)}

        def fake_scan():
            if not stopped['value']:
                return table(4242, (4243,))
            return table(6262, (6263,))

        with mock.patch.object(win, 'scan', side_effect=fake_scan), \
                mock.patch.object(win, 'port_open', side_effect=[0, 111, 0]), \
                mock.patch.object(win, 'pending_clients', return_value=[]), \
                mock.patch.object(win, 'run', side_effect=fake_run), \
                mock.patch.object(win, 'npu_evidence', return_value=[]), \
                mock.patch.object(win, 'wait_health', return_value={'status': 'ready'}), \
                mock.patch.object(win, 'env_evidence', return_value={'match': True}), \
                mock.patch.object(win, 'read_root_env', return_value={}), \
                mock.patch.object(win, 'joint_env', return_value={'ok': True}), \
                mock.patch.object(win, 'http_json', return_value={'status': 'ready', 'config': 's1'}), \
                mock.patch.object(win, 'post_json', return_value=(200, {'ok': True})), \
                mock.patch.object(win, 'proc_identity', return_value={
                    'pid': 5252, 'ppid': 1, 'pgid': 5252, 'state': 'S',
                    'starttime': '1', 'boot_id': 'b', 'cmdline': ['child']}), \
                mock.patch.object(win.subprocess, 'Popen', return_value=FakeProcess()):
            return win.main(self.args(execute=True))

    def test_full_success_path(self):
        self.assertEqual(self.full_success(), 0)
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertTrue(result['ok'])
        self.assertEqual(result['restore_branch'], 'aa3e-noop')
        self.assertEqual(result['source_sha256_after_restore'], self.sha)
        self.assertEqual(result['steps']['stopped']['port'], 111)

    def test_f810_to_fused_marks_isolation_failure(self):
        self.assertEqual(self.full_success('f810->fused'), 1)
        result = json.loads((self.run_dir / 'window.json').read_text())
        self.assertTrue(result['isolation_failed'])
        self.assertFalse(result['ok'])

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


if __name__ == '__main__':
    unittest.main()
