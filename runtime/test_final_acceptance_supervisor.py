"""CPU-only Linux lifecycle tests; no model import, service or NPU access."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import final_acceptance_supervisor as subject


@unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux /proc required')
class LifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['CLEF_TEST_CLEANUP_HELPER'])
        assert subject.sha(path) == subject.HELPER_SHA
        cls.helper = subject.load_cleanup_helper(path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-client-cpu-')
        self.records = Path(self.tmp.name) / 'records'
        self.spawned = []
        self.real_popen = subprocess.Popen

    def tearDown(self):
        # Only test-created, still unreaped leaders can be signalled here.
        for proc in self.spawned:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        self.tmp.cleanup()

    def run_client(self, code, timeout=3, protected=()):
        def spawn(*args, **kwargs):
            proc = self.real_popen(*args, **kwargs)
            self.spawned.append(proc)
            return proc
        with patch.object(subject.subprocess, 'Popen', side_effect=spawn):
            return subject.supervise([sys.executable, '-c', code], timeout,
                                     self.records, self.helper, protected)

    def test_normal(self):
        result = self.run_client('print("cpu-only")')
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['clients_exited'])
        self.assertEqual(result['client_returncode'], 0)
        self.assertEqual((self.records / 'client.stdout').read_text(), 'cpu-only\n')

    def test_nonzero_preserved(self):
        result = self.run_client('raise SystemExit(7)')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['clients_exited'])
        self.assertEqual(result['client_returncode'], 7)

    def test_timeout_descendant(self):
        result = self.run_client('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
            'time.sleep(30)', timeout=1.2)
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['clients_exited'])
        self.assertGreaterEqual(len(result['observed']), 2)
        self.assertEqual(result['cleanup']['unknown'], [])
        self.assertEqual(result['cleanup']['residual'], [])

    def test_orphan_observed_before_leader_exit(self):
        result = self.run_client('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
            'time.sleep(1.1)')
        self.assertTrue(result['clients_exited'])
        self.assertFalse(result['timed_out'])
        self.assertGreaterEqual(len(result['observed']), 2)

    def test_query_failure_blocks_cleanup_and_cutover(self):
        with patch.object(subject, 'scan', side_effect=PermissionError('synthetic proc query failure')):
            with patch.object(self.helper.Window, '_probe_cleanup') as cleanup:
                result = self.run_client('import time;time.sleep(30)')
        cleanup.assert_not_called()
        self.assertFalse(result['clients_exited'])
        self.assertIn('synthetic proc query failure', result['errors'][0])

    def test_changed_identity_blocks_signal(self):
        real_scan = subject.scan
        calls = 0
        def changing():
            nonlocal calls
            table = real_scan()
            calls += 1
            if calls > 1:
                table[self.spawned[-1].pid]['starttime'] = 'changed'
            return table
        with patch.object(subject, 'scan', side_effect=changing):
            with patch.object(self.helper.Window, '_probe_cleanup') as cleanup:
                result = self.run_client('import time;time.sleep(30)')
        cleanup.assert_not_called()
        self.assertFalse(result['clients_exited'])
        self.assertIn('identity changed', result['errors'][0])

    def test_service_group_overlap_blocks_signal(self):
        real_scan = subject.scan
        def overlapping():
            table = real_scan()
            own = dict(table[os.getpid()])
            own['pgid'] = self.spawned[-1].pid
            table[os.getpid()] = own
            return table
        with patch.object(subject, 'scan', side_effect=overlapping):
            with patch.object(self.helper.Window, '_probe_cleanup') as cleanup:
                result = self.run_client('import time;time.sleep(30)', protected={os.getpid()})
        cleanup.assert_not_called()
        self.assertFalse(result['clients_exited'])
        self.assertIn('service pid found', result['errors'][0])

    def test_cleanup_query_failure_preserved(self):
        with patch.object(self.helper, 'proc_identity', side_effect=PermissionError('cleanup query unavailable')):
            result = self.run_client('print("done")')
        self.assertFalse(result['clients_exited'])
        self.assertIn('cleanup query unavailable', result['errors'][0])

    def test_signal_identity_includes_pgid_and_cmdline(self):
        identity = dict(pid=1, pgid=1, starttime='123', boot_id='boot', cmdline='python fixed.py')
        self.assertTrue(subject.same_signal_target(identity, dict(identity)))
        self.assertFalse(subject.same_signal_target(identity, dict(identity, pgid=2)))
        self.assertFalse(subject.same_signal_target(identity, dict(identity, cmdline='python other.py')))

    def test_preparation_delay_uses_absolute_deadline(self):
        deadline = time.monotonic() + 0.05
        time.sleep(0.08)
        with patch.object(subject.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'deadline already elapsed'):
                subject.supervise([sys.executable, '-c', 'print(1)'], 1,
                    self.records, self.helper, deadline_monotonic=deadline)
        launch.assert_not_called()

    def test_launch_time_counts_toward_deadline(self):
        def slow_spawn(*args, **kwargs):
            time.sleep(0.25)
            proc = self.real_popen(*args, **kwargs)
            self.spawned.append(proc)
            return proc
        deadline = time.monotonic() + 0.1
        with patch.object(subject.subprocess, 'Popen', side_effect=slow_spawn):
            result = subject.supervise([sys.executable, '-c', 'import time;time.sleep(30)'],
                30, self.records, self.helper, deadline_monotonic=deadline)
        self.assertEqual(result['deadline_monotonic'], deadline)
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['clients_exited'])


if __name__ == '__main__':
    unittest.main()
