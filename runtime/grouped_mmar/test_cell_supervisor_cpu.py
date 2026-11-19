"""CPU-only tests for the source-aware supervision slice."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

import cell_supervisor as sup


class CellSupervisorTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='grouped-mmar-cell-'))
        self.source = self.root / 'clef_joint_schema.py'
        self.source.write_text('aa3e source fixture\n')
        self.sha = sup.sha256(self.source)
        self.out = self.root / 'cell'

    def cmd(self, *body, execute=True, timeout=5.0, extra=()):
        argv = ['--source-target', str(self.source), '--expected-source-sha', self.sha,
                '--run-dir', str(self.out), '--timeout', str(timeout), *extra]
        if execute:
            argv.append('--execute')
        argv.append('--')
        argv += list(body)
        return argv

    def test_plan_only_does_not_run(self):
        argv = self.cmd(sys.executable, '-c', 'print(1)', execute=False)
        self.assertEqual(sup.main(argv), 0)
        self.assertFalse(self.out.exists())

    def test_source_mismatch_blocks_before_child(self):
        argv = ['--source-target', str(self.source), '--expected-source-sha', '0' * 64,
                '--run-dir', str(self.out), '--', sys.executable, '-c', 'print(1)']
        with self.assertRaises(ValueError):
            sup.main(argv)

    def test_success_records_source_and_child(self):
        code = sup.main(self.cmd(sys.executable, '-c', 'print("ok")'))
        self.assertEqual(code, 0)
        result = json.loads((self.out / 'cell.json').read_text())
        self.assertTrue(result['ok'])
        self.assertEqual(result['child_rc'], 0)
        self.assertFalse(result['timed_out'])
        self.assertEqual(result['source_sha256_before'], self.sha)
        self.assertEqual(result['source_sha256_after'], self.sha)
        self.assertEqual(result['survivors'], [])

    def test_timeout_kills_group_and_keeps_source(self):
        code = sup.main(self.cmd(sys.executable, '-c', 'import time; time.sleep(10)',
                                 timeout=0.2, execute=True))
        self.assertEqual(code, 1)
        result = json.loads((self.out / 'cell.json').read_text())
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['ok'])
        self.assertEqual(result['survivors'], [])
        self.assertEqual(result['source_sha256_after'], self.sha)

    def test_protected_pid_is_never_signalled(self):
        calls = []
        members = {10: {'pid': 10}, 11: {'pid': 11}}
        outcome = sup.cleanup_group(10, members, {11}, kill=lambda *a: calls.append(a), grace=0)
        self.assertFalse(outcome['killed'])
        self.assertEqual(outcome['reason'], sup.PROTECTED_ERROR)
        self.assertEqual(calls, [])

    def test_cleanup_kills_when_no_protected_member(self):
        calls = []
        members = {10: {'pid': 10}}
        outcome = sup.cleanup_group(10, members, {99}, kill=lambda *a: calls.append(a), grace=0)
        self.assertTrue(outcome['killed'])
        self.assertEqual(calls[0], (10, signal.SIGTERM))

    def test_restore_and_post_check_run_after_child(self):
        restore = json.dumps([sys.executable, '-c', 'print("restore")'])
        post = json.dumps([sys.executable, '-c', 'print("check")'])
        code = sup.main(self.cmd(sys.executable, '-c', 'print("probe")',
                                 extra=('--restore-argv', restore, '--post-check-argv', post)))
        self.assertEqual(code, 0)
        result = json.loads((self.out / 'cell.json').read_text())
        self.assertEqual(result['restore']['rc'], 0)
        self.assertEqual(result['post_check']['rc'], 0)

    def test_failed_restore_fails_cell(self):
        restore = json.dumps([sys.executable, '-c', 'import sys; sys.exit(3)'])
        code = sup.main(self.cmd(sys.executable, '-c', 'print("probe")',
                                 extra=('--restore-argv', restore)))
        self.assertEqual(code, 1)
        result = json.loads((self.out / 'cell.json').read_text())
        self.assertFalse(result['ok'])
        self.assertEqual(result['restore']['rc'], 3)


if __name__ == '__main__':
    unittest.main()
