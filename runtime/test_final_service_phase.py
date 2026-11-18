"""CPU-only fault injection; every service/stop/NPU command is mocked."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import final_service_phase as phase


@unittest.skipUnless(Path('/proc/self/cmdline').exists(), 'Linux fixture argv required')
class SwitchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-phase-cpu-')
        self.root = Path(self.tmp.name)
        self.log = self.root / 'labels';self.log.mkdir()
        old_label = self.log / 'old';old_label.mkdir()
        self.pid = os.getpid();self.newpid = self.pid + 1000000
        (old_label / 'api.pid').write_text(str(self.pid))
        self.old = dict(pid=self.pid, ppid=1, pgid=self.pid, starttime='1', boot_id='cpu-fixture', state='S', cmdline='fixture')
        self.new = dict(self.old, pid=self.newpid, pgid=self.newpid, starttime='2')
        target = self.root / 'clef_joint_schema.py';target.write_text('frozen-head')
        guard = self.root / 'guard.py';guard.write_text('frozen-guard')
        code = self.root / 'code';(code / 'runtime').mkdir(parents=True)
        runner = code / 'runtime/final_acceptance.py';runner.write_text('frozen-runner')
        argv = Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        manifest = self.root / 'manifest.json';manifest.write_text(json.dumps({'phases':{'s1':{'expected_argv':argv}}}))
        restore = self.root / 'old-restore.json'
        restore.write_text(json.dumps({'label':'old', 'service_discovery':{'api_identity':self.old},
            **{k:{'rc':0,'ok':True} for k in ('health','workers','text','image')}}))
        self.args = argparse.Namespace(code=code, manifest=manifest, manifest_sha=phase.sha(manifest),
            current_restore=restore, current_restore_sha=phase.sha(restore), run_dir=self.root/'run',
            runner_sha=phase.sha(runner), phase='s1', from_phase='s1', prior_client_record=None,
            first_phase_no_client=True, execute=True)
        self.calls = []
        self.stop_fail = False
        def command(out, name, argv, timeout, env=None):
            self.calls.append((name, argv, env))
            if name == 'stop-command' and self.stop_fail:
                raise RuntimeError('synthetic stop nonzero')
            if name == 'launch':
                label_dir = self.log / argv[2];label_dir.mkdir()
                (label_dir / 'api.pid').write_text(str(self.newpid))
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        def load(path, name):
            if name == 'phase_acceptance':
                return types.SimpleNamespace(frozen_check=lambda m:None, service_check=lambda *a:{'ok':True})
            return types.SimpleNamespace(D0_MODULES={})
        def verify(cmd, log, **kw):
            log.update({k:{'rc':0,'ok':True} for k in ('health','workers','text','image')})
            return True
        self.patches = [
            patch.object(phase, 'TARGET', target), patch.object(phase, 'GUARD', guard),
            patch.object(phase, 'LOG_ROOT', self.log),
            patch.object(phase, 'HASHES', {str(target):phase.sha(target),str(guard):phase.sha(guard)}),
            patch.object(phase, 'command', side_effect=command), patch.object(phase, 'resource', return_value=[]),
            patch.object(phase, 'load', side_effect=load), patch.object(phase, 'no_pending_clients'),
            patch.object(phase.strict, 'read_identity', return_value=self.new),
            patch.object(phase.strict, 'scan', side_effect=[{self.pid:self.old},{},{self.newpid:self.new}]),
            patch.dict(sys.modules, {'d0_restore':types.SimpleNamespace(verify=verify,build_commands=lambda fn:{}),
                                    'd0_run':types.SimpleNamespace(render_request=None)}),
        ]
        for item in self.patches:item.start()

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.tmp.cleanup()

    def test_plan_only_no_commands_or_output_dir(self):
        self.args.execute = False
        self.assertEqual(phase.switch(self.args)['status'], 'plan_not_executed')
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.run_dir.exists())

    def test_success_has_exact_clean_parent_and_stop(self):
        with patch.object(phase, 'port', side_effect=[0,111,111]):
            result = phase.switch(self.args)
        self.assertTrue(result['ok'])
        self.assertEqual([r[0] for r in self.calls], ['stop-command','launch','wait'])
        self.assertEqual(self.calls[1][2], phase.CLEAN_ENV)
        self.assertEqual(set(self.calls[1][2]), {'PATH','CLEF_LAUNCH_CONFIG'})
        self.assertEqual(self.calls[0][1][1], phase.STOP)

    def test_fresh_open_port_blocks_new_launch(self):
        with patch.object(phase, 'port', side_effect=[0,111,0]):
            with self.assertRaisesRegex(RuntimeError, 'fresh port not closed'):
                phase.switch(self.args)
        self.assertEqual([r[0] for r in self.calls], ['stop-command'])

    def test_unknown_port_blocks_new_launch(self):
        with patch.object(phase, 'port', side_effect=[0,111,RuntimeError('unknown port connect_ex=5')]):
            with self.assertRaisesRegex(RuntimeError, 'unknown port'):
                phase.switch(self.args)
        self.assertEqual([r[0] for r in self.calls], ['stop-command'])

    def test_stop_failure_blocks_new_launch(self):
        self.stop_fail = True
        with patch.object(phase, 'port', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'stop nonzero'):
                phase.switch(self.args)
        self.assertEqual([r[0] for r in self.calls], ['stop-command'])

    def test_changed_source_rejected_before_commands(self):
        phase.TARGET.write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'frozen file mismatch'):
            phase.switch(self.args)
        self.assertEqual(self.calls, [])

    def test_before_stop_orphan_same_group_blocks_all_commands(self):
        orphan = dict(self.old, pid=self.pid+1, ppid=1)
        with patch.object(phase.strict, 'scan', return_value={self.pid:self.old,orphan['pid']:orphan}):
            with self.assertRaisesRegex(RuntimeError, 'unattributed old PGID'):
                phase.switch(self.args)
        self.assertEqual(self.calls, [])

    def test_after_stop_orphan_same_group_blocks_launch(self):
        orphan = dict(self.old, pid=self.pid+1, ppid=1)
        with patch.object(phase.strict, 'scan', side_effect=[{self.pid:self.old},{orphan['pid']:orphan}]):
            with patch.object(phase, 'port', return_value=0):
                with self.assertRaisesRegex(RuntimeError, 'old PGID member remains'):
                    phase.switch(self.args)
        self.assertEqual([r[0] for r in self.calls], ['stop-command'])

    def test_missing_client_mode_rejected(self):
        self.args.first_phase_no_client = False
        with self.assertRaisesRegex(RuntimeError, 'select prior client'):
            phase.switch(self.args)
        self.assertEqual(self.calls, [])

    def test_first_mode_other_phase_rejected(self):
        self.args.phase = 'erqa'
        with self.assertRaisesRegex(RuntimeError, 'requires s1 to s1'):
            phase.switch(self.args)
        self.assertEqual(self.calls, [])

    def test_client_modes_mutually_exclusive(self):
        self.args.prior_client_record = self.root / 'client.json'
        with self.assertRaisesRegex(RuntimeError, 'select prior client'):
            phase.switch(self.args)
        self.assertEqual(self.calls, [])


class ResourceParsingTest(unittest.TestCase):
    def test_exact_card_chip_pid_columns(self):
        text = 'Process id in container\n| 7 0 | 123 | VLLMWorker_TP | 999 | 123 |\n| 7 1 | 124 | VLLMWorker_TP | 888 | 124 |'
        self.assertEqual([r['pid'] for r in phase.npu_rows(text)], [123,124])

    def test_query_unknown_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):phase.npu_rows('query failed')

    def test_other_card_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'outside assigned'):
            phase.npu_rows('Process id in container\n| 6 0 | 123 | python | 1 | 123 |')


class ClientBoundaryTest(unittest.TestCase):
    def check(self, record, table):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'client.json';p.write_text(json.dumps(record))
            phase.no_pending_clients(table, p)

    def test_unknown_client_exit_blocks(self):
        with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
            self.check({'clients_exited':False}, {})

    def test_escaped_observed_client_blocks(self):
        with self.assertRaisesRegex(RuntimeError, 'observed client PID remains'):
            self.check({'clients_exited':True,'client_pid':111,'observed':{'222':{}}},
                       {222:{'pgid':999,'state':'S'}})

    def test_confirmed_gone_client_passes(self):
        self.check({'clients_exited':True,'client_pid':111,'observed':{'222':{}}}, {})


if __name__ == '__main__':unittest.main()
