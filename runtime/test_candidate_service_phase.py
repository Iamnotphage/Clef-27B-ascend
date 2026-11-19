"""CPU-only fault injection for the candidate deploy/restore entry.

Every service, stop, installer, /proc and NPU command is a fixture object; the
real module pins are exercised separately by build_deps on the device.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import candidate_service_phase as cand

F810 = cand.F810_HEAD
AA3E = cand.AA3E_HEAD
PRE = cand.GUARD_PRE
POST = cand.GUARD_POST


def row(pid, pgid, ppid=1, starttime='1', state='S'):
    return dict(pid=pid, ppid=ppid, pgid=pgid, starttime=starttime,
                boot_id='boot-fixture', state=state, cmdline='fixture')


class World:
    def __init__(self, log_root, target_path, guard_path):
        self.log_root = log_root
        self.target_path = target_path
        self.guard_path = guard_path
        self.head = F810
        self.guard = PRE
        self.port = 0
        self.table = {4242: row(4242, 4242), 4343: row(4343, 4242)}
        self.tree = {4242, 4343}
        self.env = {cand.FLAG_ENV: '1', cand.ALLOW_ENV: AA3E}
        self.pending = False
        self.fail_on = set()
        self.port_after_stop = 111
        self.events = []
        self.old_argv = ['old', 'argv']
        self.new_pid = 5252
        self.new_child = 5353


class FakeStrict:
    def __init__(self, world):
        self.world = world

    def scan(self):
        return dict(self.world.table)

    def read_identity(self, pid):
        return self.world.table.get(pid)

    def same_process(self, a, b):
        return bool(a and b and all(a[k] == b[k] for k in ('pid', 'pgid', 'starttime', 'boot_id')))


class FakePhase:
    PYTHON = '/fixture/python'
    STOP = '/fixture/stop.py'
    OVERRIDES = {'s1': [], 's2_b8': ['--max-num-seqs', '8'],
                 's2_b16': ['--max-num-seqs', '16'], 'erqa': ['--max-model-len', '16384']}

    def __init__(self, world):
        self.world = world
        self.TARGET = world.target_path
        self.GUARD = world.guard_path

    def tree(self, table, root):
        return set(self.world.tree)

    def no_pending_clients(self, table, prior=None):
        self.world.events.append(('clients', bool(prior)))
        if self.world.pending:
            raise RuntimeError('pending runner')

    def port(self):
        return self.world.port

    def resource(self, out, name, service_pids=None):
        self.world.events.append(('resource', name, service_pids))

    def command(self, out, name, argv, timeout, env=None):
        self.world.events.append((name, tuple(argv), None if env is None else tuple(sorted(env.items()))))
        if name in self.world.fail_on:
            raise RuntimeError(f'synthetic {name} failure')
        if name == 'stop-command':
            self.world.table = {}
            self.world.port = self.world.port_after_stop
        if name == 'launch':
            label = argv[2]
            label_dir = self.world.log_root / label
            label_dir.mkdir()
            (label_dir / 'api.pid').write_text(str(self.world.new_pid))
            (label_dir / 'launch-manifest.json').write_text('{}')
            self.world.table = {self.world.new_pid: row(self.world.new_pid, self.world.new_pid),
                                self.world.new_child: row(self.world.new_child, self.world.new_pid)}
            self.world.tree = {self.world.new_pid, self.world.new_child}
            self.world.port = 0
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')


class FakeAB:
    def __init__(self, world, run_dir, log_root):
        self.world = world
        self.run_dir = run_dir
        self.args = types.SimpleNamespace(service_log_root=str(log_root),
                                          launch='/fixture/launch.sh', wait='/fixture/wait.sh',
                                          clean_env_expect='/fixture/expect.json')

    def stage_d0(self):
        d0 = self.run_dir / 'stage' / 'd0'
        d0.mkdir(parents=True, exist_ok=True)
        (d0 / 'd0_restore.py').write_text(
            'def build_commands(render):\n    return {}\n'
            'def verify(cmds, log, runner=None, timeout=120, service_pids=None):\n    return True\n')
        (d0 / 'd0_run.py').write_text('def render_request(*a, **k):\n    return None\n')
        self.world.events.append(('stage_d0',))

    def stage_candidate(self):
        self.world.events.append(('stage_candidate',))

    def apply_service_check(self):
        self.world.events.append(('adapter_apply',))
        if 'adapter_apply' in self.world.fail_on:
            raise RuntimeError('synthetic adapter failure')
        self.world.guard = POST

    def revert_service_check(self):
        self.world.events.append(('adapter_revert',))
        if 'adapter_revert' in self.world.fail_on:
            raise RuntimeError('synthetic revert failure')
        self.world.guard = PRE

    def switch_source(self, target, tag):
        self.world.events.append(('switch', target, tag))
        if 'switch' in self.world.fail_on:
            self.world.head = 'unknown-head'
            raise RuntimeError('synthetic switch failure')
        if target in self.world.fail_on:
            raise RuntimeError('synthetic switch failure')
        self.world.head = F810 if target == 'f810' else AA3E

    def restore_prod(self):
        self.world.events.append(('restore_prod',))
        if 'restore_prod' in self.world.fail_on:
            raise RuntimeError('synthetic restore failure')
        self.world.head, self.world.guard, self.world.port = F810, PRE, 0
        label = 'restored-label'
        label_dir = self.world.log_root / label
        label_dir.mkdir(exist_ok=True)
        (label_dir / 'api.pid').write_text(str(self.world.new_pid))
        (label_dir / 'launch-manifest.json').write_text('{}')
        self.world.table = {self.world.new_pid: row(self.world.new_pid, self.world.new_pid),
                            self.world.new_child: row(self.world.new_child, self.world.new_pid)}
        self.world.tree = {self.world.new_pid, self.world.new_child}
        return {'label': label, 'verify_ok': True, 'service_pids': [self.world.new_pid, self.world.new_child],
                'restore_env': {'match': True}}


class FakeAcceptance:
    def __init__(self, world):
        self.world = world

    def frozen_check(self, manifest):
        self.world.events.append(('frozen', manifest.get('_tag')))

    def service_check(self, manifest, phase, identities, launch):
        self.world.events.append(('service_check', phase, manifest.get('_tag'), str(identities)))
        return {'ok': True}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-candidate-cpu-')
        self.root = Path(self.tmp.name)
        self.log_root = self.root / 'labels'
        self.log_root.mkdir()
        self.world = World(self.log_root, self.root / 'head.py', self.root / 'guard.py')
        self.world.target_path.write_text('fixture-head')
        self.world.guard_path.write_text('fixture-guard')
        self.deps = types.SimpleNamespace(
            strict=FakeStrict(self.world), phase=FakePhase(self.world), ab=None,
            acceptance=FakeAcceptance(self.world),
            manifest={'_tag': 'cand', 'phases': {p: {'expected_argv': ['old', 'argv']} for p in cand.PHASES}},
            base_manifest={'_tag': 'f810', 'phases': {p: {'expected_argv': ['old', 'argv']} for p in cand.PHASES}})
        self.patches = [
            patch.object(cand, 'sha', side_effect=self.fake_sha),
            patch.object(cand, 'proc_cmdline', side_effect=lambda pid: list(self.world.old_argv)),
            patch.object(cand, 'proc_environ', side_effect=lambda pid: dict(self.world.env)),
        ]
        for p in self.patches:
            p.start()
        self.deploy_run = self.root / 'deploy-run'
        self.deps.ab = FakeAB(self.world, self.deploy_run, self.log_root)
        self.old_restore = self.root / 'old-restore.json'
        self.old_restore.write_text(json.dumps({
            'label': 'old', 'service_discovery': {'api_identity': row(4242, 4242)},
            **{k: {'rc': 0, 'ok': True} for k in ('health', 'workers', 'text', 'image')}}))

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def fake_sha(self, path):
        text = str(path)
        if text == str(self.world.target_path):
            return self.world.head
        if text == str(self.world.guard_path):
            return self.world.guard
        return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()

    def deploy_args(self, execute=True, **kw):
        self.deps.ab = FakeAB(self.world, self.deploy_run, self.log_root)
        a = argparse.Namespace(
            action='deploy', run_dir=self.deploy_run, phase='s1', from_phase='s1',
            current_restore=self.old_restore, current_restore_sha=cand.sha(self.old_restore),
            deploy_record=None, deploy_record_sha=None, prior_client_record=None,
            first_phase_no_client=True, execute=execute, manifest_sha=cand.MANIFEST_SHA)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def restore_args(self, record, execute=True, **kw):
        self.deps.ab = FakeAB(self.world, self.root / 'restore-run', self.log_root)
        a = argparse.Namespace(
            action='restore', run_dir=self.root / 'restore-run', phase='s1', from_phase='s1',
            deploy_record=record, deploy_record_sha=cand.sha(record), prior_client_record=self.root / 'supervision.json',
            first_phase_no_client=False, execute=execute, manifest_sha=cand.MANIFEST_SHA)
        for k, v in kw.items():
            setattr(a, k, v)
        return a


class DeployTest(Base):
    def test_plan_only_does_not_mutate(self):
        result = cand.deploy(self.deploy_args(execute=False), self.deps)
        self.assertEqual(result['status'], 'plan_not_executed')
        self.assertFalse(self.deploy_run.exists())
        self.assertEqual(self.world.events, [])
        self.assertEqual(result['plan']['launch_env'][cand.FLAG_ENV], '1')

    def test_deploy_success_binds_candidate(self):
        result = cand.deploy(self.deploy_args(), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], AA3E)
        self.assertEqual(result['guard'], POST)
        self.assertEqual(self.world.head, AA3E)
        self.assertEqual(self.world.guard, POST)
        complete = json.loads((self.deploy_run / 'complete.json').read_text())
        self.assertEqual(complete['action'], 'deploy')
        names = [e[0] for e in self.world.events]
        self.assertLess(names.index('stop-command'), names.index('adapter_apply'))
        self.assertLess(names.index('adapter_apply'), names.index('switch'))
        self.assertIn(('switch', 'aa3e', 'deploy'), self.world.events)
        self.assertIn(('frozen', 'cand'), self.world.events)
        launch = next(e for e in self.world.events if e[0] == 'launch')
        env = dict(launch[2])
        self.assertEqual(env[cand.FLAG_ENV], '1')
        self.assertEqual(env[cand.ALLOW_ENV], AA3E)
        self.assertEqual(env['CLEF_LAUNCH_CONFIG'], 's1')
        self.assertNotIn('ASCEND_CUSTOM_OPP_PATH', env)

    def test_stop_failure_stops_before_install(self):
        self.world.fail_on.add('stop-command')
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('adapter_apply', names)
        self.assertNotIn('launch', names)
        self.assertFalse((self.deploy_run / 'recovery.json').exists())

    def test_launch_failure_recovers_to_f810(self):
        self.world.fail_on.add('launch')
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'restored_prod')
        self.assertEqual(self.world.head, F810)
        self.assertEqual(self.world.guard, PRE)
        self.assertIn(('restore_prod',), self.world.events)

    def test_launch_failure_with_open_port_blocks(self):
        self.world.fail_on.add('launch')
        self.world.port_after_stop = 0
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'blocked_service_or_source_unknown')
        self.assertNotIn(('restore_prod',), self.world.events)

    def test_unknown_source_after_stop_blocks(self):
        self.world.fail_on.add('switch')
        self.world.head = 'unknown-head'
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'blocked_service_or_source_unknown')
        self.assertNotIn(('restore_prod',), self.world.events)

    def test_deploy_erqa_uses_frozen_overrides(self):
        cand.deploy(self.deploy_args(phase='erqa'), self.deps)
        launch = next(e for e in self.world.events if e[0] == 'launch')
        self.assertIn('--max-model-len', launch[1])
        self.assertEqual(launch[1][launch[1].index('--max-model-len') + 1], '16384')

    def test_incomplete_old_restore_blocks_before_stop(self):
        doc = json.loads(self.old_restore.read_text())
        doc['health']['ok'] = False
        self.old_restore.write_text(json.dumps(doc))
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        self.assertEqual(self.world.events, [])

    def test_unattributed_pgid_member_blocks(self):
        self.world.table[9999] = row(9999, 4242)
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('stop-command', names)

    def test_pending_client_blocks(self):
        self.world.pending = True
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('stop-command', names)


class RestoreTest(Base):
    def make_record(self):
        ids = self.root / 'ready.json'
        ids.write_text(json.dumps({'root': row(4242, 4242), 'descendants': [row(4343, 4242)]}))
        record = self.root / 'deploy-complete.json'
        record.write_text(json.dumps({'ok': True, 'action': 'deploy', 'phase': 's1',
                                      'source': AA3E, 'guard': POST, 'identities': str(ids)}))
        return record

    def test_restore_success_returns_production(self):
        record = self.make_record()
        self.world.head, self.world.guard = AA3E, POST
        result = cand.restore(self.restore_args(record), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], F810)
        self.assertEqual(result['guard'], PRE)
        names = [e[0] for e in self.world.events]
        self.assertLess(names.index('stop-command'), names.index('adapter_revert'))
        self.assertLess(names.index('adapter_revert'), names.index('switch'))
        self.assertIn(('switch', 'f810', 'restore'), self.world.events)
        self.assertIn(('frozen', 'f810'), self.world.events)
        self.assertIn(('restore_prod',), self.world.events)
        expected_ids = str((self.root / 'restore-run/restored-identities.json').resolve())
        self.assertIn(('service_check', 's1', 'f810', expected_ids), self.world.events)

    def test_restore_rejects_wrong_record(self):
        record = self.make_record()
        doc = json.loads(record.read_text())
        doc['action'] = 'restore'
        record.write_text(json.dumps(doc))
        with self.assertRaises(RuntimeError):
            cand.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_rejects_phase_mismatch(self):
        record = self.make_record()
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            cand.restore(self.restore_args(record, from_phase='erqa'), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_stop_failure_does_not_launch(self):
        record = self.make_record()
        self.world.head, self.world.guard = AA3E, POST
        self.world.fail_on.add('stop-command')
        with self.assertRaises(RuntimeError):
            cand.restore(self.restore_args(record), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('adapter_revert', names)
        self.assertNotIn('restore_prod', names)

    def test_restore_adapter_failure_recovers_without_launch_when_unknown(self):
        record = self.make_record()
        self.world.head, self.world.guard = AA3E, POST
        self.world.fail_on.add('adapter_revert')
        with self.assertRaises(RuntimeError):
            cand.restore(self.restore_args(record), self.deps)
        recovery = json.loads((self.root / 'restore-run/recovery.json').read_text())
        self.assertEqual(recovery['action'], 'restore_failed')
        self.assertNotIn(('restore_prod',), self.world.events)


class CliTest(Base):
    def test_restore_rejects_first_phase_mode(self):
        dummy = self.root / 'dummy-record.json'
        dummy.write_text('{}')
        a = self.restore_args(dummy)
        a.first_phase_no_client = True
        a.prior_client_record = None
        with patch.object(cand, 'build_deps', side_effect=AssertionError('must not build')):
            with self.assertRaises(RuntimeError):
                cand.main(['--action', 'restore', '--code', str(self.root), '--owner-code', str(self.root),
                           '--ab-code', str(self.root), '--repo', str(self.root), '--manifest', str(self.root / 'm.json'),
                           '--base-manifest', str(self.root / 'b.json'), '--run-dir', str(self.root / 'r'),
                           '--deploy-record', str(self.root / 'd.json'), '--deploy-record-sha', 'x',
                           '--first-phase-no-client'])

    def test_deploy_requires_current_restore(self):
        with patch.object(cand, 'build_deps', side_effect=AssertionError('must not build')):
            with self.assertRaises(RuntimeError):
                cand.main(['--action', 'deploy', '--code', str(self.root), '--owner-code', str(self.root),
                           '--ab-code', str(self.root), '--repo', str(self.root), '--manifest', str(self.root / 'm.json'),
                           '--base-manifest', str(self.root / 'b.json'), '--run-dir', str(self.root / 'r'),
                           '--first-phase-no-client'])


if __name__ == '__main__':
    unittest.main()
