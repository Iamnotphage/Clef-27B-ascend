"""CPU-only tests for the typed S2/ERQA phase variant.

The frozen a1ca entry and its test stay byte-identical; this file only covers
the added typed gates of `candidate_service_phase_typed.py`.  Every service,
stop, launch, installer, /proc and NPU command is a fixture object bound to a
mkdtemp tree/guard/service-log-root.  Real frozen files (manifest, phase table,
a1ca entry) are read from this directory; real record copies, when 32 provides
them, are read read-only from `CLEF_REAL_RECORDS` and checked by SHA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import candidate_service_phase_typed as typed

F810 = typed.F810_HEAD
AA3E = typed.AA3E_HEAD
PRE = typed.GUARD_PRE
POST = typed.GUARD_POST
HERE = Path(__file__).resolve().parent


def row(pid, pgid, ppid=1, starttime='1', state='S'):
    return dict(pid=pid, ppid=ppid, pgid=pgid, starttime=starttime,
                boot_id='boot-fixture', state=state, cmdline='fixture')


def manifest_doc(candidate):
    phases = {}
    for phase in typed.PHASES:
        env = {'CLEF_MODEL_NAME': 'clef', 'CLEF_CONFIG': 's1'}
        if candidate:
            env[typed.FLAG_ENV] = '1'
            env[typed.ALLOW_ENV] = AA3E
        phases[phase] = {
            'expected_argv': ['old', 'argv'], 'effective': {'max_num_seqs': 1},
            'config': 's1', 'concurrencies': [1], 'rounds': 1,
            'launch_overrides': list(FakePhase.OVERRIDES[phase]),
            'service_environment': env}
    forbidden = ['CLEF_JOINT_SMALLQ'] if candidate else [typed.FLAG_ENV, 'CLEF_JOINT_SMALLQ']
    return {'_tag': 'cand' if candidate else 'f810', 'phases': phases,
            'forbidden_environment': forbidden}


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
        self.env = {typed.FLAG_ENV: '1', typed.ALLOW_ENV: AA3E}
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
    OVERRIDES = {'s1': [],
                 's2_b8': ['--max-num-seqs', '8', '--max-num-batched-tokens', '8192'],
                 's2_b16': ['--max-num-seqs', '16', '--max-num-batched-tokens', '16384'],
                 'erqa': ['--max-model-len', '16384', '--max-num-batched-tokens', '16384',
                          '--limit-mm-per-prompt', '{"image":16,"video":0}']}

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
        self.run_dir = Path(run_dir)
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
        record = {'label': label,
                  'service_discovery': {'api_identity': row(self.world.new_pid, self.world.new_pid),
                                        'pids': sorted(self.world.tree)},
                  **{k: {'rc': 0, 'ok': True} for k in typed.FOUR_CHECKS}}
        if 'incomplete_restore_record' in self.world.fail_on:
            record['health'] = {'rc': 1, 'ok': False}
        (self.run_dir / 'restore.json').write_text(json.dumps(record))
        return {'label': label, 'verify_ok': True,
                'service_pids': [self.world.new_pid, self.world.new_child],
                'restore_env': {'match': 'restore_env_mismatch' not in self.world.fail_on}}


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
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-typed-cpu-')
        self.root = Path(self.tmp.name)
        self.log_root = self.root / 'labels'
        self.log_root.mkdir()
        self.world = World(self.log_root, self.root / 'head.py', self.root / 'guard.py')
        self.world.target_path.write_text('fixture-head')
        self.world.guard_path.write_text('fixture-guard')
        self.manifest_path = self.root / 'candidate-manifest.json'
        self.manifest_path.write_text(json.dumps(manifest_doc(True)))
        self.deps = types.SimpleNamespace(
            strict=FakeStrict(self.world), phase=FakePhase(self.world), ab=None,
            acceptance=FakeAcceptance(self.world),
            manifest=manifest_doc(True), base_manifest=manifest_doc(False))
        self.patches = [
            patch.object(typed, 'sha', side_effect=self.fake_sha),
            patch.object(typed, 'proc_cmdline', side_effect=lambda pid: list(self.world.old_argv)),
            patch.object(typed, 'proc_environ', side_effect=lambda pid: dict(self.world.env)),
        ]
        for p in self.patches:
            p.start()
        self.deploy_run = self.root / 'deploy-run'
        self.deps.ab = FakeAB(self.world, self.deploy_run, self.log_root)
        self.old_restore = self.root / 'old-restore.json'
        self.old_restore.write_text(json.dumps({
            'label': 'old', 'service_discovery': {'api_identity': row(4242, 4242)},
            **{k: {'rc': 0, 'ok': True} for k in typed.FOUR_CHECKS}}))
        (self.log_root / 'old').mkdir()
        (self.log_root / 'old' / 'api.pid').write_text('4242')
        self.supervision = self.root / 'supervision.json'

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
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_supervision(self, record, phase='s1', manifest=None, exited=True, tamper=None):
        argv = ['/fixture/python', '/fixture/final_acceptance.py',
                '--manifest', str(manifest or self.manifest_path), '--phase', phase,
                '--output-dir', '/fixture/out',
                '--identities', record['identities'],
                '--launch-manifest', record['launch_manifest'],
                '--owner-evidence', record['owner_evidence'], '--execute']
        doc = {'schema': 'final-client-supervision/1', 'clients_exited': exited,
               'client_pid': 9999, 'observed': {}, 'argv': argv}
        if tamper:
            tamper(doc)
        self.supervision.write_text(json.dumps(doc))
        return self.supervision

    def deploy_args(self, execute=True, **kw):
        self.deps.ab = FakeAB(self.world, self.deploy_run, self.log_root)
        a = argparse.Namespace(
            action='deploy', run_dir=self.deploy_run, phase='s1', from_phase='s1',
            manifest=self.manifest_path,
            current_restore=self.old_restore, current_restore_sha=typed.sha(self.old_restore),
            deploy_record=None, deploy_record_sha=None, prior_client_record=None,
            first_phase_no_client=True, execute=execute, manifest_sha=typed.MANIFEST_SHA)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def make_deploy_record(self, phase='s1', label='cand-label', root=4242):
        ids = self.root / 'ready.json'
        ids.write_text(json.dumps({'root': row(root, root), 'descendants': [row(4343, root)]}))
        label_dir = self.log_root / label
        label_dir.mkdir(exist_ok=True)
        (label_dir / 'api.pid').write_text(str(root))
        launch_manifest = label_dir / 'launch-manifest.json'
        launch_manifest.write_text('{}')
        owner = self.root / 'deploy-restore.json'
        owner.write_text('{}')
        record = self.root / 'deploy-complete.json'
        record.write_text(json.dumps({
            'ok': True, 'action': 'deploy', 'phase': phase, 'label': label, 'root': root,
            'source': AA3E, 'guard': POST, 'identities': str(ids),
            'launch_manifest': str(launch_manifest), 'owner_evidence': str(owner),
            'manifest': str(self.manifest_path.resolve()), 'manifest_sha': typed.MANIFEST_SHA}))
        return record

    def restore_args(self, record, execute=True, **kw):
        self.deps.ab = FakeAB(self.world, self.root / 'restore-run', self.log_root)
        a = argparse.Namespace(
            action='restore', run_dir=self.root / 'restore-run', phase='s1', from_phase='s1',
            manifest=self.manifest_path,
            deploy_record=record, deploy_record_sha=typed.sha(record),
            prior_client_record=self.supervision,
            first_phase_no_client=False, execute=execute, manifest_sha=typed.MANIFEST_SHA)
        for k, v in kw.items():
            setattr(a, k, v)
        return a


class DeployTest(Base):
    def test_plan_only_does_not_mutate(self):
        result = typed.deploy(self.deploy_args(execute=False), self.deps)
        self.assertEqual(result['status'], 'plan_not_executed')
        self.assertFalse(self.deploy_run.exists())
        self.assertEqual(self.world.events, [])
        self.assertEqual(result['plan']['launch_env'][typed.FLAG_ENV], '1')
        self.assertEqual(result['plan']['source'], F810)
        self.assertEqual(result['plan']['guard'], PRE)

    def test_deploy_success_keeps_audited_primitive_order(self):
        result = typed.deploy(self.deploy_args(), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], AA3E)
        self.assertEqual(result['guard'], POST)
        self.assertEqual(self.world.head, AA3E)
        self.assertEqual(self.world.guard, POST)
        names = [e[0] for e in self.world.events]
        self.assertLess(names.index('stop-command'), names.index('adapter_apply'))
        self.assertLess(names.index('adapter_apply'), names.index('switch'))
        launch = next(e for e in self.world.events if e[0] == 'launch')
        env = dict(launch[2])
        self.assertEqual(env[typed.FLAG_ENV], '1')
        self.assertEqual(env[typed.ALLOW_ENV], AA3E)
        self.assertEqual(env['CLEF_LAUNCH_CONFIG'], 's1')

    def test_each_candidate_phase_uses_exact_frozen_overrides(self):
        for phase in typed.CANDIDATE_PHASES:
            self.deploy_run = self.root / f'deploy-{phase}'
            self.world.table = {4242: row(4242, 4242), 4343: row(4343, 4242)}
            self.world.tree = {4242, 4343}
            self.world.port = 0
            self.world.head, self.world.guard = F810, PRE
            self.world.events.clear()
            result = typed.deploy(self.deploy_args(
                phase=phase, from_phase='s1', first_phase_no_client=False,
                prior_client_record=self.supervision), self.deps)
            launch = next(e for e in self.world.events if e[0] == 'launch')
            self.assertEqual(list(launch[1][-len(FakePhase.OVERRIDES[phase]):]),
                             FakePhase.OVERRIDES[phase])
            self.assertEqual(result['phase'], phase)
            self.assertEqual(result['from_phase'], 's1')
            self.assertEqual(result['overrides'], FakePhase.OVERRIDES[phase])
            self.assertEqual(result['manifest'], str(self.manifest_path.resolve()))
            self.assertIn(('service_check', phase, 'cand'),
                          [e[:3] for e in self.world.events])

    def test_deploy_rejects_non_s1_start(self):
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(
                phase='s2_b8', from_phase='s2_b8', first_phase_no_client=False,
                prior_client_record=self.supervision), self.deps)
        self.assertEqual(self.world.events, [])

    def test_deploy_rejects_non_f810_start_source(self):
        self.world.head = AA3E
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertEqual(self.world.events, [])

    def test_deploy_rejects_non_preimage_start_guard(self):
        self.world.guard = POST
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertEqual(self.world.events, [])

    def test_old_label_api_pid_mismatch_blocks(self):
        (self.log_root / 'old' / 'api.pid').write_text('9999')
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_old_label_missing_blocks(self):
        (self.log_root / 'old' / 'api.pid').unlink()
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_old_restore_without_label_blocks(self):
        doc = json.loads(self.old_restore.read_text())
        doc.pop('label')
        self.old_restore.write_text(json.dumps(doc))
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_service_label_alias_is_accepted(self):
        doc = json.loads(self.old_restore.read_text())
        doc['service_label'] = doc.pop('label')
        self.old_restore.write_text(json.dumps(doc))
        result = typed.deploy(self.deploy_args(), self.deps)
        self.assertTrue(result['ok'])

    def test_old_restore_sha_mismatch_blocks(self):
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(current_restore_sha='0' * 64), self.deps)
        self.assertEqual(self.world.events, [])

    def test_incomplete_old_restore_blocks(self):
        doc = json.loads(self.old_restore.read_text())
        doc['health']['ok'] = False
        self.old_restore.write_text(json.dumps(doc))
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertEqual(self.world.events, [])

    def test_old_root_pgid_change_blocks(self):
        self.world.table[4242] = row(4242, 4243)
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_unattributed_pgid_member_blocks(self):
        self.world.table[9999] = row(9999, 4242)
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_pending_client_blocks(self):
        self.world.pending = True
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        self.assertNotIn('stop-command', [e[0] for e in self.world.events])

    def test_stop_failure_does_not_install_or_launch(self):
        self.world.fail_on.add('stop-command')
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('adapter_apply', names)
        self.assertNotIn('launch', names)
        self.assertFalse((self.deploy_run / 'recovery.json').exists())

    def test_launch_failure_recovers_to_f810(self):
        self.world.fail_on.add('launch')
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'restored_prod')
        self.assertEqual(self.world.head, F810)
        self.assertEqual(self.world.guard, PRE)

    def test_launch_failure_with_open_port_blocks(self):
        self.world.fail_on.add('launch')
        self.world.port_after_stop = 0
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'blocked_service_or_source_unknown')
        self.assertNotIn(('restore_prod',), self.world.events)

    def test_unknown_source_after_stop_blocks(self):
        self.world.fail_on.add('switch')
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.deploy_run / 'recovery.json').read_text())
        self.assertEqual(recovery['action'], 'blocked_service_or_source_unknown')
        self.assertNotIn(('restore_prod',), self.world.events)


class RestoreTest(Base):
    def test_restore_success_returns_production_and_handoff(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()))
        self.world.head, self.world.guard = AA3E, POST
        result = typed.restore(self.restore_args(record), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], F810)
        self.assertEqual(result['guard'], PRE)
        names = [e[0] for e in self.world.events]
        self.assertLess(names.index('stop-command'), names.index('adapter_revert'))
        self.assertLess(names.index('adapter_revert'), names.index('switch'))
        self.assertIn(('switch', 'f810', 'restore'), self.world.events)
        self.assertIn(('frozen', 'f810'), self.world.events)
        current = Path(result['current_restore'])
        self.assertTrue(current.is_file())
        self.assertEqual(typed.sha(current), result['current_restore_sha'])
        doc = json.loads(current.read_text())
        for key in typed.FOUR_CHECKS:
            self.assertEqual(doc[key], {'rc': 0, 'ok': True})
        self.assertEqual(doc['service_discovery']['api_identity']['pid'], self.world.new_pid)
        chained = typed.deploy(self.deploy_args(
            phase='erqa', from_phase='s1', first_phase_no_client=False,
            prior_client_record=self.supervision,
            current_restore=current, current_restore_sha=result['current_restore_sha']), self.deps)
        self.assertEqual(chained['phase'], 'erqa')

    def test_restore_rejects_wrong_record(self):
        record = self.make_deploy_record()
        doc = json.loads(record.read_text())
        doc['action'] = 'restore'
        record.write_text(json.dumps(doc))
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_rejects_phase_mismatch(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()))
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record, from_phase='erqa'), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_derives_missing_record_manifest_from_supervision(self):
        record = self.make_deploy_record()
        doc = json.loads(record.read_text())
        doc.pop('manifest')
        record.write_text(json.dumps(doc))
        self.write_supervision(doc)
        self.world.head, self.world.guard = AA3E, POST
        result = typed.restore(self.restore_args(record), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['manifest_source'], 'prior_supervision')
        self.assertEqual(result['manifest'], str(self.manifest_path.resolve()))

    def test_restore_rejects_missing_manifest_with_wrong_supervision(self):
        record = self.make_deploy_record()
        doc = json.loads(record.read_text())
        doc.pop('manifest')
        record.write_text(json.dumps(doc))
        other = self.root / 'other-manifest.json'
        other.write_text('{}')
        self.write_supervision(doc, manifest=other)
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_rejects_record_manifest_mismatch(self):
        record = self.make_deploy_record()
        doc = json.loads(record.read_text())
        other = self.root / 'other-manifest.json'
        other.write_text('{}')
        doc['manifest'] = str(other.resolve())
        record.write_text(json.dumps(doc))
        self.write_supervision(doc)
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_requires_supervision_phase(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()), phase='s2_b8')
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_requires_supervision_manifest(self):
        record = self.make_deploy_record()
        other = self.root / 'other-manifest.json'
        other.write_text('{}')
        self.write_supervision(json.loads(record.read_text()), manifest=other)
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_requires_supervision_identities(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()),
                               tamper=lambda d: d['argv'].__setitem__(
                                   d['argv'].index('--identities') + 1, '/fixture/other.json'))
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_requires_supervision_launch_manifest(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()),
                               tamper=lambda d: d['argv'].__setitem__(
                                   d['argv'].index('--launch-manifest') + 1, '/fixture/other.json'))
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_requires_supervision_exited(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()), exited=False)
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_rejects_protected_root_in_client_group(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()),
                               tamper=lambda d: d.__setitem__('observed', {'4242': row(4242, 9999)}))
        self.world.head, self.world.guard = AA3E, POST
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertEqual(self.world.events, [])

    def test_restore_incomplete_restore_record_blocks(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()))
        self.world.head, self.world.guard = AA3E, POST
        self.world.fail_on.add('incomplete_restore_record')
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertFalse((self.root / 'restore-run/complete.json').exists())
        recovery = json.loads((self.root / 'restore-run/recovery.json').read_text())
        self.assertEqual(recovery['action'], 'blocked_service_or_source_unknown')

    def test_restore_opp_ld_mismatch_blocks(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()))
        self.world.head, self.world.guard = AA3E, POST
        self.world.fail_on.add('restore_env_mismatch')
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        self.assertFalse((self.root / 'restore-run/complete.json').exists())

    def test_restore_stop_failure_does_not_launch(self):
        record = self.make_deploy_record()
        self.write_supervision(json.loads(record.read_text()))
        self.world.head, self.world.guard = AA3E, POST
        self.world.fail_on.add('stop-command')
        with self.assertRaises(RuntimeError):
            typed.restore(self.restore_args(record), self.deps)
        names = [e[0] for e in self.world.events]
        self.assertNotIn('adapter_revert', names)
        self.assertNotIn('restore_prod', names)


class BindingTest(unittest.TestCase):
    def setUp(self):
        self.candidate = manifest_doc(True)
        self.base = manifest_doc(False)
        self.phase = types.SimpleNamespace(OVERRIDES=FakePhase.OVERRIDES)

    def test_accepts_exact_candidate_delta(self):
        typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_argv_drift(self):
        self.candidate['phases']['s2_b8']['expected_argv'] = ['other']
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_effective_or_rounds_drift(self):
        for key, value in (('effective', {'max_num_seqs': 2}),
                           ('rounds', 3), ('concurrencies', [8]),
                           ('config', 's2')):
            candidate = manifest_doc(True)
            candidate['phases']['s2_b16'][key] = value
            with self.assertRaises(RuntimeError):
                typed.verify_phase_bindings(candidate, self.base, self.phase)

    def test_rejects_override_drift(self):
        self.candidate['phases']['erqa']['launch_overrides'] = ['--max-model-len', '8192']
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_phase_table_drift(self):
        phase = types.SimpleNamespace(OVERRIDES={'s1': []})
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, phase)

    def test_rejects_missing_phase(self):
        self.candidate['phases'].pop('erqa')
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_extra_environment_key(self):
        self.candidate['phases']['s2_b8']['service_environment']['CLEF_JOINT_SMALLQ'] = '1'
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_wrong_allow_sha(self):
        self.candidate['phases']['s2_b16']['service_environment'][typed.ALLOW_ENV] = F810
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)

    def test_rejects_forbidden_environment_drift(self):
        self.candidate['forbidden_environment'] = []
        with self.assertRaises(RuntimeError):
            typed.verify_phase_bindings(self.candidate, self.base, self.phase)


class CliTest(Base):
    def main_argv(self, *extra):
        return ['--action', 'deploy', '--code', str(self.root), '--owner-code', str(self.root),
                '--ab-code', str(self.root), '--repo', str(self.root),
                '--manifest', str(self.root / 'm.json'),
                '--base-manifest', str(self.root / 'b.json'),
                '--run-dir', str(self.root / 'r'), *extra]

    def test_invalid_phase_rejected(self):
        with self.assertRaises(SystemExit):
            typed.main(self.main_argv('--phase', 'bogus', '--first-phase-no-client'))

    def test_non_s1_first_phase_rejected(self):
        with patch.object(typed, 'build_deps', side_effect=AssertionError('must not build')):
            with self.assertRaises(RuntimeError):
                typed.main(self.main_argv('--phase', 's2_b8', '--from-phase', 's1',
                                          '--first-phase-no-client'))

    def test_deploy_requires_current_restore(self):
        with patch.object(typed, 'build_deps', side_effect=AssertionError('must not build')):
            with self.assertRaises(RuntimeError):
                typed.main(self.main_argv('--first-phase-no-client'))

    def test_deploy_rejects_non_s1_from_phase(self):
        with patch.object(typed, 'build_deps', side_effect=AssertionError('must not build')):
            with self.assertRaises(RuntimeError):
                typed.main(self.main_argv('--phase', 's2_b8', '--from-phase', 's1',
                                          '--first-phase-no-client'))

    def test_restore_requires_prior_client(self):
        with self.assertRaises(SystemExit):
            typed.main(['--action', 'restore', '--code', str(self.root),
                        '--owner-code', str(self.root), '--ab-code', str(self.root),
                        '--repo', str(self.root), '--manifest', str(self.root / 'm.json'),
                        '--base-manifest', str(self.root / 'b.json'),
                        '--run-dir', str(self.root / 'r'),
                        '--deploy-record', str(self.root / 'd.json'),
                        '--deploy-record-sha', 'x'])


class BuildDepsTest(unittest.TestCase):
    def test_build_deps_pins_a1ca_and_phase_bindings(self):
        with tempfile.TemporaryDirectory(prefix='clef-typed-build-') as tmp:
            root = Path(tmp)
            owner = root / 'owner'; owner.mkdir()
            code = root / 'code' / 'runtime'; code.mkdir(parents=True)
            ab = root / 'ab'; ab.mkdir()
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(manifest_doc(True)))
            base = root / 'base-manifest.json'
            base.write_text(json.dumps(manifest_doc(False)))
            paths = {
                owner / 'final_acceptance_supervisor.py': typed.SUPERVISOR_SHA,
                owner / 'final_service_phase.py': typed.PHASE_SHA,
                code / 'final_acceptance.py': typed.RUNNER_SHA,
                manifest: typed.MANIFEST_SHA,
                base: typed.BASE_MANIFEST_SHA,
                ab / 'joint_fused_service_ab.py': typed.BASE_SHA,
                ab / 'joint_fused_source_control.py': typed.SOURCE_CONTROL_SHA,
                ab / 'joint_fused_attention_window.py': typed.WINDOW_SHA,
                ab / 'apply_joint_fused_attention.py': 'b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406',
                ab / 'apply_joint_service_check.py': '1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c',
            }
            for path in paths:
                if path not in (manifest, base):
                    path.write_text('fixture')
            paths = {path.resolve(): digest for path, digest in paths.items()}

            class FakeSourceControl:
                def __init__(self, ns):
                    self.args = ns
                    self.run_dir = Path(ns.run_dir)

                def switch_source(self, target, tag):
                    self.state['source_switches'].append({'target': target, 'tag': tag})

            fake_base = types.SimpleNamespace(now=lambda: 'T',
                                              __file__=str((ab / 'joint_fused_service_ab.py').resolve()))
            fake_sc = types.SimpleNamespace(SourceControl=FakeSourceControl, base=fake_base)
            fake_phase = types.SimpleNamespace(strict=types.SimpleNamespace(),
                                               OVERRIDES=FakePhase.OVERRIDES)
            a = argparse.Namespace(
                owner_code=owner, code=root / 'code', manifest=manifest,
                manifest_sha=typed.MANIFEST_SHA, base_manifest=base,
                ab_code=ab, run_dir=root / 'run', repo=root, tree=str(root / 'tree'),
                python='/py', activate='/act', launch='/launch', wait='/wait',
                stop_script='/tmp/d0cc/d0_stop.py', service_log_root=str(root / 'labels'),
                clean_env_expect=root / 'expect.json', port=18143,
                stop_timeout=120.0, wait_timeout=900.0)
            real_sha = typed.sha

            def sha_side_effect(path):
                resolved = Path(path).resolve()
                if resolved == Path(typed.__file__).with_name('candidate_service_phase.py').resolve():
                    return real_sha(resolved)
                return paths[resolved]

            with patch.object(typed, 'sha', side_effect=sha_side_effect), \
                 patch.object(typed, 'load', side_effect=[fake_phase, fake_sc, types.SimpleNamespace()]):
                deps = typed.build_deps(a)
            self.assertEqual(deps.ab.state, {'schema': 'joint-fused-source-control/1', 'at': 'T',
                                             'args': {}, 'rounds': [], 'steps': {},
                                             'source_switches': [], 'overall_rc': 1})
            self.assertEqual(deps.ab.prod_env, {})
            deps.ab.switch_source('aa3e', 'fixture')
            self.assertEqual(deps.ab.state['source_switches'],
                             [{'target': 'aa3e', 'tag': 'fixture'}])

            drift = manifest_doc(True)
            drift['phases']['s2_b8']['launch_overrides'] = ['drift']
            manifest.write_text(json.dumps(drift))
            with patch.object(typed, 'sha', side_effect=sha_side_effect), \
                 patch.object(typed, 'load', side_effect=[fake_phase, fake_sc, types.SimpleNamespace()]):
                with self.assertRaises(RuntimeError):
                    typed.build_deps(a)


class RealFrozenFilesTest(unittest.TestCase):
    """Local real files: a1ca entry and the four-phase manifest/phase table."""

    def load_phase_module(self):
        return typed.load(HERE / 'final_service_phase.py', 'typed_real_phase')

    def test_a1ca_entry_bytes_unchanged(self):
        self.assertEqual(typed.sha(HERE / 'candidate_service_phase.py'), typed.BASE_ENTRY_SHA)

    def test_candidate_manifest_bytes_unchanged(self):
        self.assertEqual(typed.sha(HERE / 'final_acceptance_candidate_manifest.json'),
                         typed.MANIFEST_SHA)

    def test_real_overrides_equal_manifest(self):
        phase = self.load_phase_module()
        self.assertEqual(tuple(phase.OVERRIDES), typed.PHASES)
        manifest = json.loads((HERE / 'final_acceptance_candidate_manifest.json').read_text())
        for name in typed.PHASES:
            self.assertEqual(manifest['phases'][name]['launch_overrides'], phase.OVERRIDES[name])
            env = manifest['phases'][name]['service_environment']
            self.assertEqual(env[typed.FLAG_ENV], '1')
            self.assertEqual(env[typed.ALLOW_ENV], AA3E)
        self.assertNotIn(typed.FLAG_ENV, manifest['forbidden_environment'])

    def test_real_scan_points_unchanged(self):
        manifest = json.loads((HERE / 'final_acceptance_candidate_manifest.json').read_text())
        expect = {'s1': ([1], 3), 's2_b8': ([8, 16], 1),
                  's2_b16': ([32, 64], 1), 'erqa': ([1], 1)}
        for name, (conc, rounds) in expect.items():
            phase = manifest['phases'][name]
            self.assertEqual(phase['concurrencies'], conc)
            self.assertEqual(phase['rounds'], rounds)

    @unittest.skipUnless(os.environ.get('CLEF_BASE_MANIFEST'),
                         'frozen 0338f87b base manifest not provided')
    def test_real_phase_bindings_against_frozen_base(self):
        base = json.loads(Path(os.environ['CLEF_BASE_MANIFEST']).read_text())
        manifest = json.loads((HERE / 'final_acceptance_candidate_manifest.json').read_text())
        typed.verify_phase_bindings(manifest, base, self.load_phase_module())


@unittest.skipUnless(os.environ.get('CLEF_REAL_RECORDS'),
                     '32 real record copies not provided (CLEF_REAL_RECORDS)')
class RealRecordShapeTest(unittest.TestCase):
    """32-provided read-only copies of real acceptance JSON (no request raw).

    Two layouts are accepted:
      full   typed-fixtures-20261119: manifest.json + SHA256SUMS + 16 records
      simple typed-records-20261119: supervision.json, deploy-complete.json,
             restore-complete.json, restore-record.json, ready-identities.json,
             SHA256SUMS
    All path checks compare strings/paths recorded in the real files; the
    /workspace paths never exist here and are never read.
    """

    def setUp(self):
        self.dir = Path(os.environ['CLEF_REAL_RECORDS'])
        self.full = (self.dir / 'manifest.json').is_file()
        if self.full:
            self.names = dict(supervision='cand-s1-supervision.json',
                              deploy='cand-deploy-complete.json',
                              restore_complete='cand-finally-restore-complete.json',
                              restore_records=['cand-deploy-restore.json',
                                               'cand-finally-restore.json',
                                               'phase-s2b8-restore.json'],
                              identities='cand-deploy-ready-identities.json')
        else:
            self.names = dict(supervision='supervision.json', deploy='deploy-complete.json',
                              restore_complete='restore-complete.json',
                              restore_records=['restore-record.json'],
                              identities='ready-identities.json')

    def read(self, name):
        return json.loads((self.dir / name).read_text())

    def argv_value(self, argv, flag):
        return argv[argv.index(flag) + 1]

    def test_manifest_and_sha256sums_match(self):
        if self.full:
            doc = self.read('manifest.json')
            self.assertEqual(doc['schema'], 'typed-fixtures/1')
            for name, entry in doc['files'].items():
                data = (self.dir / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'], name)
                self.assertEqual(len(data), entry['bytes'], name)
        for line in (self.dir / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split()
            self.assertEqual(
                hashlib.sha256((self.dir / name.lstrip('*')).read_bytes()).hexdigest(), digest, name)

    def test_real_supervision_shape(self):
        doc = self.read(self.names['supervision'])
        self.assertEqual(doc['schema'], 'final-client-supervision/1')
        self.assertEqual(doc['status'], 'complete')
        self.assertIs(doc['clients_exited'], True)
        self.assertIsInstance(doc['argv'], list)
        for flag in ('--manifest', '--phase', '--identities',
                     '--launch-manifest', '--owner-evidence'):
            self.assertIn(flag, doc['argv'])
        self.assertIsInstance(doc['observed'], dict)
        self.assertTrue(all(isinstance(key, str) for key in doc['observed']))
        self.assertEqual(doc['errors'], [])

    def test_real_d0_restore_record_shape(self):
        for name in self.names['restore_records']:
            doc = self.read(name)
            typed.record_label(doc)
            disc = doc['service_discovery']
            self.assertEqual(disc['api_pid'], disc['api_identity']['pid'], name)
            self.assertIs(disc['root_identity_stable'], True, name)
            self.assertTrue(disc['pids'], name)
            for key in typed.FOUR_CHECKS:
                self.assertEqual(doc[key]['rc'], 0, f'{name}:{key}')
                self.assertIs(doc[key]['ok'], True, f'{name}:{key}')

    def test_real_deploy_record_binds_supervision_via_derived_manifest(self):
        record = self.read(self.names['deploy'])
        prior = self.read(self.names['supervision'])
        ids = self.read(self.names['identities'])
        rows = [ids['root'], *ids['descendants']]
        for key in ('label', 'root', 'identities', 'launch_manifest', 'owner_evidence',
                    'manifest_sha', 'phase', 'source', 'guard'):
            self.assertIn(key, record)
        # Historical a1ca schema carries no manifest path; typed derives it
        # from this deploy's own supervision argv and records the source.
        self.assertNotIn('manifest', record)
        derived = self.argv_value(prior['argv'], '--manifest')
        self.assertTrue(derived.endswith('/final_acceptance_candidate_manifest.json'))
        self.assertEqual(record['manifest_sha'], typed.MANIFEST_SHA)
        self.assertIs(prior['clients_exited'], True)
        self.assertEqual(ids['root']['pid'], record['root'])
        self.assertNotIn(str(record['root']), prior['observed'])
        typed.verify_prior_supervision(dict(record, manifest=derived), rows, prior)

    def test_real_restore_complete_carries_opp_ld_compare(self):
        doc = self.read(self.names['restore_complete'])
        self.assertEqual(doc['action'], 'restore')
        self.assertEqual(doc['source'], typed.F810_HEAD)
        self.assertEqual(doc['guard'], typed.GUARD_PRE)
        compare = doc['restore_env']
        self.assertEqual(compare['schema'], 'restore-env-compare/1')
        self.assertIs(compare['match'], True)
        self.assertEqual(compare['expected_sha256'],
                         'fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230')
        self.assertEqual(compare['counts'], {'ASCEND_CUSTOM_OPP_PATH': 9, 'LD_LIBRARY_PATH': 12})
        # The typed entry adds the next-phase handoff fields.
        self.assertNotIn('current_restore', doc)
        self.assertNotIn('current_restore_sha', doc)

    def test_real_phase_and_acceptance_shapes(self):
        if not self.full:
            self.skipTest('simple record layout carries no phase/acceptance records')
        phase = self.read('phase-s2b8-complete.json')
        self.assertEqual(phase['phase'], 's2_b8')
        for key in ('label', 'root', 'identities', 'launch_manifest', 'owner_evidence'):
            self.assertIn(key, phase)
        # The frozen phase switch record is not a candidate deploy record.
        self.assertNotIn('source', phase)
        self.assertNotIn('manifest_sha', phase)
        plan = self.read('phase-s2b8-plan.json')
        for key in ('phase', 'label', 'launch', 'clean_parent', 'source_sha256'):
            self.assertIn(key, plan)
        acceptance = self.read('cand-s1-acceptance.json')
        self.assertEqual(acceptance['scope'], 'final_s1')
        self.assertEqual(acceptance['completed_rounds'], 3)
        self.assertIs(acceptance['final_s1_complete'], True)
        self.assertIs(acceptance['quality_passed'], True)
        low, high = acceptance['p95_range_ms']
        self.assertLessEqual(low, acceptance['p95_median_ms'])
        self.assertLessEqual(acceptance['p95_median_ms'], high)



if __name__ == '__main__':
    unittest.main()
