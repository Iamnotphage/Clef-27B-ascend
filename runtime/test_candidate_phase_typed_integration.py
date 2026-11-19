"""Real-method CPU integration for the typed S2/ERQA phase variant.

Requires local fixture copies (the harness builds them in a mkdtemp from
read-only frozen sources); skipped otherwise:

  CLEF_TYPED_BUNDLE     joint-fused-source-control-ebca769 bundle root
  CLEF_TYPED_OWNER      dir with final_acceptance_supervisor.py + final_service_phase.py
  CLEF_TYPED_CODE       dir containing runtime/final_acceptance.py (cc7 runner)
  CLEF_TYPED_MANIFEST   frozen f810 manifest (0338f87b...)
  CLEF_TYPED_CANDIDATE  candidate manifest (31202d27...)
  CLEF_TYPED_F810       f810 head file (f8101b98...)
  CLEF_TYPED_GUARD      guard preimage (76595657...)
  CLEF_TYPED_CLEAN      clean expected launch-manifest (fcbdcb...)
  CLEF_TYPED_PYTHON     interpreter used by the real installers

The real `SourceControl`/`ServiceAB` methods run against a temporary tree,
guard root, service-log root and backup dirs.  Only external boundaries are
substitutes: stop/launch/wait commands, /proc and NPU scans, HTTP four-check,
and d0 service discovery/verify.  Every spawned command is guarded: no ssh,
scp, sftp, curl or npu-smi, and every absolute path must stay inside the
mkdtemp fixture.  No production file, service or NPU is touched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import candidate_service_phase_typed as typed

ORIGINAL_RUN = subprocess.run
FORBIDDEN = {'ssh', 'scp', 'sftp', 'curl', 'npu-smi'}
REQUIRED = ('CLEF_TYPED_BUNDLE', 'CLEF_TYPED_OWNER', 'CLEF_TYPED_CODE', 'CLEF_TYPED_MANIFEST',
            'CLEF_TYPED_CANDIDATE', 'CLEF_TYPED_F810', 'CLEF_TYPED_GUARD', 'CLEF_TYPED_CLEAN',
            'CLEF_TYPED_PYTHON')


def row(pid, pgid, ppid=1, starttime='1', state='S'):
    return dict(pid=pid, ppid=ppid, pgid=pgid, starttime=starttime,
                boot_id='boot-fixture', state=state, cmdline='fixture')


@unittest.skipUnless(all(os.environ.get(k) for k in REQUIRED),
                     'local frozen fixture copies not provided')
class RealMethodsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-typed-real-')
        self.root = Path(self.tmp.name).resolve()
        bundle = Path(os.environ['CLEF_TYPED_BUNDLE']).resolve()
        self.service = self.root / 'service'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        guard.parent.mkdir(parents=True)
        guard.write_bytes(Path(os.environ['CLEF_TYPED_GUARD']).read_bytes())
        self.assertEqual(typed.sha(guard), typed.GUARD_PRE)
        self.tree = self.root / 'tree'
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(Path(os.environ['CLEF_TYPED_F810']).read_bytes())
        self.assertEqual(typed.sha(target), typed.F810_HEAD)
        self.logs = self.root / 'labels'; self.logs.mkdir()
        self.proc = self.root / 'proc'; self.proc.mkdir()
        self.fused_backup = self.root / 'fused-backup'
        self.adapter_backup = self.root / 'adapter-backup'
        self.port = 0
        self.old_root, self.old_child = 4242, 4343
        self.cand_root, self.cand_child = 5252, 5353
        self.prod_root, self.prod_child = 6262, 6363
        self.table = {self.old_root: row(self.old_root, self.old_root),
                      self.old_child: row(self.old_child, self.old_root)}
        self.ptree = {self.old_root, self.old_child}
        self.cand_env = {typed.FLAG_ENV: '1', typed.ALLOW_ENV: typed.AA3E_HEAD}
        self.old_argv = []
        self.events = []
        self.fail_apply_after = False
        self.service_check_calls = []
        self.bundle = bundle
        self.deps = self.build(self.root / 'deploy-run')
        self.install(self.deps)

    def tearDown(self):
        if hasattr(self, 'base_mod'):
            self.base_mod.subprocess.run = ORIGINAL_RUN
        if hasattr(self, 'saved_proc'):
            typed.proc_cmdline, typed.proc_environ = self.saved_proc
        self.tmp.cleanup()

    def build(self, run_dir):
        a = argparse.Namespace(
            owner_code=Path(os.environ['CLEF_TYPED_OWNER']).resolve(),
            code=Path(os.environ['CLEF_TYPED_CODE']).resolve(),
            manifest=Path(os.environ['CLEF_TYPED_CANDIDATE']).resolve(),
            manifest_sha=typed.MANIFEST_SHA,
            base_manifest=Path(os.environ['CLEF_TYPED_MANIFEST']).resolve(),
            ab_code=(self.bundle / 'runtime').resolve(), run_dir=run_dir,
            repo=self.bundle, tree=str(self.tree), python=os.environ['CLEF_TYPED_PYTHON'],
            activate='/fixture/activate.sh',
            launch='/fixture/launch.sh', wait='/fixture/wait.sh',
            stop_script='/fixture/d0_stop.py', service_log_root=str(self.logs),
            clean_env_expect=Path(os.environ['CLEF_TYPED_CLEAN']).resolve(),
            port=18143, stop_timeout=120.0, wait_timeout=900.0)
        self.assert_fixture_paths(a)
        deps = typed.build_deps(a)
        self.old_argv = list(deps.manifest['phases']['s1']['expected_argv'])
        self.table[self.old_root]['cmdline'] = ' '.join(self.old_argv)
        self.table[self.old_child]['cmdline'] = 'child'
        return deps

    def assert_fixture_paths(self, a):
        fixture_root = Path(os.environ.get('CLEF_TYPED_FIXTURE_ROOT') or self.root).resolve()
        for value in (Path(a.tree), Path(a.service_log_root)):
            resolved = Path(value).resolve()
            self.assertTrue(resolved.is_relative_to(self.root),
                            f'tree/log root escapes the test mkdtemp: {resolved}')
        for value in (a.owner_code, a.code, a.ab_code, a.repo, Path(a.clean_env_expect),
                      a.manifest, a.base_manifest, a.run_dir):
            resolved = Path(value).resolve()
            self.assertFalse(str(resolved).startswith(('/workspace', '/tmp/d0cc')),
                             f'production path in fixture: {resolved}')
            self.assertTrue(resolved.is_relative_to(self.root) or
                            resolved.is_relative_to(fixture_root),
                            f'fixture path escapes the explicit roots: {resolved}')

    def install(self, deps):
        deps.phase.TARGET = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        deps.phase.GUARD = self.service / 'runtime/apply_joint_single_field_patch.py'
        deps.phase.command = self.command
        deps.phase.resource = lambda out, name, service_pids=None: self.events.append(('resource', name))
        deps.phase.no_pending_clients = lambda table, prior=None: self.events.append(('clients', prior))
        deps.phase.port = lambda: self.port
        deps.phase.tree = lambda table, root: set(self.ptree)
        strict = deps.strict
        strict.scan = lambda: dict(self.table)
        strict.read_identity = lambda pid: self.table.get(pid)
        strict.same_process = lambda a, b: bool(a and b and all(
            a[k] == b[k] for k in ('pid', 'pgid', 'starttime', 'boot_id')))
        deps.ab.win.port_open = lambda port: 'closed'
        deps.ab.win.PROC_ROOT = self.proc
        self.saved_proc = (typed.proc_cmdline, typed.proc_environ)
        typed.proc_cmdline = lambda pid: list(self.old_argv)
        typed.proc_environ = lambda pid: dict(self.cand_env)
        deps.acceptance.frozen_check = lambda m: self.events.append(('frozen', m.get('schema_version')))
        deps.acceptance.service_check = self.service_check
        sc_globals = type(deps.ab).switch_source.__globals__
        base_mod = sc_globals['base']
        base_mod.SERVICE_ROOT = str(self.service)
        real_call = deps.ab.call

        def call(cmd, timeout=120, env=None):
            cmd = list(cmd)
            text = ' '.join(str(x) for x in cmd)
            if 'apply_joint_fused_attention.py' in text:
                cmd += ['--backup-dir', str(self.fused_backup)]
            if 'apply_joint_service_check.py' in text:
                cmd += ['--backup-dir', str(self.adapter_backup)]
            result = real_call(cmd, timeout, env)
            if (self.fail_apply_after and '--apply' in cmd and
                    'apply_joint_fused_attention.py' in text):
                raise RuntimeError('synthetic post-apply failure')
            return result

        deps.ab.call = call
        self.base_mod = base_mod
        self.real_run = self.guarded_run
        base_mod.subprocess.run = self.dispatch_run
        self.proc.mkdir(exist_ok=True)
        self.inject_d0()

    def guarded_run(self, cmd, **kw):
        cmd = list(cmd)
        names = {Path(str(x)).name for x in cmd if not str(x).startswith('-')}
        self.assertFalse(names & FORBIDDEN, f'forbidden command in fixture: {cmd}')
        for token in cmd:
            text = str(token)
            if text.startswith('/'):
                resolved = Path(text).resolve()
                allowed = resolved.is_relative_to(self.root)
                if not allowed:
                    self.assertEqual(text, os.environ['CLEF_TYPED_PYTHON'],
                                     f'path escapes the temporary fixture: {text}')
        return ORIGINAL_RUN(cmd, **kw)

    def inject_d0(self):
        fake_restore = types.ModuleType('d0_restore')
        fake_restore.build_commands = lambda render: {}
        fake_restore.discover_service_pids = lambda label, before=None: {
            'pid_file': str(self.logs / label / 'api.pid'), 'exists': True,
            'api_pid': self.prod_root, 'api_identity': row(self.prod_root, self.prod_root),
            'root_identity_stable': True, 'pids': [self.prod_root, self.prod_child],
            'descendant_count': 1,
            'tree': [row(self.prod_root, self.prod_root), row(self.prod_child, self.prod_root)],
            'error': None}

        def verify(cmds, log, runner=None, timeout=120, service_pids=None):
            for stage in typed.FOUR_CHECKS:
                log[stage] = {'started': 'fixture', 'finished': 'fixture', 'rc': 0, 'ok': True}
            return True

        fake_restore.verify = verify
        fake_run = types.ModuleType('d0_run')
        fake_run.render_request = lambda *a, **k: None
        sys.modules['d0_restore'] = fake_restore
        sys.modules['d0_run'] = fake_run

    def expected_env(self):
        doc = json.loads(Path(os.environ['CLEF_TYPED_CLEAN']).read_text())
        env = doc.get('service_environment', doc)
        return {k: str(env[k]).encode() for k in ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH')}

    def dispatch_run(self, cmd, **kw):
        cmd = list(cmd)
        if cmd[:1] == ['/bin/bash'] and len(cmd) >= 2:
            if cmd[1] == self.deps.ab.args.launch:
                label = cmd[2]
                label_dir = self.logs / label; label_dir.mkdir(exist_ok=True)
                (label_dir / 'api.pid').write_text(str(self.prod_root))
                (label_dir / 'launch-manifest.json').write_text('{}')
                pdir = self.proc / str(self.prod_root); pdir.mkdir(exist_ok=True)
                env = self.expected_env()
                (pdir / 'environ').write_bytes(
                    b'\0'.join(k.encode() + b'=' + v for k, v in env.items()) + b'\0')
                self.table = {self.prod_root: row(self.prod_root, self.prod_root),
                              self.prod_child: row(self.prod_child, self.prod_root)}
                self.ptree = {self.prod_root, self.prod_child}
                self.port = 0
                return types.SimpleNamespace(returncode=0, stdout='', stderr='')
            if cmd[1] == self.deps.ab.args.wait:
                return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        return self.real_run(cmd, **kw)

    def command(self, out, name, argv, timeout, env=None):
        self.events.append((name, tuple(argv), None if env is None else tuple(sorted(env.items()))))
        (Path(out) / (name + '.json')).write_text(json.dumps({'rc': 0}))
        if name == 'stop-command':
            self.table = {}
            self.ptree = set()
            self.port = 111
        elif name == 'launch':
            label = argv[2]
            label_dir = self.logs / label; label_dir.mkdir()
            (label_dir / 'api.pid').write_text(str(self.cand_root))
            (label_dir / 'launch-manifest.json').write_text('{}')
            self.table = {self.cand_root: row(self.cand_root, self.cand_root),
                          self.cand_child: row(self.cand_child, self.cand_root)}
            self.ptree = {self.cand_root, self.cand_child}
            self.port = 0
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')

    def service_check(self, manifest, phase, identities, launch):
        self.service_check_calls.append((phase, str(identities)))
        return {'ok': True}

    def restore_record(self, label='old'):
        label_dir = self.logs / label
        label_dir.mkdir(exist_ok=True)
        (label_dir / 'api.pid').write_text(str(self.old_root))
        return {'label': label,
                'service_discovery': {'api_identity': row(self.old_root, self.old_root)},
                **{k: {'rc': 0, 'ok': True} for k in typed.FOUR_CHECKS}}

    def bind_run_dir(self, path):
        self.deps.ab.run_dir = Path(path).resolve()
        return Path(path)

    def deploy_args(self, run_dir, phase='s2_b8', **kw):
        self.bind_run_dir(run_dir)
        old = self.root / f'current-restore-{phase}.json'
        old.write_text(json.dumps(self.restore_record()))
        supervision = self.root / f'supervision-{phase}.json'
        supervision.write_text(json.dumps({'clients_exited': True, 'client_pid': 9999, 'observed': {}}))
        args = argparse.Namespace(
            action='deploy', run_dir=run_dir, phase=phase, from_phase='s1',
            manifest=Path(os.environ['CLEF_TYPED_CANDIDATE']).resolve(),
            current_restore=old, current_restore_sha=typed.sha(old),
            deploy_record=None, deploy_record_sha=None,
            prior_client_record=supervision, first_phase_no_client=False,
            execute=True, manifest_sha=typed.MANIFEST_SHA)
        for key, value in kw.items():
            setattr(args, key, value)
        return args

    def supervision_for(self, record, phase):
        doc = json.loads(Path(record).read_text())
        prior = self.root / f'client-{phase}.json'
        prior.write_text(json.dumps({
            'schema': 'final-client-supervision/1', 'clients_exited': True,
            'client_pid': 7777, 'observed': {},
            'argv': [os.environ['CLEF_TYPED_PYTHON'], '/fixture/final_acceptance.py',
                     '--manifest', doc['manifest'], '--phase', phase,
                     '--output-dir', '/fixture/out',
                     '--identities', doc['identities'],
                     '--launch-manifest', doc['launch_manifest'],
                     '--owner-evidence', doc['owner_evidence'], '--execute']}))
        return prior

    def test_deploy_s2_b8_uses_real_installers_and_state(self):
        result = typed.deploy(self.deploy_args(self.root / 'deploy-s2b8'), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['phase'], 's2_b8')
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        self.assertEqual(typed.sha(target), typed.AA3E_HEAD)
        self.assertEqual(typed.sha(guard), typed.GUARD_POST)
        self.assertEqual(self.deps.ab.state['source_switches'][0]['target'], 'aa3e')
        launch = next(e for e in self.events if e[0] == 'launch')
        overrides = list(self.deps.manifest['phases']['s2_b8']['launch_overrides'])
        self.assertEqual(list(launch[1][-len(overrides):]), overrides)
        complete = json.loads((self.root / 'deploy-s2b8/complete.json').read_text())
        for key in ('label', 'root', 'identities', 'launch_manifest', 'owner_evidence',
                    'manifest', 'phase', 'from_phase'):
            self.assertIn(key, complete)

    def test_deploy_s2_b16_uses_exact_phase_argv(self):
        result = typed.deploy(self.deploy_args(self.root / 'deploy-s2b16', phase='s2_b16'),
                              self.deps)
        self.assertEqual(result['phase'], 's2_b16')
        launch = next(e for e in self.events if e[0] == 'launch')
        overrides = list(self.deps.manifest['phases']['s2_b16']['launch_overrides'])
        self.assertEqual(list(launch[1][-len(overrides):]), overrides)

    def test_restore_and_chained_erqa_from_handoff(self):
        first = typed.deploy(self.deploy_args(self.root / 'deploy-a'), self.deps)
        record = self.root / 'deploy-a/complete.json'
        prior = self.supervision_for(record, 's2_b8')
        self.bind_run_dir(self.root / 'restore-a')
        self.old_argv = list(self.deps.manifest['phases']['s2_b8']['expected_argv'])
        restored = typed.restore(argparse.Namespace(
            action='restore', run_dir=self.root / 'restore-a', phase='s1', from_phase='s2_b8',
            manifest=Path(os.environ['CLEF_TYPED_CANDIDATE']).resolve(),
            deploy_record=record, deploy_record_sha=typed.sha(record),
            prior_client_record=prior, first_phase_no_client=False, execute=True,
            manifest_sha=typed.MANIFEST_SHA), self.deps)
        self.assertEqual(restored['source'], typed.F810_HEAD)
        self.assertEqual(typed.sha(self.tree / 'vllm/model_executor/models/clef_joint_schema.py'),
                         typed.F810_HEAD)
        current = Path(restored['current_restore'])
        self.assertEqual(typed.sha(current), restored['current_restore_sha'])
        doc = json.loads(current.read_text())
        for key in typed.FOUR_CHECKS:
            self.assertIs(doc[key]['ok'], True)
        self.bind_run_dir(self.root / 'deploy-b')
        self.old_argv = list(self.deps.manifest['phases']['s1']['expected_argv'])
        second = typed.deploy(argparse.Namespace(
            action='deploy', run_dir=self.root / 'deploy-b', phase='erqa', from_phase='s1',
            manifest=Path(os.environ['CLEF_TYPED_CANDIDATE']).resolve(),
            current_restore=current, current_restore_sha=restored['current_restore_sha'],
            deploy_record=None, deploy_record_sha=None,
            prior_client_record=self.supervision_for(record, 's2_b8'),
            first_phase_no_client=False, execute=True,
            manifest_sha=typed.MANIFEST_SHA), self.deps)
        self.assertEqual(second['phase'], 'erqa')
        self.assertEqual(second['from_phase'], 's1')

    def test_post_stop_switch_exception_recovers_to_f810(self):
        self.fail_apply_after = True
        with self.assertRaises(RuntimeError):
            typed.deploy(self.deploy_args(self.root / 'deploy-fail'), self.deps)
        recovery = json.loads((self.root / 'deploy-fail/recovery.json').read_text())
        self.assertEqual(recovery['action'], 'restored_prod')
        self.assertTrue(recovery['source_f810'])
        self.assertEqual(typed.sha(self.tree / 'vllm/model_executor/models/clef_joint_schema.py'),
                         typed.F810_HEAD)
        self.assertEqual(typed.sha(self.service / 'runtime/apply_joint_single_field_patch.py'),
                         typed.GUARD_PRE)

    def test_main_cli_deploy_requires_explicit_fixture_paths(self):
        old = self.root / 'cli-restore.json'
        old.write_text(json.dumps(self.restore_record()))
        supervision = self.root / 'cli-supervision.json'
        supervision.write_text(json.dumps({'clients_exited': True, 'client_pid': 9999, 'observed': {}}))
        real_build = typed.build_deps

        def build(a):
            self.assert_fixture_paths(a)
            deps = real_build(a)
            self.deps = deps
            self.old_argv = list(deps.manifest['phases']['s1']['expected_argv'])
            self.table[self.old_root]['cmdline'] = ' '.join(self.old_argv)
            self.install(deps)
            return deps

        argv = ['--action', 'deploy', '--code', os.environ['CLEF_TYPED_CODE'],
                '--owner-code', os.environ['CLEF_TYPED_OWNER'],
                '--ab-code', str(self.bundle / 'runtime'), '--repo', str(self.bundle),
                '--manifest', os.environ['CLEF_TYPED_CANDIDATE'],
                '--manifest-sha', typed.MANIFEST_SHA,
                '--base-manifest', os.environ['CLEF_TYPED_MANIFEST'],
                '--run-dir', str(self.root / 'cli-run'), '--phase', 's2_b8', '--from-phase', 's1',
                '--current-restore', str(old), '--current-restore-sha', typed.sha(old),
                '--prior-client-record', str(supervision), '--execute',
                '--tree', str(self.tree), '--service-log-root', str(self.logs),
                '--port', '18143', '--python', os.environ['CLEF_TYPED_PYTHON'],
                '--activate', '/fixture/activate.sh',
                '--launch', '/fixture/launch.sh', '--wait', '/fixture/wait.sh',
                '--stop-script', '/fixture/d0_stop.py',
                '--clean-env-expect', os.environ['CLEF_TYPED_CLEAN']]
        with patch.object(typed, 'build_deps', side_effect=build):
            rc = typed.main(argv)
        self.assertEqual(rc, 0)
        self.assertTrue((self.root / 'cli-run/complete.json').is_file())
        self.assertEqual(typed.sha(Path(os.environ['CLEF_TYPED_F810'])), typed.F810_HEAD,
                         'fixture preimage must remain untouched')

    def test_main_cli_without_explicit_paths_is_rejected_by_fixture_guard(self):
        """The fixture wrapper refuses any default production path."""
        a = argparse.Namespace(
            owner_code=Path(os.environ['CLEF_TYPED_OWNER']), code=Path(os.environ['CLEF_TYPED_CODE']),
            ab_code=self.bundle / 'runtime', repo=self.bundle,
            tree='/workspace/yangchen/work/src/vllm', service_log_root=str(self.logs),
            clean_env_expect=Path(os.environ['CLEF_TYPED_CLEAN']),
            manifest=Path(os.environ['CLEF_TYPED_CANDIDATE']),
            base_manifest=Path(os.environ['CLEF_TYPED_MANIFEST']))
        with self.assertRaises(AssertionError):
            self.assert_fixture_paths(a)


if __name__ == '__main__':
    unittest.main()
