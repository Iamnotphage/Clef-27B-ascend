"""Real-method CPU integration for the candidate deploy/restore entry.

Requires the frozen remote fixtures and is skipped otherwise:

  CLEF_REAL_BUNDLE=/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769
  CLEF_REAL_OWNER=/workspace/yangchen/work/final-phase-owner27-d4464ac...
  CLEF_REAL_CODE=/workspace/yangchen/work/final-acceptance-cc7f32842...
  CLEF_REAL_AB=$CLEF_REAL_BUNDLE/runtime
  CLEF_REAL_REPO=$CLEF_REAL_BUNDLE
  CLEF_REAL_MANIFEST=$CLEF_REAL_CODE/runtime/final_acceptance_manifest.json
  CLEF_REAL_CANDIDATE_MANIFEST=<stage>/runtime/final_acceptance_candidate_manifest.json
  CLEF_REAL_F810=/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py
  CLEF_REAL_GUARD=/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py
  CLEF_REAL_CLEAN_EXPECT=/workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json
  CLEF_REAL_PYTHON=/workspace/yangchen/venvs/clef-py312/bin/python

The real `SourceControl`/`ServiceAB` methods run against a temporary tree,
guard root and backup dirs.  Only external boundaries are substitutes: stop /
launch / wait commands, /proc and NPU scans, HTTP four-check, and the d0
service-discovery/verify calls.  No production file, service or NPU is touched.
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

import candidate_service_phase as cand

ORIGINAL_RUN = subprocess.run


REQUIRED = ('CLEF_REAL_BUNDLE', 'CLEF_REAL_OWNER', 'CLEF_REAL_CODE', 'CLEF_REAL_AB',
            'CLEF_REAL_REPO', 'CLEF_REAL_MANIFEST', 'CLEF_REAL_CANDIDATE_MANIFEST',
            'CLEF_REAL_F810', 'CLEF_REAL_GUARD', 'CLEF_REAL_CLEAN_EXPECT', 'CLEF_REAL_PYTHON')


def row(pid, pgid, ppid=1, starttime='1', state='S'):
    return dict(pid=pid, ppid=ppid, pgid=pgid, starttime=starttime,
                boot_id='boot-fixture', state=state, cmdline='fixture')


@unittest.skipUnless(all(os.environ.get(k) for k in REQUIRED),
                     'real frozen bundle fixtures not provided')
class RealMethodsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='clef-real-integration-')
        self.root = Path(self.tmp.name)
        self.tree = self.root / 'tree'
        source = Path(os.environ['CLEF_REAL_F810'])
        self.assertEqual(cand.sha(source), cand.F810_HEAD,
                         'real f810 fixture is not the frozen preimage; refusing to run')
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        self.service = self.root / 'service'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        guard.parent.mkdir(parents=True)
        guard.write_bytes(Path(os.environ['CLEF_REAL_GUARD']).read_bytes())
        self.logs = self.root / 'labels'; self.logs.mkdir()
        self.proc = self.root / 'proc'; self.proc.mkdir()
        self.fused_backup = self.root / 'fused-backup'
        self.adapter_backup = self.root / 'adapter-backup'
        self.old_argv = None
        self.port = 0
        self.old_root, self.old_child = 4242, 4343
        self.cand_root, self.cand_child = 5252, 5353
        self.prod_root, self.prod_child = 6262, 6363
        self.table = {self.old_root: row(self.old_root, self.old_root),
                      self.old_child: row(self.old_child, self.old_root)}
        self.ptree = {self.old_root, self.old_child}
        self.cand_env = {cand.FLAG_ENV: '1', cand.ALLOW_ENV: cand.AA3E_HEAD}
        self.old_argv = []
        self.events = []
        self.fail_apply_after = False
        self.service_check_calls = []
        self.deps = self.build(self.root / 'deploy-run')
        self.install(self.deps)

    def tearDown(self):
        if hasattr(self, 'base_mod'):
            self.base_mod.subprocess.run = ORIGINAL_RUN
        self.tmp.cleanup()

    def build(self, run_dir):
        a = argparse.Namespace(
            owner_code=Path(os.environ['CLEF_REAL_OWNER']), code=Path(os.environ['CLEF_REAL_CODE']),
            manifest=Path(os.environ['CLEF_REAL_CANDIDATE_MANIFEST']),
            manifest_sha=cand.MANIFEST_SHA,
            base_manifest=Path(os.environ['CLEF_REAL_MANIFEST']),
            ab_code=Path(os.environ['CLEF_REAL_AB']), run_dir=run_dir,
            repo=Path(os.environ['CLEF_REAL_REPO']), tree=str(self.tree),
            python=os.environ['CLEF_REAL_PYTHON'],
            activate='/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh',
            launch='/workspace/yangchen/logs/perf-wave2-20261118/launch.sh',
            wait='/workspace/yangchen/logs/perf-wave2-20261118/wait.sh',
            stop_script='/tmp/d0cc/d0_stop.py', service_log_root=str(self.logs),
            clean_env_expect=Path(os.environ['CLEF_REAL_CLEAN_EXPECT']),
            port=18143, stop_timeout=120.0, wait_timeout=900.0)
        deps = cand.build_deps(a)
        self.old_argv = list(deps.manifest['phases']['s1']['expected_argv'])
        self.table[self.old_root]['cmdline'] = ' '.join(self.old_argv)
        self.table[self.old_child]['cmdline'] = 'child'
        return deps

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
        cand.proc_cmdline = lambda pid: list(self.old_argv)
        cand.proc_environ = lambda pid: dict(self.cand_env)
        deps.acceptance.frozen_check = lambda m: self.events.append(('frozen', m.get('schema_version')))
        deps.acceptance.service_check = self.service_check
        sc_globals = type(deps.ab).switch_source.__globals__
        base_mod = sc_globals['base']
        base_mod.SERVICE_ROOT = str(self.service)
        real_run = ORIGINAL_RUN
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
        self.real_run = real_run
        base_mod.subprocess.run = self.dispatch_run
        self.proc.mkdir(exist_ok=True)
        self.inject_d0()

    def inject_d0(self):
        fake_restore = types.ModuleType('d0_restore')
        fake_restore.build_commands = lambda render: {}
        fake_restore.discover_service_pids = lambda label, before=None: {
            'pids': [self.prod_root], 'error': None}
        fake_restore.verify = lambda cmds, log, runner=None, timeout=120, service_pids=None: True
        fake_run = types.ModuleType('d0_run')
        fake_run.render_request = lambda *a, **k: None
        sys.modules['d0_restore'] = fake_restore
        sys.modules['d0_run'] = fake_run

    def expected_env(self):
        doc = json.loads(Path(os.environ['CLEF_REAL_CLEAN_EXPECT']).read_text())
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

    def deploy_args(self):
        old = self.root / 'old-restore.json'
        old.write_text(json.dumps({'label': 'old',
            'service_discovery': {'api_identity': row(self.old_root, self.old_root)},
            **{k: {'rc': 0, 'ok': True} for k in ('health', 'workers', 'text', 'image')}}))
        return argparse.Namespace(
            action='deploy', run_dir=self.root / 'deploy-run', phase='s1', from_phase='s1',
            current_restore=old, current_restore_sha=cand.sha(old),
            deploy_record=None, deploy_record_sha=None, prior_client_record=None,
            first_phase_no_client=True, execute=True, manifest_sha=cand.MANIFEST_SHA)

    def restore_args(self, record):
        return argparse.Namespace(
            action='restore', run_dir=self.root / 'restore-run', phase='s1', from_phase='s1',
            deploy_record=record, deploy_record_sha=cand.sha(record),
            prior_client_record=self.root / 'supervision.json',
            first_phase_no_client=False, execute=True, manifest_sha=cand.MANIFEST_SHA)

    def test_deploy_success_uses_real_installers_and_state(self):
        result = cand.deploy(self.deploy_args(), self.deps)
        self.assertTrue(result['ok'])
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        self.assertEqual(cand.sha(target), cand.AA3E_HEAD)
        self.assertEqual(cand.sha(guard), cand.GUARD_POST)
        self.assertIn('state', self.deps.ab.__dict__)
        self.assertIn('prod_env', self.deps.ab.__dict__)
        self.assertEqual(self.deps.ab.state['source_switches'][0]['target'], 'aa3e')
        self.assertEqual(self.deps.ab.active_source, 'aa3e')
        self.assertTrue((self.root / 'deploy-run/complete.json').is_file())

    def test_post_stop_switch_exception_recovers_and_restores(self):
        self.fail_apply_after = True
        with self.assertRaises(RuntimeError):
            cand.deploy(self.deploy_args(), self.deps)
        recovery = json.loads((self.root / 'deploy-run/recovery.json').read_text())
        self.assertEqual(recovery['action'], 'restored_prod')
        self.assertTrue(recovery['source_f810'])
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        self.assertEqual(cand.sha(target), cand.F810_HEAD)
        self.assertEqual(cand.sha(guard), cand.GUARD_PRE)
        self.assertIn('restore_env', self.deps.ab.state['steps'])
        self.assertTrue(self.deps.ab.state['steps']['restore_env']['ok'])

    def test_main_deploy_cli_success(self):
        real_build = cand.build_deps

        def build(a):
            self.assertEqual(Path(a.tree), self.tree, 'CLI tree must stay in the fixture')
            deps = real_build(a)
            self.deps = deps
            self.old_argv = list(deps.manifest['phases']['s1']['expected_argv'])
            self.table[self.old_root]['cmdline'] = ' '.join(self.old_argv)
            self.install(deps)
            return deps

        old = self.root / 'cli-old-restore.json'
        old.write_text(json.dumps({'label': 'old',
            'service_discovery': {'api_identity': row(self.old_root, self.old_root)},
            **{k: {'rc': 0, 'ok': True} for k in ('health', 'workers', 'text', 'image')}}))
        argv = ['--action', 'deploy', '--code', os.environ['CLEF_REAL_CODE'],
                '--owner-code', os.environ['CLEF_REAL_OWNER'], '--ab-code', os.environ['CLEF_REAL_AB'],
                '--repo', os.environ['CLEF_REAL_REPO'], '--manifest', os.environ['CLEF_REAL_CANDIDATE_MANIFEST'],
                '--manifest-sha', cand.MANIFEST_SHA, '--base-manifest', os.environ['CLEF_REAL_MANIFEST'],
                '--run-dir', str(self.root / 'cli-run'), '--phase', 's1', '--from-phase', 's1',
                '--current-restore', str(old), '--current-restore-sha', cand.sha(old),
                '--first-phase-no-client', '--execute',
                '--tree', str(self.tree), '--service-log-root', str(self.logs),
                '--port', '18143', '--python', os.environ['CLEF_REAL_PYTHON'],
                '--activate', '/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh',
                '--launch', '/workspace/yangchen/logs/perf-wave2-20261118/launch.sh',
                '--wait', '/workspace/yangchen/logs/perf-wave2-20261118/wait.sh',
                '--stop-script', '/tmp/d0cc/d0_stop.py',
                '--clean-env-expect', os.environ['CLEF_REAL_CLEAN_EXPECT']]
        with patch.object(cand, 'build_deps', side_effect=build):
            rc = cand.main(argv)
        self.assertEqual(rc, 0)
        self.assertTrue((self.root / 'cli-run/complete.json').is_file())
        self.assertEqual(cand.sha(self.tree / 'vllm/model_executor/models/clef_joint_schema.py'),
                         cand.AA3E_HEAD)
        self.assertEqual(cand.sha(Path(os.environ['CLEF_REAL_F810'])), cand.F810_HEAD,
                         'production head must remain untouched')

    def test_independent_restore_after_deploy(self):
        deployed = cand.deploy(self.deploy_args(), self.deps)
        with open(self.root / 'deploy-run/complete.json') as f:
            record = self.root / 'deploy-record.json'
            record.write_text(f.read())
        result = cand.restore(self.restore_args(record), self.deps)
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], cand.F810_HEAD)
        self.assertEqual(result['guard'], cand.GUARD_PRE)
        target = self.tree / 'vllm/model_executor/models/clef_joint_schema.py'
        guard = self.service / 'runtime/apply_joint_single_field_patch.py'
        self.assertEqual(cand.sha(target), cand.F810_HEAD)
        self.assertEqual(cand.sha(guard), cand.GUARD_PRE)
        self.assertTrue((self.root / 'restore-run/complete.json').is_file())
        self.assertTrue(self.service_check_calls)


if __name__ == '__main__':
    unittest.main()
