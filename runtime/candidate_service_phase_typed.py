#!/usr/bin/env python3
"""Typed S2/ERQA phase variant of the frozen a1ca candidate entry.

Derived from `candidate_service_phase.py` (SHA a1ca26cf..., required beside
this file at build time).  Only the typed phase cycle is added:

  fresh f810 s1 (four-checked, f810 head + guard preimage)
    -> deploy one candidate phase (s2_b8 | s2_b16 | erqa)
    -> frozen v5 supervisor with the same `--phase`
    -> restore to the same clean f810 s1 (four-check + OPP/LD byte compare)

No candidate-to-candidate source transition is accepted.  Deployment start is
pinned to f810/guard-pre/s1 argv; launch overrides must equal the frozen phase
table and the candidate manifest may differ from the frozen f810 manifest only
in the two candidate environment keys.  The restore's prior supervision must
bind the same protected root/label/identities/manifest/phase as this deploy.

Default is plan-only.  Production f810 keeps using the frozen
`final_service_phase.py` unchanged.  Every device action is an already-audited
primitive: original d0 stop, original launcher/wait, audited guard adapter and
source installer, cc7 service_check, d0 four-check, phase-v3 PGID/client/port/
NPU gates.  Any error after the old service was proven stopped goes through the
explicit recovery gate; otherwise nothing is signalled or launched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

F810_HEAD = 'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863'
AA3E_HEAD = 'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'
GUARD_PRE = '76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9'
GUARD_POST = '36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7'
SUPERVISOR_SHA = '172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876'
PHASE_SHA = '888a8d7fac79bd243fe894c25324aca0f498e311cab91be77df1dc2cfdc0de44'
RUNNER_SHA = '7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab'
BASE_SHA = '127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648'
SOURCE_CONTROL_SHA = 'a1c502a3a709b836d31dc994a156f702d6a826ce5f2854beeec9ac50701f7299'
WINDOW_SHA = '700cf10cdc3fda18d1ae7f113158342e1ef6ec114569e52b15b134949bfe8473'
MANIFEST_SHA = '31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4'
BASE_MANIFEST_SHA = '0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2'
BASE_ENTRY_SHA = 'a1ca26cf34cd2ca0ff47787d10e06bf6e395b794882c59c938691b05d62e1d44'
FLAG_ENV = 'CLEF_JOINT_FUSED_ATTN'
ALLOW_ENV = 'CLEF_JOINT_ALLOWED_POSTIMAGE_SHA'
PATH_KEYS = ('PATH', 'CLEF_LAUNCH_CONFIG')
CANDIDATE_ENV = {
    'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
    'CLEF_LAUNCH_CONFIG': 's1',
    FLAG_ENV: '1',
    ALLOW_ENV: AA3E_HEAD,
}
PHASES = ('s1', 's2_b8', 's2_b16', 'erqa')
CANDIDATE_PHASES = ('s2_b8', 's2_b16', 'erqa')
FOUR_CHECKS = ('health', 'workers', 'text', 'image')
MANIFEST_PHASE_KEYS = ('expected_argv', 'effective', 'config', 'concurrencies', 'rounds')
RECORD_BINDINGS = (('--identities', 'identities'),
                   ('--launch-manifest', 'launch_manifest'),
                   ('--owner-evidence', 'owner_evidence'))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path: Path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def proc_cmdline(pid):
    return (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')


def proc_environ(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    return dict((part.split(b'=', 1)[0].decode(), part.split(b'=', 1)[1].decode())
                for part in raw.split(b'\0') if b'=' in part)


def initial_state(base):
    """The ebca main() normally seeds this before any switch/restore call."""
    return {'schema': 'joint-fused-source-control/1', 'at': base.now(), 'args': {},
            'rounds': [], 'steps': {}, 'source_switches': [], 'overall_rc': 1}


def verify_phase_bindings(manifest, base_manifest, phase):
    """The candidate manifest may differ from frozen f810 only in the two
    candidate environment keys; argv, scoring, concurrency and rounds stay
    byte-identical.  Launch overrides must equal the frozen phase table."""
    require(set(manifest.get('phases', {})) == set(PHASES) == set(base_manifest.get('phases', {})),
            'phase set mismatch')
    require(tuple(phase.OVERRIDES) == PHASES, 'frozen phase override table mismatch')
    for name in PHASES:
        got = manifest['phases'][name]
        ref = base_manifest['phases'][name]
        require(got.get('launch_overrides') == phase.OVERRIDES[name],
                f'{name} launch overrides differ from the frozen phase table')
        for key in MANIFEST_PHASE_KEYS:
            require(got.get(key) == ref.get(key),
                    f'{name}.{key} differs from the frozen f810 manifest')
        env = dict(got.get('service_environment', {}))
        ref_env = dict(ref.get('service_environment', {}))
        require(env.pop(FLAG_ENV, None) == '1', f'{name} lacks the fused flag')
        require(env.pop(ALLOW_ENV, None) == AA3E_HEAD, f'{name} lacks the exact allow SHA')
        require(env == ref_env, f'{name} service environment drift outside the two candidate keys')
    require(manifest.get('forbidden_environment') ==
            [key for key in base_manifest.get('forbidden_environment', []) if key != FLAG_ENV],
            'forbidden environment drift outside the fused flag')


def record_label(record):
    """Exact service label of a restore/deploy record."""
    label = record.get('label')
    alias = record.get('service_label')
    require(not (label and alias and label != alias), 'record carries two different labels')
    label = label or alias
    require(isinstance(label, str) and label and Path(label).name == label,
            'record lacks its exact service label')
    return label


def argv_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def build_deps(a):
    owner = a.owner_code.resolve()
    require(sha(Path(__file__).with_name('candidate_service_phase.py')) == BASE_ENTRY_SHA,
            'frozen a1ca S1 entry missing or changed beside this variant')
    require(sha(owner / 'final_acceptance_supervisor.py') == SUPERVISOR_SHA, 'supervisor SHA mismatch')
    require(sha(owner / 'final_service_phase.py') == PHASE_SHA, 'phase module SHA mismatch')
    require(sha(a.code / 'runtime/final_acceptance.py') == RUNNER_SHA, 'acceptance runner SHA mismatch')
    require(sha(a.manifest) == a.manifest_sha, 'candidate manifest SHA mismatch')
    require(a.manifest_sha == MANIFEST_SHA, 'candidate manifest is not the frozen derived one')
    require(a.base_manifest.is_file() and sha(a.base_manifest) == BASE_MANIFEST_SHA,
            'frozen f810 manifest SHA mismatch')
    ab_dir = a.ab_code.resolve()
    require(sha(ab_dir / 'joint_fused_service_ab.py') == BASE_SHA, 'AB base SHA mismatch')
    require(sha(ab_dir / 'joint_fused_source_control.py') == SOURCE_CONTROL_SHA,
            'source-control runner SHA mismatch')
    require(sha(ab_dir / 'joint_fused_attention_window.py') == WINDOW_SHA, 'AB window SHA mismatch')
    for name, digest in (('apply_joint_fused_attention.py', 'b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406'),
                         ('apply_joint_service_check.py', '1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c')):
        require(sha(ab_dir / name) == digest, f'{name} SHA mismatch')
    sys.path.insert(0, str(owner))
    phase = load(owner / 'final_service_phase.py', 'candidate_phase_gates')
    strict = phase.strict
    sc_mod = load(ab_dir / 'joint_fused_source_control.py', 'candidate_source_control')
    require(sc_mod.base.__file__ == str(ab_dir / 'joint_fused_service_ab.py'), 'AB base resolution drift')
    acceptance = load(a.code / 'runtime/final_acceptance.py', 'candidate_acceptance')
    ns = SimpleNamespace(
        run_dir=a.run_dir.resolve(), repo=str(a.repo), tree=a.tree, python=a.python,
        activate=a.activate, launch=a.launch, wait=a.wait, stop_script=a.stop_script,
        service_log_root=a.service_log_root, clean_env_expect=str(a.clean_env_expect),
        data='', short='od-0105', long='od-1534', image_source='', text_url='',
        image_url='', port=a.port, rounds=0, reps=0, warmup=0, http_timeout=0.0,
        stop_timeout=a.stop_timeout, wait_timeout=a.wait_timeout)
    ab = sc_mod.SourceControl(ns)
    # Complete context for every real SourceControl/ServiceAB method this entry
    # calls (ebca main() normally seeds these before use):
    #   switch_source        state["source_switches"], source_switches, active_source
    #   restore_prod         state["steps"], prod_env (saved/restored only; the
    #                        launch env is the explicit clean dict)
    #   verify_restored_env  state["steps"], win.PROC_ROOT, args.clean_env_expect
    # __init__ already provides args/run_dir/win, active_source/source_switches,
    # service_check_applied, attribution_unknown and cleanup_unconfirmed.
    ab.state = initial_state(sc_mod.base)
    ab.prod_env = {}
    manifest, base_manifest = read(a.manifest), read(a.base_manifest)
    verify_phase_bindings(manifest, base_manifest, phase)
    return SimpleNamespace(strict=strict, phase=phase, ab=ab, acceptance=acceptance,
                           manifest=manifest, base_manifest=base_manifest)


def verify_old_service(deps, old, from_phase):
    """Frozen phase-v3 gates for the service that is about to be replaced."""
    strict, require_ = deps.strict, require
    saved = old['service_discovery']['api_identity']
    root = saved['pid']
    label = record_label(old)
    pid_file = Path(deps.ab.args.service_log_root) / label / 'api.pid'
    require_(pid_file.is_file(), 'old label api.pid missing')
    require_(int(pid_file.read_text().strip()) == root, 'old label api.pid mismatch')
    table = strict.scan()
    live = table.get(root)
    require_(live and live['state'] != 'Z' and live['pgid'] == root and live['ppid'] == 1,
             'old root unavailable or group/parent changed')
    require_(all(live[k] == saved[k] for k in ('pid', 'starttime', 'boot_id')),
             'old root lifetime changed')
    require_(proc_cmdline(root) == deps.manifest['phases'][from_phase]['expected_argv'],
             'old final argv mismatch')
    pids = deps.phase.tree(table, root)
    require_(all(table[p]['pgid'] == root for p in pids), 'old descendant escaped group')
    old_group = {pid for pid, row in table.items() if row['pgid'] == root and row['state'] != 'Z'}
    require_(old_group.issubset(pids), 'unattributed old PGID member before stop')
    return table, root, pids


def stop_and_gate(deps, run_dir, old, prior=None, state=None):
    """Original exact stop plus the frozen post-stop emptiness gates."""
    strict, phase, ab = deps.strict, deps.phase, deps.ab
    table, root, pids = verify_old_service(deps, old, old.get('_from_phase', 's1'))
    deps.phase.no_pending_clients(table, prior)
    ids = dict(root=table[root], descendants=[table[p] for p in sorted(pids - {root})])
    write(run_dir / 'old-identities.json', ids)
    write(run_dir / 'root-identity.json', table[root])
    phase.resource(run_dir, 'before-npu', pids)
    require(phase.port() == 0, 'old service port not open')
    phase.command(run_dir, 'stop-command', [deps.phase.PYTHON, deps.phase.STOP,
        '--api-pid', str(root), '--saved-identity', str(run_dir / 'root-identity.json'),
        '--out', str(run_dir / 'stop.json')], 120)
    if state is not None:
        state['stopped'] = True
    table = strict.scan()
    require(all(p not in table or table[p]['state'] == 'Z' for p in pids), 'old tree remains')
    require(not any(row['pgid'] == root and row['state'] != 'Z' for row in table.values()),
            'old PGID member remains after stop')
    phase.no_pending_clients(table, prior)
    require(phase.port() == 111, 'port not closed before launch')
    phase.resource(run_dir, 'stopped-npu')
    return root, pids


def launch_candidate(deps, run_dir, label, phase_name):
    strict, phase = deps.strict, deps.phase
    label_dir = Path(deps.ab.args.service_log_root) / label
    require(not label_dir.exists(), 'fresh label collision')
    before = set(strict.scan())
    require(phase.port() == 111, 'fresh port not closed immediately before launch')
    # Exactly the frozen phase-v3 per-phase launcher overrides.
    phase.command(run_dir, 'launch',
                  ['/bin/bash', deps.ab.args.launch, label, *phase.OVERRIDES[phase_name]], 60,
                  env=dict(CANDIDATE_ENV))
    pid = int((label_dir / 'api.pid').read_text())
    initial = strict.read_identity(pid)
    require(initial and pid not in before and initial['pgid'] == pid,
            'new root identity unavailable')
    write(run_dir / 'launch-identity.json', initial)
    phase.command(run_dir, 'wait', ['timeout', '600', '/bin/bash', deps.ab.args.wait], 630)
    table = strict.scan()
    final = table.get(pid)
    require(strict.same_process(initial, final), 'new root changed during wait')
    pids = phase.tree(table, pid)
    require(all(table[p]['pgid'] == pid for p in pids), 'new descendant escaped group')
    ids = dict(root=final, descendants=[table[p] for p in sorted(pids - {pid})])
    write(run_dir / 'ready-identities.json', ids)
    check = deps.acceptance.service_check(
        deps.manifest, phase_name, run_dir / 'ready-identities.json',
        label_dir / 'launch-manifest.json')
    write(run_dir / 'service-check.json', check)
    require(sha(deps.phase.TARGET) == AA3E_HEAD, 'candidate head not installed at ready')
    require(sha(deps.phase.GUARD) == GUARD_POST, 'candidate guard not installed at ready')
    env = proc_environ(pid)
    require(env.get(FLAG_ENV) == '1' and env.get(ALLOW_ENV) == AA3E_HEAD,
            'ready root lacks the exact candidate environment')
    phase.resource(run_dir, 'ready-npu', pids)
    return label, pid, ids, label_dir / 'launch-manifest.json'


def four_check(deps, run_dir, label, root, pids):
    d0 = run_dir / 'stage' / 'd0'
    require(d0.is_dir(), 'd0 staging missing')
    sys.path.insert(0, str(d0))
    import d0_restore, d0_run
    table = deps.strict.scan()
    log = dict(label=label, service_discovery=dict(
        api_pid=root, api_identity=table[root],
        root_identity_stable=True, pids=sorted(pids),
        tree=[table[p] for p in sorted(pids)]))
    ok = d0_restore.verify(d0_restore.build_commands(d0_run.render_request), log,
                           service_pids=pids)
    write(run_dir / 'restore.json', log)
    require(ok, 'four-check verification failed')
    return log


def recover(deps, run_dir, prior=None):
    """Audited recovery after the old service was proven stopped; no fallback."""
    strict, phase, ab = deps.strict, deps.phase, deps.ab
    state = {}
    try:
        table = strict.scan()
        state['port_closed'] = phase.port() == 111
        phase.no_pending_clients(table, prior)
        state['clients_clear'] = True
        target = Path(deps.phase.TARGET)
        guard = Path(deps.phase.GUARD)
        head = sha(target) if target.is_file() else None
        guard_sha = sha(guard) if guard.is_file() else None
        state.update(head=head, guard=guard_sha)
        state['known'] = head in (F810_HEAD, AA3E_HEAD) and guard_sha in (GUARD_PRE, GUARD_POST)
        if not (state['port_closed'] and state['known']):
            state.update(action='blocked_service_or_source_unknown')
            write(run_dir / 'recovery.json', state)
            return False
        if guard_sha == GUARD_POST:
            ab.revert_service_check()
            state['adapter_reverted'] = sha(guard) == GUARD_PRE
        if sha(target) != F810_HEAD:
            ab.switch_source('f810', 'recover')
        state['source_f810'] = sha(target) == F810_HEAD
        require(state['source_f810'] and sha(guard) == GUARD_PRE, 'source/guard not clean')
        restored = ab.restore_prod()
        state.update(action='restored_prod', restored=restored)
        write(run_dir / 'recovery.json', state)
        return True
    except BaseException as exc:
        state.update(action=state.get('action', 'restore_failed'),
                     error=f'{type(exc).__name__}: {exc}')
        try:
            write(run_dir / 'recovery.json', state)
        except BaseException:
            pass
        return False


def deploy(a, deps):
    run_dir = a.run_dir.resolve()
    require(not run_dir.exists(), 'new run directory required')
    require(a.phase in CANDIDATE_PHASES or a.phase == 's1',
            'unknown candidate phase')
    require(a.from_phase == 's1',
            'every candidate phase deploys from the restored f810 s1 service')
    require(sha(deps.phase.TARGET) == F810_HEAD, 'deploy start source is not f810')
    require(sha(deps.phase.GUARD) == GUARD_PRE, 'deploy start guard is not the preimage')
    require(sha(a.current_restore) == a.current_restore_sha, 'old restore SHA mismatch')
    old = read(a.current_restore)
    require(all(old[k]['rc'] == 0 and old[k]['ok'] for k in FOUR_CHECKS),
            'old restore not complete')
    old_label = record_label(old)
    old['_from_phase'] = a.from_phase
    label = f'd0-cand-{utc_stamp()}-{uuid.uuid4().hex[:8]}'
    manifest = str(a.manifest.resolve())
    plan = dict(action='deploy', phase=a.phase, label=label, run_dir=str(run_dir),
                current_restore=str(a.current_restore), current_restore_sha=a.current_restore_sha,
                current_restore_label=old_label,
                manifest=manifest, manifest_sha=a.manifest_sha,
                source=F810_HEAD, source_after=AA3E_HEAD, guard=GUARD_PRE, guard_after=GUARD_POST,
                launch=['/bin/bash', str(deps.ab.args.launch), label,
                        *deps.phase.OVERRIDES[a.phase]],
                overrides=list(deps.phase.OVERRIDES[a.phase]),
                launch_env=dict(CANDIDATE_ENV), from_phase=a.from_phase,
                client_gate=('prior:' + str(a.prior_client_record)) if a.prior_client_record
                else 'first-phase-no-client')
    if not a.execute:
        return dict(plan=plan, status='plan_not_executed')
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / 'plan.json', plan)
    stop_state = {}
    try:
        deps.ab.stage_d0()
        deps.ab.stage_candidate()
        stop_and_gate(deps, run_dir, old, a.prior_client_record, stop_state)
        deps.ab.apply_service_check()
        deps.ab.switch_source('aa3e', 'deploy')
        require(sha(deps.phase.TARGET) == AA3E_HEAD, 'candidate head not installed')
        require(sha(deps.phase.GUARD) == GUARD_POST, 'candidate guard adapter not installed')
        # Full frozen-file check now covers the installed candidate head/adapter.
        deps.acceptance.frozen_check(deps.manifest)
        label, pid, ids, launch_manifest = launch_candidate(deps, run_dir, label, a.phase)
        log = four_check(deps, run_dir, label, pid,
                         {row['pid'] for row in [ids['root'], *ids['descendants']]})
        result = dict(ok=True, action='deploy', phase=a.phase, label=label, root=pid,
                      source=AA3E_HEAD, guard=GUARD_POST, flag_env={FLAG_ENV: '1', ALLOW_ENV: AA3E_HEAD},
                      identities=str(run_dir / 'ready-identities.json'),
                      launch_manifest=str(launch_manifest),
                      owner_evidence=str(run_dir / 'restore.json'),
                      manifest=manifest, manifest_sha=a.manifest_sha,
                      from_phase=a.from_phase, overrides=list(deps.phase.OVERRIDES[a.phase]))
        write(run_dir / 'complete.json', result)
        return result
    except BaseException:
        (run_dir / 'error.txt').write_text(traceback.format_exc())
        if stop_state.get('stopped'):
            recover(deps, run_dir, a.prior_client_record)
        raise


def verify_prior_supervision(record, record_rows, prior):
    """The prior supervision must be the last client group of exactly this
    candidate deploy: same protected root/label/identities/manifest/phase."""
    argv = prior.get('argv') or []
    require(isinstance(argv, list) and argv, 'prior client record lacks its argv')
    require(prior.get('clients_exited') is True, 'prior client supervision did not exit')
    require(argv_value(argv, '--phase') == record['phase'],
            'prior client supervision is not this candidate phase')
    require(Path(argv_value(argv, '--manifest') or '').resolve() == Path(record['manifest']).resolve(),
            'prior client supervision manifest mismatch')
    for flag, key in RECORD_BINDINGS:
        require(Path(argv_value(argv, flag) or '').resolve() == Path(record[key]).resolve(),
                f'prior client supervision {key} mismatch')
    root = record_rows[0]['pid']
    require(record.get('root') == root, 'deploy record root mismatch')
    observed = {str(key) for key in (prior.get('observed') or {})}
    require(str(root) not in observed, 'protected root appears in the prior client group')


def restore(a, deps):
    run_dir = a.run_dir.resolve()
    require(not run_dir.exists(), 'new run directory required')
    require(sha(a.deploy_record) == a.deploy_record_sha, 'deploy record SHA mismatch')
    record = read(a.deploy_record)
    require(record.get('action') == 'deploy' and record.get('ok') is True,
            'not a successful candidate deploy record')
    require(record.get('source') == AA3E_HEAD and record.get('guard') == GUARD_POST,
            'deploy record is not the aa3e candidate')
    require(record.get('phase') in PHASES and record.get('phase') == a.from_phase,
            'restore from-phase must equal the deployed candidate phase')
    manifest = str(a.manifest.resolve())
    require(record.get('manifest_sha') == a.manifest_sha,
            'deploy record manifest SHA mismatch')
    record_label(record)
    require(a.prior_client_record is not None, 'restore requires the last candidate client record')
    ids_file = Path(record['identities'])
    require(ids_file.is_file(), 'deployed ready identities missing')
    ids = read(ids_file)
    rows = [ids['root'], *ids['descendants']]
    prior = read(a.prior_client_record)
    recorded = record.get('manifest')
    bound = recorded if recorded is not None else argv_value(prior.get('argv') or [], '--manifest')
    require(Path(bound or '').resolve() == Path(manifest),
            'deploy record manifest binding mismatch')
    manifest_source = 'deploy_record' if recorded is not None else 'prior_supervision'
    verify_prior_supervision(dict(record, manifest=manifest), rows, prior)
    label = f'd0-restore-{utc_stamp()}-{uuid.uuid4().hex[:8]}'
    plan = dict(action='restore', phase=a.from_phase, label=label, run_dir=str(run_dir),
                deploy_record=str(a.deploy_record), deploy_record_sha=a.deploy_record_sha,
                deploy_label=record['label'], deploy_root=record['root'],
                manifest=manifest, manifest_sha=a.manifest_sha,
                manifest_source=manifest_source,
                ready_pids=[row['pid'] for row in rows], source=F810_HEAD, guard=GUARD_PRE,
                client_gate='prior:' + str(a.prior_client_record))
    if not a.execute:
        return dict(plan=plan, status='plan_not_executed')
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / 'plan.json', plan)
    stop_state = {}
    old = dict(label=record['label'],
               service_discovery={'api_identity': ids['root']}, _from_phase=a.from_phase)
    try:
        deps.ab.stage_d0()
        deps.ab.stage_candidate()
        stop_and_gate(deps, run_dir, old, a.prior_client_record, stop_state)
        guard = Path(deps.phase.GUARD)
        if sha(guard) == GUARD_POST:
            deps.ab.revert_service_check()
        require(sha(guard) == GUARD_PRE, 'original guard not restored')
        deps.ab.switch_source('f810', 'restore')
        require(sha(deps.phase.TARGET) == F810_HEAD, 'f810 source not restored')
        deps.acceptance.frozen_check(deps.base_manifest)
        restored = deps.ab.restore_prod()
        require(restored.get('restore_env', {}).get('match') is True,
                'clean restore OPP/LD compare not matched')
        label_dir = Path(deps.ab.args.service_log_root) / restored['label']
        pid = int((label_dir / 'api.pid').read_text())
        table = deps.strict.scan()
        pids = deps.phase.tree(table, pid)
        ids = dict(root=table[pid], descendants=[table[p] for p in sorted(pids - {pid})])
        write(run_dir / 'restored-identities.json', ids)
        check = deps.acceptance.service_check(
            deps.base_manifest, 's1', run_dir / 'restored-identities.json',
            label_dir / 'launch-manifest.json')
        write(run_dir / 'service-check.json', check)
        record_path = run_dir / 'restore.json'
        require(record_path.is_file(), 'clean restore record missing')
        restore_record = read(record_path)
        require(all(restore_record.get(k, {}).get('rc') == 0 and
                    restore_record[k].get('ok') is True for k in FOUR_CHECKS),
                'clean restore four-check incomplete')
        require(restore_record.get('service_discovery', {}).get('api_identity', {}).get('pid') == pid,
                'clean restore record root mismatch')
        result = dict(ok=True, action='restore', phase=a.from_phase, label=restored['label'],
                      source=F810_HEAD, guard=GUARD_PRE, deploy_record=str(a.deploy_record),
                      deploy_label=record['label'], deploy_root=record['root'],
                      manifest=manifest, manifest_sha=a.manifest_sha,
                      manifest_source=manifest_source,
                      restore_env=restored.get('restore_env'),
                      identities=str(run_dir / 'restored-identities.json'),
                      launch_manifest=str(label_dir / 'launch-manifest.json'),
                      owner_evidence=str(record_path),
                      current_restore=str(record_path),
                      current_restore_sha=sha(record_path))
        write(run_dir / 'complete.json', result)
        return result
    except BaseException:
        (run_dir / 'error.txt').write_text(traceback.format_exc())
        if stop_state.get('stopped'):
            recover(deps, run_dir, a.prior_client_record)
        raise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('deploy', 'restore'), required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--owner-code', type=Path, required=True)
    p.add_argument('--ab-code', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha', default=MANIFEST_SHA)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--phase', choices=PHASES, default='s1')
    p.add_argument('--from-phase', choices=PHASES, default='s1')
    p.add_argument('--current-restore', type=Path)
    p.add_argument('--current-restore-sha')
    p.add_argument('--deploy-record', type=Path)
    p.add_argument('--deploy-record-sha')
    client = p.add_mutually_exclusive_group(required=True)
    client.add_argument('--prior-client-record', type=Path)
    client.add_argument('--first-phase-no-client', action='store_true')
    p.add_argument('--tree', default='/workspace/yangchen/work/src/vllm')
    p.add_argument('--python', default='/workspace/yangchen/venvs/clef-py312/bin/python')
    p.add_argument('--activate', default='/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh')
    p.add_argument('--launch', default='/workspace/yangchen/logs/perf-wave2-20261118/launch.sh')
    p.add_argument('--wait', default='/workspace/yangchen/logs/perf-wave2-20261118/wait.sh')
    p.add_argument('--stop-script', default='/tmp/d0cc/d0_stop.py')
    p.add_argument('--service-log-root', default='/workspace/yangchen/logs/perf-wave2-20261118')
    p.add_argument('--clean-env-expect', type=Path,
                   default=Path('/workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json'))
    p.add_argument('--port', type=int, default=18143)
    p.add_argument('--stop-timeout', type=float, default=120.0)
    p.add_argument('--wait-timeout', type=float, default=900.0)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'deploy':
        require(a.from_phase == 's1',
                'every candidate phase deploys from the restored f810 s1 service')
        require(bool(a.current_restore) == bool(a.current_restore_sha),
                'deploy requires --current-restore and --current-restore-sha')
        require(a.current_restore is not None, 'deploy requires the current verified restore record')
        require(a.deploy_record is None, 'deploy must not take a deploy record')
    else:
        require(a.first_phase_no_client is False, 'restore does not accept first-phase mode')
        require(bool(a.deploy_record) == bool(a.deploy_record_sha),
                'restore requires --deploy-record and --deploy-record-sha')
        require(a.deploy_record is not None, 'restore requires the candidate deploy record')
        require(a.current_restore is None, 'restore must not take --current-restore')
    if a.first_phase_no_client:
        require(a.action == 'deploy' and a.from_phase == a.phase == 's1',
                'first phase mode requires a deploy from s1 to s1')
    deps = build_deps(a)
    result = deploy(a, deps) if a.action == 'deploy' else restore(a, deps)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
