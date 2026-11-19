# Review packet: candidate S1 minimal combo (stage 640ac23)

Status: **CPU review evidence, device not executed**. This packet is self-contained for independent audit by 32. It does not authorize device execution; 31 remains the only service owner. Frozen f810 entry (`final_service_phase.py` `888a8d7f…`, supervisor `172c2838…`, cc7 runner/manifest) is untouched.

## Errata rev2 (32 finding: call parameter only)

The earlier example used `--repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb`; **the correct value is the bundle root** `/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769`. `stage_candidate()` and `stage_d0()` copy from `repo/runtime`; the bundle root holds all four pinned artifacts (`b09cd602…` fusion installer, `1ed35f0a…` guardadapter, `aa3e1758…` postimage, `36eaff44…` guard postimage) and the six pinned d0 modules. `5c045eb` is still correct only for `--helper joint_fused_attention_window.py` (`9245e1a6…`).

31's failed attempt reached `stage_d0/stage_candidate` and failed **before** the stop gate: no signal, source or guard change; the service tree survived. Code, tests, manifest and gates are unchanged (`candidate_service_phase.py` still `1f98dba9…`); this revision only corrects the call parameter and adds this errata. The code-identical execution stage is the commit containing this packet revision; verify with section 0/5.

Preflight before any deploy (any mismatch blocks):

```bash
B=/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769
sha256sum "$B/runtime/apply_joint_fused_attention.py" \
          "$B/runtime/apply_joint_service_check.py" \
          "$B/runtime/joint_fused_attention/clef_joint_schema.py" \
          "$B/runtime/joint_service_check/apply_joint_single_field_patch.py"
# b09cd602… / 1ed35f0a… / aa3e1758… / 36eaff44…
```

## 0. Frozen identities

| Object | Path | SHA-256 |
| --- | --- | --- |
| code stage (original) | `/workspace/yangchen/work/final-acceptance-cc34-640ac23cac9aaa9a937832c2361dddd74007d195` | commit `640ac23` |
| docs-corrected stage | commit containing this rev2 packet (code/manifest SHAs identical to 640ac23) | path in 34 steer |
| candidate entry | stage `runtime/candidate_service_phase.py` | `1f98dba90c4cb466fd3ae3804ab1dacbaee8bd9799b8ef2110e22bc5489164d7` |
| candidate manifest | stage `runtime/final_acceptance_candidate_manifest.json` | `31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4` |
| manifest generator | stage `runtime/candidate_manifest.py` | `6348ab527ce03e025263e23f6745a304351e8ab72e34ce04d4af3f43f256c2b5` |
| diff report | stage `runtime/final_acceptance_candidate_manifest.diff.json` | `646ba8d89a07c9a647bea1a6429b617d22b409fb6aa9ca9f68b1fc37c6c9a9ab` |
| phase/supervisor gates | d4464ac stage / 063109a stage | `888a8d7f…` / `172c2838…` |
| source-control base | ebca runtime | `joint_fused_source_control.py a1c502a3…`, `joint_fused_service_ab.py 127e47ae…`, window `700cf10c…` |
| installers | ebca runtime | fusion `b09cd602…`, guardadapter `1ed35f0a…` |
| stop/launch/wait/activate | original | `d6d49d75…` / `a7373861…` / `e3fb0888…` / `3b363394…` |

## 1. What is new and what is reused

New files only (f810 files 0 diff):
- `runtime/candidate_manifest.py` — deterministic 13-leaf transform from cc7 `0338f87b…` to candidate `31202d27…`; rejects any other drift.
- `runtime/candidate_service_phase.py` — minimal `deploy`/`restore` orchestration for the candidate S1. No new stop, signal, source-install or guard logic: it calls the audited `d0_stop`, original launcher/wait, both audited installers with owned manifests, frozen phase-v3 PGID/client/port/NPU gates, cc7 `service_check`, and d0 four-check.
- Tests + `runtime/verify_final_acceptance_combo.py` + derived manifest/diff.
- No four-phase candidate framework; the CLI accepts `--phase` but this round only S1 is frozen and reviewed.

## 2. Entry full file: `runtime/candidate_service_phase.py`

```python
#!/usr/bin/env python3
"""Owner-side deploy/restore of the exact aa3e candidate acceptance service.

Default is plan-only.  This is the minimal compatibility entry for the candidate
arm of the final acceptance; production f810 keeps using the frozen
`final_service_phase.py` unchanged.  It adds no stop, signal, source-install or
guard logic of its own: every device action is an already-audited primitive.

  stop            /tmp/d0cc/d0_stop.py through the frozen phase gates
  guard adapter   audited apply_joint_service_check.py (preimage/postimage pinned)
  source switch   audited apply_joint_fused_attention.py with an owned manifest
  launch/wait     original perf-wave2 launch.sh / wait.sh, clean two-key parent
                  plus only CLEF_JOINT_FUSED_ATTN=1 and the exact allow SHA
  verification    cc7 final_acceptance.service_check with the derived candidate
                  manifest, then the original d0 four-check and NPU card gate
  restore         audited adapter revert -> f810 source -> clean production
                  launch -> four-check -> byte compare of OPP/LD against the
                  frozen clean-launch manifest

Any error after the old service was proven stopped goes through an explicit
recovery gate: port closed, no pending client runners, known source/guard state.
Only then is the audited restore attempted; otherwise the run is recorded as
blocked and nothing is signalled or launched.
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


def build_deps(a):
    owner = a.owner_code.resolve()
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
    return SimpleNamespace(strict=strict, phase=phase, ab=ab, acceptance=acceptance,
                           manifest=read(a.manifest), base_manifest=read(a.base_manifest))


def verify_old_service(deps, old, from_phase):
    """Frozen phase-v3 gates for the service that is about to be replaced."""
    strict, require_ = deps.strict, require
    saved = old['service_discovery']['api_identity']
    root = saved['pid']
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
    require(sha(a.current_restore) == a.current_restore_sha, 'old restore SHA mismatch')
    old = read(a.current_restore)
    require(all(old[k]['rc'] == 0 and old[k]['ok'] for k in ('health', 'workers', 'text', 'image')),
            'old restore not complete')
    old['_from_phase'] = a.from_phase or 's1'
    label = f'd0-cand-{utc_stamp()}-{uuid.uuid4().hex[:8]}'
    plan = dict(action='deploy', label=label, run_dir=str(run_dir),
                current_restore=str(a.current_restore), current_restore_sha=a.current_restore_sha,
                manifest_sha=a.manifest_sha, source=AA3E_HEAD, guard=GUARD_POST,
                launch=['/bin/bash', str(deps.ab.args.launch), label,
                        *deps.phase.OVERRIDES[a.phase]],
                launch_env=dict(CANDIDATE_ENV), from_phase=old['_from_phase'],
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
                      manifest_sha=a.manifest_sha)
        write(run_dir / 'complete.json', result)
        return result
    except BaseException:
        (run_dir / 'error.txt').write_text(traceback.format_exc())
        if stop_state.get('stopped'):
            recover(deps, run_dir, a.prior_client_record)
        raise


def restore(a, deps):
    run_dir = a.run_dir.resolve()
    require(not run_dir.exists(), 'new run directory required')
    require(sha(a.deploy_record) == a.deploy_record_sha, 'deploy record SHA mismatch')
    record = read(a.deploy_record)
    require(record.get('action') == 'deploy' and record.get('ok') is True,
            'not a successful candidate deploy record')
    require(record.get('source') == AA3E_HEAD and record.get('guard') == GUARD_POST,
            'deploy record is not the aa3e candidate')
    require(record.get('phase') == a.from_phase,
            'restore from-phase must equal the deployed candidate phase')
    require(a.prior_client_record is not None, 'restore requires the last candidate client record')
    ids_file = Path(record['identities'])
    require(ids_file.is_file(), 'deployed ready identities missing')
    ids = read(ids_file)
    rows = [ids['root'], *ids['descendants']]
    label = f'd0-restore-{utc_stamp()}-{uuid.uuid4().hex[:8]}'
    plan = dict(action='restore', label=label, run_dir=str(run_dir),
                deploy_record=str(a.deploy_record), deploy_record_sha=a.deploy_record_sha,
                ready_pids=[row['pid'] for row in rows], source=F810_HEAD, guard=GUARD_PRE,
                client_gate='prior:' + str(a.prior_client_record))
    if not a.execute:
        return dict(plan=plan, status='plan_not_executed')
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / 'plan.json', plan)
    stop_state = {}
    old = dict(service_discovery={'api_identity': ids['root']}, _from_phase=a.from_phase)
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
        result = dict(ok=True, action='restore', label=restored['label'],
                      source=F810_HEAD, guard=GUARD_PRE, deploy_record=str(a.deploy_record),
                      restore_env=restored.get('restore_env'),
                      identities=str(run_dir / 'restored-identities.json'),
                      launch_manifest=str(label_dir / 'launch-manifest.json'),
                      owner_evidence=str(run_dir / 'restore.json'))
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
```

## 3. CPU boundary logs (25 tests, stage itself)

Command (run on the remote Linux host in the stage):

```bash
cd /workspace/yangchen/work/final-acceptance-cc34-640ac23cac9aaa9a937832c2361dddd74007d195/runtime
/workspace/yangchen/venvs/clef-py312/bin/python -m unittest -v test_candidate_service_phase test_candidate_manifest
```

```
test_deploy_requires_current_restore (test_candidate_service_phase.CliTest.test_deploy_requires_current_restore) ... ok
test_restore_rejects_first_phase_mode (test_candidate_service_phase.CliTest.test_restore_rejects_first_phase_mode) ... ok
test_deploy_erqa_uses_frozen_overrides (test_candidate_service_phase.DeployTest.test_deploy_erqa_uses_frozen_overrides) ... ok
test_deploy_success_binds_candidate (test_candidate_service_phase.DeployTest.test_deploy_success_binds_candidate) ... ok
test_incomplete_old_restore_blocks_before_stop (test_candidate_service_phase.DeployTest.test_incomplete_old_restore_blocks_before_stop) ... ok
test_launch_failure_recovers_to_f810 (test_candidate_service_phase.DeployTest.test_launch_failure_recovers_to_f810) ... ok
test_launch_failure_with_open_port_blocks (test_candidate_service_phase.DeployTest.test_launch_failure_with_open_port_blocks) ... ok
test_pending_client_blocks (test_candidate_service_phase.DeployTest.test_pending_client_blocks) ... ok
test_plan_only_does_not_mutate (test_candidate_service_phase.DeployTest.test_plan_only_does_not_mutate) ... ok
test_stop_failure_stops_before_install (test_candidate_service_phase.DeployTest.test_stop_failure_stops_before_install) ... ok
test_unattributed_pgid_member_blocks (test_candidate_service_phase.DeployTest.test_unattributed_pgid_member_blocks) ... ok
test_unknown_source_after_stop_blocks (test_candidate_service_phase.DeployTest.test_unknown_source_after_stop_blocks) ... ok
test_restore_adapter_failure_recovers_without_launch_when_unknown (test_candidate_service_phase.RestoreTest.test_restore_adapter_failure_recovers_without_launch_when_unknown) ... ok
test_restore_rejects_phase_mismatch (test_candidate_service_phase.RestoreTest.test_restore_rejects_phase_mismatch) ... ok
test_restore_rejects_wrong_record (test_candidate_service_phase.RestoreTest.test_restore_rejects_wrong_record) ... ok
test_restore_stop_failure_does_not_launch (test_candidate_service_phase.RestoreTest.test_restore_stop_failure_does_not_launch) ... ok
test_restore_success_returns_production (test_candidate_service_phase.RestoreTest.test_restore_success_returns_production) ... ok
test_base_forbidden_requires_flag (test_candidate_manifest.DeriveTest.test_base_forbidden_requires_flag) ... ok
test_base_must_be_f810 (test_candidate_manifest.DeriveTest.test_base_must_be_f810) ... ok
test_base_must_be_preimage_guard (test_candidate_manifest.DeriveTest.test_base_must_be_preimage_guard) ... ok
test_derive_binds_candidate_exactly (test_candidate_manifest.DeriveTest.test_derive_binds_candidate_exactly) ... ok
test_derive_is_deterministic (test_candidate_manifest.DeriveTest.test_derive_is_deterministic) ... ok
test_drift_outside_whitelist_rejected (test_candidate_manifest.DeriveTest.test_drift_outside_whitelist_rejected) ... ok
test_extra_phase_env_rejected (test_candidate_manifest.DeriveTest.test_extra_phase_env_rejected) ... ok
test_frozen_base_sha_and_candidate_sha (test_candidate_manifest.FrozenManifestTest.test_frozen_base_sha_and_candidate_sha) ... skipped 'real frozen manifest not provided'

----------------------------------------------------------------------
Ran 25 tests in 0.041s

OK (skipped=1)
```

Real frozen base manifest test (8/8, candidate SHA recomputed):

```bash
CLEF_BASE_MANIFEST=/workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  /workspace/yangchen/venvs/clef-py312/bin/python -m unittest -v test_candidate_manifest
```

```
test_base_forbidden_requires_flag (test_candidate_manifest.DeriveTest.test_base_forbidden_requires_flag) ... ok
test_base_must_be_f810 (test_candidate_manifest.DeriveTest.test_base_must_be_f810) ... ok
test_base_must_be_preimage_guard (test_candidate_manifest.DeriveTest.test_base_must_be_preimage_guard) ... ok
test_derive_binds_candidate_exactly (test_candidate_manifest.DeriveTest.test_derive_binds_candidate_exactly) ... ok
test_derive_is_deterministic (test_candidate_manifest.DeriveTest.test_derive_is_deterministic) ... ok
test_drift_outside_whitelist_rejected (test_candidate_manifest.DeriveTest.test_drift_outside_whitelist_rejected) ... ok
test_extra_phase_env_rejected (test_candidate_manifest.DeriveTest.test_extra_phase_env_rejected) ... ok
test_frozen_base_sha_and_candidate_sha (test_candidate_manifest.FrozenManifestTest.test_frozen_base_sha_and_candidate_sha) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.005s

OK
```

Real CLI plan-only deploy/restore + wrong-manifest block (no service touched, no run dir created):

```
{
 "deploy_status": "plan_not_executed",
 "deploy_plan": {
  "action": "deploy",
  "label": "d0-cand-20261119T003054Z-a59d0c31",
  "run_dir": "/tmp/cc34-revpkt-2YZb4G/deploy-run",
  "current_restore": "/tmp/cc34-revpkt-2YZb4G/old-restore.json",
  "current_restore_sha": "9b3a68076aa49f17f6e72b20faadb727b6688c243f30f793602d88d865a0e846",
  "manifest_sha": "31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4",
  "source": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503",
  "guard": "36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7",
  "launch": [
   "/bin/bash",
   "/workspace/yangchen/logs/perf-wave2-20261118/launch.sh",
   "d0-cand-20261119T003054Z-a59d0c31"
  ],
  "launch_env": {
   "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
   "CLEF_LAUNCH_CONFIG": "s1",
   "CLEF_JOINT_FUSED_ATTN": "1",
   "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
  },
  "from_phase": "s1",
  "client_gate": "first-phase-no-client"
 },
 "restore_status": "plan_not_executed",
 "restore_plan": {
  "action": "restore",
  "label": "d0-restore-20261119T003055Z-f28d1818",
  "run_dir": "/tmp/cc34-revpkt-2YZb4G/restore-run",
  "deploy_record": "/tmp/cc34-revpkt-2YZb4G/deploy-complete.json",
  "deploy_record_sha": "951ae3c29d73907b5b7871be58dc990b38365db2e2ce60d23582511675ae88b7",
  "ready_pids": [
   1,
   2,
   3
  ],
  "source": "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863",
  "guard": "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9",
  "client_gate": "prior:/tmp/cc34-revpkt-2YZb4G/supervision.json"
 },
 "bad_manifest_rc": 1,
 "bad_manifest_stderr_tail": [
  "    raise RuntimeError(message)",
  "RuntimeError: candidate manifest SHA mismatch"
 ]
}
```

## 4. Manifest machine diff (13 leaves)

Generator check command:

```bash
/workspace/yangchen/venvs/clef-py312/bin/python "$STAGE/runtime/candidate_manifest.py" \
  --base /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  --out "$STAGE/runtime/final_acceptance_candidate_manifest.json" --check
# -> {"status": "verified", "candidate_sha256": "31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4"}
```

Full diff report `runtime/final_acceptance_candidate_manifest.diff.json`:

```json
{
  "schema": "candidate-manifest-diff/1",
  "base_sha256": "0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2",
  "candidate_sha256": "31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4",
  "indexes": {
    "head_index": 5,
    "guard_index": 47
  },
  "changed": [
    "forbidden_environment",
    "phases.erqa.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA",
    "phases.erqa.service_environment.CLEF_JOINT_FUSED_ATTN",
    "phases.s1.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA",
    "phases.s1.service_environment.CLEF_JOINT_FUSED_ATTN",
    "phases.s2_b16.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA",
    "phases.s2_b16.service_environment.CLEF_JOINT_FUSED_ATTN",
    "phases.s2_b8.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA",
    "phases.s2_b8.service_environment.CLEF_JOINT_FUSED_ATTN",
    "source_files[47].sha256",
    "source_files[5].sha256",
    "source_selection.guard_sha",
    "source_selection.production_head_sha"
  ],
  "values": {
    "forbidden_environment": {
      "before": [
        "CLEF_JOINT_FUSED_ATTN",
        "CLEF_JOINT_SMALLQ",
        "CLEF_JOINT_SMALLQ_KERNEL",
        "CLEF_GDN_FLAG_FILE",
        "CLEF_PROFILE_MARKERS",
        "CLEF_JOINT_TIMING",
        "CLEF_PROFILE_OBSERVE_DIR",
        "CLEF_PROFILE_SCOPE_FILTER",
        "CLEF_GDN_NORM_BLOCK_M"
      ],
      "after": [
        "CLEF_JOINT_SMALLQ",
        "CLEF_JOINT_SMALLQ_KERNEL",
        "CLEF_GDN_FLAG_FILE",
        "CLEF_PROFILE_MARKERS",
        "CLEF_JOINT_TIMING",
        "CLEF_PROFILE_OBSERVE_DIR",
        "CLEF_PROFILE_SCOPE_FILTER",
        "CLEF_GDN_NORM_BLOCK_M"
      ]
    },
    "phases.erqa.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": {
      "before": "<absent>",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    },
    "phases.erqa.service_environment.CLEF_JOINT_FUSED_ATTN": {
      "before": "<absent>",
      "after": "1"
    },
    "phases.s1.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": {
      "before": "<absent>",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    },
    "phases.s1.service_environment.CLEF_JOINT_FUSED_ATTN": {
      "before": "<absent>",
      "after": "1"
    },
    "phases.s2_b16.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": {
      "before": "<absent>",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    },
    "phases.s2_b16.service_environment.CLEF_JOINT_FUSED_ATTN": {
      "before": "<absent>",
      "after": "1"
    },
    "phases.s2_b8.service_environment.CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": {
      "before": "<absent>",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    },
    "phases.s2_b8.service_environment.CLEF_JOINT_FUSED_ATTN": {
      "before": "<absent>",
      "after": "1"
    },
    "source_files[47].sha256": {
      "before": "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9",
      "after": "36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7"
    },
    "source_files[5].sha256": {
      "before": "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    },
    "source_selection.guard_sha": {
      "before": "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9",
      "after": "36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7"
    },
    "source_selection.production_head_sha": {
      "before": "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863",
      "after": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
    }
  }
}
```

The 13 leaves are exactly: `source_files[5].sha256` (head), `source_files[47].sha256` (guard), `source_selection.production_head_sha`, `source_selection.guard_sha`, `forbidden_environment`, and `CLEF_JOINT_FUSED_ATTN` + `CLEF_JOINT_ALLOWED_POSTIMAGE_SHA` in each of the four phase environments. Data, denominators, author scorer, 2pp gates, argv, effective config and OPP/LD are byte-identical.

## 5. Local prep + remote pin verification

Command:

```bash
python3 runtime/verify_final_acceptance_combo.py --stage . --ssh-host clef --json
```

```
{
  "schema": "final-acceptance-combo-check/1",
  "stage": "/Users/chenclay/.ao/data/worktrees/clef-27b-ascend/clef-27b-ascend-34",
  "local": {
    "checked": 8,
    "failed": 0,
    "ok": true,
    "mismatches": []
  },
  "remote": {
    "checked": 13,
    "failed": 0,
    "ok": true,
    "mismatches": []
  },
  "ok": true
}
```

8 local prep files and 13 remote stage/runtime pins all match (0 failures). The 13 remote pins are the f810 runner/manifest, supervisor, phase, source-control base, both installers, cleanup window, stop, launch, wait, activate and the 12,933 B clean expected manifest.

## 6. Safety gates kept

1. Original `d0_stop.py` only; identity-bound saved root; no pgrep/wide kill.
2. Pre-stop: full `/proc` scan, old root PID/PGID/starttime/boot/argv, full PGID membership, no pending runner, port open, NPU card 7 rows subset.
3. Post-stop: old tree and whole PGID have no non-Z member, port 111, card 7 empty, clients clear.
4. Candidate launch: clean two-key parent + only `CLEF_JOINT_FUSED_ATTN=1` and `CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e…`; identical launcher and per-phase overrides.
5. Ready: new root identity stable, PGID tree complete, full argv/env vs candidate manifest, health/info, d0 four-check, TP2 dual worker mapping.
6. Any failure after a proven stop: recovery only if port 111, clients clear, source/guard known; otherwise writes `blocked` and launches/signals nothing. Recovery reverts adapter, restores f810, clean-launches production, four-checks, byte-compares OPP/LD.
7. `--single-baseline-round` never accepted for candidate; S1 is 3 rounds.

## 7. Reproduction checklist for 32

```bash
STAGE=/workspace/yangchen/work/final-acceptance-cc34-640ac23cac9aaa9a937832c2361dddd74007d195
cd "$STAGE/runtime"
/workspace/yangchen/venvs/clef-py312/bin/python -m unittest -v test_candidate_service_phase test_candidate_manifest
CLEF_BASE_MANIFEST=/workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json   /workspace/yangchen/venvs/clef-py312/bin/python -m unittest -v test_candidate_manifest
cd "$STAGE" && /workspace/yangchen/venvs/clef-py312/bin/python runtime/verify_final_acceptance_combo.py --stage . --json
sha256sum runtime/candidate_service_phase.py runtime/final_acceptance_candidate_manifest.json runtime/candidate_manifest.py runtime/final_acceptance_candidate_manifest.diff.json
```

Expected: 25 tests OK (1 skipped), 8 real-base tests OK, prep 8/8, SHA list matches section 0.

## 8. Known boundaries

- Device execution is not performed here; the packet proves CPU behavior and frozen identity only.
- `deploy` runs `frozen_check` after the candidate is installed (guard post/head aa3e); a pre-stop check against the f810 manifest is not run, so an environment/data drift would be caught after stop and trigger the audited recovery path. This is deliberate: the recovery path is safer than a second preflight variant, and the frozen `frozen_check` is the same one used by phase v3.
- The candidate restore always restores f810 s1; it does not restore a candidate winner. If candidate S1 shows stable gain, the four-phase variant and winner restore need a separate frozen stage.
- No S2/ERQA candidate execution is prepared in this round.

## 9. Feedback protocol

Report only concrete findings with file/line or command/output evidence. For an accepted finding, 34 will make the minimal fix in a new independent stage/SHA under `/workspace/yangchen/work/final-acceptance-cc34-<newcommit>/`, regenerate this packet, and never modify the frozen f810 entry or the locked 640ac23 stage. 31 executes only after 32 closes and 20 schedules.

