#!/usr/bin/env python3
"""Bound one frozen final-acceptance client group; never signal the service.

Default is plan-only. Execution uses the reviewed 5c cleanup function in an
isolated module with strict identity/query adapters and pre/post inventories.
The adapters are part of this supervisor's review, distinct from the frozen
helper file. The service owner handles launch, stop and resource gates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HELPER_SHA = '9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5'
RUNNER_SHA = '7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab'
PYTHON = '/workspace/yangchen/venvs/clef-py312/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        argv = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        after = (path / 'stat').read_text()
        after_fields = after[after.rfind(')') + 2:].split()
    except FileNotFoundError:
        require(not path.exists(), f'partial /proc read: {path}')
        return None
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(bool(boot), 'boot identity unavailable')
    require(fields[19] == after_fields[19] and fields[2] == after_fields[2],
            f'identity changed during read: {pid}')
    return dict(pid=int(pid), ppid=int(after_fields[1]), pgid=int(after_fields[2]),
                starttime=after_fields[19], boot_id=boot, state=after_fields[0], cmdline=argv)


def scan():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        ident = read_identity(int(path.name))
        if ident is not None:
            result[ident['pid']] = ident
    return result


def same_process(a, b):
    return bool(a and b and all(a[k] == b[k] for k in ('pid', 'pgid', 'starttime', 'boot_id')))


def same_signal_target(a, b):
    return same_process(a, b) and a['cmdline'] == b['cmdline']


def load_cleanup_helper(path):
    require(sha(path) == HELPER_SHA, 'cleanup helper hash mismatch')
    spec = importlib.util.spec_from_file_location('strict_client_cleanup', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Keep the frozen function body; replace its query dependencies explicitly.
    # This avoids its original permissive OSError -> None conversion.
    helper.proc_identity = read_identity
    # Full cmdline is checked for the whole group immediately before cleanup.
    # During KILL teardown cmdline may become empty before state becomes Z;
    # the frozen loop therefore uses kernel lifetime identity, plus PGID.
    helper.identity_matches = same_process
    helper.group_members = lambda pgid: {p: i for p, i in scan().items() if i['pgid'] == pgid}
    return helper


def supervise(argv, seconds, records, helper, protected_pids=(), *, deadline_monotonic=None):
    deadline = deadline_monotonic if deadline_monotonic is not None else time.monotonic() + seconds
    require(seconds > 0 and time.monotonic() < deadline, 'deadline already elapsed')
    records.mkdir(parents=True, exist_ok=False)
    state = dict(schema='final-client-supervision/1', started_utc=now(), argv=argv,
                 timeout_seconds=seconds, deadline_monotonic=deadline,
                 status='starting', observed={}, errors=[])
    record = records / 'supervision.json'
    write(record, state)
    proc = None
    saved = {}
    leader = None
    timed_out = False
    allowed_cleanup = True
    try:
        with (records / 'client.stdout').open('xb') as stdout, (records / 'client.stderr').open('xb') as stderr:
            require(time.monotonic() < deadline, 'deadline elapsed before client launch')
            proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            state['client_pid'] = proc.pid
            timed_out = time.monotonic() >= deadline
            while True:
                inventory = scan()
                live = inventory.get(proc.pid)
                group = {pid: item for pid, item in inventory.items() if item['pgid'] == proc.pid}
                require(not set(group).intersection(protected_pids), 'service pid found in client group')
                require(proc.pid != os.getpgrp(), 'client group equals supervisor group')
                if leader is None and live is not None:
                    require(live['pgid'] == proc.pid, 'client did not create its own group')
                    leader = dict(live)
                    saved[proc.pid] = dict(live)
                if live is not None and live['state'] != 'Z':
                    require(same_process(leader, live), 'client leader identity changed')
                    require(live['cmdline'] == leader['cmdline'], 'client leader command changed')
                    reachable = {proc.pid}
                    while True:
                        found = {pid for pid, item in inventory.items() if item['ppid'] in reachable}
                        if found.issubset(reachable):
                            break
                        reachable.update(found)
                    require(all(pid in reachable or pid in saved for pid in group), 'unattributed client group member')
                    require(all(inventory[pid]['pgid'] == proc.pid for pid in reachable), 'client child escaped group')
                    for pid, item in group.items():
                        require(pid not in saved or same_process(saved[pid], item), 'observed client pid reused')
                        saved[pid] = dict(item)
                else:
                    require(all(pid in saved and same_process(saved[pid], item) for pid, item in group.items()),
                            'unknown group member after leader exit')
                state['observed'] = {str(pid): item for pid, item in saved.items()}
                state['status'] = 'running'
                write(record, state)
                if timed_out or time.monotonic() >= deadline:
                    timed_out = True
                    break
                if proc.poll() is not None:
                    break
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    except BaseException:
        state['errors'].append(traceback.format_exc())
        # An unknown inventory/identity must never be converted to an empty set.
        allowed_cleanup = False
    finally:
        if proc is not None and allowed_cleanup:
            try:
                require(leader is not None or proc.poll() is not None, 'no client leader identity')
                fresh = scan()
                current_group = {pid: item for pid, item in fresh.items() if item['pgid'] == proc.pid and item['state'] != 'Z'}
                require(not set(current_group).intersection(protected_pids), 'protected pid in cleanup group')
                require(all(pid in saved and same_signal_target(saved[pid], item) for pid, item in current_group.items()),
                        'unknown or changed cleanup group member')
                ordered = ([saved[proc.pid]] if proc.pid in saved else []) + [saved[p] for p in sorted(saved) if p != proc.pid]
                cleanup = helper.Window._probe_cleanup(object.__new__(helper.Window), proc, ordered, timed_out)
                final = scan()
                remaining = [item for item in final.values() if item['pgid'] == proc.pid and item['state'] != 'Z']
                state['cleanup'] = cleanup
                state['remaining_group'] = remaining
                state['clients_exited'] = bool(cleanup['reaped'] and not cleanup['unknown'] and not cleanup['residual'] and not remaining)
            except BaseException:
                state['errors'].append(traceback.format_exc())
                state['clients_exited'] = False
        else:
            state['clients_exited'] = proc is None
        state['timed_out'] = timed_out
        state['client_returncode'] = proc.poll() if proc is not None else None
        state['finished_utc'] = now()
        state['status'] = ('complete' if state['clients_exited'] and not timed_out and
                           not state['errors'] and state['client_returncode'] == 0 else 'failed')
        write(record, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha', required=True)
    p.add_argument('--phase', choices=('s1', 's2_b8', 's2_b16', 'erqa'), required=True)
    p.add_argument('--single-baseline-round', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--records-dir', type=Path, required=True)
    p.add_argument('--identities', type=Path, required=True)
    p.add_argument('--launch-manifest', type=Path, required=True)
    p.add_argument('--owner-evidence', type=Path, required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--helper', type=Path, default=Path('/workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py'))
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    runner = a.code.resolve() / 'runtime/final_acceptance.py'
    require(sha(runner) == RUNNER_SHA, 'acceptance runner hash mismatch')
    require(sha(a.manifest) == a.manifest_sha, 'manifest hash mismatch')
    require(sha(a.helper) == HELPER_SHA, 'cleanup helper hash mismatch')
    require(not a.output_dir.exists() and not a.records_dir.exists(), 'fresh output and supervision directories required')
    deadline = datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
    require(deadline.tzinfo is not None, 'deadline must include UTC offset')
    anchor_monotonic = time.monotonic()
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    deadline_monotonic = anchor_monotonic + seconds
    require(seconds > 0, 'deadline already elapsed')
    argv = [PYTHON, str(runner), '--manifest', str(a.manifest.resolve()), '--phase', a.phase,
            '--output-dir', str(a.output_dir.resolve()), '--identities', str(a.identities.resolve()),
            '--launch-manifest', str(a.launch_manifest.resolve()), '--owner-evidence', str(a.owner_evidence.resolve()), '--execute']
    if a.single_baseline_round:
        require(a.phase == 's1', 'single baseline round requires s1')
        argv.append('--single-baseline-round')
    if not a.execute:
        print(json.dumps(dict(status='plan_not_executed', argv=argv, deadline_utc=deadline.isoformat(), timeout_seconds=seconds), indent=2))
        return 0
    ids = json.loads(a.identities.read_text())
    rows = ids if isinstance(ids, list) else [ids['root'], *ids['descendants']]
    protected = {row['pid'] for row in rows}
    helper = load_cleanup_helper(a.helper)
    state = supervise(argv, seconds, a.records_dir, helper, protected,
                      deadline_monotonic=deadline_monotonic)
    print(json.dumps({k: state[k] for k in ('status', 'client_returncode', 'timed_out', 'clients_exited', 'errors')}, indent=2))
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
