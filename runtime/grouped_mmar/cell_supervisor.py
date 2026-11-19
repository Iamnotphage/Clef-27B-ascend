#!/usr/bin/env python3
"""Source-aware pure supervision window for one probe child process group.

Runs exactly one command in a new session/PGID with a hard timeout, refuses to
signal any protected service PID, verifies the source tree is byte-unchanged
(aa3e), and optionally runs 31's restore/post-check commands afterwards.  It
never stops, starts or signals the service itself; service lifecycle stays
with 31.  Plan-only by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROTECTED_ERROR = 'blocked_protected_in_child_group'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_pids(values):
    out = set()
    for value in values or ():
        for item in str(value).split(','):
            item = item.strip()
            if item:
                out.add(int(item))
    return out


def parse_argv(raw, name):
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError(f'{name} must be a non-empty JSON string array')
    return value


def pgid_members(pgid):
    members = {}
    proc = Path('/proc')
    if not proc.is_dir():
        return members
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
        except (OSError, IndexError):
            continue
        if len(fields) < 3:
            continue
        state, ppid, pgrp = fields[0], int(fields[1]), int(fields[2])
        if pgrp == pgid and state != 'Z':
            members[int(entry.name)] = {'pid': int(entry.name), 'ppid': ppid,
                                        'pgid': pgrp, 'state': state}
    return members


def cleanup_group(pgid, members, protected, kill=os.killpg, grace=2.0,
                  wait=None, sleeper=time.sleep):
    """Kill the child group only when no protected PID is inside it."""
    if set(members) & set(protected):
        return {'killed': False, 'reason': PROTECTED_ERROR,
                'intersection': sorted(set(members) & set(protected))}
    kill(pgid, signal.SIGTERM)
    sleeper(grace)
    if wait is not None:
        wait(grace)
    remaining = pgid_members(pgid)
    if remaining:
        kill(pgid, signal.SIGKILL)
        if wait is not None:
            wait(grace)
        remaining = pgid_members(pgid)
    return {'killed': True, 'reason': None, 'remaining': sorted(remaining)}


def run_aux(argv, timeout, log_path):
    try:
        with Path(log_path).open('xb') as log:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        return {'argv': argv, 'rc': proc.returncode, 'log': str(log_path)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'rc': None, 'log': str(log_path),
                'error': f'{type(exc).__name__}: {exc}'}


def build_plan(a):
    if not Path(a.source_target).is_absolute():
        raise ValueError('source-target must be absolute')
    if not Path(a.source_target).is_file():
        raise ValueError(f'source-target missing: {a.source_target}')
    actual = sha256(a.source_target)
    if actual != a.expected_source_sha:
        raise ValueError(f'source SHA mismatch: {actual} != {a.expected_source_sha}')
    if not a.command:
        raise ValueError('no child command supplied (use -- <argv...>)')
    if not Path(a.run_dir).is_absolute():
        raise ValueError('run-dir must be absolute')
    if a.cwd is not None and not Path(a.cwd).is_absolute():
        raise ValueError('cwd must be absolute')
    return {
        'schema': 'grouped-mmar-cell-plan/1',
        'command': list(a.command), 'cwd': a.cwd, 'timeout_seconds': a.timeout,
        'source_target': str(a.source_target), 'expected_source_sha256': a.expected_source_sha,
        'source_sha256_before': actual,
        'protected_pids': sorted(parse_pids(a.protected_pid)),
        'restore_argv': parse_argv(a.restore_argv, 'restore-argv'),
        'post_check_argv': parse_argv(a.post_check_argv, 'post-check-argv'),
        'run_dir': str(Path(a.run_dir).resolve()),
        'no_service_lifecycle': True,
        'notes': ['child runs in its own session/PGID', 'protected PIDs are never signalled',
                  'source SHA must be identical before and after', 'service lifecycle stays with 31'],
    }


def execute(a, plan):
    run_dir = Path(plan['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=False)
    protected = set(plan['protected_pids'])
    result = {'schema': 'grouped-mmar-cell-result/1', 'plan': plan,
              'started_monotonic_ns': time.monotonic_ns(), 'ok': False}
    state_path = run_dir / 'cell.json'
    flush = lambda: (state_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n'))

    def fail(message):
        result['error'] = message
        result['finished_monotonic_ns'] = time.monotonic_ns()
        flush()
        return 1

    before = sha256(a.source_target)
    result['source_sha256_before'] = before
    if before != a.expected_source_sha:
        return fail(f'source changed before child: {before}')
    flush()
    stdout = run_dir / 'child.stdout'
    stderr = run_dir / 'child.stderr'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        proc = subprocess.Popen(list(a.command), cwd=a.cwd, stdout=out, stderr=err,
                                start_new_session=True)
    result.update(child_pid=proc.pid, child_pgid=proc.pid)
    if proc.pid in protected:
        return fail('child root is a protected PID')
    try:
        if os.getpgid(proc.pid) != proc.pid:
            return fail('child did not create its own process group')
    except OSError as exc:
        return fail(f'child group lookup failed: {exc}')
    timed = False
    try:
        rc = proc.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        timed = True
        members = pgid_members(proc.pid)
        cleanup = cleanup_group(proc.pid, members, protected, wait=proc.wait)
        result['cleanup'] = cleanup
        if cleanup['reason'] == PROTECTED_ERROR:
            return fail(PROTECTED_ERROR)
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rc = None
    members = pgid_members(proc.pid)
    result.update(child_rc=rc, timed_out=timed, survivors=sorted(members),
                  stdout=str(stdout), stderr=str(stderr))
    after = sha256(a.source_target)
    result['source_sha256_after'] = after
    if after != a.expected_source_sha:
        return fail(f'source changed during cell: {after}')
    restore = plan['restore_argv']
    post = plan['post_check_argv']
    if restore:
        result['restore'] = run_aux(restore, a.restore_timeout, run_dir / 'restore.log')
        if result['restore']['rc'] != 0:
            return fail(f'restore rc={result["restore"]["rc"]}')
    if post:
        result['post_check'] = run_aux(post, a.restore_timeout, run_dir / 'post-check.log')
        if result['post_check']['rc'] != 0:
            return fail(f'post-check rc={result["post_check"]["rc"]}')
    result['ok'] = bool(rc == 0 and not timed and not members and after == a.expected_source_sha)
    result['finished_monotonic_ns'] = time.monotonic_ns()
    flush()
    return 0 if result['ok'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-target', type=Path, required=True)
    p.add_argument('--expected-source-sha', required=True)
    p.add_argument('--timeout', type=float, default=240.0)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--cwd', type=Path)
    p.add_argument('--protected-pid', action='append', default=[])
    p.add_argument('--restore-argv', help='JSON string array run after the child')
    p.add_argument('--post-check-argv', help='JSON string array run after restore')
    p.add_argument('--restore-timeout', type=float, default=300.0)
    p.add_argument('--execute', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    plan = build_plan(a)
    if not a.execute:
        plan['status'] = 'plan_not_executed'
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    return execute(a, plan)


if __name__ == '__main__':
    raise SystemExit(main())
