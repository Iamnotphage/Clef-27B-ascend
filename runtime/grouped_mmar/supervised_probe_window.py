#!/usr/bin/env python3
"""Pure supervised window for one Grouped MMAR probe cell.

Responsibilities only: verify the live service identity, stop it with the
original d0_stop, run one child in its own PGID with a hard timeout, then
restore aa3e/s1 with the official start.sh and check the documented gates.
It has no installer/guard/vendor/tree write logic of its own.  Plan-only by
default; execute requires explicit temporary paths.

A normal cell must end with joint-patch-record branch == aa3e-noop and the
source tree unchanged at aa3e.  ``f810->fused`` is rescue-only and marks the
cell isolation-failed; it never satisfies the success gate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

AA3E = 'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'
CLEAN = 'fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230'
CLEAN_PATH = ('/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:'
              '/usr/games:/usr/local/games:/snap/bin')
_RESTORE_LOG_HANDLES = []
RUNNER_NAMES = {'final_acceptance.py', 'bench_ready.py', 'run_cell.py',
                'window_probe.py', 'supervised_window.py',
                'joint_fused_attention_window.py', 'joint_fused_service_ab.py',
                'joint_fused_source_control.py'}
FIELDS = ('pid', 'starttime', 'boot_id', 'pgid', 'sid', 'ppid')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')


def _read_proc(pid):
    """Direct /proc identity read; tests must not be able to fake a live process."""
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        cmdline = (path / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
    except OSError:
        return None
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = ''
    return {'pid': int(pid), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
            'sid': int(fields[3]), 'state': fields[0], 'starttime': fields[19],
            'boot_id': boot, 'cmdline': cmdline}


def proc_identity(pid):
    return _read_proc(pid)


def scan():
    result = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            ident = proc_identity(int(entry.name))
            if ident is not None:
                result[ident['pid']] = ident
    return result


def tree(table, root):
    pids = {root}
    while True:
        children = {p for p, row in table.items() if row['ppid'] in pids}
        if children.issubset(pids):
            return pids
        pids |= children


def atomic_write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(temp, path)


def enable_subreaper():
    """Linux only: orphaned descendants reparent to this window, not to init."""
    if not sys.platform.startswith('linux'):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except Exception:  # noqa: BLE001
        return False


def _scan_direct():
    """Real /proc scan; mocked scan() can never make a fake pid signalable."""
    if not Path('/proc').is_dir():
        return {}
    result = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            ident = _read_proc(int(entry.name))
            if ident is not None:
                result[ident['pid']] = ident
    return result


OWNER_ENV = 'CLEF_WINDOW_OWNER'


def _has_env_token(pid, token):
    if not token:
        return False
    try:
        raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    except OSError:
        return False
    return f'{OWNER_ENV}={token}'.encode() in raw.split(b'\0')


class DescendantRegistry:
    """Records child descendants while they are still reachable by PPID."""

    def __init__(self, root_pid, path, protected, interval=0.2, token=None):
        self.root_pid = int(root_pid)
        self.path = Path(path)
        self.protected = set(protected)
        self.interval = interval
        self.token = token
        self._entries = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._stopped = False

    def _record(self):
        table = scan()
        with self._lock:
            known = {entry['pid'] for entry in self._entries.values()}
        pids = tree(table, self.root_pid) | known
        self_pid = os.getpid()
        for pid, row in table.items():
            if (row['ppid'] == self_pid and pid != self.root_pid and
                    _has_env_token(pid, self.token)):
                pids.add(pid)  # adopted orphan that provably carries the child's token
        # Expand again so children of an adopted orphan are recorded in the same pass.
        while True:
            children = {p for p, row in table.items() if row['ppid'] in pids}
            if children.issubset(pids):
                break
            pids |= children
        changed = False
        now = time.monotonic_ns()
        with self._lock:
            for pid in pids:
                row = table.get(pid)
                if not row or row['state'] == 'Z' or pid in self.protected or pid == self_pid:
                    continue
                key = (pid, row['starttime'], row['boot_id'])
                if key not in self._entries:
                    self._entries[key] = {
                        'pid': pid, 'ppid': row['ppid'], 'pgid': row['pgid'],
                        'sid': row['sid'], 'starttime': row['starttime'],
                        'boot_id': row['boot_id'], 'cmdline': row.get('cmdline') or [],
                        'first_seen_ns': now, 'last_seen_ns': now}
                    changed = True
                else:
                    self._entries[key]['last_seen_ns'] = now
                    self._entries[key]['ppid'] = row['ppid']
        if changed:
            self.write()

    def write(self):
        with self._lock:
            payload = {'schema': 'grouped-mmar-descendants/1',
                       'root_pid': self.root_pid,
                       'entries': sorted(self._entries.values(), key=lambda e: e['pid'])}
        atomic_write_json(self.path, payload)

    def start(self):
        try:
            self._record()
        except Exception:  # noqa: BLE001
            pass
        self._thread = threading.Thread(target=self._loop, name='descendant-registry',
                                        daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self._record()
            except Exception:  # noqa: BLE001
                pass

    def record_now(self):
        try:
            self._record()
        except Exception:  # noqa: BLE001
            pass
        self.write()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        # Short settle plus repeat so an adopted intermediate and its children both land.
        for _ in range(3):
            before = len(self.snapshot())
            self.record_now()
            if len(self.snapshot()) == before:
                break
            time.sleep(0.1)
        self.write()

    def snapshot(self):
        with self._lock:
            return [dict(entry) for entry in self._entries.values()]


def _live_verified(entries, reader=None):
    read = reader or _read_proc
    verified, drift = [], []
    for entry in entries:
        live = read(entry['pid'])
        if live is None or live['state'] == 'Z':
            continue
        if same_process(entry, live):
            verified.append(live)
        else:
            drift.append({'saved': entry, 'live': live})
    return verified, drift


def _group_members(pid, table):
    return {p for p, row in table.items() if row['pgid'] == pid and row['state'] != 'Z'}


def _attributed_targets(entries, table=None, reader=None):
    """Verified group leaders and singles; unknown group members are never signaled."""
    table = _scan_direct() if table is None else table
    verified, drift = _live_verified(entries, reader)
    verified_pids = {row['pid'] for row in verified}
    groups, singles, unknown = {}, set(), set()
    for live in verified:
        if live['pgid'] == live['pid']:
            groups[live['pid']] = _group_members(live['pid'], table)
        else:
            singles.add(live['pid'])
    for gid, members in list(groups.items()):
        outside = members - verified_pids
        if outside:
            unknown |= outside
            groups.pop(gid)
            singles |= {pid for pid in verified_pids
                        if table.get(pid) and table[pid]['pgid'] == gid}
    # A registered leader may already be dead while its group still holds members;
    # those members remain attributed and must block the restore gate.
    for entry in entries:
        if entry.get('pgid') == entry.get('pid'):
            unknown |= _group_members(entry['pgid'], table) - verified_pids
    return {'verified': verified, 'drift': drift, 'groups': groups,
            'singles': singles, 'unknown': sorted(unknown)}


def _signal_targets(targets, sig):
    for gid in targets['groups']:
        try:
            os.killpg(gid, sig)
        except ProcessLookupError:
            pass
    for pid in targets['singles']:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _reap(entries):
    for entry in entries:
        try:
            os.waitpid(entry['pid'], os.WNOHANG)
        except (ChildProcessError, OSError):
            pass


def cleanup_attributed(child_pid, child_identity, registry, protected, *, wait=1.0,
                       reap=False):
    """Kill only identity-verified child groups, including independent rank PGIDs."""
    protected = set(protected)
    entries = registry.snapshot() if registry is not None else []
    if child_identity is not None:
        entries.append(child_identity)
    targets = _attributed_targets(entries)
    overlap = {row['pid'] for row in targets['verified']} & protected
    if overlap:
        return {'killed': False, 'reason': 'protected_overlap',
                'members': sorted(overlap), 'groups': [], 'singles': [],
                'unknown': [], 'drift': targets['drift']}
    if targets['verified'] or targets['unknown']:
        _signal_targets(targets, signal.SIGTERM)
        time.sleep(wait)
        targets = _attributed_targets(entries)
        _signal_targets(targets, signal.SIGKILL)
        time.sleep(wait)
    # Reap every registered candidate, including PIDs that are already zombies:
    # _live_verified excludes Z entries, so reaping only "verified" misses them.
    _reap(entries)
    time.sleep(0.05)
    final = _attributed_targets(entries)
    members = ({row['pid'] for row in final['verified']} |
               set(final['unknown']) |
               {row['live']['pid'] for row in final['drift'] if row.get('live')} |
               {p for members_set in final['groups'].values() for p in members_set})
    return {'killed': True, 'reason': None, 'members': sorted(members),
            'groups': sorted(final['groups']), 'singles': sorted(final['singles']),
            'unknown': final['unknown'], 'drift': final['drift']}


def attributed_survivors(registry, child_pid=None, child_identity=None, table=None,
                         reader=None):
    entries = registry.snapshot() if registry is not None else []
    if child_identity is not None:
        entries.append(child_identity)
    if not entries:
        return []
    targets = _attributed_targets(entries, table=table, reader=reader)
    live = {row['pid'] for row in targets['verified']} | set(targets['unknown'])
    # A PID reuse / start-time drift is not signalable, but it must still block restore.
    live |= {row['live']['pid'] for row in targets['drift'] if row.get('live')}
    return sorted(live)


def cleanup_blocks_restore(outcome):
    return bool(outcome and outcome.get('reason') == 'protected_overlap')


def same_identity(saved, live):
    return bool(live and live['state'] != 'Z' and
                all(live[k] == saved[k] for k in FIELDS if k in saved))


def same_process(saved, live):
    """Identity check for registered descendants: PPID may legitimately reparent."""
    fields = ('pid', 'starttime', 'boot_id', 'pgid', 'sid')
    return bool(live and live['state'] != 'Z' and
                all(live[k] == saved[k] for k in fields if k in saved))


def pending_clients(table, prior=None):
    pending = []
    for pid, row in table.items():
        if pid == os.getpid() or row['state'] == 'Z':
            continue
        argv = row.get('cmdline') or []
        if any(Path(x).name in RUNNER_NAMES for x in argv if x):
            pending.append(pid)
    if prior:
        record = read_json(prior)
        if record.get('clients_exited') is not True:
            raise RuntimeError('prior client supervision did not exit')
        if any(r['pgid'] == record['client_pid'] and r['state'] != 'Z' for r in table.values()):
            raise RuntimeError('prior client group still present')
    return pending


def port_open(port):
    with socket.socket() as s:
        s.settimeout(3)
        rc = s.connect_ex(('127.0.0.1', port))
    if rc not in (0, 111):
        raise RuntimeError(f'unknown port connect_ex={rc}')
    return rc


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(base, suffix, timeout=10):
    with _OPENER.open(base + suffix, timeout=timeout) as response:
        return json.loads(response.read())


def post_json(base, suffix, body, timeout=120):
    data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    request = urllib.request.Request(base + suffix, data=data,
                                     headers={'Content-Type': 'application/json'})
    with _OPENER.open(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def npu_rows(text):
    if 'Process id in container' not in text:
        raise RuntimeError('NPU process table unavailable')
    rows = []
    for line in text.split('Process id in container', 1)[1].splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 5:
            continue
        loc = cells[0].split()
        if len(loc) == 2 and all(x.isdigit() for x in loc):
            if not (cells[1].isdigit() and cells[4].isdigit()):
                raise RuntimeError('invalid NPU row')
            rows.append({'card': int(loc[0]), 'chip': int(loc[1]), 'name': cells[2], 'pid': int(cells[4])})
    return rows


def npu_evidence(expected_pids=None):
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30)
    rows = npu_rows(result.stdout)
    if expected_pids is None:
        if rows or 'No running processes found in NPU 7' not in result.stdout:
            raise RuntimeError('card7 is not empty')
        return rows
    if not rows or not {r['pid'] for r in rows}.issubset(expected_pids):
        raise RuntimeError('unattributed NPU process')
    workers = [r for r in rows if r['name'] == 'VLLMWorker_TP']
    if len(workers) != 2 or {r['chip'] for r in workers} != {0, 1}:
        raise RuntimeError('TP2 worker mapping mismatch')
    return rows


def read_root_env(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    out = {}
    for part in raw.split(b'\0'):
        if b'=' in part:
            key, value = part.split(b'=', 1)
            out[key.decode()] = value
    return out


def env_evidence(pid, clean_expect):
    expected_doc = read_json(clean_expect)
    expected = expected_doc.get('service_environment', expected_doc)
    live = read_root_env(pid)
    checks = {}
    for key in ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH'):
        want = str(expected[key]).encode()
        got = live.get(key)
        checks[key] = {'match': got == want,
                       'count': len(got.split(b':')) if got else 0,
                       'expected_count': len(want.split(b':'))}
    return {'clean_expect_sha256': sha256(clean_expect), 'checks': checks,
            'match': all(v['match'] for v in checks.values())}


def joint_env(live):
    wanted = {'CLEF_JOINT_FUSED_ATTN': b'1',
              'CLEF_JOINT_ALLOWED_POSTIMAGE_SHA': AA3E.encode(),
              'CLEF_JOINT_SINGLE_FIELD': b'1'}
    forbidden = ('CLEF_JOINT_SMALLQ', 'CLEF_JOINT_SMALLQ_KERNEL')
    return {'keys': {k: live.get(k) == v for k, v in wanted.items()},
            'forbidden_absent': all(k not in live for k in forbidden),
            'ok': all(live.get(k) == v for k, v in wanted.items()) and
                  all(k not in live for k in forbidden)}


def run(argv, timeout, log):
    with Path(log).open('xb') as handle:
        proc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False)
    return {'argv': argv, 'rc': proc.returncode, 'log': str(log)}


def build_restore_argv(a):
    inner = (f'cd {a.service_root} && source {a.activate} && exec bash {a.official_start} '
             f'--model-dir {a.model_dir} --output-dir {a.restore_output}')
    return ['env', '-i', f'PATH={CLEAN_PATH}', 'CLEF_LAUNCH_CONFIG=s1',
            'setsid', '/bin/bash', '-c', inner]


def build_child_argv(a):
    if a.child_argv:
        value = json.loads(a.child_argv)
        if not isinstance(value, list) or not value:
            raise ValueError('child-argv must be a non-empty JSON array')
        return [str(x) for x in value]
    if a.arm == 'A12':
        if not a.a12_binary:
            raise ValueError('A12 requires --a12-binary')
        return ['bash', str(a.a12_runner), str(a.a12_binary)]
    argv = ['torchrun', '--nproc_per_node=2', '--master_port', str(a.master_port),
            str(Path(__file__).with_name('run_probe.py')), '--arm', a.arm,
            '--temp-root', str(a.probe_temp_root), '--out', str(a.probe_out),
            '--k', str(a.k), '--n', str(a.n), '--rows', str(a.rows), '--seed', str(a.seed),
            '--arm-timeout', str(a.child_timeout), '--comm-timeout', str(a.comm_timeout)]
    if a.arm != 'A8':
        argv += ['--adapter', str(a.adapter)]
    argv.append('--execute')
    return argv


def build_plan(a):
    for value, name in ((a.run_dir, 'run-dir'), (a.restore_output, 'restore-output')):
        if not Path(value).is_absolute():
            raise ValueError(f'{name} must be absolute')
    if Path(a.run_dir).exists():
        raise ValueError('run-dir must be fresh')
    if not Path(a.stop_script).is_file():
        raise ValueError(f'stop script missing: {a.stop_script}')
    if Path(a.restore_output).exists():
        raise ValueError('restore-output must be fresh')
    if not Path(a.source_target).is_file():
        raise ValueError('source-target missing')
    if sha256(a.source_target) != a.expected_source_sha:
        raise ValueError('source SHA mismatch')
    identity = read_json(a.root_identity)
    for key in ('pid', 'starttime', 'boot_id', 'pgid'):
        if key not in identity:
            raise ValueError(f'root identity lacks {key}')
    if not a.child_argv and not a.arm:
        raise ValueError('supply --child-argv or --arm')
    if not Path(a.text_body).is_file() or not Path(a.image_body).is_file():
        raise ValueError('text/image probe bodies must exist')
    child = build_child_argv(a)
    return {'schema': 'grouped-mmar-window-plan/1', 'run_dir': str(Path(a.run_dir).resolve()),
            'root_identity': identity, 'stop_argv': [a.python, str(a.stop_script),
            '--api-pid', str(identity['pid']), '--saved-identity', str(a.root_identity),
            '--out', str(Path(a.run_dir) / 'stop.json')],
            'child_argv': child, 'child_timeout': a.child_timeout,
            'restore_argv': build_restore_argv(a), 'restore_output': str(a.restore_output),
            'source_target': str(a.source_target), 'expected_source_sha256': a.expected_source_sha,
            'port': a.port, 'clean_expect': str(a.clean_env_expect),
            'probes': {'text': str(a.text_body), 'image': str(a.image_body)},
            'success': {'child_rc': 0, 'not_timed_out': True, 'no_survivors': True,
                        'source_sha256': AA3E, 'branch': 'aa3e-noop',
                        'four_checks': True, 'env': True, 'joint_env': True,
                        'two_workers': True, 'clients_empty': True},
            'failure_branches': ['preflight identity/port/client mismatch',
                                 'stop rc!=0 or post-stop port/PGID/NPU not empty',
                                 'child timeout/nonzero/unknown group member',
                                 'restore rc!=0/health timeout/unknown pid',
                                 'branch f810->fused (isolation failure, rescue only)',
                                 'source SHA drift', 'four-check/env/joint/worker mismatch'],
            'no_service_writes': True, 'no_f810_revert': True}


def cleanup_child(pid, table, protected):
    members = {p for p, row in table.items() if row['pgid'] == pid and row['state'] != 'Z'}
    if members & set(protected):
        return {'killed': False, 'reason': 'protected_in_child_group', 'members': sorted(members)}
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(1.0)
    table = scan()
    members = {p for p, row in table.items() if row['pgid'] == pid and row['state'] != 'Z'}
    if members:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1.0)
        table = scan()
        members = {p for p, row in table.items() if row['pgid'] == pid and row['state'] != 'Z'}
    return {'killed': True, 'reason': None, 'members': sorted(members)}


def wait_health(a, state):
    deadline = time.monotonic() + a.restore_timeout
    last = None
    while time.monotonic() < deadline:
        try:
            health = http_json(f'http://127.0.0.1:{a.port}', '/health')
            if health.get('status') == 'ready' and health.get('model') == 'clef':
                return health
            last = health
        except Exception as exc:  # noqa: BLE001
            last = f'{type(exc).__name__}: {exc}'
        time.sleep(1.0)
    raise RuntimeError(f'health not ready: {last}')


def _libs_for_pid(pid, names):
    try:
        text = (Path('/proc') / str(pid) / 'maps').read_text(errors='replace')
    except OSError:
        return None
    libs = []
    for line in text.splitlines():
        parts = line.split()
        path = parts[-1] if parts else ''
        if not path.startswith('/') or not any(name in Path(path).name for name in names):
            continue
        if not Path(path).is_file():
            continue
        entry = {'path': path, 'sha256': sha256(path), 'soname': None}
        try:
            text_d = subprocess.run(['readelf', '-d', path], capture_output=True, text=True,
                                    timeout=10).stdout
            match = __import__('re').search(r'\(SONAME\).*\[(.+?)\]', text_d)
            entry['soname'] = match.group(1) if match else None
        except (OSError, subprocess.SubprocessError):
            pass
        if entry not in libs:
            libs.append(entry)
    return libs


def maps_for_pids(pids, names):
    """Real /proc maps for the given pids, including independently grouped ranks."""
    out = []
    for pid in sorted(set(pids)):
        libs = _libs_for_pid(pid, names)
        if not libs:
            continue
        live = _read_proc(pid) or {}
        out.append({'pid': pid, 'cmdline': live.get('cmdline') or [], 'libs': libs})
    return out


def maps_evidence(pgid, names):
    """Per-PID maps for every member of the owned child PGID."""
    per_pid = {}
    for pid, row in scan().items():
        if row['pgid'] != pgid or row['state'] == 'Z':
            continue
        libs = _libs_for_pid(pid, names)
        if libs:
            per_pid[pid] = {'pid': pid, 'cmdline': row.get('cmdline') or [], 'libs': libs}
    return list(per_pid.values())


def all_libs(maps):
    return [lib for entry in maps for lib in entry['libs']]


def map_coverage(maps):
    names = [Path(lib['path']).name for lib in all_libs(maps)]
    return {token: any(token in name for name in names) for token in
            ('libhccl', 'opapi_math', 'opapi_transformer', 'hccl_fwk', 'hcomm',
             'libascendcl', 'libnnopbase')}


def binary_processes(maps):
    out = []
    for entry in maps:
        cmd_list = entry.get('cmdline') or []
        cmd = ' '.join(cmd_list)
        first = Path(cmd_list[0]).name if cmd_list else ''
        if 'a12_grouped_mmar' in cmd or first == 'timeout':
            out.append(entry['pid'])
    return out


def required_map_ok(arm, adapter, maps):
    names = [Path(lib['path']).name for lib in all_libs(maps)]
    if arm == 'A12':
        return (any('libhccl' in name for name in names) and
                any('opapi_math' in name or 'opapi_transformer' in name for name in names))
    if adapter is None:
        return False
    return any(Path(adapter).name == name for name in names)


def a12_evidence(stdout, stderr, rc):
    try:
        text = Path(stdout).read_text(errors='replace') + Path(stderr).read_text(errors='replace')
    except OSError:
        text = ''
    error_tokens = ('[ERROR]', 'retcode:', 'TestCall execute AICPU failed',
                    'This is an error', 'CreateTensor Failed')
    errors = [line for line in text.splitlines() if any(tok in line for tok in error_tokens)]
    cost = [line for line in text.splitlines() if 'mc2 costtime' in line]
    names = [line for line in text.splitlines() if 'hcom name is' in line]
    return {'error_lines': errors[:20], 'costtime_lines': cost,
            'hcom_lines': names,
            'ok': bool(rc == 0 and not errors and len(cost) >= 2)}


def recovery_gate(a, plan, saved, stopped, child_pid, source_after, registry=None):
    """Never start a second tree while the original root is alive."""
    table = scan()
    live = table.get(saved['pid'])
    old_alive = bool(live and live['state'] != 'Z' and same_identity(saved, live))
    try:
        port = port_open(a.port)
    except Exception:  # noqa: BLE001
        port = None
    health = None
    if old_alive and port == 0:
        try:
            health = http_json(f'http://127.0.0.1:{a.port}', '/health')
        except Exception as exc:  # noqa: BLE001
            health = {'error': f'{type(exc).__name__}: {exc}'}
    residual = sorted({p for p, row in table.items()
                       if child_pid and row['pgid'] == child_pid and row['state'] != 'Z'} |
                      set(attributed_survivors(registry, child_pid)))
    old_pgid = sorted(p for p, row in table.items()
                      if row['pgid'] == saved['pid'] and row['state'] != 'Z')
    try:
        runners = pending_clients(table)
    except Exception as exc:  # noqa: BLE001
        runners = [f'error: {type(exc).__name__}: {exc}']
    context = {'port': port, 'health': health, 'old_identity': live,
               'old_pgid_members': old_pgid, 'probe_residual': residual,
               'attributed_survivors': attributed_survivors(registry, child_pid),
               'runner_residual': runners, 'source_after': source_after}
    if old_alive:
        if port == 0 and isinstance(health, dict) and health.get('status') == 'ready':
            return False, 'skipped_service_alive_healthy', context
        return False, 'old_root_alive_partial', context
    if port != 111:
        return False, f'blocked_port_{port}', context
    if old_pgid:
        return False, 'blocked_old_pgid_members', context
    if residual:
        return False, 'blocked_probe_residual', context
    if runners:
        return False, 'blocked_runner_residual', context
    if source_after != a.expected_source_sha:
        return False, 'blocked_source_unknown', context
    return True, 'restore', context


def start_restore(plan, run_dir, result):
    log = (run_dir / 'restore.log').open('xb')
    _RESTORE_LOG_HANDLES.append(log)  # keep the foreground service log open
    proc = subprocess.Popen(plan['restore_argv'], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result['restore'] = {'wrapper_pid': proc.pid, 'argv': plan['restore_argv'],
                         'started': True, 'waited': False}
    return proc


def execute(a, plan):
    run_dir = Path(plan['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'grouped-mmar-window-result/1', 'plan': plan, 'ok': False,
              'isolation_failed': False, 'steps': {}}
    state_path = run_dir / 'window.json'
    flush = lambda: write_json(state_path, result)

    def step(name, value):
        result['steps'][name] = value
        flush()

    saved = plan['root_identity']
    before = sha256(a.source_target)
    result['source_sha256_before'] = before
    if before != a.expected_source_sha:
        result['error'] = 'source changed before preflight'
        flush()
        return 1
    table = scan()
    live = table.get(saved['pid'])
    if not same_identity(saved, live) or live['ppid'] != 1 or live['pgid'] != saved['pid']:
        result['error'] = 'root identity mismatch'
        flush()
        return 1
    if port_open(a.port) != 0:
        result['error'] = 'service port not open before stop'
        flush()
        return 1
    clients = pending_clients(table, a.prior_client_record)
    if clients:
        result['error'] = f'pending clients: {clients}'
        flush()
        return 1
    step('preflight', {'identity': live, 'port': 0, 'clients': []})

    stop_called = False
    stopped = False
    child = None
    child_state = {'started': False}
    child_ok = False
    source_ok = False
    registry = None
    cleanup_blocked = False
    subreaper = enable_subreaper()
    result['subreaper'] = subreaper
    if sys.platform.startswith('linux') and not subreaper:
        result['error'] = 'subreaper unavailable on Linux; refusing to start a cell'
        flush()
        return 1
    stop_started_ns = time.time_ns()

    try:
        stop_called = True
        stop = run(plan['stop_argv'], a.stop_timeout, run_dir / 'stop.log')
        step('stop', stop)
        if stop['rc'] != 0:
            raise RuntimeError(f'stop rc={stop["rc"]}')
        table = scan()
        old_tree = tree(table, saved['pid'])
        if any(p in table and table[p]['state'] != 'Z' for p in old_tree):
            raise RuntimeError('old tree remains after stop')
        if any(row['pgid'] == saved['pid'] and row['state'] != 'Z' for row in table.values()):
            raise RuntimeError('old PGID member remains after stop')
        if port_open(a.port) != 111:
            raise RuntimeError('port not closed after stop')
        npu_evidence(None)
        stopped = True
        step('stopped', {'port': 111, 'npu_empty': True})

        protected = {saved['pid'], *[p for p, row in table.items() if row['pgid'] == saved['pid']]}
        owner_token = secrets.token_hex(16)
        child_env = {**os.environ, OWNER_ENV: owner_token}
        with (run_dir / 'child.stdout').open('xb') as out, (run_dir / 'child.stderr').open('xb') as err:
            child = subprocess.Popen(plan['child_argv'], cwd=a.cwd, stdout=out, stderr=err,
                                     env=child_env, start_new_session=True)
        identity = proc_identity(child.pid)
        if not identity or identity['pgid'] != child.pid or child.pid in protected:
            raise RuntimeError('child is not a fresh own-PGID process')
        child_state.update(started=True, pid=child.pid, identity=identity,
                           subreaper=subreaper)
        registry = DescendantRegistry(child.pid, run_dir / 'descendants.json', protected,
                                      token=owner_token).start()
        child_state['descendants'] = str(run_dir / 'descendants.json')
        child_state['owner_token'] = True
        names = (['libhccl', 'libopapi', 'libnnopbase', 'libascendcl', 'libhcomm']
                 if a.arm == 'A12' else
                 [Path(a.adapter).name if a.adapter else 'clef_mmar_probe',
                  'libhccl', 'libascendcl'])
        maps = []
        for _ in range(30):
            maps = maps_evidence(child.pid, names)
            if maps or child.poll() is not None:
                break
            time.sleep(0.1)
        child_state['maps'] = maps
        child_state['maps_ok'] = required_map_ok(a.arm, a.adapter, maps)
        child_state['map_coverage'] = map_coverage(maps)
        child_state['binary_pids'] = binary_processes(maps)
        step('child_started', child_state)
        try:
            rc = child.wait(timeout=a.child_timeout)
            timed = False
        except subprocess.TimeoutExpired:
            timed = True
            child_state['timed_out'] = True
            registry.stop()  # final synchronous record before any cleanup
            attributed = maps_for_pids(
                {child.pid, *[entry['pid'] for entry in registry.snapshot()]}, names)
            child_state['maps_attributed'] = attributed
            cleanup = cleanup_attributed(child.pid, child_state.get('identity'), registry,
                                         protected, reap=subreaper)
            child_state['cleanup'] = cleanup
            if cleanup_blocks_restore(cleanup):
                cleanup_blocked = True
            try:
                rc = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = None
        registry.stop()  # final synchronous record before any cleanup
        attributed = maps_for_pids(
            {child.pid, *[entry['pid'] for entry in registry.snapshot()]}, names)
        child_state['maps_attributed'] = attributed
        combined = child_state['maps'] + [m for m in attributed
                                          if m not in child_state['maps']]
        child_state['maps'] = combined
        child_state['maps_ok'] = required_map_ok(a.arm, a.adapter, combined)
        child_state['map_coverage'] = map_coverage(combined)
        child_state['binary_pids'] = binary_processes(combined)
        cleanup_final = cleanup_attributed(child.pid, child_state.get('identity'), registry,
                                           protected, reap=subreaper)
        child_state['cleanup_final'] = cleanup_final
        if cleanup_blocks_restore(cleanup_final):
            cleanup_blocked = True
        members = {p for p, row in scan().items() if row['pgid'] == child.pid and row['state'] != 'Z'}
        survivors = sorted(set(members) |
                           set(attributed_survivors(registry, child.pid,
                                                    child_state.get('identity'))) |
                           set(cleanup_final.get('unknown') or []))
        child_state.update(rc=rc, timed_out=timed, survivors=survivors)
        if survivors:
            raise RuntimeError(f'child group survivors: {survivors}')
        evidence = None
        if a.arm == 'A12':
            evidence = a12_evidence(run_dir / 'child.stdout', run_dir / 'child.stderr', rc)
            child_state['a12_evidence'] = evidence
        child_ok = bool(rc == 0 and not timed and child_state['maps_ok'] and
                        (a.arm != 'A12' or child_state['binary_pids']) and
                        (evidence is None or evidence['ok']))
        if not child_ok:
            result['error'] = 'child failed or missing A12/maps evidence'
        step('child', child_state)
    except BaseException as exc:
        child_state['error'] = f'{type(exc).__name__}: {exc}'
        if registry is not None:
            try:
                registry.stop()  # final record before cleanup
            except BaseException:
                pass
        if child is not None and child.poll() is None:
            try:
                child_state['cleanup'] = cleanup_attributed(
                    child.pid, child_state.get('identity'), registry, {saved['pid']},
                    reap=subreaper)
                if cleanup_blocks_restore(child_state['cleanup']):
                    cleanup_blocked = True
            except BaseException as cleanup_error:
                child_state['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
        result['error'] = child_state['error']
        step('child', child_state)
    finally:
        if registry is not None:
            try:
                registry.stop()  # final record before any cleanup
            except BaseException:
                pass
        if child is not None and child.poll() is None:
            try:
                outcome = cleanup_attributed(child.pid, child_state.get('identity'),
                                             registry, {saved['pid']}, reap=subreaper)
                if cleanup_blocks_restore(outcome):
                    cleanup_blocked = True
            except BaseException:
                pass
        source_after = sha256(a.source_target)
        result['source_sha256_after_child'] = source_after
        source_ok = source_after == a.expected_source_sha
        if not source_ok:
            result['error'] = 'source changed during child'
        needs_restore = False
        reason = 'stop_not_called'
        context = {}
        if cleanup_blocked:
            needs_restore = False
            reason = 'blocked_protected_overlap'
            context = {'cleanup_blocked': True,
                       'attributed_survivors': attributed_survivors(
                           registry, child.pid if child is not None else None)}
            result['recovery_gate'] = {'allow': False, 'reason': reason, **context}
        elif stop_called:
            needs_restore, reason, context = recovery_gate(
                a, plan, saved, stopped, child.pid if child is not None else None,
                source_after, registry)
            result['recovery_gate'] = {'allow': needs_restore, 'reason': reason, **context}
        if needs_restore:
            try:
                start_restore(plan, run_dir, result)
                health = wait_health(a, result)
                manifest_path = Path(a.restore_output) / 'launch-manifest.json'
                if not manifest_path.is_file():
                    raise RuntimeError('restore launch-manifest missing')
                if manifest_path.stat().st_mtime_ns < stop_started_ns:
                    raise RuntimeError('restore launch-manifest predates the stop')
                manifest = read_json(manifest_path)
                if Path(manifest.get('output_dir', '')).resolve() != Path(a.restore_output).resolve():
                    raise RuntimeError('restore output_dir mismatch')
                pid = int(manifest['pid'])
                live = scan().get(pid)
                if not live or live['ppid'] != 1 or live['pgid'] != pid:
                    raise RuntimeError('restored root identity mismatch')
                pids = tree(scan(), pid)
                if not all(scan()[p]['pgid'] == pid for p in pids):
                    raise RuntimeError('restored descendant escaped group')
                record_path = Path(a.restore_output) / 'joint-patch-record.json'
                if not record_path.is_file():
                    raise RuntimeError('joint-patch-record missing after restore')
                branch = read_json(record_path).get('branch')
                result['restore_branch'] = branch
                if branch != 'aa3e-noop':
                    result['isolation_failed'] = True
                    result['error'] = f'joint branch={branch}; source drift evidence retained'
                npu_evidence(pids)
                env = env_evidence(pid, a.clean_env_expect)
                joint = joint_env(read_root_env(pid))
                info = http_json(f'http://127.0.0.1:{a.port}', '/v1/decide/info')
                text_status, text = post_json(f'http://127.0.0.1:{a.port}', '/v1/systemone',
                                              Path(a.text_body).read_text())
                image_status, image = post_json(f'http://127.0.0.1:{a.port}', '/v1/decide',
                                                Path(a.image_body).read_text())
                after_restore = sha256(a.source_target)
                result['source_sha256_after_restore'] = after_restore
                clients_after = pending_clients(scan())
                checks = {'health': health.get('status') == 'ready',
                          'info': info.get('config') == 's1',
                          'workers': True, 'text': text_status == 200,
                          'image': image_status == 200,
                          'env': env['match'], 'joint_env': joint['ok'],
                          'source': after_restore == a.expected_source_sha,
                          'clients': not clients_after}
                result['restore_checks'] = {'pid': pid, 'pids': sorted(pids), 'branch': branch,
                                             'health': health, 'info': info, 'env': env,
                                             'joint_env': joint, 'text': text, 'image': image,
                                             'checks': checks, 'clients': clients_after}
                result['ok'] = bool(child_ok and source_ok and source_after == a.expected_source_sha and
                                    after_restore == a.expected_source_sha and
                                    branch == 'aa3e-noop' and all(checks.values()) and
                                    not result['isolation_failed'])
            except BaseException as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
        else:
            result['restore'] = {'skipped': True, 'blocked': reason != 'skipped_service_alive_healthy',
                                 'reason': reason, **context}
            if stop_called:
                result['error'] = f'restore_blocked: {reason}' 
        result['finished_monotonic_ns'] = time.monotonic_ns()
        flush()
        return 0 if result['ok'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--root-identity', type=Path, required=True)
    p.add_argument('--stop-script', type=Path, default=Path('/tmp/d0cc/d0_stop.py'))
    p.add_argument('--expect-stop-sha', default='d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--child-argv', help='JSON string array; overrides --arm')
    p.add_argument('--arm', choices=('A12', 'A13', 'A14', 'A15', 'A8', 'A0', 'H1', 'H2', 'H3', 'H6', 'H7', 'H4', 'H5', 'H4K', 'H5K', 'H11'), default=None)
    p.add_argument('--a12-binary', type=Path)
    p.add_argument('--a12-runner', type=Path, default=Path(__file__).with_name('a12') / 'run_a12.sh')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--probe-temp-root', type=Path)
    p.add_argument('--probe-out', type=Path)
    p.add_argument('--k', type=int, default=8704)
    p.add_argument('--n', type=int, default=5120)
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261119)
    p.add_argument('--master-port', type=int, default=29531)
    p.add_argument('--comm-timeout', type=float, default=90.0)
    p.add_argument('--child-timeout', type=float, default=240.0)
    p.add_argument('--stop-timeout', type=float, default=120.0)
    p.add_argument('--restore-timeout', type=float, default=900.0)
    p.add_argument('--port', type=int, default=18143)
    p.add_argument('--source-target', type=Path, required=True)
    p.add_argument('--expected-source-sha', default=AA3E)
    p.add_argument('--official-start', type=Path, required=True)
    p.add_argument('--activate', type=Path, required=True)
    p.add_argument('--service-root', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--restore-output', type=Path, required=True)
    p.add_argument('--clean-env-expect', type=Path,
                   default=Path('/workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json'))
    p.add_argument('--text-body', type=Path, required=True)
    p.add_argument('--image-body', type=Path, required=True)
    p.add_argument('--prior-client-record', type=Path)
    p.add_argument('--cwd', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if a.stop_script.is_file() and sha256(a.stop_script) != a.expect_stop_sha:
        raise ValueError('stop script SHA mismatch')
    plan = build_plan(a)
    if not a.execute:
        plan['status'] = 'plan_not_executed'
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    return execute(a, plan)


if __name__ == '__main__':
    raise SystemExit(main())
