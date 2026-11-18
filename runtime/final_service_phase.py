#!/usr/bin/env python3
"""Owner27 frozen f810 phase switch. Default prints a plan without mutations.

Only a verified old restore record names the old service. Every launch uses a
new label, the original launcher and a clean two-key parent environment.
Any error preserves records and blocks subsequent actions; there is no guessed
PID, fallback signal or parallel recovery launch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone

import final_acceptance_supervisor as strict

PYTHON = '/workspace/yangchen/venvs/clef-py312/bin/python'
REPO = Path('/workspace/yangchen/work/scratch-24/joint-fused-5c045eb')
HELPER = REPO / 'runtime/joint_fused_attention_window.py'
LAUNCH = '/workspace/yangchen/logs/perf-wave2-20261118/launch.sh'
WAIT = '/workspace/yangchen/logs/perf-wave2-20261118/wait.sh'
STOP = '/tmp/d0cc/d0_stop.py'
LOG_ROOT = Path('/workspace/yangchen/logs/perf-wave2-20261118')
TARGET = Path('/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py')
GUARD = Path('/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py')
HASHES = {
    str(Path(strict.__file__)): '172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876',
    str(HELPER): strict.HELPER_SHA,
    LAUNCH: 'a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321',
    WAIT: 'e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0',
    STOP: 'd6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886',
    str(TARGET): 'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863',
    str(GUARD): '76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9',
}
CLEAN_ENV = {
    'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
    'CLEF_LAUNCH_CONFIG': 's1',
}
OVERRIDES = {
    's1': [],
    's2_b8': ['--max-num-seqs', '8', '--max-num-batched-tokens', '8192'],
    's2_b16': ['--max-num-seqs', '16', '--max-num-batched-tokens', '16384'],
    'erqa': ['--max-model-len', '16384', '--max-num-batched-tokens', '16384',
             '--limit-mm-per-prompt', '{"image":16,"video":0}', '--mm-processor-kwargs',
             '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}'],
}
RUNNER_NAMES = {'joint_fused_attention_window.py', 'joint_fused_service_ab.py',
                'joint_fused_source_control.py', 'supervised_window.py',
                'final_acceptance.py', 'final_acceptance_supervisor.py',
                'run_cell.py', 'bench_ready.py', 'window_probe.py'}
require, write, sha = strict.require, strict.write, strict.sha


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def port():
    with socket.socket() as sock:
        sock.settimeout(3)
        rc = sock.connect_ex(('127.0.0.1', 18143))
    require(rc in (0, 111), f'unknown port connect_ex={rc}')
    return rc


def tree(table, root):
    pids = {root}
    while True:
        children = {p for p, row in table.items() if row['ppid'] in pids}
        if children.issubset(pids):
            return pids
        pids |= children


def no_pending_clients(table, prior=None):
    if prior:
        record = json.loads(Path(prior).read_text())
        require(record.get('clients_exited') is True, 'previous clients not confirmed exited')
        group = record['client_pid']
        require(not any(r['pgid'] == group and r['state'] != 'Z' for r in table.values()),
                'previous client group remains')
        require(not any(int(p) in table and table[int(p)]['state'] != 'Z'
                        for p in record['observed']), 'observed client PID remains or was reused')
    pending = []
    for pid, row in table.items():
        if pid == os.getpid() or row['state'] == 'Z':
            continue
        path = Path('/proc') / str(pid)
        try:
            argv = (path / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            require(not path.exists(), f'partial client inventory query: {pid}')
            continue
        if any(Path(a.decode()).name in RUNNER_NAMES for a in argv if a):
            pending.append(row)
    require(not pending, f'pending service/experiment/client runners: {pending}')


def npu_rows(text):
    require('Process id in container' in text, 'NPU process table unavailable')
    section = text.split('Process id in container', 1)[1]
    result = []
    for line in section.splitlines():
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(cells) != 5:
            continue
        loc = cells[0].split()
        if len(loc) == 2 and all(x.isdigit() for x in loc):
            require(cells[1].isdigit() and cells[4].isdigit(), 'invalid NPU PID row')
            row = dict(card=int(loc[0]), chip=int(loc[1]), pid=int(cells[4]), name=cells[2])
            require(row['card'] == 7 and row['chip'] in (0, 1), 'process outside assigned card7')
            result.append(row)
    return result


def command(out, name, argv, timeout, env=None):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    write(out / (name + '.json'), dict(argv=argv, rc=result.returncode,
          stdout=result.stdout, stderr=result.stderr, finished_utc=strict.now()))
    require(result.returncode == 0, f'{name} rc={result.returncode}')
    return result


def resource(out, name, service_pids=None):
    result = command(out, name, ['npu-smi', 'info'], 30)
    rows = npu_rows(result.stdout)
    if service_pids is None:
        require(not rows and 'No running processes found in NPU 7' in result.stdout,
                'card7 not proven empty')
    else:
        require(rows and {r['pid'] for r in rows}.issubset(service_pids), 'unattributed NPU process')
        workers = [r for r in rows if r['name'] == 'VLLMWorker_TP']
        require(len(workers) == 2 and {r['chip'] for r in workers} == {0, 1}, 'TP2 mapping mismatch')
    return rows


def switch(a):
    require(bool(a.prior_client_record) != bool(a.first_phase_no_client),
            'select prior client record or explicit first phase mode')
    if a.first_phase_no_client:
        require(a.from_phase == a.phase == 's1', 'first phase mode requires s1 to s1')
    manifest = json.loads(a.manifest.read_text())
    require(sha(a.manifest) == a.manifest_sha, 'manifest SHA mismatch')
    require(sha(a.current_restore) == a.current_restore_sha, 'old restore SHA mismatch')
    require(not a.run_dir.exists(), 'new run directory required')
    require(sha(a.code / 'runtime/final_acceptance.py') == a.runner_sha, 'phase runner SHA mismatch')
    for path, expected in HASHES.items():
        require(sha(path) == expected, f'frozen file mismatch: {path}')
    require(not (TARGET.parent / 'clef_smallq_kernel.py').exists(), 'candidate kernel still installed')
    old = json.loads(a.current_restore.read_text())
    require(all(old[k]['rc'] == 0 and old[k]['ok'] for k in ('health', 'workers', 'text', 'image')),
            'old restore not complete')
    label = 'd0-final-' + a.phase + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    label_dir = LOG_ROOT / label
    require(not label_dir.exists(), 'fresh label collision')
    plan = dict(phase=a.phase, label=label, launch=['/bin/bash', LAUNCH, label, *OVERRIDES[a.phase]],
                clean_parent=CLEAN_ENV, source_sha256=HASHES[str(TARGET)])
    if not a.execute:
        return dict(plan=plan, status='plan_not_executed')
    a.run_dir.mkdir(parents=True, exist_ok=False)
    write(a.run_dir / 'plan.json', plan)
    try:
        acceptance = load(a.code / 'runtime/final_acceptance.py', 'phase_acceptance')
        acceptance.frozen_check(manifest)
        helper = load(HELPER, 'phase_helper')
        d0 = a.run_dir / 'd0';d0.mkdir()
        for name, expected in helper.D0_MODULES.items():
            src = REPO / 'runtime/d0' / name
            require(sha(src) == expected, f'd0 hash mismatch: {name}')
            shutil.copy2(src, d0 / name)
        sys.path.insert(0, str(d0))
        import d0_restore, d0_run
        table = strict.scan()
        no_pending_clients(table, a.prior_client_record)
        saved = old['service_discovery']['api_identity']
        root = saved['pid']
        live = table.get(root)
        require(live and live['state'] != 'Z' and live['pgid'] == root and live['ppid'] == 1,
                'old root unavailable or group/parent changed')
        require(all(live[k] == saved[k] for k in ('pid', 'starttime', 'boot_id')), 'old root lifetime changed')
        old_label = LOG_ROOT / old['label']
        require(int((old_label / 'api.pid').read_text()) == root, 'old label PID mismatch')
        argv = (Path('/proc') / str(root) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        require(argv == manifest['phases'][a.from_phase]['expected_argv'], 'old final argv mismatch')
        pids = tree(table, root)
        require(all(table[p]['pgid'] == root for p in pids), 'old descendant escaped group')
        old_group = {pid for pid, row in table.items() if row['pgid'] == root and row['state'] != 'Z'}
        require(old_group.issubset(pids), 'unattributed old PGID member before stop')
        ids = dict(root=live, descendants=[table[p] for p in sorted(pids - {root})])
        write(a.run_dir / 'old-identities.json', ids)
        write(a.run_dir / 'root-identity.json', live)
        resource(a.run_dir, 'before-npu', pids)
        require(port() == 0, 'old service port not open')
        command(a.run_dir, 'stop-command', [PYTHON, STOP, '--api-pid', str(root),
            '--saved-identity', str(a.run_dir / 'root-identity.json'),
            '--out', str(a.run_dir / 'stop.json')], 120)
        table = strict.scan()
        require(all(p not in table or table[p]['state'] == 'Z' for p in pids), 'old tree remains')
        require(not any(row['pgid'] == root and row['state'] != 'Z' for row in table.values()),
                'old PGID member remains after stop')
        no_pending_clients(table, a.prior_client_record)
        require(port() == 111, 'port not closed before launch')
        resource(a.run_dir, 'stopped-npu')
        before = set(table)
        require(port() == 111, 'fresh port not closed immediately before launch')
        command(a.run_dir, 'launch', plan['launch'], 60, env=dict(CLEAN_ENV))
        pid = int((label_dir / 'api.pid').read_text())
        initial = strict.read_identity(pid)
        require(initial and pid not in before and initial['pgid'] == pid, 'new root identity unavailable')
        write(a.run_dir / 'launch-identity.json', initial)
        command(a.run_dir, 'wait', ['timeout', '600', '/bin/bash', WAIT], 630)
        table = strict.scan()
        final = table.get(pid)
        require(strict.same_process(initial, final), 'new root changed during wait')
        pids = tree(table, pid)
        require(all(table[p]['pgid'] == pid for p in pids), 'new descendant escaped group')
        ids = dict(root=final, descendants=[table[p] for p in sorted(pids - {pid})])
        write(a.run_dir / 'ready-identities.json', ids)
        # Full final argv/env/launch/health/info checks establish expected exec.
        check = acceptance.service_check(manifest, a.phase, a.run_dir / 'ready-identities.json', label_dir / 'launch-manifest.json')
        write(a.run_dir / 'service-check.json', check)
        log = dict(label=label, service_discovery=dict(api_pid=pid, api_identity=final,
            root_identity_stable=True, pids=sorted(pids), tree=[table[p] for p in sorted(pids)]))
        ok = d0_restore.verify(d0_restore.build_commands(d0_run.render_request), log, service_pids=pids)
        write(a.run_dir / 'restore.json', log)
        require(ok, 'phase four-check verification failed')
        resource(a.run_dir, 'ready-npu', pids)
        require(sha(TARGET) == HASHES[str(TARGET)] and sha(GUARD) == HASHES[str(GUARD)], 'source changed during switch')
        result = dict(ok=True, phase=a.phase, label=label, root=pid,
                      identities=str(a.run_dir / 'ready-identities.json'),
                      launch_manifest=str(label_dir / 'launch-manifest.json'),
                      owner_evidence=str(a.run_dir / 'restore.json'))
        write(a.run_dir / 'complete.json', result)
        return result
    except BaseException:
        (a.run_dir / 'error.txt').write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('code', 'manifest', 'current-restore', 'run-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('manifest-sha', 'current-restore-sha', 'runner-sha'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--phase', choices=OVERRIDES, required=True)
    p.add_argument('--from-phase', choices=OVERRIDES, required=True)
    client = p.add_mutually_exclusive_group(required=True)
    client.add_argument('--prior-client-record', type=Path)
    client.add_argument('--first-phase-no-client', action='store_true')
    p.add_argument('--execute', action='store_true')
    print(json.dumps(switch(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
