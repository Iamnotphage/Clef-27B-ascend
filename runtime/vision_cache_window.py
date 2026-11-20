#!/usr/bin/env python3
"""Isolated ERQA encoder-cache window; frozen c80 lifecycle, GET-only arm readiness."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

FROZEN_SHA = '3178f76375e80c1f0c23525f7d0742766c1c7a531ae1550b75b89f03edd78e28'
STOP_SHA = 'd6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886'
AA3E = 'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'
CAPACITIES = (16384, 131072, 262144)
GIB = 1 << 30
ADMISSION_BYTES = 2 * GIB + (227549 - 131072) * 5120 * 2 + GIB // 2
REQUEST_FIELDS = ('id', 'record_index', 'image_count', 'option_count', 'image_sha256',
                  'image_order', 'request_structural_sha256', 'request_wire_sha256')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_frozen(path):
    path = Path(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_SHA,
            'frozen lifecycle SHA mismatch')
    spec = importlib.util.spec_from_file_location('vision_frozen_c80', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmdline_evidence(raw):
    require(isinstance(raw, bytes) and raw.endswith(b'\0'), 'incomplete raw argv')
    argv = raw[:-1].decode(errors='replace').split('\0')
    require(argv and argv[0], 'empty executable')
    return {'argv': argv, 'd0_cmdline': raw.replace(b'\0', b' ').decode(errors='replace').strip(),
            'raw_sha256': hashlib.sha256(raw).hexdigest(), 'raw_hex': raw.hex()}


def bound_root(frozen, saved, expected_argv, raw_reader=None):
    require(all(k in saved for k in frozen.FIELDS), 'incomplete saved root identity')
    require(isinstance(saved['starttime'], str) and saved['starttime'] and saved['boot_id'],
            'missing root lifetime')
    require(isinstance(expected_argv, list) and expected_argv and
            all(isinstance(x, str) for x in expected_argv), 'invalid expected argv')
    raw_reader = raw_reader or (lambda pid: (Path('/proc') / str(pid) / 'cmdline').read_bytes())
    first = frozen.proc_identity(saved['pid'])
    require(frozen.same_identity(saved, first), 'root identity drift before argv read')
    evidence = cmdline_evidence(raw_reader(saved['pid']))
    second = frozen.proc_identity(saved['pid'])
    require(frozen.same_identity(saved, second), 'root identity drift after argv read')
    require(saved['pid'] == saved['pgid'] == saved['sid'], 'root must lead its session')
    require(evidence['argv'] == expected_argv, 'root argv mismatch')
    return {**second, **evidence, 'cmdline': evidence['d0_cmdline']}


def strict_npu(frozen, expected_pids=None, *, deadline=None):
    # Keep raw command evidence; c80's PID/chip check alone does not check card.
    remaining = 30 if deadline is None else min(30, deadline - time.monotonic())
    require(remaining > 0, 'NPU proof deadline')
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=remaining)
    require(deadline is None or time.monotonic() < deadline, 'NPU proof deadline')
    require(result.returncode == 0, 'NPU inventory command failed')
    rows = frozen.npu_rows(result.stdout)
    require(all(r['card'] == 7 and r['chip'] in (0, 1) for r in rows), 'NPU card/chip drift')
    if expected_pids is None:
        require(not rows and 'No running processes found in NPU 7' in result.stdout,
                'card7 must be empty')
    else:
        require(rows and {r['pid'] for r in rows}.issubset(expected_pids), 'unknown NPU PID')
        workers = [r for r in rows if r['name'] == 'VLLMWorker_TP']
        require(len(workers) == 2 and {r['chip'] for r in workers} == {0, 1}, 'TP2 mapping drift')
    return {'rows': rows, 'stdout': result.stdout, 'stderr': result.stderr, 'rc': result.returncode}


class OwnedArm:
    """One arm owns its launchers, service, clients and sampler until settled."""

    def __init__(self, frozen, directory, protected):
        self.f = frozen
        self.directory = Path(directory)
        self.protected = set(protected)
        self.token = secrets.token_hex(16)
        self.processes = []
        self.identities = []
        self.extra_entries = []
        self.cleanup_deadline = None
        self.closed = False
        self.clean = False
        self.record = {}
        self.registry = frozen.DescendantRegistry(
            os.getpid(), self.directory / 'descendants.json', self.protected,
            token=self.token).start()

    def spawn(self, name, argv, env=None, cwd=None):
        require(not self.closed, 'arm already closed')
        require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv),
                'argv must be an exact string list')
        argv = list(argv)
        # env -i intentionally discards parent variables; carry ownership explicitly.
        if len(argv) >= 2 and Path(argv[0]).name == 'env' and argv[1] == '-i':
            argv.insert(2, f'{self.f.OWNER_ENV}={self.token}')
        environment = dict(os.environ if env is None else env)
        environment[self.f.OWNER_ENV] = self.token
        with (self.directory / (name + '.stdout')).open('xb') as out, \
                (self.directory / (name + '.stderr')).open('xb') as err:
            proc = subprocess.Popen(argv, env=environment, cwd=cwd, stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        self.processes.append(proc)  # retain even if identity capture fails
        identity = self.f.proc_identity(proc.pid)
        require(identity and identity['pgid'] == identity['sid'] == proc.pid and
                proc.pid not in self.protected, 'owned child identity unavailable')
        self.identities.append(identity)
        self.registry.record_now()
        self.f.write_json(self.directory / (name + '-identity.json'),
                          {'identity': identity, 'argv': argv, 'owner': self.token})
        return proc

    def entries(self):
        return self.registry.snapshot() + self.identities + self.extra_entries

    def close(self, port):
        # One absolute budget and one cleanup path, also on exceptions/finally.
        if self.closed:
            return self.clean
        self.closed = True
        self.cleanup_deadline = time.monotonic() + 300.0
        try:
            self.registry.stop()  # final synchronous registration precedes any signal
            entries = self.entries()
            registry = SimpleNamespace(snapshot=lambda: entries)
            cleanup = self.f.cleanup_attributed(None, None, registry, self.protected, reap=True)
            self.record['cleanup'] = cleanup
            remaining = self.cleanup_deadline - time.monotonic()
            # Even an immediate empty cleanup must obtain two empty snapshots.
            settle = self.f.settle_attributed(entries, None, deadline=max(0.0, remaining),
                                              min_empty=2, reap=True)
            self.record['settle'] = settle
            require(not self.f.cleanup_blocks_restore(cleanup), 'protected cleanup overlap')
            require(not cleanup.get('unknown') and not cleanup.get('drift'), 'unknown cleanup identity')
            require(len(self.identities) == len(self.processes), 'unbound owned process')
            require(settle['clean'] is True and settle['deadline_hit'] is False and
                    not settle['residual'] and time.monotonic() < self.cleanup_deadline,
                    'unclean or expired settle')
            table = settle['table']  # residual and table are from the same final snapshot
            require(not self.f.attributed_survivors(registry, table=table, reader=table.get),
                    'settle snapshot retains owned processes')
            require(all(p.poll() is not None for p in self.processes), 'owned runner has not exited')
            require(self.f.port_open(port) == 111, 'arm port still open')
            self.record['npu'] = strict_npu(self.f, deadline=self.cleanup_deadline)
            require(time.monotonic() < self.cleanup_deadline, 'cleanup proof deadline')
            self.clean = True
        except BaseException as exc:
            self.record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            self.f.write_json(self.directory / 'cleanup.json', self.record)
        return self.clean


def final_restore_gate(frozen, plan, saved, arms, stop_called, result):
    require(stop_called, 'stop was never called')
    require(all(a.closed and a.clean for a in arms), 'unclean arm blocks restore')
    entries = [entry for arm in arms for entry in arm.entries()]
    entries += plan.get('old_tree', [])
    registry = SimpleNamespace(snapshot=lambda: entries)
    table = frozen.scan()  # fresh gate; never reuse a stale settle snapshot
    require(not pending_runners(frozen, table), 'pending vision runner')
    require(not frozen.attributed_survivors(registry, table=table, reader=table.get),
            'fresh attributed survivor or PID drift')
    args = SimpleNamespace(port=plan['port'], expected_source_sha=AA3E)
    allowed, reason, context = frozen.recovery_gate(
        args, plan, saved, True, None, frozen.sha256(plan['source_target']),
        registry=registry, table=table)
    result['recovery_gate'] = {'allow': allowed, 'reason': reason, **context}
    require(allowed, reason)
    result['recovery_npu'] = strict_npu(frozen)


def restore_once(frozen, plan, result):
    require(result.get('restore_attempts', 0) == 0, 'restore already consumed')
    result['restore_attempts'] = 1  # consume before begin_restore, including failures
    require(frozen.begin_restore(plan, Path(plan['run_dir']), result), 'subreaper restore blocked')


def option(argv, key):
    require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv), f'missing/duplicate {key}')
    return argv[argv.index(key) + 1]


def startup_record(text, cap):
    marker = 'Clef encoder cache candidate initialized: '
    records = [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
    require(len(records) == 1, 'missing/duplicate constructor record')
    row = records[0]
    for key in ('async_scheduling', 'balance_scheduling'):
        require(row.get(key) is False, f'{key} drift')
    for key, value in {'compute_config': 16384, 'compute_effective': 16384,
                       'cache_config_before': 16384, 'cache_requested': cap,
                       'cache_effective': cap, 'unit': 'encoder_embeddings'}.items():
        require(row.get(key) == value, f'constructor {key} drift')
    cls = 'clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler'
    parent = 'vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler'
    require(row.get('parent') == parent and row.get('mro', [])[:2] == [cls, parent],
            'constructor class/MRO drift')
    return row


def verify_startup(frozen, plan, phase, manifest, identity, text):
    cap, gate = phase['cache_cli'], phase['startup_gate']
    row = startup_record(text, cap)
    for flag in ('enable_mm_processor_stats', 'enable_logging_iteration_details'):
        require(flag + '=False' in text, f'missing disabled {flag} evidence')
        require(not re.search(flag + r'\s*[=:]\s*True', text), f'{flag} enabled')
    lines = [line for line in text.splitlines() if 'Clef encoder cache candidate initialized: ' in line]
    engine = re.search(r'EngineCore[^)]*pid=(\d+)', lines[0])
    require(engine, 'constructor EngineCore PID missing')
    engine_pid = int(engine.group(1))
    table = frozen.scan()
    require(engine_pid in frozen.tree(table, identity['pid']) and engine_pid in table,
            'constructor EngineCore not in current service tree')
    for flag in ('enable_mm_processor_stats', 'enable_logging_iteration_details'):
        owners = {int(m.group(1)) for line in text.splitlines() if flag + '=False' in line
                  for m in [re.search(r'EngineCore[^)]*pid=(\d+)', line)] if m}
        require(owners == {engine_pid}, 'timing evidence belongs to another EngineCore')
    argv = manifest['argv']
    utilization = [option(phase['service_argv'], '--gpu-memory-utilization'),
                   option(argv, '--gpu-memory-utilization')]
    require(all(isinstance(value, str) for value in utilization), 'utilization argv type drift')
    utilization = [float(value) for value in utilization] + [manifest['effective'].get('gpu_memory_utilization')]
    require(all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
                for value in utilization) and utilization[0] == utilization[1] == utilization[2],
            'plan/argv/effective gpu memory utilization drift')
    for key, value in {'--scheduler-cls': gate['scheduler_cls'], '--runner': 'pooling',
                       '--dtype': 'bfloat16', '--max-model-len': '16384',
                       '--max-num-batched-tokens': '16384', '--max-num-seqs': '1',
                       '--tensor-parallel-size': '2'}.items():
        require(option(argv, key) == value, f'service {key} drift')
    require(json.loads(option(argv, '--limit-mm-per-prompt')) == {'image': 16, 'video': 0},
            'image limit drift')
    require(option(argv, '--mm-processor-kwargs') ==
            option(phase['service_argv'], '--mm-processor-kwargs'), 'processor drift')
    require(json.loads(option(argv, '--mm-processor-kwargs'))['images_kwargs']['size'] ==
            {'shortest_edge': 65536, 'longest_edge': 1048576}, 'processor image dimensions drift')
    require('--enforce-eager' in argv and '--enable-mm-processor-stats' not in argv,
            'eager/timing drift')
    require(frozen.sha256(gate['module_path']) == gate['module_sha256'] and
            manifest['effective']['encoder_cache_scheduler_sha256'] == gate['module_sha256'],
            'scheduler module SHA drift')
    live = frozen.read_root_env(identity['pid'])
    path = live.get('PYTHONPATH', b'').decode().split(':')
    require(path[0] == str(Path(gate['module_path']).parent), 'PYTHONPATH module resolution drift')
    require(live.get('CLEF_ENCODER_CACHE_TOKENS') == str(cap).encode(), 'cache environment drift')
    require(frozen.joint_env(live)['ok'], 'joint environment drift')
    require(frozen.sha256(plan['cleanup_restore']['source_target']) == AA3E, 'aa3e source drift')
    service_dir = Path(manifest['output_dir'])
    require(frozen.read_json(service_dir / 'joint-patch-record.json').get('branch') == 'aa3e-noop',
            'joint patch branch drift')
    return {'constructor': row, 'module_sha256': gate['module_sha256'], 'pythonpath': path,
            'identity': identity, 'engine_identity': table[engine_pid], 'manifest': manifest}


def json_lines(path, complete=True):
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    if not complete and raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def read_reference(frozen, plan):
    reference = plan['evaluation_frozen']['request_reference']
    path = Path(reference['remote_path'])
    require(path.stat().st_size == reference['bytes'] and frozen.sha256(path) == reference['sha256'],
            'fixed request reference SHA/bytes drift')
    rows = json_lines(path)
    require(len(rows) == 400 and [r['id'] for r in rows] == [f'erqa-{i:04}' for i in range(400)] and
            all(r['record_index'] == i and all(k in r for k in REQUEST_FIELDS) for i, r in enumerate(rows)),
            'fixed request reference coverage/order/fields drift')
    return rows


def verify_predictions(frozen, directory, expected, reference, quality=None):
    directory = Path(directory)
    rows = json_lines(directory / 'predictions.jsonl')
    require(len(rows) == len(expected) == 400, 'ERQA coverage incomplete')
    require([r['id'] for r in rows] == [f'erqa-{i:04}' for i in range(400)], 'ERQA order/IDs drift')
    require(len(reference) == 400, 'fixed request reference required')
    for row, sample, fixed in zip(rows, expected, reference):
        require(all(row[k] == fixed[k] for k in REQUEST_FIELDS), 'fixed request identity drift')
        require(row['id'] == sample['id'] and row['record_index'] == sample['record_index'],
                'ERQA input order drift')
        require(row['image_sha256'] == [img['sha256'] for img in sample['images']] and
                row['image_count'] == sample['image_count'], 'ERQA image input drift')
        require(row['status'] == 'ok', 'ERQA failed response')
        probabilities = row['probabilities']
        require(isinstance(probabilities, dict) and len(probabilities) == row['option_count'] and
                set(probabilities) == {chr(65 + i) for i in range(row['option_count'])} and
                row['prediction'] in probabilities and
                all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1
                    for v in probabilities.values()) and abs(sum(probabilities.values()) - 1) < 0.001,
                'incomplete/nonfinite probabilities')
        require(math.isfinite(row['latency_ms']) and row['latency_ms'] > 0 and
                row['started_at'] < row['finished_at'], 'ERQA timing missing')
        sd = directory / 'samples' / row['id']
        record = frozen.read_json(sd / 'record.json')
        attempts = json_lines(sd / 'attempts.jsonl')
        require(record['attempts'] == attempts and len(attempts) == 1 and
                attempts[0]['attempt'] == 1 and attempts[0]['status_code'] == 200 and
                not attempts[0].get('error'), 'ERQA retry/error')
        require(frozen.sha256(sd / 'request.json') == row['request_wire_sha256'], 'wire hash drift')
        require(frozen.sha256(sd / 'response.json') == attempts[0]['response_sha256'] and
                (sd / 'response.json').read_bytes() == (sd / 'response-attempt-1.bin').read_bytes(),
                'raw response drift')
        require(frozen.read_json(sd / 'response.json')['answers']['answer']['probabilities'] == probabilities,
                'probabilities differ from raw response')
    summary = frozen.read_json(directory / 'summary.json')
    require(all(summary['coverage'][k] == v for k, v in
                {'fixed_total': 400, 'ok': 400, 'failed': 0, 'retry_count': 0}.items()) and
            summary['run']['skipped_resumed'] == 0 and summary['run']['retries'] == 1 and
            summary['run']['scope'] == 'full', 'ERQA summary drift')
    score = frozen.read_json(directory / 'erqa-score.json')
    require(score['request_verification']['checked'] is True and
            score['request_verification']['passed'] is True and
            score['request_verification']['mismatches'] == [] and
            score['summary_comparison']['equal'] is True, 'offline request/score verification failed')
    if quality is not None:
        accuracy = score.get('accuracy', {})
        correct, total = accuracy.get('correct'), accuracy.get('total')
        valid = (type(correct) is int and 0 <= correct <= 400 and type(total) is int and total == 400 and
                 accuracy.get('value') == correct / 400 and
                 all(summary.get('accuracy', {}).get(key) == accuracy.get(key)
                     for key in ('correct', 'total', 'value')))
        passed = valid and correct >= quality['min_correct']
        reason = 'passed' if passed else ('below minimum correct' if valid else 'invalid score accuracy')
        frozen.write_json(directory / 'quality-gate.json',
                          {**quality, 'actual_correct': correct, 'actual_total': total,
                           'passed': passed, 'reason': reason})
        require(passed, f'ERQA quality gate: {reason}; correct={correct}/{total}, '
                        f"baseline={quality['baseline_correct']}/400, minimum={quality['min_correct']}")
    return reference


def hbm_values(text):
    def field(name):
        values = re.findall(r'^\s*' + re.escape(name) + r'\s*:\s*(\d+)\s*$', text, re.M)
        require(len(values) == 1, f'HBM field missing/duplicate: {name}')
        return int(values[0])
    capacity, used = field('HBM Capacity(MB)'), field('HBM Usage Rate(%)')
    require(capacity == 65536 and 0 <= used <= 100, 'HBM capacity/usage drift')
    return {'capacity_mb': capacity, 'usage_pct': used,
            'free_lower_bytes': max(0, 99 - used) * capacity * 1048576 // 100}


def sample_loop(frozen, directory, identity):
    """Owned CPU sampler, fixed 5s cadence; no synchronized model instrumentation."""
    directory = Path(directory)
    slot = time.monotonic()
    with (directory / 'hbm.jsonl').open('x') as out:
        while True:
            started = time.monotonic()
            record = {'started': started, 'ok': False, 'queries': [], 'chips': []}
            try:
                require(started - slot < 1, 'sampler cadence drift')
                table = frozen.scan()
                require(frozen.same_process(identity, table.get(identity['pid'])), 'sampler root drift')
                owned = frozen.tree(table, identity['pid'])
                ready = (directory / 'ready.json').exists()
                if not ready:
                    argv = ['/usr/local/bin/npu-smi', 'info']
                    result = subprocess.run(argv, capture_output=True, text=True, timeout=1)
                    record['queries'].append({'argv': argv, 'rc': result.returncode,
                                              'stdout': result.stdout, 'stderr': result.stderr})
                    require(result.returncode == 0, 'startup NPU query failed')
                    rows = frozen.npu_rows(result.stdout)
                    require(rows or 'No running processes found in NPU 7' in result.stdout,
                            'startup NPU empty state unproven')
                    require(all(r['card'] == 7 and r['chip'] in (0, 1) and r['pid'] in owned
                                for r in rows), 'startup NPU card/chip/PID drift')
                for chip in (0, 1):
                    for kind in (('usages', 'proc-mem') if ready else ('usages',)):
                        argv = ['/usr/local/bin/npu-smi', 'info', '-t', kind, '-i', '7', '-c', str(chip)]
                        result = subprocess.run(argv, capture_output=True, text=True,
                                                timeout=min(1.0, max(0.001, started + 4 - time.monotonic())))
                        record['queries'].append({'argv': argv, 'rc': result.returncode,
                                                  'stdout': result.stdout, 'stderr': result.stderr})
                        require(result.returncode == 0, 'HBM query failed')
                        if kind == 'usages':
                            values = {'card': 7, 'chip': chip, **hbm_values(result.stdout)}
                            require(values['free_lower_bytes'] > 2 * GIB, 'HBM runtime reserve exhausted')
                        else:
                            matches = re.findall(r'Process id:(\d+) Process name:(.*?)\s+Process memory\(MB\):(\d+)',
                                                 result.stdout)
                            require(matches, 'HBM process data missing')
                            pids = [int(pid) for pid, _, _ in matches]
                            require(set(pids).issubset(owned), 'HBM unknown PID')
                            require(all(frozen.same_process(table[pid], frozen.proc_identity(pid)) for pid in pids),
                                    'HBM PID reuse/drift')
                            workers = [int(pid) for pid, name, _ in matches if name.strip() == 'VLLMWorker_TP']
                            bound = frozen.read_json(directory / 'ready.json')['workers']
                            require(workers == [bound[str(chip)]['pid']] and
                                    frozen.same_process(bound[str(chip)], table.get(workers[0])),
                                    'HBM TP worker lifetime/mapping drift')
                            values['processes'] = [{'identity': table[int(pid)], 'name': name.strip(),
                                                    'memory_mb': int(mem)} for pid, name, mem in matches]
                    record['chips'].append(values)
                require(time.monotonic() - started < 4, 'HBM round deadline')
                record['ok'] = True
            except BaseException as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['finished'] = time.monotonic()
            out.write(json.dumps(record) + '\n')
            out.flush()
            if not record['ok']:
                return 1
            slot += 5
            time.sleep(max(0, slot - time.monotonic()))


def hbm_gate(samples, now):
    require(samples and all(s['ok'] is True for s in samples), 'HBM missing/error sample')
    require(0 <= now - samples[-1]['started'] <= 10, 'HBM stale sample')
    require(all(0 <= s['finished'] - s['started'] < 4 for s in samples), 'HBM round deadline')
    require(all(4 <= b['started'] - a['started'] <= 6 for a, b in zip(samples, samples[1:])),
            'HBM cadence/missing sample')
    minima = [float('inf'), float('inf')]
    for sample in samples:
        require([(c['card'], c['chip']) for c in sample['chips']] == [(7, 0), (7, 1)],
                'HBM card/chip drift')
        for chip in sample['chips']:
            require(chip['free_lower_bytes'] > 2 * GIB, 'HBM runtime reserve exhausted')
            minima[chip['chip']] = min(minima[chip['chip']], chip['free_lower_bytes'])
    return minima


def check_environment(frozen, pid, expected):
    live = frozen.read_root_env(pid)
    for key, item in expected.items():
        value = live.get(key)
        require(value == item['value'].encode() and
                hashlib.sha256(value).hexdigest() == item['value_sha256'] and
                len(value.split(b':')) == item['elements'], f'{key} drift')


def pending_runners(frozen, table):
    return sorted(set(frozen.pending_clients(table)) | {
        pid for pid, row in table.items() if pid != os.getpid() and row['state'] != 'Z'
        and any(Path(arg).name in {'vision_cache_window.py', 'erqa_baseline.py', 'erqa_score.py'}
                for arg in row.get('cmdline', []))})


def check_pins(frozen, plan):
    for pin in plan['runtime_file_pins']:
        path = Path(pin['path'])
        require(path.stat().st_size == pin['bytes'] and frozen.sha256(path) == pin['sha256'],
                f'runtime pin drift: {path}')
    for relative, pin in plan['candidate_files'].items():
        path = Path(plan['remote_stage']) / relative
        require(path.stat().st_size == pin['bytes'] and frozen.sha256(path) == pin['sha256'],
                f'candidate pin drift: {path}')


def planned_arm_order(plan):
    order = plan.get('arm_order', list(CAPACITIES))
    require(isinstance(order, list) and all(type(cap) is int for cap in order) and
            tuple(order) in (CAPACITIES, (131072, 16384), (131072, 262144)), 'unsupported arm_order')
    phases = plan['phases']
    require(all(type(p['cache_cli']) is int for p in phases) and
            tuple(p['cache_cli'] for p in phases) == tuple(order), 'capacity/order mismatch')
    require(len({p['name'] for p in phases}) == len(phases), 'duplicate arm names')
    return tuple(order)


def validate_plan(plan):
    require(plan['schema'] == 'clef-vision-cache-plan/2', 'plan schema mismatch')
    planned_arm_order(plan)
    if 'quality_gate' in plan:
        quality = plan['quality_gate']
        require(quality == {'baseline_correct': 219, 'total': 400, 'max_drop_pp': 2, 'min_correct': 211} and
                all(type(value) is int for value in quality.values()), 'ERQA quality policy drift')
    require(plan['candidate_commit'] == '3bf466d913ea269d4f4bb37b560a6ee7a3fdbf47', 'candidate drift')
    require(plan['memory_guard']['sample_interval_s'] == 5 and
            plan['memory_guard']['query_timeout_s'] == 1 and
            plan['memory_guard']['sample_round_timeout_s'] == 4 and
            plan['memory_guard']['max_sample_age_s'] == 10, 'sampler policy drift')
    for phase in plan['phases']:
        require(phase['ready']['method'] == 'GET' and phase['ready']['image_warmup'] == 0 and
                phase['ready']['text_warmup'] == 0, 'ready policy drift')
        require([p['label'] for p in phase['passes']] == ['cold-first', 'natural-second'], 'pass order drift')
        for trial in phase['passes']:
            require(option(trial['argv'], '--retries') == '1' and
                    option(trial['argv'], '--scope') == 'full' and '--resume' not in trial['argv'] and
                    '--limit' not in trial['argv'] and '--verify-requests' in trial['score_argv'],
                    'ERQA attempts/scope drift')
    require(plan['cleanup_restore']['max_restore_attempts'] == 1, 'restore count drift')


def service_identity(frozen, manifest, launcher, directory, launched_ns):
    path = directory / 'launch-manifest.json'
    require(path.stat().st_mtime_ns >= launched_ns, 'stale manifest')
    require(Path(manifest['output_dir']) == directory, 'manifest output mismatch')
    live = frozen.proc_identity(int(manifest['pid']))
    require(live and live['pid'] == launcher.pid and live['pgid'] == live['sid'] == live['pid'],
            'service not the owned exec process')
    argv = cmdline_evidence((Path('/proc') / str(live['pid']) / 'cmdline').read_bytes())['argv']
    require(argv == manifest['argv'], 'live service argv mismatch')
    return live


def monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, deadline):
    require(time.monotonic() < deadline, 'arm/client deadline')
    require(service.poll() is None and frozen.same_process(identity, frozen.proc_identity(identity['pid'])),
            'service exited/restarted')
    require(sampler.poll() is None, 'sampler exited')
    text = (service_dir / 'service.log').read_text(errors='replace') if (service_dir / 'service.log').exists() else ''
    require(not re.search(r'out of memory|OutOfMemoryError|memory allocation fail', text, re.I), 'service OOM')
    if (arm.directory / 'ready.json').exists():
        ready = frozen.read_json(arm.directory / 'ready.json')
        startup_record(text, ready['constructor']['cache_requested'])
        require(all(frozen.same_process(row, frozen.proc_identity(row['pid']))
                    for row in [ready['engine_identity'], *ready['workers'].values()]), 'engine/TP worker restarted')
    samples = json_lines(arm.directory / 'hbm.jsonl', complete=False)
    if samples or time.monotonic() - sampling_started >= 5:
        hbm_gate(samples, time.monotonic())
    return samples


def run_phase(frozen, plan, phase, arm, expected, reference, active_deadline, frozen_path):
    service_dir = Path(option(phase['service_argv'], '--output-dir'))
    require(not service_dir.exists(), 'fresh service directory required')
    launched_ns = time.time_ns()
    service = arm.spawn('launch', phase['launch_argv'], cwd=phase['launch_cwd'])
    ready_deadline = min(active_deadline, time.monotonic() + phase['startup_timeout_s'])
    manifest_path = service_dir / 'launch-manifest.json'
    while not manifest_path.exists():
        require(service.poll() is None and time.monotonic() < ready_deadline, 'launch/manifest failed')
        time.sleep(0.2)
    # A manifest is written immediately before exec; wait only for that transition.
    manifest = frozen.read_json(manifest_path)
    while True:
        try:
            identity = service_identity(frozen, manifest, service, service_dir, launched_ns)
            break
        except RuntimeError:
            require(service.poll() is None and time.monotonic() < ready_deadline, 'service exec failed')
            time.sleep(0.2)
    frozen.write_json(arm.directory / 'service-identity.json', identity)
    sampler = arm.spawn('sampler', [sys.executable, str(Path(__file__).resolve()), '--sample-dir',
                        str(arm.directory), '--frozen', str(frozen_path)])
    sampling_started = time.monotonic()
    base = 'http://127.0.0.1:18143'
    while True:
        monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, ready_deadline)
        try:
            health = frozen.http_json(base, '/health', timeout=2)
        except Exception:
            health = None
        if health == phase['ready']['expected_body']:
            break
        time.sleep(0.2)
    record = verify_startup(frozen, plan, phase, manifest, identity,
                            (service_dir / 'service.log').read_text(errors='replace'))
    check_environment(frozen, identity['pid'], plan['candidate_environment_expect'])
    record['npu'] = strict_npu(frozen, frozen.tree(frozen.scan(), identity['pid']))
    record['workers'] = {str(row['chip']): frozen.proc_identity(row['pid'])
                         for row in record['npu']['rows'] if row['name'] == 'VLLMWorker_TP'}
    require(all(record['workers'].values()), 'worker identity unavailable')
    frozen.write_json(arm.directory / 'ready.json', record)
    # Require a fresh ready-state sample before the first image POST.
    ready_at = time.monotonic()
    while True:
        samples = monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, ready_deadline)
        if samples and samples[-1]['started'] >= ready_at:
            break
        time.sleep(0.2)
    for trial in phase['passes']:
        output = Path(trial['output_dir'])
        require(not output.exists(), 'fresh ERQA output required')
        client = arm.spawn(trial['label'], trial['argv'])
        deadline = min(active_deadline, time.monotonic() + trial['outer_wall_deadline_s'])
        while True:
            monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, deadline)
            # The fixed runner writes predictions only at the end; progress is flushed per question.
            for row in json_lines(arm.directory / (trial['label'] + '.stdout'), complete=False):
                require('status' not in row or row['status'] == 'ok', 'ERQA live failure')
            if client.poll() is not None:
                break
            time.sleep(0.2)
        require(client.returncode == 0, 'ERQA client failed')
        score = arm.spawn(trial['label'] + '-score', trial['score_argv'])
        deadline = min(active_deadline, time.monotonic() + trial['score_timeout_s'])
        while True:
            monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, deadline)
            if score.poll() is not None:
                break
            time.sleep(0.2)
        require(score.returncode == 0, 'ERQA score failed')
        reference = verify_predictions(frozen, output, expected, reference, plan.get('quality_gate'))
        (output / 'expected-inputs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in expected))
        require(frozen.same_process(identity, frozen.proc_identity(identity['pid'])), 'service lifetime drift')
    samples = monitor(frozen, arm, service, sampler, identity, service_dir, sampling_started, active_deadline)
    return reference, hbm_gate(samples, time.monotonic())


def verify_restore(frozen, plan, result, started_ns, deadline):
    restore = plan['cleanup_restore']
    base = 'http://127.0.0.1:18143'
    while True:
        require(time.monotonic() < deadline, 'restore deadline')
        try:
            health = frozen.http_json(base, '/health', timeout=2)
        except Exception:
            health = None
        if health and health.get('status') == 'ready' and health.get('model') == 'clef':
            break
        time.sleep(0.5)
    output = Path(restore['restore_output'])
    path = output / 'launch-manifest.json'
    require(path.stat().st_mtime_ns >= started_ns, 'stale restore manifest')
    manifest = frozen.read_json(path)
    require(Path(manifest['output_dir']) == output and manifest['config'] == 's1', 'restore output/config drift')
    expected = restore['restore_source']['selected']
    require(manifest['argv'] == expected['argv'], 'restore argv drift')
    require(manifest['effective'].get('encoder_cache_tokens', 0) == 0, 'restore candidate enabled')
    pid = int(manifest['pid'])
    live = frozen.wait_restored_root(pid, timeout=min(1.0, max(0, deadline - time.monotonic())))
    require(live and live['pgid'] == live['sid'] == pid and
            live['ppid'] in (1, result['restore']['wrapper_pid']) and pid != plan['current_service']['root'],
            'restore identity/parent drift')
    bound_root(frozen, live, expected['argv'])
    require(frozen.read_json(output / 'joint-patch-record.json').get('branch') == 'aa3e-noop',
            'restore branch drift')
    environment = frozen.read_root_env(pid)
    require(environment.get('CLEF_ENCODER_CACHE_TOKENS', b'0') == b'0' and
            frozen.joint_env(environment)['ok'], 'restore environment/feature drift')
    check_environment(frozen, pid, plan['restore_environment_expect'])
    require(frozen.env_evidence(pid, restore['clean_env_expect'])['match'], 'restore original env drift')
    table = frozen.scan()
    pids = frozen.tree(table, pid)
    require(all(table[p]['pgid'] == pid for p in pids), 'restore descendant escaped PGID')
    npu = strict_npu(frozen, pids)
    info = frozen.http_json(base, '/v1/decide/info', timeout=2)
    require(info.get('config') == 's1', 'restore info drift')
    checks = {'health': health, 'info': info, 'npu': npu, 'identity': live}
    for key, endpoint in (('text', '/v1/systemone'), ('image', '/v1/decide')):
        require(time.monotonic() < deadline, 'restore probe deadline')
        status, body = frozen.post_json(base, endpoint, Path(restore[key + '_body']).read_text(),
                                       timeout=min(120, deadline - time.monotonic()))
        require(status == 200, f'restore {key} check failed')
        checks[key] = body
    require(frozen.same_identity(live, frozen.proc_identity(pid)), 'restore root changed during checks')
    require(not pending_runners(frozen, frozen.scan()), 'restore pending client')
    check_pins(frozen, plan)
    require(time.monotonic() < deadline, 'restore completion deadline')
    result['restore_checks'] = checks


def execute(frozen, plan, saved, expected, frozen_path):
    order = planned_arm_order(plan)
    run_dir = Path(plan['run_root'])
    require(not run_dir.exists() and not Path(plan['cleanup_restore']['restore_output']).exists(),
            'fresh window/restore directories required')
    check_pins(frozen, plan)
    reference = read_reference(frozen, plan)
    restore_plan = {**plan['cleanup_restore'], 'run_dir': str(run_dir), 'port': 18143}
    argv = plan['cleanup_restore']['restore_source']['selected']['argv']
    bound_root(frozen, saved, argv)
    original_subreaper = frozen.get_subreaper()
    require(original_subreaper is not None and frozen.enable_subreaper(), 'subreaper unavailable')
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {'ok': False, 'restore_attempts': 0, 'arms': [], 'arm_order': list(order),
              'by_capacity': {}, 'schema': 'vision-cache-window-result/1'}
    stop_called, arms = False, []
    global_deadline = time.monotonic() + plan['budgets_s']['overall_outer']
    active_end = global_deadline - 900 - 420 - 120
    try:
        table = frozen.scan()
        old_pids = frozen.tree(table, saved['pid'])
        require(all(table[p]['pgid'] == saved['pgid'] for p in old_pids), 'old descendant escaped group')
        require({p for p, r in table.items() if r['pgid'] == saved['pgid'] and r['state'] != 'Z'} <= old_pids,
                'unknown old PGID member')
        restore_plan['old_tree'] = [table[p] for p in old_pids]
        require(not pending_runners(frozen, table) and frozen.port_open(18143) == 0, 'preflight client/port gate')
        result['before_npu'] = strict_npu(frozen, old_pids)
        check_environment(frozen, saved['pid'], plan['restore_environment_expect'])
        initial_dir = run_dir / 'initial-stop'
        initial_dir.mkdir()
        initial = OwnedArm(frozen, initial_dir, old_pids)
        arms.append(initial)
        initial.extra_entries = restore_plan['old_tree']
        # Rebind full identity and exact argv immediately before the frozen stopper.
        identity = bound_root(frozen, saved, argv)
        frozen.write_json(run_dir / 'root-identity-d0.json', identity)
        frozen.write_json(run_dir / 'root-identity-argv.json', {'saved': saved, 'bound': identity})
        stop_called = True
        stopper = initial.spawn('stop', plan['cleanup_restore']['stop_argv_template'])
        stop_rc = stopper.wait(timeout=120)
        closed = initial.close(18143)
        require(stop_rc == 0 and closed, 'initial stop/settle failed')
        minima_by_capacity = {}
        for phase in plan['phases']:
            cap = phase['cache_cli']
            if cap == 262144:
                require(131072 in minima_by_capacity, '131072 evidence required before 262144')
                prior_min = minima_by_capacity[131072]
                if min(prior_min) < ADMISSION_BYTES:
                    record = {'capacity': cap, 'skipped': 'HBM conservative admission',
                              'prior_capacity': 131072, 'prior_min_free_bytes': prior_min,
                              'required': ADMISSION_BYTES}
                    result['arms'].append(record)
                    result['by_capacity'][str(cap)] = record
                    break
            require(time.monotonic() + phase['max_arm_active_s'] < active_end, 'insufficient global budget')
            final_restore_gate(frozen, restore_plan, saved, arms, stop_called, result)
            directory = run_dir / phase['name']
            directory.mkdir(exist_ok=False)
            arm = OwnedArm(frozen, directory, old_pids)
            arms.append(arm)
            try:
                reference, arm_min = run_phase(frozen, plan, phase, arm, expected, reference,
                                                min(active_end, time.monotonic() + phase['max_arm_active_s']),
                                                frozen_path)
                minima_by_capacity[cap] = arm_min
                record = {'capacity': cap, 'ok': True, 'min_free_lower_bytes': arm_min}
                result['arms'].append(record)
                result['by_capacity'][str(cap)] = record
            finally:
                clean = arm.close(18143)
            require(clean, 'arm cleanup/settle failed')
            check_pins(frozen, plan)
        result['experiment_ok'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for arm in arms:
            arm.close(18143)  # idempotent: never restart a cleanup/settle budget
        try:
            final_restore_gate(frozen, restore_plan, saved, arms, stop_called, result)
            check_pins(frozen, plan)
            started_ns = time.time_ns()
            restore_once(frozen, restore_plan, result)
            verify_restore(frozen, plan, result, started_ns, min(global_deadline, time.monotonic() + 900))
            result['ok'] = result.get('experiment_ok', False)
        except BaseException as exc:
            result['restore_error'] = f'{type(exc).__name__}: {exc}'
        result['subreaper_restored'] = frozen.restore_subreaper(original_subreaper)
        if not result['subreaper_restored']:
            result['ok'] = False
        frozen.write_json(run_dir / 'window.json', result)
    return 0 if result['ok'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha')
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--root-identity', type=Path)
    parser.add_argument('--expected-inputs', type=Path)
    parser.add_argument('--sample-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    frozen = load_frozen(args.frozen)
    if args.sample_dir:
        return sample_loop(frozen, args.sample_dir, frozen.read_json(args.sample_dir / 'service-identity.json'))
    require(args.plan and args.plan_sha and frozen.sha256(args.plan) == args.plan_sha, 'plan SHA mismatch')
    plan = frozen.read_json(args.plan)
    validate_plan(plan)
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'capacities': planned_arm_order(plan),
                          'frozen_sha256': FROZEN_SHA}))
        return 0
    require(args.root_identity and args.expected_inputs, 'explicit root/inputs files required')
    require(frozen.sha256(args.expected_inputs) == plan['evaluation_frozen']['expected_inputs_sha256'],
            'expected input SHA drift')
    def interrupted(signum, _frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    return execute(frozen, plan, frozen.read_json(args.root_identity),
                   json_lines(args.expected_inputs), args.frozen)


if __name__ == '__main__':
    raise SystemExit(main())
