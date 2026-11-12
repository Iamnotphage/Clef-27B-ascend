#!/usr/bin/env python3
"""Sequentially switch only the snapshotted Clef API service for experiments.

Requires run-dir/restore-service.json containing the original argv/cwd and a
credential-filtered environment. Every launched command is recorded. Run after
activate_env.sh; no global process-name kill or device reset is performed.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def descendants(pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            for line in path.read_text().splitlines():
                if line.startswith('PPid:'):
                    parents[int(path.parent.name)] = int(line.split()[1])
        except OSError:
            continue
    result, frontier = [], [pid]
    while frontier:
        children = [child for child, parent in parents.items() if parent in frontier]
        result.extend(children)
        frontier = children
    return result


def health():
    try:
        with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def option(argv, flag, value):
    result, index = [], 0
    while index < len(argv):
        if argv[index] == flag:
            index += 2
        elif argv[index].startswith(flag + '='):
            index += 1
        else:
            result.append(argv[index])
            index += 1
    if value is not None:
        result += [flag, str(value)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stop', 'start', 'restore', 'status'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--label')
    parser.add_argument('--tp', type=int, choices=[1, 2], default=2)
    parser.add_argument('--max-model-len', type=int, default=1024)
    parser.add_argument('--max-num-seqs', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--observations', action='store_true')
    parser.add_argument('--cpu-timing-only', action='store_true')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--multimodal', action='store_true')
    args = parser.parse_args()
    if min(args.max_model_len, args.max_num_seqs) < 1 or (args.cpu_timing_only and args.observations):
        parser.error('Lengths/sequences must be positive; select one observation mode')
    run = args.run_dir.resolve()
    original = json.loads((run / 'restore-service.json').read_text())
    current_path = run / 'current-service.json'
    current = json.loads(current_path.read_text()) if current_path.exists() else original
    if args.action == 'status':
        print(json.dumps({'pid': current['pid'], 'alive': alive(current['pid']), 'healthy': health()}))
        return
    if args.action == 'stop':
        pid = current['pid']
        if alive(pid):
            actual = Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
            if actual != current['argv'] or 'vllm.entrypoints.cli.main' not in actual or 'serve' not in actual:
                raise RuntimeError('API process identity changed; refusing to signal it')
            children = descendants(pid)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and any(alive(p) for p in [pid, *children]):
                time.sleep(1)
            remaining = [p for p in [pid, *children] if alive(p)]
            for child in remaining:
                os.kill(child, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and any(alive(p) for p in remaining):
                time.sleep(1)
            killed = [p for p in remaining if alive(p)]
            for child in killed:
                os.kill(child, signal.SIGKILL)
            save(run / f'stop-{pid}.json', {'pid': pid, 'descendants': children, 'forced_pids': killed,
                                          'finished_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'stopped_pid': pid, 'healthy': health()}))
        return
    if alive(current['pid']) or health():
        raise RuntimeError('An existing API service is still running; stop the recorded service first')
    label = args.label or ('restored-tp2-4096' if args.action == 'restore' else None)
    if not label or Path(label).name != label:
        parser.error('Use a single directory name for --label')
    output = run / label
    output.mkdir(exist_ok=False)
    argv = list(original['argv'])
    if args.action == 'start':
        for flag, value in [('--tensor-parallel-size', args.tp), ('--max-model-len', args.max_model_len),
                            ('--max-num-batched-tokens', args.max_model_len), ('--max-num-seqs', args.max_num_seqs)]:
            argv = option(argv, flag, value)
        argv = option(argv, '--limit-mm-per-prompt', '{"image":1,"video":1}' if args.multimodal else '{"image":1,"video":0}')
        argv = option(argv, '--profiler-config', None)
        if args.profile:
            config = {'profiler': 'torch', 'torch_profiler_dir': str(output / 'traces'),
                      'torch_profiler_with_stack': args.memory, 'torch_profiler_with_memory': args.memory,
                      'torch_profiler_record_shapes': args.observations or args.memory, 'ignore_frontend': True}
            argv = option(argv, '--profiler-config', json.dumps(config, separators=(',', ':')))
    environment = dict(os.environ)
    environment.update(original['environment'])
    for key in ['CLEF_PROFILE_MARKERS', 'CLEF_PROFILE_OBSERVE_DIR', 'CLEF_PROFILE_SCOPE_FILTER']:
        environment.pop(key, None)
    if args.observations:
        environment.update(CLEF_PROFILE_MARKERS='1', CLEF_PROFILE_OBSERVE_DIR=str(output / 'cpu-timings'))
    elif args.cpu_timing_only:
        names = ['api_handler_including_json', 'frontend_request', 'request_schema_parse',
                 'schema_tokenize_encode', 'media_decode_process', 'worker_execute_model',
                 'backbone', 'language_backbone', 'joint_pooler', 'joint_head',
                 'candidate_embedding_tp', 'request_finished_metrics', 'cache_initialize']
        environment.update(CLEF_PROFILE_MARKERS='0', CLEF_PROFILE_OBSERVE_DIR=str(output / 'cpu-timings'),
                           CLEF_PROFILE_SCOPE_FILTER=','.join('clef::' + name for name in names))
    cwd = original['cwd'] if args.action == 'restore' else str(Path(__file__).resolve().parents[1])
    with (output / 'service.log').open('wb') as log:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'argv': argv, 'cwd': cwd, 'environment': original['environment'],
              'label': label, 'profile': args.profile, 'observations': args.observations,
              'cpu_timing_only': args.cpu_timing_only,
              'memory': args.memory, 'started_at': datetime.now(timezone.utc).isoformat(),
              'log': str(output / 'service.log')}
    save(output / 'launch.json', record)
    save(current_path, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
