#!/usr/bin/env python3
"""Index retained rank traces and evidence without copying their large data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def describe(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-windows', type=int)
    parser.add_argument('--expected-traces', type=int)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows, traces, files = [], [], []
    for path in sorted(run.glob('*/window-*/manifest.json')):
        window = read(path)
        if 'new_rank_directories' not in window:
            continue
        directories = [Path(value).resolve() for value in window['new_rank_directories']]
        window_ok = bool(window.get('passed') and window.get('trace_completeness_passed') and
                         len(directories) == window['expected_ranks'])
        rank_ids = []
        for directory in directories:
            if not directory.is_relative_to(run):
                raise ValueError('Trace directory is outside run-dir')
            info = [read(p) for p in sorted(directory.glob('profiler_info*.json'))]
            rank = next((value.get('rank_id') for value in info if value.get('rank_id') is not None), None)
            rank_ids.append(rank)
            indexed = [describe(p) for p in sorted(directory.rglob('*')) if p.is_file()]
            files.extend(indexed)
            exported = directory / 'ASCEND_PROFILER_OUTPUT'
            raw_nonempty = any(row['bytes'] > 0 and ('/FRAMEWORK/' in row['path'] or '/PROF_' in row['path']) for row in indexed)
            trace_ok = bool(raw_nonempty and (exported / 'kernel_details.csv').is_file() and
                            any(p.stat().st_size > 0 for p in exported.glob('*trace*.json*')))
            traces.append({'window': str(path.parent), 'rank': rank, 'directory': str(directory),
                           'passed': trace_ok, 'file_count': len(indexed),
                           'bytes': sum(row['bytes'] for row in indexed),
                           'file_index_sha256': hashlib.sha256(json.dumps(indexed, sort_keys=True).encode()).hexdigest()})
            window_ok = window_ok and trace_ok
        window_ok = window_ok and set(rank_ids) == set(range(window['expected_ranks']))
        windows.append({'manifest': describe(path), 'passed': window_ok, 'ranks': rank_ids,
                        'request_report': describe(path.parent / 'profile-requests.json'),
                        'start': window.get('start'), 'stop': window.get('stop')})
    file_index = output / 'raw-file-index.json'
    write(file_index, {'scope': 'Retained files after the fixed wrapper data_simplification/export process', 'files': files})
    restoration = read(run / 'restoration-verification.json')
    aggregate = read(run / 'aggregate-final/summary.json')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': run.name,
              'remote_run_directory': str(run), 'hardware_execution_verified': True,
              'window_count': len(windows), 'rank_trace_count': len(traces),
              'retained_trace_bytes': sum(row['bytes'] for row in traces),
              'windows': windows, 'rank_traces': traces,
              'raw_file_index': describe(file_index), 'restoration': restoration,
              'benchmark_report_count': aggregate['report_count'],
              'benchmark_classifications': aggregate['classification_counts'],
              'benchmark_read_errors': aggregate['read_errors'],
              'source_provenance': {}, 'evidence': []}
    for name in ('project', 'vllm', 'vllm_ascend'):
        path = run / 'before' / (name + '-head.txt')
        report['source_provenance'][name + '_head'] = path.read_text().strip()
    report['source_provenance']['original_project_head'] = (run / 'before/original-project-head.txt').read_text().strip()
    for pattern in ('before/manifest.json', 'after/manifest.json', 'provenance/controllers.json',
                    'instrumentation*/instrumentation-manifest.json', 'instrumentation*/*-profiling.patch',
                    'workloads/manifest.json', 'workloads/encoding-evidence.json', 'workloads/matrix-plan.json',
                    'uncached-media/manifest.json', 'uncached-media/semantic-results.json',
                    'aggregate-final/*', 'restored-tp2-4096/*.json', 'micro-hardware/*summary*.json'):
        report['evidence'].extend(describe(p) for p in sorted(run.glob(pattern)) if p.is_file())
    report['passed'] = bool(windows and all(row['passed'] for row in windows) and restoration['passed'] and not aggregate['read_errors'])
    if args.expected_windows is not None:
        report['passed'] &= len(windows) == args.expected_windows
    if args.expected_traces is not None:
        report['passed'] &= len(traces) == args.expected_traces
    write(output / 'run-manifest.json', report)
    print(json.dumps({key: report[key] for key in ('passed', 'window_count', 'rank_trace_count', 'retained_trace_bytes', 'benchmark_report_count')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
