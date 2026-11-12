#!/usr/bin/env python3
"""Read fixed-307 Clef batching evidence; never send requests or execute NPU work.

python scripts/analyze_systemone_batching.py --run-root /absolute/profiling/run \
    --output-dir /absolute/profiling/run/batching-analysis

Formal Prometheus windows and profiled trace windows are separate populations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


TOKENS = 307
LAYERS = 64
BACKBONE_REDUCTIONS = 2 * LAYERS + 1


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source(path, **extra):
    return dict(path=str(path.resolve()), sha256=sha(path), **extra)


def read(path):
    return json.loads(path.read_text())


def metric_values(path):
    values, lines = {}, {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r'^vllm:iteration_tokens_total_bucket\{([^}]+)\} ([^ ]+)', line)
        if match:
            limit = float(re.search(r'le="([^"]+)"', match[1])[1])
            if limit in values:
                raise ValueError(f'Multiple histogram series require explicit selection: {path}:{number}')
            values[limit] = float(match[2]); lines[str(limit)] = number
        for kind in ('sum', 'count', 'created'):
            if line.startswith('vllm:iteration_tokens_total_'+kind+'{'):
                if kind in values:
                    raise ValueError(f'Multiple histogram series: {path}:{number}')
                values[kind] = float(line.rsplit(' ', 1)[1]); lines[kind] = number
    return values, lines


def solve_histogram(before, after, warmup_count, measured_count, maximum):
    delta = {key: after[key] - value for key, value in before.items() if key != 'created'}
    total = warmup_count + measured_count
    one = delta[512.0] - delta[256.0]
    two_three = delta[1024.0] - delta[512.0]
    four = delta[2048.0] - delta[1024.0]
    three = total - one - 2 * two_three - 4 * four
    batches = {1: one, 2: two_three - three, 3: three, 4: four}
    zero = delta[256.0]
    checks = {
        'same_counter_creation': before['created'] == after['created'],
        'token_sum_delta': delta['sum'] == total * TOKENS,
        'nonnegative_integer_batches': all(value >= 0 and value.is_integer() for value in batches.values()),
        'max_num_seqs_bound': all(count == 0 for batch, count in batches.items() if batch > maximum),
        'all_buckets_match': all(delta[limit] == zero + sum(count for batch, count in batches.items() if batch * TOKENS <= limit)
                                 for limit in before if isinstance(limit, float)),
        'count_matches': delta['count'] == zero + sum(batches.values()),
        'sequential_warmup_available': one >= warmup_count,
    }
    formal = dict(batches); formal[1] -= warmup_count
    checks['formal_request_sum'] = sum(batch * count for batch, count in formal.items()) == measured_count
    valid = all(checks.values())
    return {'passed': valid, 'checks': checks, 'histogram_deltas': {str(key): value for key, value in delta.items()},
            'all_batch_histogram': {key: int(value) for key, value in batches.items()} if valid else None,
            'formal_batch_histogram': {key: int(value) for key, value in formal.items()} if valid else None}


def benchmark_fields(document):
    summary = document['summary']
    return {'requests_per_second': summary['successful_requests_per_second'],
            'latency_mean_ms': summary['successful_request_latency_ms']['mean'],
            'latency_p50_ms': summary['successful_request_latency_ms']['p50'],
            'latency_p95_ms': summary['successful_request_latency_ms']['p95'],
            'request_template_sha256': document.get('request_template_sha256')}


def formal_windows(root):
    cases, unavailable = [], []
    for name, maximum in [('tp2-4096-matrix', 1), ('tp2-4096-batch2', 2), ('tp2-4096-batch4', 4)]:
        for concurrency in (1, 2, 4):
            directory = root / name
            before = directory / f'concurrency-{concurrency}-metrics-before.txt'
            after = directory / f'concurrency-{concurrency}-metrics-after.txt'
            benchmark = directory / f'concurrency-{concurrency}.json'
            if not all(path.exists() for path in (before, after, benchmark)):
                unavailable.append({'configuration': name, 'concurrency': concurrency, 'reason': 'No complete before/after/benchmark triple'})
                continue
            document = read(benchmark); warmup, measured = document['warmup'], document['requests']
            a, a_lines = metric_values(before); z, z_lines = metric_values(after)
            solved = solve_histogram(a, z, len(warmup), len(measured), maximum)
            solved['checks']['requests_valid_and_fixed307'] = all(row['passed'] and row['actual_input_tokens'] == TOKENS for row in warmup + measured)
            solved['passed'] = all(solved['checks'].values())
            histogram = solved['formal_batch_histogram'] if solved['passed'] else None
            forward_count = sum(histogram.values()) if histogram else None
            derived_calls = BACKBONE_REDUCTIONS * forward_count + len(measured) if histogram else None
            cases.append(dict(measurement_scope='formal_unprofiled', configuration=name, max_num_seqs=maximum,
                              concurrency=concurrency, request_count=len(measured), warmup_count=len(warmup),
                              expected_token_sum_delta=(len(warmup) + len(measured)) * TOKENS,
                              batch_histogram=histogram, forward_count=forward_count,
                              collectives_per_request=derived_calls / len(measured) if derived_calls is not None else None,
                              collective_source='derived 129*F+N; independently validated by profiled communication exports',
                              sources=[source(before, line_numbers=a_lines), source(after, line_numbers=z_lines), source(benchmark)],
                              **benchmark_fields(document), **solved))
    return cases, unavailable


def trace_windows(root):
    cases = []
    for name, relative, maximum in [('b1', 'tp2-1024-native/window-1', 1),
                                     ('max_seqs2', 'tp2-4096-batch2/window-batch-c4', 2),
                                     ('max_seqs4', 'tp2-4096-batch4/window-batch-c4', 4)]:
        window = root / relative; manifest_path = window / 'manifest.json'; benchmark = window / 'profile-requests.json'
        manifest, document = read(manifest_path), read(benchmark)
        count = len(document['requests']); ranks = []
        for directory in map(Path, manifest['new_rank_directories']):
            info = read(next(directory.glob('profiler_info*.json')))
            kernel = directory / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
            by_m, examples = Counter(), {}
            with kernel.open(encoding='utf-8-sig', newline='') as stream:
                for line, row in enumerate(csv.DictReader(stream), 2):
                    if not ('Matmul' in row['Name'] or 'MatMul' in row['Name']):
                        continue
                    match = re.fullmatch(r'(\d+),5120;17408,5120', row.get('Input Shapes', '').strip('"'))
                    if match:
                        m = int(match[1]); by_m[m] += 1
                        examples.setdefault(m, {'csv_row': line, 'name': row['Name'], 'shape': row['Input Shapes'], 'dtype': row['Input Data Types']})
            valid = bool(by_m) and all(m % TOKENS == 0 and calls % LAYERS == 0 for m, calls in by_m.items())
            histogram = {m // TOKENS: calls // LAYERS for m, calls in by_m.items()} if valid else {}
            comm = directory / 'ASCEND_PROFILER_OUTPUT/communication.json'
            operations = [(step, category, op, value) for step, groups in read(comm).items()
                          for category, values in groups.items() for op, value in values.items() if not op.lower().startswith('total')]
            sizes, types = Counter(), Counter()
            for _, _, op, value in operations:
                types[re.sub(r'__\d+.*', '', op)] += 1
                sizes[round(value['Communication Bandwidth Info'].get('SIO', {}).get('Transit Size(MB)', 0) * 1024**2)] += 1
            total_bytes = sum(size * calls for size, calls in sizes.items())
            checks = {'shape_counts_integral': valid, 'request_sum': sum(batch * n for batch, n in histogram.items()) == count,
                      'max_num_seqs_bound': all(batch <= maximum for batch in histogram),
                      'communication_formula_129F_plus_N': len(operations) == BACKBONE_REDUCTIONS * sum(histogram.values()) + count}
            ranks.append({'rank': info['rank_id'], 'batch_histogram': histogram, 'mlp_calls_by_M': dict(by_m),
                          'forward_count': sum(histogram.values()), 'shape_examples': examples,
                          'collective_count': len(operations), 'collectives_per_request': len(operations) / count,
                          'collective_types': dict(types), 'reported_transit_bytes_histogram': dict(sizes),
                          'total_reported_transit_bytes': total_bytes, 'reported_transit_bytes_per_request': total_bytes / count,
                          'checks': checks, 'passed': all(checks.values()),
                          'sources': [source(kernel), source(comm)],
                          'communication_examples': [{'json_path': list(row[:3])} for row in operations[:3]]})
        agree = all((row['batch_histogram'], row['reported_transit_bytes_histogram']) == (ranks[0]['batch_histogram'], ranks[0]['reported_transit_bytes_histogram']) for row in ranks)
        cases.append(dict(measurement_scope='profiled_trace', configuration=name, max_num_seqs=maximum,
                          concurrency=document['configuration']['concurrency'], request_count=count,
                          warmup_count=len(document['warmup']), warmup_scope='External controller warmup is recorded in the window manifest',
                          ranks=ranks, ranks_agree=agree, passed=manifest['passed'] and agree and all(row['passed'] for row in ranks),
                          sources=[source(manifest_path), source(benchmark)], **benchmark_fields(document)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    formal, unavailable = formal_windows(args.run_root)
    traces = trace_windows(args.run_root)
    baseline_bytes = traces[0]['ranks'][0]['reported_transit_bytes_per_request']
    baseline_hash = traces[0]['request_template_sha256']
    checks = {'same_transit_bytes_per_request_in_native_traces': all(row['reported_transit_bytes_per_request'] == baseline_bytes for case in traces for row in case['ranks']),
              'same_request_body': bool(baseline_hash) and all(case['request_template_sha256'] == baseline_hash for case in formal + traces)}
    rows = []
    for case in formal:
        case['reported_transit_bytes_per_request'] = baseline_bytes
        case['byte_source'] = 'derived from invariant native message signature; total bytes per request remain constant'
        rows.append({key: value for key, value in case.items() if key not in ('histogram_deltas', 'checks', 'all_batch_histogram', 'formal_batch_histogram')})
    for case in traces:
        for rank in case['ranks']:
            rows.append({key: value for key, value in case.items() if key != 'ranks'} | rank |
                        {'collective_source': 'measured communication.json', 'byte_source': 'measured SIO transport; duplicate SDMA aliases excluded'})
    result = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'source_script': source(Path(__file__)), 'run_root': str(args.run_root.resolve()),
              'methods': {'layers': LAYERS, 'tokens_per_request': TOKENS, 'MLP_up_selector': 'M,5120;17408,5120',
                          'trace_batches': 'Per-M call count / 64; B=M/307; sum(B*count) must equal profiled requests',
                          'formal_B1': 'delta(le512)-delta(le256)', 'formal_B2_plus_B3': 'delta(le1024)-delta(le512)',
                          'formal_B4': 'delta(le2048)-delta(le1024)', 'formal_B3': 'N-B1-2*(B2+B3)-4*B4',
                          'formal_B2': '(B2+B3)-B3', 'warmup': 'Subtract actual sequential benchmark warmup count from B1',
                          'validation': 'Counter creation; token sum; all buckets/count; nonnegative integers; sequence count; scheduler maximum',
                          'communication': '129 all_reduce per forward plus one lexical candidate all_reduce per request, checked against independent native traces',
                          'bytes': 'SIO reported MB * 1024**2; duplicate SDMA aliases excluded; transit bytes differ from HBM weight bytes',
                          'population': 'Formal unprofiled metrics and profiled trace windows are reported separately'},
              'formal_metric_windows': formal, 'profiled_trace_windows': traces, 'unavailable': unavailable,
              'cross_checks': checks, 'passed': bool(formal) and all(case['passed'] for case in formal + traces) and all(checks.values())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'batching.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fields = ['measurement_scope', 'configuration', 'max_num_seqs', 'concurrency', 'rank', 'request_count', 'warmup_count',
              'batch_histogram', 'forward_count', 'requests_per_second', 'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms',
              'collective_count', 'collectives_per_request', 'collective_source', 'reported_transit_bytes_per_request', 'byte_source', 'passed', 'sources']
    with (args.output_dir / 'batching.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n'); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    print(json.dumps({'passed': result['passed'], 'formal_windows': len(formal), 'trace_windows': len(traces), 'output': str(args.output_dir)}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
