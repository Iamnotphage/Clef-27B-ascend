#!/usr/bin/env python3
"""Check every saved heterogeneous concurrent response against the CPU reference.

No inference is executed. The input benchmark must use the five existing smoke
case names as sample names. Token/span source evidence remains separately linked.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from reference_systemone import compare, digest
from verify_systemone import validate
from verify_systemone_precision import request_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-report', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--encoding-report', type=Path, required=True)
    parser.add_argument('--criteria', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/evidence/precision-criteria-2026-11-12.json')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'scope': 'Every heterogeneous benchmark response compared with saved CPU BF16 reference; no model run by comparator',
              'encoding_scope': 'Saved exact IDs/span/mapping evidence for each request plus per-response usage token check; fresh encoding audit follows source restoration',
              'inputs': {}, 'requests': []}
    try:
        documents = {}
        for name in ('benchmark_report', 'reference_report', 'encoding_report', 'criteria'):
            path = getattr(args, name)
            raw = path.read_bytes()
            documents[name] = json.loads(raw)
            result['inputs'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
        benchmark, reference, encoding, criteria = [documents[key] for key in ('benchmark_report', 'reference_report', 'encoding_report', 'criteria')]
        if not benchmark.get('passed') or reference.get('status') != 'complete' or not encoding.get('passed'):
            raise ValueError('Successful benchmark, complete CPU reference and passed saved encoding are required')
        if encoding.get('reference_report_sha256') != result['inputs']['reference_report']['sha256']:
            raise ValueError('Encoding evidence refers to a different CPU reference')
        refs = {case['name']: case for case in reference['cases']}
        enc = {case['name']: case for case in encoding['cases']}
        measured_names = set()
        probability_max, score_max, decisions, matched = 0.0, 0.0, 0, 0
        for phase in ('warmup', 'requests'):
            for row in benchmark[phase]:
                name = row['sample_name']
                if name not in refs or name not in enc:
                    raise ValueError(f'Unknown smoke case {name}')
                expected, aligned = refs[name], enc[name]
                request, response = row['request'], row['response']
                if not row['passed'] or digest(request) != expected['request_sha256'] or request_order(request) != request_order(expected['request']):
                    raise ValueError(f'Request mismatch or failed request: {phase}/{row["index"]}')
                if not aligned['passed'] or digest(aligned['request']) != digest(request):
                    raise ValueError(f'Encoding mismatch: {name}')
                if validate(request, response, benchmark['configuration']['validation_atol']):
                    raise ValueError(f'Protocol mismatch: {name}')
                if response['usage']['input_tokens'] != expected['encoding']['input_tokens']:
                    raise ValueError(f'Actual input token count mismatch: {name}')
                fields = compare(expected['semantic_answers'], response)
                for qid, field in fields.items():
                    kind = request['questions'][qid]['type']
                    same = field['same_argmax_level'] if kind == 'score' else field['same_decision']
                    probability_max = max(probability_max, field['max_probability_absolute_difference'])
                    score_max = max(score_max, field.get('score_absolute_difference', 0))
                    decisions += 1
                    matched += int(same)
                    field['passed'] = bool(same and field['max_probability_absolute_difference'] <= criteria['candidate_probability_max_absolute_difference'] and field.get('score_absolute_difference', 0) <= criteria['score_max_absolute_difference'])
                result['requests'].append({'phase': phase, 'index': row['index'], 'sample_name': name,
                                           'client_request_id': row['client_request_id'], 'fields': fields,
                                           'input_tokens': response['usage']['input_tokens'],
                                           'passed': all(field['passed'] for field in fields.values())})
                if phase == 'requests':
                    measured_names.add(name)
        result['summary'] = {'total_compared_requests': len(result['requests']),
                             'measured_requests': len(benchmark['requests']),
                             'measured_case_names': sorted(measured_names), 'decisions': decisions,
                             'matching_discrete_decisions': matched,
                             'max_probability_absolute_difference': probability_max,
                             'max_score_absolute_difference': score_max}
        result['passed'] = bool(measured_names == set(refs) and result['requests'] and all(row['passed'] for row in result['requests']))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'summary': result.get('summary'), 'error': result.get('error'), 'report': str(args.report)}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
