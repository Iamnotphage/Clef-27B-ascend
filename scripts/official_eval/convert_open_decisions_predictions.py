#!/usr/bin/env python3
"""Export stored EvalScope raw probabilities; no model calls or probability changes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

DATA_SHA256 = '78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6'
SCORER_SHA256 = '478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True,
                        help='Unmodified author score.py from the pinned dataset revision')
    args = parser.parse_args()
    blob = args.data.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DATA_SHA256:
        parser.error('data SHA256 does not match fixed OpenDecisions revision')
    tasks = [json.loads(line) for line in blob.splitlines() if line.strip()]
    by_id = {task['id']: task for task in tasks}
    assert len(tasks) == len(by_id) == 2352
    predictions = {}
    seen = set()
    # JSON strings can contain literal Unicode separators such as U+2028.
    # Split only on file line endings so these characters remain in the text.
    for line_number, line in enumerate(args.predictions.open(encoding='utf-8'), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row['metadata']['source_id']
        if source_id not in by_id or source_id in seen:
            raise ValueError(f'line {line_number}: unknown or duplicate source_id {source_id}')
        seen.add(source_id)
        task = by_id[source_id]
        metadata = (row.get('model_output') or {}).get('metadata') or {}
        raw = metadata.get('choice_response')
        if raw is None:
            continue
        request = metadata['choice_request']
        question = {key: task['question'][key] for key in ('type', 'instructions')}
        if task['question'].get('criteria') is not None:
            question['criteria'] = task['question']['criteria']
        if request['state'] != task['state'] or request['questions'] != {'answer': question}:
            raise ValueError(f'{source_id}: stored request does not match the fixed source item')
        answer = raw['answers']['answer']
        probabilities = answer.get('probabilities')
        if probabilities is None and question['type'] == 'noul':
            p_yes = answer['noul']
            probabilities = {'yes': p_yes, 'no': 1 - p_yes}
        if not isinstance(probabilities, dict) or set(probabilities) != set(task['labels']):
            raise ValueError(f'{source_id}: raw probability labels do not match source labels')
        predictions[source_id] = {'id': source_id, 'probs': probabilities}
    rows = [predictions[task['id']] for task in tasks if task['id'] in predictions]
    # Reuse the unmodified author's validation before writing any exported predictions.
    if hashlib.sha256(args.scorer.read_bytes()).hexdigest() != SCORER_SHA256:
        parser.error('scorer SHA256 does not match fixed OpenDecisions revision')
    spec = importlib.util.spec_from_file_location('opendecisions_author_score', args.scorer)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    result = scorer.score(tasks, rows)
    outcomes = [scorer.item_outcome(task, predictions.get(task['id'], {}).get('probs'))
                for task in tasks]
    valid = [outcome for outcome in outcomes if outcome['valid']]
    score_valid = [outcome for outcome in valid if outcome['type'] == 'score']
    correct = sum(outcome['correct'] for outcome in valid)
    derived = {
        'accuracy': correct / len(valid) if valid else None,
        'correct': correct,
        'accuracy_denominator': len(valid),
        'score_mae': (sum(outcome['nmae'] * (outcome['k'] - 1) for outcome in score_valid)
                      / len(score_valid)) if score_valid else None,
        'score_mae_denominator': len(score_valid),
    }
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps({'data_sha256': DATA_SHA256, 'input_rows': len(seen),
                      'exported_predictions': len(rows), 'missing_predictions': len(tasks) - len(rows),
                      'scoring_result': result, 'derived_metrics': derived},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
