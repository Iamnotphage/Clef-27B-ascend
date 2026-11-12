#!/usr/bin/env python3
"""Select device kernels through explicit joint_head CPU-to-NPU flows only."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time


def ns(value):
    return int(Decimal(str(value).strip()) * 1000)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run-root', type=Path, required=True)
    a.add_argument('--output-dir', type=Path, required=True)
    args = a.parse_args(); started = time.monotonic(); rows = []; checks = []; source_index = []
    for config, relative in [('tp1-fixed307', 'tp1-4096-observed/analysis-agent'), ('tp2-fixed307', 'tp2-4096-observed/analysis-fixed307-agent')]:
        summary_path = args.run_root/relative/'summary.json'; summary = json.loads(summary_path.read_text())
        source_index.append({'path': str(summary_path), 'sha256': sha(summary_path)})
        for rank in summary['ranks']:
            trace_meta = rank['timeline_metadata'][0]
            scopes = [scope for scope in trace_meta['module_flow_correlation']['module_instances'] if scope['module'] == 'clef::joint_head' and scope.get('request_ids')]
            trace = Path(summary['profile_root'])/trace_meta['file']
            with trace.open() as f: document = json.load(f, parse_float=str)
            events = document.get('traceEvents', []) if isinstance(document, dict) else document
            sources = {}
            for index, event in enumerate(events):
                if event.get('ph') != 's' or event.get('cat') != 'async_npu': continue
                point = ns(event['ts'])
                matching = [scope for scope in scopes if str(scope['pid']) == str(event['pid']) and str(scope['tid']) == str(event['tid']) and scope['cpu_start_ns'] <= point <= scope['cpu_end_ns']]
                if len(matching) == 1:
                    sources[event['id']] = {'cpu_flow_start_event_index': index, 'head_scope_event_index': matching[0]['scope_event_index'], 'request_ids': matching[0]['request_ids']}
            targets = defaultdict(list)
            for index, event in enumerate(events):
                if event.get('ph') == 'f' and event.get('cat') == 'async_npu' and event.get('id') in sources:
                    key = (str(event['pid']), str(event['tid']), ns(event['ts']))
                    targets[key].append(sources[event['id']] | {'cpu_flow_end_event_index': index, 'flow_id': event['id']})
            kernel = Path(rank['directory'])/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'; by_key = defaultdict(list)
            with kernel.open(encoding='utf-8-sig', newline='') as f:
                for number, row in enumerate(csv.DictReader(f), 2):
                    key = (row['Name'], row['Task ID'], row['Stream ID'])
                    by_key[key].append((number, row, ns(row['Start Time(us)']), ns(row['Duration(us)'])))
            selected = {}; excluded_sync = 0; unmatched = 0
            for index, event in enumerate(events):
                if event.get('ph') != 'X' or 'dur' not in event: continue
                point = (str(event.get('pid')), str(event.get('tid')), ns(event['ts']))
                flow = targets.get(point)
                if not flow: continue
                if re.search(r'event[ _]?(wait|record)|notify[ _]?(wait|record)|stream[ _]?wait', event.get('name',''), re.I):
                    excluded_sync += 1; continue
                attributes = event.get('args', {}); task = attributes.get('Task Id'); stream = attributes.get('Physic Stream Id', event['tid'])
                key = (event.get('name'), str(task), str(stream))
                matches = [candidate for candidate in by_key.get(key, []) if abs(candidate[2] - point[2]) <= 1000 and abs(candidate[3] - ns(event['dur'])) <= 1000]
                if len(matches) != 1:
                    unmatched += 1; continue
                number, row, start, duration = matches[0]
                selected[number] = (row, duration, flow[0] | {'device_event_index': index, 'task_id': task, 'stream_id': stream, 'start_ns': start})
            groups = defaultdict(list)
            for number, (row, duration, flow) in selected.items():
                groups[(row['Name'], row['Type'], row.get('Input Shapes'), row.get('Input Data Types'))].append((number, duration, flow))
            total = 0
            for (name, kind, shape, dtype), group in groups.items():
                duration = sum(item[1] for item in group); total += duration
                family = ('attention kernel' if 'attention' in name.lower() else 'LayerNorm' if 'layernorm' in name.lower() else 'softmax' if 'softmax' in name.lower() else 'linear/matmul' if any(v in name.lower() for v in ['matmul','addmm']) else 'other')
                rows.append({'configuration': config, 'rank': rank['rank'], 'operator': name, 'op_type': kind, 'shape': shape, 'dtype': dtype,
                             'operator_family': family, 'count': len(group), 'cumulative_us': duration/1000, 'mean_us': duration/len(group)/1000,
                             'per_request_us': duration/len(scopes)/1000, 'source_csv': str(kernel), 'csv_rows': sorted(item[0] for item in group),
                             'cpu_flow_examples': [item[2] for item in group[:1]], 'source_trace': str(trace)})
            expected = next(module for module in trace_meta['module_flow_correlation']['modules'] if module['module'] == 'clef::joint_head')
            checks.append({'configuration': config, 'rank': rank['rank'], 'matched_request_scopes': len(scopes), 'selected_csv_kernels': len(selected),
                           'selected_cumulative_us': total/1000, 'selected_per_request_us': total/len(scopes)/1000,
                           'summary_head_compute_union_us': expected['compute_kernel_union_us'], 'excluded_sync_device_events': excluded_sync,
                           'unmatched_device_events': unmatched, 'scope': 'Unmatched events include non-CSV DMA tasks; no global same-op/shape groups are added'})
            source_index.extend([{'path': str(kernel), 'sha256': sha(kernel)}, {'path': str(trace), 'sha256': sha(trace)}])
            del document, events
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row['configuration'], row['rank'], -row['cumulative_us']))
    with (args.output_dir/'joint-head-operations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n'); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    report = {'method': 'Same PID/TID CPU async_npu flow source inside an identified joint_head scope -> flow target device task -> exact name/task/stream CSV key, time/duration rounding tolerance <=1us. Sync wait/record excluded; each CSV row deduplicated.',
              'subcomponents': 'MHA, TransformerDecoder and FFN are not separately marked; operator families provide no guessed subcomponent assignment.',
              'source_script_sha256': sha(Path(__file__)), 'checks': checks, 'sources': source_index, 'elapsed_cpu_wall_seconds': time.monotonic()-started,
              'passed': all(item['matched_request_scopes']==5 and item['selected_csv_kernels']>0 for item in checks)}
    (args.output_dir/'joint-head-operations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
