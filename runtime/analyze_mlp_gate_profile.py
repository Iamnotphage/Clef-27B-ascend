#!/usr/bin/env python3
"""Offline task-level attribution. All timestamps below use the exported trace clock.

Host ranges, CPU API unions and device intervals can overlap. Never sum them.
Only explicit native flow destinations on the exact stream identify device tasks;
queue-worker association uses explicit enqueue/dequeue flow, never nearest time.
"""
import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import statistics


def intervals(xs):
    result=[]
    for a,b in sorted(xs):
        if b<=a: continue
        if result and a<=result[-1][1]: result[-1]=(result[-1][0],max(b,result[-1][1]))
        else: result.append((a,b))
    return result


def union(xs): return sum(b-a for a,b in intervals(xs))


def overlap(xs, ys):
    return union((max(a,c),min(b,d)) for a,b in intervals(xs) for c,d in intervals(ys) if max(a,c)<min(b,d))


def parse_trace(path, divisor=3, sample_prefix='clef40/sample/'):
    if divisor <= 0: raise ValueError("positive layer divisor required")
    doc=json.loads(Path(path).read_text());events=doc.get('traceEvents',[]) if isinstance(doc,dict) else doc
    processes={e['pid']:e['args']['name'] for e in events if e.get('name')=='process_name'}
    # Subtract the epoch with decimal arithmetic before using float durations.
    # Absolute epoch microseconds lose sub-us task gaps in a binary float.
    origin=min(Decimal(str(e['ts'])) for e in events if 'ts' in e)
    stamp=lambda e:float(Decimal(str(e['ts']))-origin)
    spans=[];flows=defaultdict(dict)
    for e in events:
        if e.get('ph')=='X':
            e=dict(e);e['start']=stamp(e);e['end']=e['start']+float(e.get('dur',0));spans.append(e)
        elif e.get('ph') in ('s','f'):
            flows[(e.get('cat'),str(e.get('id')))][e['ph']]=dict(e,ts=stamp(e))
    def contains(e, t): return e['start']<=float(t)<=e['end']
    def bounds(es): return [(e['start'],e['end']) for e in es]
    samples=[e for e in spans if e['name'].startswith(sample_prefix)]
    tasks=[e for e in spans if processes.get(e['pid'])=='Ascend Hardware' and
           e.get('args',{}).get('Task Type') not in ('PROFILING_ENABLE','PROFILING_DISABLE','EVENT RECORD','EVENT WAIT') and e['end']>e['start']]
    controls=[e for e in spans if processes.get(e['pid'])=='Ascend Hardware' and e.get('args',{}).get('Task Type')=='EVENT RECORD']
    waits=[e for e in spans if processes.get(e['pid'])=='Ascend Hardware' and e.get('args',{}).get('Task Type')=='EVENT WAIT']
    native=[e for e in spans if processes.get(e['pid'])=='CANN']
    rows=[]
    for sample in samples:
        phase,pair,arm=sample['name'][len(sample_prefix):].split('/')
        scopes=[e for e in spans if e['pid']==sample['pid'] and e['tid']==sample['tid'] and
                sample['start']<=e['start'] and e['end']<=sample['end'] and e is not sample]
        def scoped(name): return [e for e in scopes if e['name']=='clef40/'+name]
        # Explicit queue edges: main-thread enqueue -> worker dequeue interval.
        worker=[]
        for (cat,_),flow in flows.items():
            if cat!='async_task_queue' or not {'s','f'}<=flow.keys(): continue
            s,f=flow['s'],flow['f']
            if s['pid']==sample['pid'] and s['tid']==sample['tid'] and contains(sample,s['ts']):
                choices=[e for e in spans if e.get('cat')=='dequeue' and e['pid']==f['pid'] and e['tid']==f['tid'] and contains(e,f['ts'])]
                if choices: worker.append(min(choices,key=lambda e:e['end']-e['start']))
        def host_owned(endpoint):
            # CANN uses a synthetic pid but preserves the OS thread ID.
            if endpoint['tid']==sample['tid'] and contains(sample,endpoint['ts']): return True
            return any(endpoint['tid']==w['tid'] and contains(w,endpoint['ts']) for w in worker)
        matched={};event_records={};links=[]
        for (cat,fid),flow in flows.items():
            if cat not in ('async_npu','HostToDevice') or not {'s','f'}<=flow.keys(): continue
            s,f=flow['s'],flow['f']
            if not host_owned(s): continue
            targets=[e for e in tasks if e['pid']==f['pid'] and e['tid']==f['tid'] and e['start']<=f['ts']<e['end']]
            if not targets:
                records=[e for e in controls if e['pid']==f['pid'] and e['tid']==f['tid'] and e['start']==f['ts']]
                for e in records:event_records[(e['pid'],e['tid'],e['start'])]=e
                continue
            task=min(targets,key=lambda e:e['end']-e['start'])
            key=(task['pid'],task['tid'],task['start'],task['end']);matched[key]=task
            api=[e for e in native if e['tid']==s['tid'] and contains(e,s['ts'])]
            launches=[e for e in api if 'launch' in e['name'].lower()]
            enclosing=min(launches or api,key=lambda e:e['end']-e['start']) if api else None
            links.append(dict(category=cat,id=fid,source_us=float(s['ts']),task_start_us=task['start'],task_end_us=task['end'],
                stream=task['tid'],name=task['name'],source_to_start_us=task['start']-float(s['ts']),
                native_api=enclosing['name'] if enclosing else None,
                launch_end_to_start_us=task['start']-enclosing['end'] if enclosing else None))
        kernels=sorted(matched.values(),key=lambda e:e['start'])
        if not kernels: raise ValueError('No explicitly correlated tasks for '+sample['name'])
        k=bounds(kernels); first=min(a for a,b in k);last=max(b for a,b in k)
        sync=bounds(scoped('sync')); submit=bounds(scoped('submit'))
        apis=[e for e in native if host_owned(dict(tid=e['tid'],ts=e['start'])) and
              sample['start']<=e['start'] and e['end']<=sample['end']]
        syncapis=[e for e in apis if 'synchroniz' in e['name'].lower()]
        otherapis=[e for e in apis if e not in syncapis]
        # A disjoint elapsed partition, kept separate from overlapping host/API views.
        partition=dict(before_first_us=max(0,first-sample['start'])/divisor,kernel_union_us=union(k)/divisor,
                       between_kernel_gaps_us=(last-first-union(k))/divisor,after_last_us=max(0,sample['end']-last)/divisor)
        markers=sorted(event_records.values(),key=lambda e:e['start'])
        event_partition=None
        if len(markers)==2:
            event_partition=dict(before_first_us=(first-markers[0]['start'])/divisor,kernel_union_us=union(k)/divisor,
                between_kernel_gaps_us=(last-first-union(k))/divisor,after_last_us=(markers[1]['start']-last)/divisor)
        gap=[(markers[0]['start'],first)] if len(markers)==2 else []
        host_ranges=submit+bounds(otherapis)
        gap_covered=overlap(gap,host_ranges)/divisor
        named={name:union(bounds(scoped(name)))/divisor for name in ['submit','sync','v3_alloc','v3_bridge','v3_get_workspace','v3_launch']}
        # Self of coarse bridge excludes its nested named ranges and native CPU API spans on the main OS thread.
        submit_children=bounds([e for e in scopes if e['name']!='clef40/submit'])
        submit_self=(union(submit)-overlap(submit,submit_children))/divisor
        bridge=bounds(scoped('v3_bridge'))
        children=bounds(scoped('v3_get_workspace')+scoped('v3_launch'))+bounds([e for e in apis if e['tid']==sample['tid']])
        rows.append(dict(phase=phase,pair=int(pair),arm=arm,elapsed_us=(sample['end']-sample['start'])/divisor,
            scope_inclusive_us=named,submit_self_us=submit_self,event_partition_us=event_partition,
            before_first_host_range_coverage_us=gap_covered,
            before_first_uncovered_us=union(gap)/divisor-gap_covered,
            event_marker_count=len(markers),event_span_us=(markers[1]['start']-markers[0]['start'])/divisor if len(markers)==2 else None,bridge_self_us=(union(bridge)-overlap(bridge,children))/divisor,
            native_non_sync_api_union_us=union(bounds(otherapis))/divisor,native_sync_api_union_us=union(bounds(syncapis))/divisor,
            explicit_sync_overlap_kernel_us=overlap(sync,k)/divisor,
            last_kernel_to_sync_return_us=max(0,max((b for a,b in sync),default=sample['end'])-last)/divisor,
            partition_us=partition,task_count=len(kernels),task_names=dict(Counter(e['name'] for e in kernels)),
            task_sum_us=sum(e['end']-e['start'] for e in kernels)/divisor,streams=sorted(set(str(e['tid']) for e in kernels)),
            linked_tasks=len(matched),links=links,worker_dequeues=len(worker),
            task_keys=[dict(stream=str(e['tid']),task_id=str(e.get('args',{}).get('Task Id')),name=e['name'],start_us=e['start'],end_us=e['end'],duration_us=e['end']-e['start'],task_type=e.get('args',{}).get('Task Type')) for e in kernels],
            native_api_names=dict(Counter(e['name'] for e in apis)),
            unknown='Task internal pipeline/per-core stalls unavailable; launch-to-start is elapsed latency, not automatically avoidable wait'))
    return dict(path=str(path),trace_origin_us=str(origin),time_unit='us relative to trace_origin_us',processes=processes,samples=rows,hardware_tasks_total=len(tasks),
                all_task_names=dict(Counter(e['name'] for e in tasks)),device_wait_control_count=len(waits),
                flow_categories=dict(Counter(k[0] for k in flows)))


def join_kernel_details(trace, path):
    """Join task-level PMU export by exact stream/task IDs, never nearest time.

    CSV rows retain original names/units and collector aggregation semantics.
    They do not establish per-core observations or mutually exclusive pipelines.
    """
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    index={}
    for row in rows:
        key=(str(row['Stream ID']),str(row['Task ID']))
        if key in index:raise ValueError('duplicate task key in kernel CSV')
        index[key]=row
    count=0
    for sample in trace['samples']:
        selected=[]
        for task in sample['task_keys']:
            row=index[(task['stream'],task['task_id'])]
            assert row['Name']==task['name']
            if row.get('Start Time(us)'):
                offset=float(Decimal(row['Start Time(us)'].strip())-Decimal(trace['trace_origin_us']))
                assert abs(offset-task['start_us'])<.1
            selected.append(row);count+=1
        sample['kernel_csv_rows']=selected
    trace['kernel_csv']=str(path);trace['csv_matched_tasks']=count
    trace['csv_metric_columns']=[c for c in (rows[0] if rows else []) if c.startswith(('aic_','aiv_'))]
    return trace


def summarize_samples(rows):
    result={}
    for phase in sorted({r['phase'] for r in rows}):
        selected=[r for r in rows if r['phase']==phase]
        # Slow rank per pair/arm/clock first, then paired differences/medians.
        result[phase]={}
        for clock in ('device_ms','wall_ms','cpu_ms'):
            arms=defaultdict(dict)
            for r in selected: arms[r['arm']][r['pair']]=max(arms[r['arm']].get(r['pair'],0),r[clock])
            medians={arm:statistics.median(values.values()) for arm,values in arms.items()}
            ab=[arms['v3'][i]-arms['vendor'][i] for i in arms['v3']]
            aa=[arms['vendor_repeat'][i]-arms['vendor'][i] for i in arms.get('vendor_repeat',{})]
            result[phase][clock]=dict(medians=medians,paired_delta=ab,median_delta=statistics.median(ab),
                aa_delta=aa,aa_max_abs=max(map(abs,aa),default=None))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,action='append',default=[])
    p.add_argument('--samples',type=Path,action='append',default=[]);p.add_argument('--kernel-csv',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-prefix',default='clef40/sample/')
    p.add_argument('--divisor',type=float,default=3)
    a=p.parse_args();out=dict(traces=[parse_trace(x,a.divisor,a.sample_prefix) for x in a.trace])
    if a.kernel_csv:
        assert len(a.kernel_csv)==len(out['traces'])
        for trace,path in zip(out['traces'],a.kernel_csv):join_kernel_details(trace,path)
    if a.samples:
        docs=[json.loads(x.read_text()) for x in a.samples]
        assert all(x['status']=='complete' for x in docs)
        rows=[dict(r,rank=d['rank']) for d in docs for r in d['samples']]
        out.update(timing=summarize_samples(rows),rank_raw=docs)
    a.output.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
