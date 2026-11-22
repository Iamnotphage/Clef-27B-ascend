#!/usr/bin/env python3
"""Offline explicit-flow GDN task comparison; no device or export invocation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics as st
from analyze_mlp_gate_profile import parse_trace


def paired(rows, metric):
    grouped=defaultdict(dict)
    ranks=defaultdict(lambda:defaultdict(list))
    for row in rows:
        arm,pair=row['arm'],row['pair'];value=row[metric]
        grouped[arm][pair]=max(value,grouped[arm].get(pair,0))
        ranks[arm][row['rank']].append(value)
    arms={arm:dict(values) for arm,values in grouped.items()}
    keys=sorted(set(arms['stock']) & set(arms['reuse']) & set(arms['stock_repeat']))
    delta=[arms['reuse'][i]-arms['stock'][i] for i in keys]
    aa=[arms['stock_repeat'][i]-arms['stock'][i] for i in keys]
    return dict(rank_medians={arm:{str(r):st.median(v) for r,v in values.items()} for arm,values in ranks.items()},
                slow_rank_medians={arm:st.median(v.values()) for arm,v in arms.items()},
                paired_delta=delta,median_delta=st.median(delta),aa_delta=aa,
                aa_max_abs=max(map(abs,aa)),paired_gain_pct=[(1-arms['reuse'][i]/arms['stock'][i])*100 for i in keys],
                aa_max_pct=max(abs(arms['stock_repeat'][i]/arms['stock'][i]-1)*100 for i in keys))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--samples',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if len(a.trace)!=len(a.samples) or a.output.exists():p.error('matching trace/sample lists and NEW output required')
    output=dict(schema='clef-gdn-task/1',traces=[],timing={},tasks={},
        semantics='duration of one explicitly correlated complete GDN task; internal stall included; host/API/kernel intervals overlap',
        excluded='no pipeline/per-core counters, no task-stage causal split, no formal P95')
    all_tasks=[];samples=[]
    for trace_file,sample_file in zip(a.trace,a.samples):
        doc=json.loads(sample_file.read_text());assert doc['status']=='profile_complete'
        rank=doc['rank'];trace=parse_trace(trace_file,divisor=1)
        for row in trace['samples']:
            assert row['phase'] in ('profile_wrapper','profile_prebuilt')
            main_tasks=[]
            for task in row['task_keys']:
                compact=re.sub('[^a-z0-9]','',task['name'].lower())
                if any(x in compact for x in ('clefgdnkktstock','clefgdnkktreuse','chunkgateddeltarulefwd')):
                    main_tasks.append(task)
            if len(main_tasks)!=1:raise ValueError('expected one GDN task: '+str((rank,row['phase'],row['arm'],row['task_keys'])))
            row.update(rank=rank,main_task=main_tasks[0],main_task_us=main_tasks[0]['duration_us'])
            all_tasks.append(row)
        output['traces'].append(trace)
        samples.extend(dict(row,rank=rank) for row in doc['samples'])
    for phase in ('profile_wrapper','profile_prebuilt'):
        output['tasks'][phase]=paired([x for x in all_tasks if x['phase']==phase],'main_task_us')
    for phase in ('pre','profile','post'):
        output['timing'][phase]={}
        for mode in ('wrapper','prebuilt'):
            selected=[x for x in samples if x['phase']==phase and x['mode']==mode]
            output['timing'][phase][mode]={metric:paired(selected,metric) for metric in ('device_ms','wall_ms')}
    inventory={}
    for phase in ('profile_wrapper','profile_prebuilt'):
        inventory[phase]={}
        for arm in {x['arm'] for x in all_tasks if x['phase']==phase}:
            rows=[x for x in all_tasks if x['phase']==phase and x['arm']==arm]
            inventory[phase][arm]=dict(task_counts=[x['task_count'] for x in rows],
                names=dict(Counter(t['name'] for x in rows for t in x['task_keys'])),
                types=dict(Counter(str(t['task_type']) for x in rows for t in x['task_keys'])))
    output.update(inventory=inventory,matched_samples=len(all_tasks),
                  mapped_main_tasks=len(all_tasks),ranks=sorted({x['rank'] for x in all_tasks}))
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(matched_samples=len(all_tasks),task_comparison=output['tasks']),indent=2))


if __name__=='__main__':main()
