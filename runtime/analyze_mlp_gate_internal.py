#!/usr/bin/env python3
"""Offline msopprof CSV/tiling inventory; preserve raw units and PMU grouping.

Per-core times describe load balance, never a sum-to-wall model. Ratios of
pipelines may overlap. Missing vendor schema is UNKNOWN, with no guessed tile.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct


def number(x):
    try:return float(x.strip().rstrip('%'))
    except (ValueError,AttributeError):return None


def stats(values):
    values=[v for v in values if v is not None]
    if not values:return None
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values),
                spread_over_max=(max(values)-min(values))/max(values) if max(values)>0 else None)


def inventory(root, schema):
    out=dict(root=str(root),files=[],tiling=[],limitations=[])
    for p in sorted(root.rglob('*.csv')):
        if p.stat().st_size>10*1024*1024:continue
        with p.open() as f:rows=list(csv.DictReader(f))
        if not rows:continue
        cols=list(rows[0]);record=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),columns=cols,row_count=len(rows))
        if 'PipeUtilization' in p.name:
            cycles=next((c for c in cols if c=='aic_total_cycles'),None)
            duration=next((c for c in cols if c=='aic_time(us)'),None)
            active=[r for r in rows if (number(r.get(cycles,'')) or number(r.get(duration,'')) or 0)>0]
            record.update(active_aic_rows=len(active),core_ids=[{k:r[k] for k in ('block_id','sub_block_id') if k in r} for r in active],
                per_core_time_us=stats([number(r.get(duration,'')) for r in active]) if duration else None,
                per_core_total_cycles=stats([number(r.get(cycles,'')) for r in active]) if cycles else None,
                raw_core_rows=active)
        record['numeric_columns']={c:stats([number(r[c]) for r in rows]) for c in cols if any(number(r[c]) is not None for r in rows)}
        if p.name=='OpBasicInfo.csv':record['raw_rows']=rows
        out['files'].append(record)
    for p in sorted(root.rglob('*tiling*.bin')):
        raw=p.read_bytes();record=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),decode='UNKNOWN: no proven vendor schema')
        # Only decode the known v3 layout, with exact size, invariant fields and suffix.
        if schema and root.name=='v3' and len(raw)==schema['v3_size']:
            fields={name:struct.unpack_from('<i',raw,offset)[0] for name,offset in schema['offsets'].items()}
            suffix=struct.unpack_from('<III',raw,schema['cube_size'])
            if [fields.get(k) for k in ('M','N','Ka','Kb')]==[967,17408,5120,5120] and suffix==(967,8,3):
                record.update(decode='v3 exact SDK structure and invariant match',fields=fields,tokens=suffix[0],mTasks=suffix[1],nTasks=suffix[2])
        out['tiling'].append(record)
    if not out['files']:out['limitations'].append('No CSV exported by collector')
    return out


def validate_basic(root, app_root, arm):
    """Fail closed on any target mismatch. Shape/dtype provenance is explicit.

    msopprof BasicInfo exports task name/block/count; live tensor metadata and
    SHA come from the single selected first launch in the native MSTX range.
    """
    paths=[p for p in root.rglob('*.csv') if p.stem.replace('_','').lower()=='opbasicinfo']
    rows=[(p,row) for p in paths for row in csv.DictReader(p.open())]
    if len(rows)!=1:raise ValueError(f'{arm}: expected one BasicInfo task, got {len(rows)}')
    path,row=rows[0];cols={''.join(c for c in k.lower() if c.isalnum()):v for k,v in row.items()}
    name=cols.get('opname','')+' '+cols.get('optype','')+' '+cols.get('kernelname','')
    slug=''.join(c for c in name.lower() if c.isalnum())
    if arm=='v3':assert 'clefmlpgategemm' in slug, name
    else:assert 'matmul' in slug and 'clefmlp' not in slug, name
    block=cols.get('blockdim',cols.get('blocknum',''))
    assert number(block)==24, block
    duration=number(cols.get('taskdurationus',''))
    assert duration and duration>0, row
    apps=[(p,json.loads(p.read_text())) for p in app_root.glob('*.json')]
    assert apps, 'no application mapping'
    for p,j in apps:
        t=j.get('target',{})
        assert j['status']=='complete' and j.get('mstx_ready') is True and 'mstx_native_id' in j
        assert j['arm']==arm and t.get('launch_index_in_range')==0 and t.get('layer')==31
        assert t.get('input_shape')==[967,5120] and t.get('input_dtype')=='torch.bfloat16'
        assert t.get('weight_shape')==[17408,5120] and t.get('weight_dtype')=='torch.bfloat16' and t.get('weight_format')==29
        assert t.get('output_shape')==[967,17408] and t.get('output_dtype')=='torch.bfloat16' and t.get('output_format')==2
        assert len(t.get('prefix_sha256',''))==64 and t.get('weight_rank_hashes')
    return dict(arm=arm,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_row=row,
                task_duration_us=duration,block_dim=24,task_count=1,
                provenance='task identity/block/count from collector; tensor shape/dtype/SHA from matched first native-range launch',
                applications=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_range_id=j['mstx_native_id'],target=j['target']) for p,j in apps])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--schema',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    schema=json.loads(a.schema.read_text()) if a.schema else None
    a.output.write_text(json.dumps(dict(inventories=[inventory(r,schema) for r in a.root]),indent=2)+'\n')

if __name__=='__main__':main()
