#!/usr/bin/env python3
"""Owner-only bounded gate collector. Default prints plan; no SSH or installation.

The private control.py must hold a freshly verified restoration reference.
Single vendor BasicInfo attempt; v3 only after its hit; one paired Pipe group
only after both hits and sufficient time. Any miss/timeout restores immediately.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from probe_mlp_gate_internal import capture_failure
from analyze_mlp_gate_internal import validate_basic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True);p.add_argument('--used-s',type=float,required=True)
    p.add_argument('--remaining-s',type=float,required=True);p.add_argument('--cap-s',type=float,default=112)
    p.add_argument('--collector',choices=['msopprof','torch-pipe'],default='msopprof')
    a=p.parse_args()
    if not (0<a.cap_s<=a.remaining_s-8 and a.used_s+a.remaining_s<=240.001):p.error('budget must leave cleanup margin within cumulative240s')
    if not a.execute:print(json.dumps(dict(status='plan',collector=a.collector,basic_s=22,pipe_s=27,cap_s=a.cap_s,remaining_s=a.remaining_s)));return
    R=a.work;O=a.output;C=R/'control';O.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('f','/workspace/yangchen/work/vision-cache-owner27-3bf466d913ea269d4f4bb37b560a6ee7a3fdbf47/control/supervised_probe_window.py');f=importlib.util.module_from_spec(spec);sys.modules['f']=f;spec.loader.exec_module(f)
    record=dict(status='prepared',used_before_s=a.used_s,remaining_before_s=a.remaining_s,cap_s=a.cap_s,commands=[],basic_hits=[],pipe_hits=[])
    children=[];owned={};stopped=False
    def save():(O/'window.json').write_text(json.dumps(record,indent=2))
    def control(mode):subprocess.run([sys.executable,str(C/'control.py'),mode],check=True,timeout=60)
    def terminate():
        table=f.scan()
        for proc in children:
            if proc.poll() is None:
                for pid in f.tree(table,proc.pid):
                    if pid in table:owned[pid]=table[pid]
        for pid,identity in owned.items():
            if f.same_identity(identity,f.proc_identity(pid)):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc in children:
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                for pid,identity in owned.items():
                    if f.same_identity(identity,f.proc_identity(pid)):
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                proc.wait(timeout=2)
    def run(argv,label,limit,app=None):
        with (O/(label+'.stdout')).open('xb') as out,(O/(label+'.stderr')).open('xb') as err:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
        children.append(proc);item=dict(label=label,argv=argv,pid=proc.pid);record['commands'].append(item);save();begin=time.monotonic()
        while proc.poll() is None:
            table=f.scan()
            for pid in f.tree(table,proc.pid):
                if pid in table:owned[pid]=table[pid]
            if app:
                for path in app.glob('*.json'):
                    try:j=json.loads(path.read_text())
                    except (ValueError,OSError):continue
                    error=capture_failure(j)
                    if error:raise RuntimeError(label+': '+error)
            if time.monotonic()-begin>limit or time.monotonic()-start>a.cap_s:raise TimeoutError(label)
            time.sleep(.1)
        item.update(returncode=proc.returncode,wall_s=time.monotonic()-begin);save()
        if proc.returncode:raise RuntimeError(label+' failed')
    base=[sys.executable,str(R/'probe_mlp_gate_internal.py'),'--execute','--vendor',str(a.vendor)]
    def collect(metric,arm,label,limit):
        dst=O/label/arm;app=O/(label+'-app')/arm
        argv=['/usr/local/Ascend/cann-9.2.1/bin/msprof','op','--aic-metrics='+metric,'--replay-mode=application','--warm-up=0','--launch-count=1','--mstx=true','--mstx-include=CLEF40_TARGET','--output='+str(dst)]+base+['--mode','capture','--arm',arm,'--mstx-lib',str(R/'mstx_gate_range.so'),'--output-dir',str(app)]
        run(argv,label+'-'+arm,limit,app);return validate_basic(dst,app,arm)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('outer termination')))
    try:
        control('preflight');control('stop');stopped=True;start=time.monotonic();record['status']='running';save()
        if a.collector=='torch-pipe':
            argv=[sys.executable,str(R/'profile_mlp_gate.py'),'--execute','--solo','--aic-metrics','pipe','--rank','0','--vendor',str(a.vendor),'--output',str(O/'rank0.json')]
            run(argv,'torch-pipe',max(1,a.cap_s-2))
            result=json.loads((O/'rank0.json').read_text());assert result['status']=='complete'
            infos=list((O/'traces-r0').glob('*_ascend_pt/profiler_info*.json'));assert len(infos)==1
            info=json.loads(infos[0].read_text())
            actual=info['config']['experimental_config']['_aic_metrics']
            assert actual=='ACL_AICORE_PIPE_UTILIZATION',actual
            record['capture']=dict(path=str(O/'rank0.json'),profiler_info=str(infos[0]),actual_metric=actual,export='raw only, offline after restoration')
            record['status']='complete'
        else:
            run(base+['--mode','control','--output-dir',str(O/'pre')],'pre',18)
            pre=json.loads(next((O/'pre').glob('*.json')).read_text());assert pre['status']=='complete'
            arms={arm:{r['pair']:r['device_ms'] for r in pre['samples'] if r['arm']==arm} for arm in ('vendor','vendor_repeat','v3')}
            delta=[arms['v3'][i]-arms['vendor'][i] for i in range(3)];noise=max(abs(arms['vendor_repeat'][i]-arms['vendor'][i]) for i in range(3))
            record['pre_gap']=dict(paired_delta_ms=delta,median_delta_ms=statistics.median(delta),aa_max_ms=noise);save()
            if statistics.median(delta)<=max(noise,.2):raise RuntimeError('gap not reproduced')
            for arm in ('vendor','v3'):
                hit=collect('BasicInfo',arm,'basic',22);record['basic_hits'].append(hit);save();print('BASIC_HIT',arm,hit['task_duration_us'],flush=True)
            record['basic_delta_us']=record['basic_hits'][1]['task_duration_us']-record['basic_hits'][0]['task_duration_us'];save()
            if record['basic_delta_us']<200:raise RuntimeError('BasicInfo task gap not representative')
            if a.cap_s-(time.monotonic()-start)>=68:
                for arm in ('vendor','v3'):
                    hit=collect('PipeUtilization',arm,'pipe',27);record['pipe_hits'].append(hit);save();print('PIPE_HIT',arm,hit['task_duration_us'],flush=True)
            else:record['pipe_skipped']='insufficient time for paired group plus post control'
            run(base+['--mode','control','--output-dir',str(O/'post')],'post',12);record['status']='complete'
    except BaseException as exc:record.update(status='stopped',error=type(exc).__name__+': '+str(exc))
    finally:
        terminate();record['returncodes']=[proc.returncode for proc in children]
        if stopped:
            # Driver contexts can release after process exit. Observe fresh emptiness,
            # never signal a PID inferred only from an old npu-smi row.
            release=time.monotonic()
            while time.monotonic()-release<8:
                result=subprocess.run(['/usr/local/bin/npu-smi','info'],capture_output=True,text=True,timeout=3)
                if result.returncode==0 and not f.npu_rows(result.stdout):break
                time.sleep(.25)
            record['new_compute_collection_s']=time.monotonic()-start
            record['cumulative_compute_collection_s']=a.used_s+record['new_compute_collection_s']
        save()
        if stopped:control('launch')
        print(json.dumps({k:v for k,v in record.items() if k not in ('commands','basic_hits','pipe_hits')}),flush=True)

if __name__=='__main__':main()
