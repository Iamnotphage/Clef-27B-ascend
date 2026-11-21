#!/usr/bin/env python3
"""CPU-default plan for fixed M967/rank0 gate PMU attribution.

Run only in the owner's bounded window. msprof op application replay preserves
application preparation. MSTX selects layer31, with real layers0/63 conditioning;
no kernel/tiling/source install, per-op sync, down execution or tile sweep.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=['control','capture'],default='control')
    p.add_argument('--arm',choices=['vendor','v3'],default='vendor')
    p.add_argument('--vendor',type=Path)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    plan=dict(schema='clef-gate-internal/1',rank=0,m=967,layers=[31,0,63],capture_layer=31,
        precision='original BF16/NZ -> canonical ND',pairs=3,warmup_passes_each_arm=5,
        mode=a.mode,arm=a.arm,metrics=['PipeUtilization','Memory','MemoryL0','ResourceConflictRatio'],
        replay='application, warm-up=0 in collector; explicit full application warmup; MSTX CLEF40_TARGET',
        stop='numeric failure; gap not reproduced; tool unsupported; aggregate device/collection budget 240s')
    if not a.execute: print(json.dumps(plan,indent=2));return
    if not a.vendor or not a.output_dir:p.error('execute requires vendor and private output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(plan,status='running',pid=os.getpid(),preparation=[],numeric=[],warmup=[],samples=[])
    outpath=a.output_dir/(str(os.getpid())+'-'+uuid.uuid4().hex[:8]+'.json')
    def save():outpath.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.environ['ACLNN_CACHE_LIMIT']='1000000'
    os.environ['ASCEND_CUSTOM_OPP_PATH']=str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
    import torch
    import torch_npu
    import torch.nn.functional as F
    import probe_mlp_ffn as helper
    from fla_npu.ops.ascendc import _runtime
    torch.set_num_threads(1);helper.initialize_npu(torch,torch_npu,0)
    lib=ctypes.CDLL(str(a.vendor/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_LOCAL|os.RTLD_NOW)
    _runtime.runtime()._libraries.insert(0,lib)
    def gate(x,w):
        y=torch.empty((967,17408),dtype=torch.bfloat16,device=x.device)
        _runtime.call_aclnn('aclnnClefMlpGateGemm',lambda ctx:[ctx.tensor(x,'x'),
            ctx.tensor(w,'weight',acl_format_override=29,storage_shape_override=(320,1088,16,16)),ctx.tensor(y,'y')],(y,),
            get_workspace_argtypes=[ctypes.c_void_p]*3+[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_void_p)])
        return y
    inputs=[];weights=[]
    try:
        captures=json.loads(Path('/workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json').read_text())
        with torch.inference_mode():
            for layer in (31,0,63):
                start=time.perf_counter();cpu,down,identity=helper.load_weights(Path('/workspace/yangchen/models/Clef-27B'),torch,layer,0)
                w=torch_npu.npu_format_cast(cpu.to('npu:0'),29);d=torch_npu.npu_format_cast(down.to('npu:0'),29)
                item=next(t for c in captures['cases'] if c['name']=='od-0677' for t in c['tensors'] if t['metadata']['rank']==0 and t['metadata']['layer']==layer)
                x=torch.load(item['path'],map_location='cpu',weights_only=True)['input']
                digest=hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
                assert digest==item['input_sha256'] and x.dtype==torch.bfloat16 and x.shape==(1595,5120)
                x=x[:967].contiguous().to('npu:0');ref=F.linear(x,w);y=gate(x,w)
                assert y.shape==ref.shape==(967,17408) and y.dtype==ref.dtype==torch.bfloat16
                assert y.is_contiguous() and torch_npu.get_npu_format(y)==torch_npu.get_npu_format(ref)==2
                n=dict(layer=layer,input_sha256=digest,input_kind='explicit_prefix',halves={})
                for label,aa,bb in [('gate',y[:,:8704],ref[:,:8704]),('up',y[:,8704:],ref[:,8704:])]:
                    af,bf=aa.float(),bb.float();assert bool(torch.isfinite(af).all() & torch.isfinite(bf).all())
                    delta=af-bf;rmse=float(delta.square().mean().sqrt()/bf.square().mean().sqrt().clamp_min(1e-30));norm=float(delta.abs().max()/bf.abs().max().clamp_min(1e-30))
                    assert rmse<=.01 and norm<=.02;n['halves'][label]=dict(relative_rmse=rmse,normalized_max=norm)
                n['repeat_equal']=bool(torch.equal(gate(x,w),y));assert n['repeat_equal']
                report['numeric'].append(n);report['preparation'].append(dict(layer=layer,wall_s=time.perf_counter()-start,weight_identity=identity))
                inputs.append(x);weights.append((w,d));save()
            def one(arm,i):return gate(inputs[i],weights[i][0]) if arm=='v3' else F.linear(inputs[i],weights[i][0])
            def ring(arm):
                for i in range(3):mid=one(arm,i);out=mid
                return out
            for arm in ['vendor','v3']:
                start=time.perf_counter()
                for _ in range(5):ring(arm)
                torch.npu.synchronize();report['warmup'].append(dict(arm=arm,calls=15,wall_s=time.perf_counter()-start))
            b=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True)
            b.record();e.record();torch.npu.synchronize();b.elapsed_time(e)
            if a.mode=='control':
                for pair in range(3):
                    for arm in (['vendor','vendor_repeat','v3'] if pair%2==0 else ['v3','vendor_repeat','vendor']):
                        # Include ring allocation/dispatch, as in e05ec7a; no per-layer sync.
                        b=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True)
                        start=time.perf_counter();b.record();ring('vendor' if arm=='vendor_repeat' else arm);e.record();torch.npu.synchronize()
                        report['samples'].append(dict(pair=pair,arm=arm,device_ms=b.elapsed_time(e)/3,wall_ms=(time.perf_counter()-start)*1000/3))
            else:
                # A full ring of this arm immediately precedes capture, ending at layer63.
                ring(a.arm);torch.npu.synchronize()
                tag=torch_npu.npu.mstx.range_start('CLEF40_TARGET',torch.npu.current_stream())
                if not tag:
                    raise RuntimeError('MSTX range inactive (id=0); abort before target submission')
                y=one(a.arm,0)
                torch_npu.npu.mstx.range_end(tag)
                # Finish the same real three-layer sequence; no sync inside the ring.
                for i in (1,2):mid=one(a.arm,i);out=mid
                torch.npu.synchronize();report['mstx_range_id']=tag
        report['status']='complete';save()
    except BaseException as exc:
        report.update(status='stopped',error=type(exc).__name__+': '+str(exc));save();raise

if __name__=='__main__':main()
