#!/usr/bin/env python3
"""CPU-default plan for gate GEMM-only and measured SwiGLU budget screening.

Execution is reserved for the device owner in a later authorized window. No
SSH, service calls, source installation, compiler invocation or down tuning.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('numeric','timing'),default='numeric')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--vendor',type=Path)
    p.add_argument('--model-dir',type=Path,default=Path('/workspace/yangchen/models/Clef-27B'))
    p.add_argument('--captures',type=Path,default=Path('/workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json'))
    p.add_argument('--ranks',default='0,1')
    p.add_argument('--tokens',type=int,choices=(307,967,1595),default=967)
    p.add_argument('--pairs',type=int,choices=(3,4,5),default=5)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    a.ranks=[int(r) for r in a.ranks.split(',')]
    if not a.ranks or len(a.ranks)!=len(set(a.ranks)) or set(a.ranks)-{0,1}:p.error('unique ranks 0/1 only')
    if a.execute and (a.vendor is None or a.output is None):p.error('execution needs vendor and a new output')
    if a.output and a.output.exists():p.error('output exists')
    return a


def plan(a):
    return dict(schema='clef-gate-budget/1',status='cpu_prepared_uncompiled_unmeasured',
                phase=a.phase,ranks_serial=a.ranks,device_equals_rank=True,m=a.tokens,k=5120,n=17408,
                layers=[31,0,63],dtype='BF16',weight_format='NZ29',
                output_contract='canonical [gate8704,up8704], BF16 [M,17408]; actual physical format/stride recorded',
                arms=['vendor_gate','vendor_gate_repeat','candidate_gate','vendor_swiglu','vendor_chain','candidate_gate_chain'],
                modes=['warm','read_flush_proxy_256MiB','streaming_3_real_layers'],pairs=a.pairs,
                preparation='one NZ conversion of original weights shared by both arms; no paired-weight reorder',
                timing='allocations/layout handling/dispatch included; preparation and conditioning outside timing',
                candidate='synchronous pure-Cube API IterateAll, automatic MultiCore tiling, canonical GM output',
                excluded=['communication','new down implementation','service','quality claim','async fusion'],
                stop='numeric/finite/repeatability failure; gate slowdown consumes measured SwiGLU budget; no automatic integration')


def execute(a,report):
    os.environ['ACLNN_CACHE_LIMIT']='1000000'
    os.environ['ASCEND_CUSTOM_OPP_PATH']=str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
    import torch
    import torch_npu
    import torch.nn.functional as F
    import probe_mlp_ffn as helper
    from fla_npu.ops.ascendc import _runtime
    torch.set_num_threads(1)
    lib=ctypes.CDLL(str(a.vendor/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_LOCAL|os.RTLD_NOW)
    _runtime.runtime()._libraries.insert(0,lib)
    captures=json.loads(a.captures.read_text())
    report['capture_manifest_sha256']=hashlib.sha256(a.captures.read_bytes()).hexdigest()
    report['ranks']=[]
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def numeric(actual,ref):
        af,rf=actual.float(),ref.float()
        finite=bool(torch.isfinite(af).all() & torch.isfinite(rf).all())
        if not finite:return dict(finite=False,pass_screen=False)
        d=af-rf
        rmse=float(d.square().mean().sqrt()/rf.square().mean().sqrt().clamp_min(1e-30))
        norm=float(d.abs().max()/rf.abs().max().clamp_min(1e-30))
        return dict(finite=True,bit_equal=bool(torch.equal(actual,ref)),relative_rmse=rmse,
                    max_abs=float(d.abs().max()),normalized_max=norm,pass_screen=rmse<=.01 and norm<=.02)
    def gate(x,w):
        y=torch.empty((x.shape[0],17408),dtype=torch.bfloat16,device=x.device)
        _runtime.call_aclnn('aclnnClefMlpGateGemm',
          lambda ctx:[ctx.tensor(x,'x'),ctx.tensor(w,'weight',acl_format_override=29,
                      storage_shape_override=(320,1088,16,16)),ctx.tensor(y,'y')],(y,),
          get_workspace_argtypes=[ctypes.c_void_p]*3+[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_void_p)])
        return y
    def run_rank(rank):
        helper.initialize_npu(torch,torch_npu,rank)
        weights=[];x_values=[];preacts=[];record=dict(rank=rank,device=rank,preparation=[],numeric=[],warmup=[],samples=[])
        report['ranks'].append(record);save()
        for layer in (31,0,63):
            cpu,down,identity=helper.load_weights(a.model_dir,torch,layer,rank)
            start=time.perf_counter()
            w=torch_npu.npu_format_cast(cpu.to(f'npu:{rank}'),29)
            d=torch_npu.npu_format_cast(down.to(f'npu:{rank}'),29)
            torch.npu.synchronize()
            record['preparation'].append(dict(layer=layer,seconds=time.perf_counter()-start,identity=identity))
            case=next(c for c in captures['cases'] if c['name']==('od-0105' if a.tokens==307 else 'od-0677'))
            item=next(t for t in case['tensors'] if t['metadata']['layer']==layer and t['metadata']['rank']==rank)
            payload=torch.load(item['path'],map_location='cpu',weights_only=True);x=payload['input']
            digest=hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
            if digest!=item['input_sha256'] or x.dtype!=torch.bfloat16 or x.shape[1]!=5120 or len(x)<a.tokens:
                raise ValueError('capture hash/shape/BF16 mismatch')
            x=x[:a.tokens].contiguous().to(f'npu:{rank}')
            ref=F.linear(x,w);actual=gate(x,w)
            if actual.shape!=ref.shape or actual.dtype!=ref.dtype or not actual.is_contiguous():
                raise ValueError('gate output contract mismatch')
            n=numeric(actual,ref)
            n.update(layer=layer,input_sha256=digest,input_kind='full_capture' if a.tokens in (307,1595) else 'explicit_prefix',
                     reference_format=int(torch_npu.get_npu_format(ref)),candidate_format=int(torch_npu.get_npu_format(actual)),
                     reference_stride=list(ref.stride()),candidate_stride=list(actual.stride()))
            n['chain']=numeric(F.linear(torch_npu.npu_swiglu(actual),d),
                               F.linear(torch_npu.npu_swiglu(ref),d))
            record['numeric'].append(n);save()
            if not n['pass_screen'] or not n['chain']['pass_screen'] or not torch.equal(gate(x,w),actual):raise ValueError('gate numerical/stability screen failed')
            weights.append((w,d));x_values.append(x);preacts.append(ref)
        if a.phase=='numeric':return
        flush=torch.ones(256*1024*1024//2,dtype=torch.bfloat16,device=f'npu:{rank}')
        record['allocated_tensor_bytes']=int(torch.npu.memory_allocated(rank));save()
        if record['allocated_tensor_bytes']>1536*1024*1024:raise RuntimeError('1.5GiB active tensor cap exceeded')
        def call(arm,streaming):
            if arm=='vendor_gate_repeat':arm='vendor_gate'
            for i in (range(3) if streaming else range(1)):
                w,d=weights[i];x=x_values[i]
                if arm=='vendor_swiglu':out=torch_npu.npu_swiglu(preacts[i])
                else:
                    mid=gate(x,w) if arm.startswith('candidate') else F.linear(x,w)
                    out=mid if arm.endswith('_gate') else F.linear(torch_npu.npu_swiglu(mid),d)
            return out
        for mode in report['modes']:
            streaming=mode.startswith('streaming');count=3 if streaming else 1
            for arm in report['arms']:
                start=time.perf_counter()
                for _ in range(5):call(arm,streaming)
                torch.npu.synchronize()
                record['warmup'].append(dict(mode=mode,arm=arm,calls=5*count,wall_s=time.perf_counter()-start))
            for pair in range(a.pairs):
                order=report['arms'] if pair%2==0 else list(reversed(report['arms']))
                for arm in order:
                    if mode.startswith('read_flush'):torch.sum(flush);torch.npu.synchronize()
                    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    wall=time.perf_counter();begin.record();call(arm,streaming);end.record();torch.npu.synchronize()
                    record['samples'].append(dict(mode=mode,pair=pair,arm=arm,device_ms=begin.elapsed_time(end)/count,
                        wall_ms=(time.perf_counter()-wall)*1000/count))
            save()
    report['status']='running';save()
    try:
        with torch.inference_mode():
            for rank in a.ranks:run_rank(rank);torch.npu.synchronize();torch.npu.empty_cache()
    except Exception as error:
        report.update(status='stopped',error=str(error));save();raise
    report['status']='complete';save()


if __name__=='__main__':
    args=arguments();report=plan(args)
    if args.execute:execute(args,report)
    else:print(json.dumps(report,indent=2))
