#!/usr/bin/env python3
"""CPU-default plan for one fixed CATLASS cross-output first-K preload switch.

No SSH/install/service actions. Execute only inside the owner's bounded window.
All output allocation/descriptor/workspace/launch/layout/cast costs are included.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--phase', choices=['numeric','timing','profile'], default='numeric')
    ap.add_argument('--tokens', default='967')
    ap.add_argument('--ranks', default='0,1')
    ap.add_argument('--vendor', type=Path)
    ap.add_argument('--model-dir', type=Path, default=Path('/workspace/yangchen/models/Clef-27B'))
    ap.add_argument('--captures', type=Path, default=Path('/workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json'))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--limit-s', type=float, default=65)
    a=ap.parse_args();tokens=[int(x) for x in a.tokens.split(',')];ranks=[int(x) for x in a.ranks.split(',')]
    assert len(set(tokens))==len(tokens) and set(tokens)<={307,967,1595}
    assert len(set(ranks))==len(ranks) and set(ranks)<={0,1}
    arms=['vendor_gate','vendor_gate_repeat','stock_gate','stock_gate_repeat','prefetch_gate',
          'vendor_chain','vendor_chain_repeat','stock_chain','prefetch_chain','vendor_swiglu']
    report=dict(status='plan_only',shape=dict(k=5120,n=17408,tokens=tokens),ranks=[],
        arms=arms,pairs=5,modes=['warm_layer31','streaming_layers31_0_63'],
        contract='BF16 original NZ29; K-order FP32 accumulation; canonical BF16 ND GM output',
        variable='same ELF/tile/output; prefetch int64 0/1 only',
        timing='Event includes host dispatch; wall includes completion; all in-call allocation/layout/launch counted',
        preparation='original weights converted to NZ once per rank/shape; same weights/inputs shared by all arms',
        input='real capture; 967 explicitly uses od-0677/1595 prefix',device_budget_s=240)
    if not a.execute: print(json.dumps(report));return
    assert a.vendor and a.output and not a.output.exists()
    start=time.monotonic()
    def save():a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def deadline():
        if time.monotonic()-start>a.limit_s:raise TimeoutError('bounded probe deadline')
    os.environ['ACLNN_CACHE_LIMIT']='1000000'
    os.environ['ASCEND_CUSTOM_OPP_PATH']=str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
    import torch
    import torch_npu
    import torch.nn.functional as F
    import probe_mlp_ffn as helper
    from fla_npu.ops.ascendc import _runtime
    torch.set_num_threads(1)
    libpath=a.vendor/'op_api/lib/libcust_opapi.so'
    lib=ctypes.CDLL(str(libpath),mode=os.RTLD_LOCAL|os.RTLD_NOW)
    _runtime.runtime()._libraries.insert(0,lib)
    report.update(status='running',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  op_api_sha256=hashlib.sha256(libpath.read_bytes()).hexdigest(),
                  capture_manifest_sha256=hashlib.sha256(a.captures.read_bytes()).hexdigest())
    manifest=json.loads(a.captures.read_text())
    def compare(actual,ref):
        af,rf=actual.float(),ref.float();diff=af-rf
        finite=bool(torch.isfinite(af).all() & torch.isfinite(rf).all())
        if not finite:return dict(finite=False,pass_screen=False)
        rel=float(diff.square().mean().sqrt()/rf.square().mean().sqrt().clamp_min(1e-30))
        norm=float(diff.abs().max()/rf.abs().max().clamp_min(1e-30))
        return dict(finite=True,bit_equal=bool(torch.equal(actual,ref)),relative_rmse=rel,
                    max_abs=float(diff.abs().max()),normalized_max=norm,pass_screen=rel<=.01 and norm<=.02)
    def gate(x,w,prefetch):
        out=torch.empty((len(x),17408),device=x.device,dtype=torch.bfloat16)
        _runtime.call_aclnn('aclnnClefMlpCrossPrefetch',
            lambda ctx:[ctx.tensor(x,'x'),ctx.tensor(w,'weight',acl_format_override=29,
                        storage_shape_override=(320,1088,16,16)),prefetch,ctx.tensor(out,'out')],(out,),
            get_workspace_argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_void_p)])
        return out
    def rank_point(rank,m):
        deadline();helper.initialize_npu(torch,torch_npu,rank)
        record=dict(rank=rank,m=m,preparation=[],numeric=[],warmup=[],samples=[])
        report['ranks'].append(record);save();weights=[];inputs=[];mids=[]
        for layer in (31,0,63):
            deadline();before=time.perf_counter()
            cpu,down,identity=helper.load_weights(a.model_dir,torch,layer,rank)
            w=torch_npu.npu_format_cast(cpu.to(f'npu:{rank}'),29)
            d=torch_npu.npu_format_cast(down.to(f'npu:{rank}'),29)
            case=next(c for c in manifest['cases'] if c['name']==('od-0105' if m==307 else 'od-0677'))
            item=next(t for t in case['tensors'] if t['metadata']['layer']==layer and t['metadata']['rank']==rank)
            payload=torch.load(item['path'],map_location='cpu',weights_only=True);full=payload['input']
            sha=hashlib.sha256(memoryview(full.contiguous().view(torch.uint8).numpy())).hexdigest()
            assert sha==item['input_sha256'] and full.dtype==torch.bfloat16 and full.shape[1]==5120 and len(full)>=m
            x=full[:m].contiguous().to(f'npu:{rank}');torch.npu.synchronize()
            record['preparation'].append(dict(layer=layer,wall_s=time.perf_counter()-before,
                identity=identity,input_sha256=sha,input_kind='explicit_prefix' if m==967 else 'full_capture'))
            ref=F.linear(x,w);stock=gate(x,w,0);on=gate(x,w,1)
            n=dict(layer=layer,stock=compare(stock,ref),prefetch=compare(on,ref),on_off=compare(on,stock),
                   gate_half=compare(on[:,:8704],ref[:,:8704]),up_half=compare(on[:,8704:],ref[:,8704:]),
                   reference_format=int(torch_npu.get_npu_format(ref)),output_format=int(torch_npu.get_npu_format(on)),
                   output_stride=list(on.stride()))
            n['repeat_equal']=[bool(torch.equal(gate(x,w,1),on)),bool(torch.equal(gate(x,w,0),stock))]
            n['chain']=compare(F.linear(torch_npu.npu_swiglu(on),d),F.linear(torch_npu.npu_swiglu(ref),d))
            record['numeric'].append(n);save()
            assert on.shape==ref.shape and on.dtype==torch.bfloat16 and on.is_contiguous()
            assert all(n[k]['pass_screen'] for k in ['stock','prefetch','gate_half','up_half','chain'])
            assert n['on_off']['bit_equal'] and n['on_off']['finite'] and all(n['repeat_equal']), 'movement-only ON/OFF must be exact and stable'
            weights.append((w,d));inputs.append(x);mids.append(ref)
        if a.phase=='numeric':return
        record['active_tensor_bytes']=int(torch.npu.memory_allocated(rank))
        assert record['active_tensor_bytes']<=1536*1024*1024
        def call(arm,streaming):
            arm=arm.replace('_repeat','')
            for i in (range(3) if streaming else range(1)):
                w,d=weights[i];x=inputs[i]
                if arm=='vendor_swiglu':out=torch_npu.npu_swiglu(mids[i]);continue
                mid=F.linear(x,w) if arm.startswith('vendor') else gate(x,w,int(arm.startswith('prefetch')))
                out=F.linear(torch_npu.npu_swiglu(mid),d) if arm.endswith('_chain') else mid
            return out
        profiled=False
        def samples(phase,modes,pairs,selected):
            for mode in modes:
                streaming=mode.startswith('streaming');count=3 if streaming else 1
                for arm in selected:
                    before=time.perf_counter()
                    for _ in range(5):call(arm,streaming)
                    torch.npu.synchronize()
                    record['warmup'].append(dict(phase=phase,mode=mode,arm=arm,calls=5*count,wall_s=time.perf_counter()-before))
                for pair in range(pairs):
                    for arm in (selected if pair%2==0 else list(reversed(selected))):
                        deadline();begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        from contextlib import nullcontext
                        scope=torch.profiler.record_function(f'crosspref/{phase}/{pair}/{arm}') if profiled else nullcontext()
                        with scope:
                            before=time.perf_counter();begin.record();call(arm,streaming);end.record();torch.npu.synchronize()
                        record['samples'].append(dict(phase=phase,mode=mode,pair=pair,arm=arm,layers=count,
                            event_ms=begin.elapsed_time(end)/count,wall_ms=(time.perf_counter()-before)*1000/count))
                save()
        if a.phase=='timing':samples('unprofiled',report['modes'],5,arms)
        else:
            selected=['vendor_gate','vendor_gate_repeat','stock_gate','prefetch_gate']
            samples('pre',['streaming_layers31_0_63'],2,selected)
            cfg=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,data_simplification=False,
                record_op_args=False,export_type=[torch_npu.profiler.ExportType.Text])
            trace=a.output.parent/f'traces-r{rank}'
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False,profile_memory=False,
                with_stack=False,with_modules=False,experimental_config=cfg,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace),worker_name=f'crosspref-r{rank}',analyse_flag=False)):
                profiled=True;samples('profile',['streaming_layers31_0_63'],3,selected);profiled=False
            samples('post',['streaming_layers31_0_63'],2,selected);record['trace_directory']=str(trace)
    try:
        with torch.inference_mode():
            # M967 on both ranks precedes the explicitly requested guard points.
            for m in tokens:
                for rank in ranks:rank_point(rank,m);torch.npu.synchronize();torch.npu.empty_cache()
        report['status']='complete'
    except BaseException as exc:
        report.update(status='stopped',error=type(exc).__name__+': '+str(exc));raise
    finally:
        report['wall_s']=time.monotonic()-start;save()
        print(json.dumps(dict(status=report['status'],phase=a.phase,wall_s=report['wall_s'])),flush=True)


if __name__=='__main__':main()
