#!/usr/bin/env python3
"""Isolated BF16 NZ matmul/SwiGLU screening; never patches the service.

967 is an explicit prefix of the historical 1595-token capture. Conversion is
one-time preparation; each timed call includes output allocation, ACLNN dispatch,
the candidate epilogue, and the unchanged down GEMM. TP allreduce is excluded.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def paired_weights(packed):
    """Keep gate first within every 256-column output tile (no arithmetic)."""
    gate, up = packed.chunk(2, dim=0)
    return __import__('torch').stack((gate.view(68, 128, 5120),
                                     up.view(68, 128, 5120)), dim=1).reshape(17408, 5120)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('numeric','timing'), default='numeric')
    p.add_argument('--execute', action='store_true', help='Explicitly opt into the isolated device probe')
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/workspace/yangchen/models/Clef-27B'))
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--device', type=int, choices=(0, 1), required=True)
    p.add_argument('--layers', default='31,0,63')
    p.add_argument('--tokens', default='967,307,1595')
    p.add_argument('--pairs', type=int, default=5)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.device != args.rank:
        p.error('This window pins rank 0/1 to logical NPU 0/1')
    if not args.execute:
        print(json.dumps(dict(execute=False, rank=args.rank, device=args.device, tokens=args.tokens,
                              status='experimental; prior v1 failed numeric screen; v2 requires explicit screening',
                              scope='isolated full-chain probe; no service integration')))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        p.error('output exists; choose a new output path')
    os.environ['ACLNN_CACHE_LIMIT'] = '1000000'
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = str(args.vendor) + ':' + os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    import torch
    import torch_npu
    import torch.nn.functional as F
    from fla_npu.ops.ascendc import _runtime
    import probe_mlp_ffn as probe
    torch.set_num_threads(1)
    init = probe.initialize_npu(torch, torch_npu, args.device)
    runtime = _runtime.runtime()
    lib = ctypes.CDLL(str(args.vendor / 'op_api/lib/libcust_opapi.so'),
                      mode=os.RTLD_LOCAL | os.RTLD_NOW | getattr(os, 'RTLD_NODELETE', 0))
    runtime._libraries.insert(0, lib)
    captures = json.loads(args.captures.read_text())
    report = dict(rank=args.rank, device=args.device, initialization=init,
                  captures_sha256=hashlib.sha256(args.captures.read_bytes()).hexdigest(),
                  reference='F.linear(NZ)->npu_swiglu->F.linear(NZ)',
                  phase=args.phase,
                  candidate='NZ paired matmul->BF16 UB->FP32 SiLU(gate)*up->BF16 compact->F.linear(NZ)',
                  allreduce=False, quantization=False, preparation=[], numeric=[], timing=[])
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def compare(a, b):
        a, b = a.float(), b.float()
        d = a-b
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        return dict(finite=finite, bit_equal=bool(torch.equal(a, b)),
                    max_abs=float(d.abs().max()),
                    relative_rmse=float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30)),
                    normalized_max=float(d.abs().max()/b.abs().max().clamp_min(1e-30)))
    def candidate(x, w, stage=0):
        y = torch.empty((x.shape[0], 8704), dtype=torch.bfloat16, device=x.device)
        _runtime.call_aclnn('aclnnClefMlpSwiglu',
            lambda ctx: [ctx.tensor(x, 'x'),
                         ctx.tensor(w, 'weight', acl_format_override=29,
                                    storage_shape_override=(320,1088,16,16)),
                         ctypes.c_int64(stage), ctx.tensor(y, 'y')], (y,),
            get_workspace_argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p] +
                [ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_void_p)])
        return y
    weights = []
    for layer in map(int, args.layers.split(',')):
        cpu, down, identity = probe.load_weights(args.model_dir, torch, layer, args.rank)
        start=time.perf_counter()
        packed=cpu.to(f'npu:{args.device}')
        reordered=paired_weights(cpu).to(f'npu:{args.device}')
        gate= torch_npu.npu_format_cast(packed,29)
        fused= torch_npu.npu_format_cast(reordered,29)
        down= torch_npu.npu_format_cast(down.to(f'npu:{args.device}'),29)
        torch.npu.synchronize()
        report['preparation'].append(dict(layer=layer, seconds=time.perf_counter()-start,
                                          identity=identity, formats=[torch_npu.get_npu_format(t) for t in (gate,fused,down)]))
        weights.append((layer,gate,fused,down))
        del packed,reordered,cpu
    flush_buf=torch.ones(256*1024*1024//2,dtype=torch.bfloat16,device=f'npu:{args.device}')
    lengths=[int(x) for x in args.tokens.split(',')]
    with torch.inference_mode():
        for m in lengths:
            xs=[]
            for layer,gate,fused,down in weights:
                case=next(c for c in captures['cases'] if c['name']==('od-0105' if m==307 else 'od-0677'))
                item=next(t for t in case['tensors'] if t['metadata']['layer']==layer and t['metadata']['rank']==args.rank)
                saved=torch.load(item['path'],map_location='cpu',weights_only=True)
                x=saved['input']; meta=saved['metadata']
                if meta['layer']!=layer or meta['rank']!=args.rank or x.dtype!=torch.bfloat16 or x.shape[1]!=5120 or len(x)<m:
                    raise ValueError('Capture identity/shape mismatch')
                digest=hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
                if digest != item['input_sha256']: raise ValueError('Captured input checksum mismatch')
                xs.append(x[:m].contiguous().to(f'npu:{args.device}'))
                if layer==weights[0][0]:
                    reference_packed=F.linear(xs[-1],gate)
                    reference_gate,reference_up=reference_packed.chunk(2,dim=-1)
                    gate_diff=compare(candidate(xs[-1],fused,1),reference_gate)
                    up_diff=compare(candidate(xs[-1],fused,2),reference_up)
                    ref_mid=torch_npu.npu_swiglu(reference_packed)
                    actual_mid=candidate(xs[-1],fused)
                    torch.npu.synchronize()
                    a=compare(actual_mid,ref_mid)
                    b=compare(F.linear(actual_mid,down),F.linear(ref_mid,down))
                    row=dict(tokens=m,layer=layer,input_file=item['path'],input_sha256=digest,
                             input_kind='full_capture' if len(x)==m else 'explicit_prefix',
                             gate=gate_diff,up=up_diff,mid=a,chain=b)
                    report['numeric'].append(row); save()
                    print(json.dumps(row),flush=True)
                    if not all(v['finite'] and v['relative_rmse']<=0.01 and v['normalized_max']<=0.02 for v in (gate_diff,up_diff,a,b)):
                        report['stopped']='numeric_screen_failed';report['first_failed_stage']=next(k for k in ('gate','up','mid','chain') if not (row[k]['finite'] and row[k]['relative_rmse']<=0.01 and row[k]['normalized_max']<=0.02));save();return 2
                    # Repeat-output stability independently from reference tolerance.
                    for _ in range(2):
                        if not torch.equal(candidate(xs[-1],fused),actual_mid):
                            report['stopped']='unstable_output';save();return 2
            if args.phase == 'numeric':
                del xs
                continue
            def run(arm, streaming):
                selected=range(len(weights)) if streaming else range(1)
                for i in selected:
                    _,gate,fused,down=weights[i]
                    middle=candidate(xs[i],fused) if arm=='candidate' else torch_npu.npu_swiglu(F.linear(xs[i],gate))
                    out=F.linear(middle,down)
                return out
            # Full preparation warmup is recorded separately for each arm/mode.
            for mode in ('warm','cold_proxy_256MiB','streaming_3_real_layers'):
                streaming=mode.startswith('streaming')
                count=len(weights) if streaming else 1
                for arm in ('reference','candidate'):
                    start=time.perf_counter()
                    for _ in range(args.warmup):run(arm,streaming)
                    torch.npu.synchronize()
                    report.setdefault('warmup',[]).append(dict(tokens=m,mode=mode,arm=arm,calls=args.warmup*count,wall_s=time.perf_counter()-start))
                for pair in range(args.pairs):
                    order=('reference','candidate') if pair%2==0 else ('candidate','reference')
                    for arm in order:
                        if mode=='cold_proxy_256MiB':
                            torch.sum(flush_buf);torch.npu.synchronize()
                        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                        wall=time.perf_counter();start.record()
                        repeats=1 if mode=='cold_proxy_256MiB' else args.repeats
                        for _ in range(repeats):run(arm,streaming)
                        end.record();torch.npu.synchronize()
                        report['timing'].append(dict(tokens=m,mode=mode,pair=pair,arm=arm,
                            calls=repeats*count,device_ms=start.elapsed_time(end)/(repeats*count),
                            wall_ms=(time.perf_counter()-wall)*1000/(repeats*count)))
                save()
            del xs
            if m == 967:
                ratios=[]
                for mode in ('warm','cold_proxy_256MiB','streaming_3_real_layers'):
                    rows=[r for r in report['timing'] if r['tokens']==m and r['mode']==mode]
                    for clock in ('device_ms','wall_ms'):
                        values=[next(r[clock] for r in rows if r['pair']==i and r['arm']=='reference') /
                                next(r[clock] for r in rows if r['pair']==i and r['arm']=='candidate')
                                for i in range(args.pairs)]
                        median=statistics.median(values)
                        mad=statistics.median(abs(x-median) for x in values)
                        ratios.append(dict(mode=mode,clock=clock,ratios=values,median=median,mad=mad,
                                           stable_win=median > 1+max(0.01,3*mad) and min(values)>1))
                report['mainpoint_screen']=ratios;save()
                print(json.dumps({'mainpoint_screen':ratios}),flush=True)
                if not all(r['stable_win'] for r in ratios):
                    report['stopped']='mainpoint_full_chain_has_no_stable_win';save();return 3
    report['completed']=True;save();return 0

if __name__=='__main__':
    raise SystemExit(main())
