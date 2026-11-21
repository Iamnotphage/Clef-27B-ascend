#!/usr/bin/env python3
"""M967 three-layer gate attribution; CPU plan unless explicitly executed.

No SSH/service mutation. Parent must bound the entire two-child run. Fixed v3
kernel/tiling, original BF16 NZ weights, canonical ND output, no per-op sync.
"""
import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--rank', type=int, choices=(0, 1), default=0)
    p.add_argument('--vendor', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--barrier', type=Path)
    a = p.parse_args()
    plan = dict(schema='clef-gate-profile/1', tokens=967, layers=[31, 0, 63],
                pairs=3, warmup_passes=5, arms=['vendor', 'vendor_repeat', 'v3'],
                phases=['concurrent_pre', 'serial_pre', 'profile', 'serial_post', 'concurrent_post'],
                original_schedule='4a51f71 ranks_serial [0,1]',
                profile='CPU+NPU Level1/AiCoreNone; raw first, offline analyse after restore',
                semantics='same original NZ weights, BF16 canonical ND, allocation/dispatch included; 3 layers/sample',
                excluded=['down', 'new kernel/tiling', 'per-op synchronization', 'shape/stack/memory collection'])
    if not a.execute:
        print(json.dumps(plan, indent=2)); return
    if not all((a.vendor, a.output, a.barrier)) or a.output.exists():
        p.error('execute requires vendor, NEW output and NEW shared barrier directory')
    os.environ['ACLNN_CACHE_LIMIT'] = '1000000'
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    import torch
    import torch_npu
    import torch.nn.functional as F
    import probe_mlp_ffn as helper
    from fla_npu.ops.ascendc import _runtime
    torch.set_num_threads(1)
    helper.initialize_npu(torch, torch_npu, a.rank)
    lib = ctypes.CDLL(str(a.vendor/'op_api/lib/libcust_opapi.so'), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    _runtime.runtime()._libraries.insert(0, lib)
    report = dict(plan, rank=a.rank, pid=os.getpid(), status='running', numeric=[], warmup=[], samples=[],
                  cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def signal(name): (a.barrier/(name+f'.{a.rank}')).touch(exist_ok=False)
    def wait(name, rank):
        end = time.monotonic()+100
        while not (a.barrier/(name+f'.{rank}')).exists():
            if time.monotonic()>end: raise TimeoutError(name)
            time.sleep(.01)
    def barrier(name): signal(name); wait(name, 1-a.rank)
    profiled = False
    def scope(name): return torch.profiler.record_function('clef40/'+name) if profiled else nullcontext()
    def gate(x, w):
        with scope('v3_alloc'):
            y = torch.empty((967, 17408), dtype=torch.bfloat16, device=x.device)
        with scope('v3_bridge'):
            _runtime.call_aclnn('aclnnClefMlpGateGemm',
                lambda ctx: [ctx.tensor(x, 'x'), ctx.tensor(w, 'weight', acl_format_override=29,
                    storage_shape_override=(320, 1088, 16, 16)), ctx.tensor(y, 'y')], (y,),
                get_workspace_argtypes=[ctypes.c_void_p]*3 + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
        return y
    weights = []; inputs = []
    try:
        captures = json.loads(Path('/workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json').read_text())
        for layer in (31, 0, 63):
            start = time.perf_counter()
            cpu, down, identity = helper.load_weights(Path('/workspace/yangchen/models/Clef-27B'), torch, layer, a.rank)
            w = torch_npu.npu_format_cast(cpu.to(f'npu:{a.rank}'), 29)
            # Keep the original probe's down weights resident; no down execution.
            d = torch_npu.npu_format_cast(down.to(f'npu:{a.rank}'), 29)
            item = next(t for c in captures['cases'] if c['name']=='od-0677' for t in c['tensors']
                        if t['metadata']['layer']==layer and t['metadata']['rank']==a.rank)
            x = torch.load(item['path'], map_location='cpu', weights_only=True)['input']
            digest = hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
            assert digest==item['input_sha256'] and x.dtype==torch.bfloat16 and x.shape==(1595,5120)
            x = x[:967].contiguous().to(f'npu:{a.rank}')
            ref = F.linear(x, w); y = gate(x, w)
            assert y.shape==ref.shape==(967,17408) and y.dtype==ref.dtype==torch.bfloat16
            assert y.is_contiguous() and torch_npu.get_npu_format(y)==torch_npu.get_npu_format(ref)==2
            n = dict(layer=layer, input_sha256=digest, input_kind='explicit_prefix', preparation_s=time.perf_counter()-start)
            for label, aa, bb in [('all', y, ref), ('gate', y[:,:8704], ref[:,:8704]), ('up', y[:,8704:], ref[:,8704:])]:
                af, bf = aa.float(), bb.float(); assert bool(torch.isfinite(af).all() & torch.isfinite(bf).all())
                delta = af-bf; rmse = float(delta.square().mean().sqrt()/bf.square().mean().sqrt().clamp_min(1e-30))
                norm = float(delta.abs().max()/bf.abs().max().clamp_min(1e-30))
                n[label] = dict(relative_rmse=rmse, normalized_max=norm)
                assert rmse<=.01 and norm<=.02
            n['repeat_equal'] = bool(torch.equal(gate(x,w), y)); assert n['repeat_equal']
            report['numeric'].append(n); weights.append((w,d)); inputs.append(x); save()
        def call(arm):
            # Preserve 4a51f71's mid/out lifetimes and allocation inside each layer.
            for i in range(3):
                with scope(f'layer/{i}/{arm}'):
                    mid = gate(inputs[i],weights[i][0]) if arm=='v3' else F.linear(inputs[i],weights[i][0])
                    out = mid
            return out
        with torch.inference_mode():
            for arm in ('vendor','vendor_repeat','v3'):
                start=time.perf_counter()
                for _ in range(5): call(arm)
                torch.npu.synchronize()
                report['warmup'].append(dict(arm=arm, calls=15, wall_s=time.perf_counter()-start))
            # Initialize Event machinery outside measured samples, recorded protocol difference.
            b=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
            b.record();e.record();torch.npu.synchronize();b.elapsed_time(e)
            def samples(phase):
                arms = ['vendor','v3'] if phase=='profile' else ['vendor','vendor_repeat','v3']
                for pair in range(3):
                    for arm in (arms if pair%2==0 else list(reversed(arms))):
                        if phase.startswith('concurrent'): barrier(f'{phase}-{pair}-{arm}')
                        b=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True)
                        usage=resource.getrusage(resource.RUSAGE_SELF)
                        with scope(f'sample/{phase}/{pair}/{arm}'):
                            wall=time.perf_counter();b.record()
                            with scope('submit'): call(arm)
                            e.record()
                            with scope('sync'): torch.npu.synchronize()
                            elapsed=(time.perf_counter()-wall)*1000/3
                        after=resource.getrusage(resource.RUSAGE_SELF)
                        report['samples'].append(dict(phase=phase,pair=pair,arm=arm,device_ms=b.elapsed_time(e)/3,
                            wall_ms=elapsed, cpu_ms=(after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime)*1000/3,
                            involuntary_switches=after.ru_nivcsw-usage.ru_nivcsw,voluntary_switches=after.ru_nvcsw-usage.ru_nvcsw))
                save()
            barrier('numeric_ready');samples('concurrent_pre');barrier('concurrent_pre_done')
            if a.rank==1: wait('serial_done',0)
            samples('serial_pre')
            # Only this child's warmed symbols are wrapped; never change shared runtime files.
            runtime=_runtime.runtime(); original={}
            for symbol, label in [('aclnnClefMlpGateGemmGetWorkspaceSize','v3_get_workspace'),('aclnnClefMlpGateGemm','v3_launch')]:
                original[symbol]=runtime._symbols[symbol]
                def wrapped(*args, _fn=original[symbol], _label=label):
                    with scope(_label): return _fn(*args)
                runtime._symbols[symbol]=wrapped
            try:
                config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone, data_simplification=False,
                    record_op_args=False, export_type=[torch_npu.profiler.ExportType.Text])
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False,profile_memory=False,with_stack=False,with_modules=False,
                    experimental_config=config,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(a.output.parent/f'traces-r{a.rank}'),worker_name=f'gate-r{a.rank}',analyse_flag=False)):
                    profiled=True;samples('profile');profiled=False
            finally:
                profiled=False;runtime._symbols.update(original)
            samples('serial_post');signal('serial_done')
            barrier('both_serial_done');samples('concurrent_post')
        report['status']='complete';report['allocated_tensor_bytes']=int(torch.npu.memory_allocated(a.rank));save()
    except BaseException as exc:
        report.update(status='stopped',error=type(exc).__name__+': '+str(exc));save();raise

if __name__=='__main__': main()
