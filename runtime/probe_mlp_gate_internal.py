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


class NativeRange:
    """One checked host SDK path shared by capture and CPU entry tests."""
    def __init__(self, path, stream):
        self.lib=ctypes.CDLL(str(path),mode=os.RTLD_LOCAL|os.RTLD_NOW)
        self.lib.clef40_range_start.argtypes=[ctypes.c_char_p,ctypes.c_void_p]
        self.lib.clef40_range_start.restype=ctypes.c_uint64
        self.lib.clef40_range_end.argtypes=[ctypes.c_uint64]
        self.lib.clef40_range_end.restype=None
        self.lib.clef40_range_ready.argtypes=[]
        self.lib.clef40_range_ready.restype=ctypes.c_int
        self.stream=stream;self.native_id=None;self.ready=None;self.ended=False
    def start(self):
        self.native_id=int(self.lib.clef40_range_start(b'CLEF40_TARGET',self.stream))
        self.ready=bool(self.lib.clef40_range_ready())
        if self.ready is not True:
            raise RuntimeError('native SDK MSTX callbacks not initialized')
        return self.native_id
    def end(self):
        if self.ready is True and not self.ended:
            self.lib.clef40_range_end(self.native_id);self.ended=True
    def evidence(self):
        return dict(mstx_native_id=self.native_id,mstx_ready=self.ready,mstx_end_called=self.ended)


def capture_failure(record):
    # Missing registration evidence is UNKNOWN; numeric ID0 is accepted.
    if record.get('mstx_ready') is False:return 'SDK callbacks not initialized'
    if record.get('status')=='stopped':return record.get('error','application stopped')
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--marker-self-test',action='store_true',help='CPU-only: exercise the same NativeRange path with injected test callbacks')
    p.add_argument('--mode',choices=['control','capture'],default='control')
    p.add_argument('--arm',choices=['vendor','v3'],default='vendor')
    p.add_argument('--vendor',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--mstx-lib',type=Path,help='host-only bridge compiled against installed CANN MSTX SDK')
    a=p.parse_args()
    if a.marker_self_test:
        if not a.mstx_lib:p.error('marker test requires --mstx-lib')
        marker=NativeRange(a.mstx_lib,None)
        try:
            marker.start();marker.end();e=marker.evidence()
            assert capture_failure(dict(e,status='complete')) is None
            print(json.dumps(dict(e,status='complete')));return
        except RuntimeError as exc:
            e=marker.evidence();assert capture_failure(dict(e,status='stopped',error=str(exc)))
            print(json.dumps(dict(e,status='stopped',error=str(exc))));raise SystemExit(2)
    plan=dict(schema='clef-gate-internal/1',rank=0,m=967,layers=[31,0,63],capture_layer=31,
        precision='original BF16/NZ -> canonical ND',pairs=3,warmup_passes_each_arm=5,
        mode=a.mode,arm=a.arm,metrics=['BasicInfo first','PipeUtilization only after both BasicInfo hits'],
        replay='application, collector warm-up=0; explicit application warmup; native SDK MSTX range covers complete ring through existing sync, launch-count=1 selects layer31',
        stop='numeric failure, observed unbound callbacks or BasicInfo miss; owner wrapper supplies remaining cumulative budget')
    if not a.execute: print(json.dumps(plan,indent=2));return
    if not a.vendor or not a.output_dir:p.error('execute requires vendor and private output directory')
    if a.mode=='capture' and not a.mstx_lib:p.error('capture requires --mstx-lib')
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
                prefix=x[:967].contiguous();prefix_sha=hashlib.sha256(memoryview(prefix.view(torch.uint8).numpy())).hexdigest()
                x=prefix.to('npu:0');ref=F.linear(x,w);y=gate(x,w)
                assert y.shape==ref.shape==(967,17408) and y.dtype==ref.dtype==torch.bfloat16
                assert y.is_contiguous() and torch_npu.get_npu_format(y)==torch_npu.get_npu_format(ref)==2
                n=dict(layer=layer,input_sha256=digest,prefix_sha256=prefix_sha,input_kind='explicit_prefix',halves={})
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
                marker=NativeRange(a.mstx_lib,_runtime.current_stream_ptr(inputs[0].device))
                report['target']=dict(layer=31,launch_index_in_range=0,range_gate_launches=3,
                    input_shape=list(inputs[0].shape),input_dtype=str(inputs[0].dtype),
                    weight_shape=list(weights[0][0].shape),weight_dtype=str(weights[0][0].dtype),
                    weight_format=int(torch_npu.get_npu_format(weights[0][0])),
                    input_sha256=report['numeric'][0]['input_sha256'],prefix_sha256=report['numeric'][0]['prefix_sha256'],
                    weight_rank_hashes={k:v['sha256_rank_tensor_bytes'] for k,v in report['preparation'][0]['weight_identity']['weights'].items()})
                report['marker_backend']='CANN SDK mstxRangeStartA/End; uint64_t native id'
                try:
                    marker.start()
                finally:
                    report.update(marker.evidence());save()
                try:
                    y=one(a.arm,0)
                    for i in (1,2):mid=one(a.arm,i);out=mid
                    # Existing ring-end sync keeps vendor worker-thread launches inside the range.
                    torch.npu.synchronize()
                finally:
                    marker.end();report.update(marker.evidence())
                report['target'].update(output_shape=list(y.shape),output_dtype=str(y.dtype),
                    output_format=int(torch_npu.get_npu_format(y)),output_stride=list(y.stride()))
        report['status']='complete';save()
    except BaseException as exc:
        report.update(status='stopped',error=type(exc).__name__+': '+str(exc));save();raise

if __name__=='__main__':main()
