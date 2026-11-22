#!/usr/bin/env python3
"""Private GDN KKT screen. Default is CPU plan; explicit execution is bounded.

No service, SSH, installation or deployment actions. Raw captures stay in tmp.
The SDK-generated L0 diagnostic wrapper includes its own layout/allocation cost;
production wheel output and wrapper latency are independent acceptance checks.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time


def eligible(shape, dtype, cu, ci, *, state=False, final=False):
    if dtype != 'torch.bfloat16' or len(shape) != 4:
        return False
    b, t, h, d = shape
    expected = tuple(x for j in range((t + 127)//128) for x in (0, j))
    return (b, h, d) == (1, 8, 128) and 0 < t <= 1595 and not state and not final and (
        (cu is None and ci is None) or (tuple(cu) == (0, t) and tuple(ci) == expected))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--phase', choices=('numeric', 'timing'), default='numeric')
    p.add_argument('--vendor', type=Path)
    p.add_argument('--captures', type=Path)
    p.add_argument('--synthetic', action='store_true', help='screen only; cannot justify adoption')
    p.add_argument('--output', type=Path)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--limit-s', type=float, default=90)
    a = p.parse_args()
    report = dict(schema='clef-gdn-kkt-screen/1', status='plan', rank=a.rank,
                  phase=a.phase, lengths=[967, 127, 128, 129, 307, 1595] if a.phase == 'numeric' else [967], pairs=5,
                  arms=['wheel', 'stock', 'stock_repeat', 'reuse'],
                  device_budget_s=a.limit_s, no_device_actions=not a.execute,
                  task_metric='prebuilt L0 launch event interval; task-only trace not implied',
                  wrapper_metric='allocations, beta BF16->FP32/BHT, metadata, dispatch included')
    if not a.execute:
        print(json.dumps(report)); return
    if not a.vendor or not a.output or a.output.exists() or not (a.captures or a.synthetic):
        p.error('execution requires vendor, NEW output, captures or explicit synthetic screen')
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    os.environ['ACLNN_CACHE_LIMIT'] = '1000000'
    import torch
    import torch_npu
    from fla_npu.ops.ascendc import _runtime, chunk_gated_delta_rule_fwd
    torch.set_num_threads(1)
    torch.npu.set_device(a.rank)
    lib = C.CDLL(str(a.vendor/'op_api/lib/libcust_opapi.so'), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    rt = _runtime.runtime(); rt._libraries.insert(0, lib)
    dev = f'npu:{a.rank}'
    start = time.monotonic()
    def check_time():
        if time.monotonic()-start > a.limit_s:
            raise TimeoutError('rank computation limit reached')
    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        report['elapsed_s'] = time.monotonic()-start
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def comparison(x, y):
        x, y = x.cpu(), y.cpu()
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        return dict(finite=finite, equal=bool(torch.equal(x, y)),
                    max_abs=float((x.float()-y.float()).abs().max()) if finite else None)
    def wheel(data):
        return chunk_gated_delta_rule_fwd(*data, initial_state=None, output_final_state=False,
            chunk_size=128, scale=128**-.5, layout='BSND', use_exp2=False,
            use_qk_l2norm_in_kernel=False, allow_neg_eigval=False, disable_recompute=False,
            cu_seqlens=(0, data[0].shape[1]),
            chunk_indices=tuple(x for j in range((data[0].shape[1]+127)//128) for x in (0, j)),
            state_v_first=False)[0]
    def prepare(data, arm, observe=False):
        q, k, v, g, beta = data; t = q.shape[1]
        cu = (0, t); ci = tuple(x for j in range((t+127)//128) for x in (0, j))
        assert eligible(tuple(q.shape), str(q.dtype), cu, ci)
        assert k.shape == q.shape and v.shape == (1, t, 24, 128)
        assert g.dtype == torch.float32 and g.shape == beta.shape == (1, t, 24)
        # Match the public DAV_2201 L2 conversions, including full output allocations.
        bt = beta.float().transpose(1, 2).contiguous()
        storage = torch.empty((1, 24, t, 128), device=dev, dtype=q.dtype)
        out = torch.empty_like(v)
        final = torch.empty((1,), device=dev, dtype=torch.float32)
        gc = torch.empty((1, t, 24), device=dev, dtype=torch.float32)
        A = torch.empty_like(storage)
        ctx = _runtime._CallContext(rt, q.device)
        tensors = [q, k, v, bt, storage, g, None, None]
        args = [ctx.tensor(x) for x in tensors]
        # SDK ValueDepend inputs use the generated aclIntArray overload, exactly
        # as the production L2 call; avoid additional per-call H2D tensors.
        args += [ctx.int_array(cu), ctx.int_array(ci)]
        args += [False, 128, 128**-.5, False, 1, 1, 1]
        args += [ctx.tensor(x) for x in (out, final, gc, A)]
        name = 'aclnnClefGdnKkt'+('Reuse' if arm == 'reuse' else 'Stock')
        get = rt.symbol(name+'GetWorkspaceSize'); launch = rt.symbol(name)
        get.restype = C.c_int
        get.argtypes = ([C.c_void_p]*10 + [C.c_bool, C.c_int64, C.c_double,
            C.c_bool, C.c_int64, C.c_int64, C.c_int64] + [C.c_void_p]*4 +
            [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])
        ws_size, executor = C.c_uint64(), C.c_void_p()
        try:
            result = get(*args, C.byref(ws_size), C.byref(executor))
            if result: raise RuntimeError(rt._aclnn_error(name+'GetWorkspaceSize', result))
            # End canary is outside the size handed to the SDK, same launch/workspace contract.
            backing = torch.full((ws_size.value+512,), 0xA5, device=dev, dtype=torch.uint8) if observe else (
                torch.empty((ws_size.value+512,), device=dev, dtype=torch.uint8))
            launch.restype = C.c_int
            launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
            def run():
                r = launch(C.c_void_p(backing.data_ptr()), ws_size, executor,
                           C.c_void_p(_runtime.current_stream_ptr(q.device)))
                if r: raise RuntimeError(rt._aclnn_error(name, r))
            # ctx and all args stay alive through completion, outside timed launch.
            return dict(run=run, out=out, A=A, gc=gc, ws=backing, size=ws_size.value,
                        ctx=ctx, tensors=tensors, storage=storage, final=final)
        except BaseException:
            ctx.destroy(); raise
    def invoke(data, arm):
        if arm == 'wheel': return wheel(data)
        obj = prepare(data, arm)
        try: obj['run'](); return obj['out']
        finally: obj['ctx'].destroy()
    def stage(obj, meta, t):
        # One-op generated ACLNN API: the launch workspace begins with SDK system workspace.
        # Declared offsets must also reproduce FP32 KKT; no offset scan is permitted.
        base = int(meta['sysBytes'])
        ws = obj['ws']; nt = (t+127)//128
        raw = ws[base+int(meta['scoreOffset']):base+int(meta['scoreOffset'])+int(meta['scoreBytes'])]
        raw = raw.cpu().view(torch.float32).reshape(-1, nt, 128, 128)
        off = base+int(meta['solveInputOffset'])
        gated = ws[off:off+24*t*128*4].cpu().view(torch.float32).reshape(24, t, 128)
        return raw, gated
    report.update(status='running', input_kind='synthetic_screen' if a.synthetic else 'production_capture',
                  numeric=[], warmup=[], samples=[], launch_samples=[])
    save()
    try:
        source = None
        if a.captures:
            path = a.captures/f'rank{a.rank}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            report['capture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = [payload[x] for x in ('q', 'k', 'v', 'g', 'beta')]
            assert source[0].shape[1] >= 1595
        main_data = None
        for t in report['lengths']:
            check_time()
            if source:
                data = [x[:, :t].contiguous().to(dev) for x in source]
            else:
                gen = torch.Generator().manual_seed(20261122 + a.rank*10000+t)
                q = torch.nn.functional.normalize(torch.randn((1, t, 8, 128), generator=gen), dim=-1).bfloat16()
                k = torch.nn.functional.normalize(torch.randn((1, t, 8, 128), generator=gen), dim=-1).bfloat16()
                v = torch.randn((1, t, 24, 128), generator=gen).bfloat16()
                g = -.01 - .09*torch.rand((1, t, 24), generator=gen)
                beta = torch.rand((1, t, 24), generator=gen).bfloat16()
                data = [x.to(dev) for x in (q, k, v, g, beta)]
            metadata = a.output.with_suffix('.tiling.log')
            os.environ['CLEF_GDN_KKT_META'] = str(metadata)
            stock, reuse = prepare(data, 'stock', True), prepare(data, 'reuse', True)
            stock['run'](); reuse['run'](); ref = wheel(data); torch.npu.synchronize()
            del os.environ['CLEF_GDN_KKT_META']
            meta = {}
            for line in metadata.read_text().splitlines():
                name, *fields = line.split()
                values = dict(x.split('=') for x in fields)
                if int(values['T']) == t: meta['reuse' if 'Reuse' in name else 'stock'] = values
            assert set(meta) == {'stock', 'reuse'}
            assert all(int(x['key']) == 3 for x in meta.values()) and int(meta['reuse']['reuse']) == 1
            raw_s, gate_s = stage(stock, meta['stock'], t)
            raw_r, gate_r = stage(reuse, meta['reuse'], t)
            row = dict(t=t, meta=meta, final=comparison(reuse['out'], stock['out']),
                       stock_wheel=comparison(stock['out'], ref), raw=[], gate=[], repeats=[],
                       canary=bool((stock['ws'][stock['size']:] == 0xA5).all() and
                                   (reuse['ws'][reuse['size']:] == 0xA5).all()),
                       input_kind='explicit_prefix' if source else 'synthetic')
            for hk in range(8):
                for chunk in range((t+127)//128):
                    lo=chunk*128; n=min(128,t-lo)
                    x = data[1][0,lo:lo+n,hk].cpu().float()
                    expected = x @ x.T
                    independent = comparison(raw_r[hk,chunk,:n,:n], expected)
                    independent.update(kind='independent_cpu_fp32',
                        pass_check=independent['finite'] and torch.allclose(
                            raw_r[hk,chunk,:n,:n], expected, rtol=2e-5, atol=2e-6))
                    row['raw'].append(independent)
                    for j in range(3):
                        row['raw'].append(comparison(raw_r[hk,chunk,:n,:n], raw_s[hk*3+j,chunk,:n,:n]))
            for hv in range(24):
                for chunk in range((t+127)//128):
                    lo=chunk*128; n=min(128,t-lo)
                    row['gate'].append(comparison(gate_s[hv,lo:lo+n,:n],gate_r[hv,lo:lo+n,:n]))
            for _ in range(2):
                row['repeats'].append(comparison(invoke(data,'reuse'), reuse['out']))
            report['numeric'].append(row); save()
            stock['ctx'].destroy(); reuse['ctx'].destroy()
            checks=[row['final'], row['stock_wheel'], *row['raw'], *row['gate'], *row['repeats']]
            if not row['canary'] or not all(x.get('pass_check', x['equal']) and x['finite'] for x in checks):
                raise ValueError('stage/production/finite/stability/canary check failed')
            if t == 967: main_data=data
        if a.phase == 'numeric':
            report['status']='numeric_passed'; save(); return
        # Strict screen passed: warmups are separate, all five AA/AB pairs retained.
        data = main_data
        for arm in report['arms']:
            wall=time.perf_counter()
            for _ in range(5): invoke(data, arm)
            torch.npu.synchronize()
            report['warmup'].append(dict(arm=arm, calls=5, wall_s=time.perf_counter()-wall))
        for pair in range(5):
            for arm in (report['arms'] if pair%2==0 else list(reversed(report['arms']))):
                check_time()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                wall=time.perf_counter(); begin.record(); out=invoke(data,arm); end.record(); end.synchronize()
                report['samples'].append(dict(pair=pair, arm=arm, device_ms=begin.elapsed_time(end),
                                              wall_ms=(time.perf_counter()-wall)*1000))
                if arm != 'wheel':
                    obj=prepare(data,arm); torch.npu.synchronize()
                    begin.record(); obj['run'](); end.record(); end.synchronize()
                    report['launch_samples'].append(dict(pair=pair,arm=arm,device_ms=begin.elapsed_time(end)))
                    obj['ctx'].destroy()
            save()
        report['status']='screen_passed'; save()
    except BaseException as exc:
        report.update(status='stopped', error=type(exc).__name__+': '+str(exc)); save(); raise


if __name__ == '__main__':
    main()
