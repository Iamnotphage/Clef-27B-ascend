#!/usr/bin/env python3
"""Bounded frozen-real-input comparison; defaults to a CPU-only plan.

Event encloses host dispatch and device work. Wall includes completion wait;
neither is a pure task duration. All allocation/layout/copy in the call counts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def plan():
    return dict(reference='aa3e stock', paths=['stock', 'view', 'contiguous'],
                aa='stock_repeat', pairs=5, single='decoder0 incl norms/self-attn/FFN',
                budget_s=240, input='armed real head capture; checkpoint weights SHA checked')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--rank', type=int, choices=[0, 1], default=0)
    ap.add_argument('--phase', choices=['single', 'full', 'profile'], default='single')
    ap.add_argument('--captures', type=Path)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--weights', type=Path, default=Path('/workspace/yangchen/models/Clef-27B/joint_head.safetensors'))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--limit-s', type=float, default=45)
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(plan())); return
    assert args.output and not args.output.exists()
    os.environ['ACLNN_CACHE_LIMIT'] = '1000000'
    import torch
    import torch_npu
    from safetensors.torch import load_file
    from head_packed_kv import load_reference, Selection, cross, projections, probabilities, metadata
    os.environ['CLEF_JOINT_FUSED_ATTN'] = '1'
    os.environ['CLEF_JOINT_SINGLE_FIELD'] = '1'
    torch.set_num_threads(1)
    torch.npu.set_device(args.rank)
    device = f'npu:{args.rank}'
    start = time.monotonic()
    result = dict(plan=plan(), rank=args.rank, phase=args.phase, status='started',
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  measurements=[], numerical=[], warmup=[], capture_source_ranks=[])
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    def deadline():
        if time.monotonic()-start > args.limit_s:
            raise TimeoutError('bounded probe limit')
    def cmp(a, b):
        af, bf = a.float(), b.float()
        delta = af-bf
        finite = bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
        return dict(bit_equal=bool(torch.equal(a, b)), finite=finite,
                    max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                    rel_rmse=float(delta.square().mean().sqrt() / af.square().mean().sqrt().clamp_min(1e-12)))
    def check(stage, a, b, exact=False):
        c = cmp(a, b); result['numerical'].append(dict(stage=stage, **c));save()
        assert c['finite'] and c['rel_rmse'] <= .02, (stage, c)
        if exact: assert c['bit_equal'], (stage, c)
        return c
    def restore(t, meta=None):
        if meta is None: return t.to(device)
        assert meta['dtype'] == str(t.dtype)
        shape, stride, offset = meta['shape'], meta['stride'], meta['storage_offset']
        size = offset + 1 + sum((n-1)*s for n, s in zip(shape, stride))
        flat = torch.empty(size, device=device, dtype=t.dtype)
        out = flat.as_strided(shape, stride, offset)
        out.copy_(t)
        fmt = meta['format']
        if torch_npu.get_npu_format(out) != fmt:
            out = torch_npu.npu_format_cast(out, fmt)
        return out
    def payload(label):
        # Prefer each rank's real capture; if pooler only runs one rank, retain
        # actual provenance when replaying that frozen real input on both NPUs.
        paths = sorted(args.captures.glob(label+'.rank*.pt'))
        assert paths, label
        path = next((p for p in paths if f'.rank{args.rank}.' in p.name), paths[0])
        data = torch.load(path, map_location='cpu', weights_only=True)
        result['capture_source_ranks'].append(dict(label=label, source_rank=data['source_rank'],
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        hidden = restore(data['hidden'], data['metadata']['hidden'])
        lexical = restore(data['lexical'], data['metadata']['lexical'])
        schema = reference.SchemaIndices(**{k: v.to(device) for k,v in data['schema'].items()})
        return data, hidden, lexical, schema
    def timed(fn, calls):
        deadline()
        torch.npu.synchronize()
        a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin = time.perf_counter(); a.record()
        for _ in range(calls): out = fn()
        b.record(); b.synchronize()
        wall = (time.perf_counter()-begin)*1000/calls
        return dict(event_ms=a.elapsed_time(b)/calls, wall_ms=wall)
    def paired(scope, fn, selected, calls):
        for mode in ('stock', 'view', 'contiguous'):
            selection.set(mode, selected)
            result['warmup'].append(dict(scope=scope, arm=mode, calls=8, **timed(fn, 8)))
        for pair in range(5):
            order = ['stock', 'view', 'stock_repeat', 'contiguous']
            if pair % 2: order.reverse()
            for arm in order:
                selection.set('stock' if arm == 'stock_repeat' else arm, selected)
                row = dict(scope=scope, pair=pair, arm=arm, calls=calls, **timed(fn, calls))
                result['measurements'].append(row)
            save()
        selection.set()
    try:
        reference = load_reference(args.source)
        selection = Selection(reference)
        assert hashlib.sha256(args.weights.read_bytes()).hexdigest() == 'a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953'
        weights = load_file(str(args.weights))
        head = reference.JointSchemaHead(5120, 1024, 2, 4, 16, 4096).to(dtype=torch.bfloat16)
        head.load_state_dict(weights, strict=True)
        head.to(device).eval()
        with torch.inference_mode():
            data, hidden, lexical, schema = payload('od-1534')
            assert len(weights) == len(data['weights']) == 122
            for name, value in weights.items():
                assert str(value.dtype) == data['weights'][name]['dtype']
                assert hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() == data['weights'][name]['sha256'], name
                param = head.state_dict()[name]
                fmt = data['weights'][name]['format']
                if torch_npu.get_npu_format(param) != fmt:
                    # state_dict tensors alias storage, but set parameter data
                    # explicitly for a format cast which allocates new storage.
                    parent, attr = name.rsplit('.', 1) if '.' in name else ('', name)
                    owner = head.get_submodule(parent) if parent else head
                    getattr(owner, attr).data = torch_npu.npu_format_cast(param, fmt)
            result['weight_formats'] = sorted({x['format'] for x in data['weights'].values()})
            result['setup_s'] = time.monotonic()-start
            stock_logits = head(hidden, lexical, schema)
            check('stock_vs_actual_aa3e_logits', data['logits'].to(device), stock_logits, exact=True)
            check('stock_repeat_logits', stock_logits, head(hidden, lexical, schema), exact=True)
            modules = dict(head.named_modules())
            captured = next(x for x in data['attentions'] if x['name']=='layers.0.multihead_attn')
            attn = modules[captured['name']]
            query, memory = (restore(captured[k], captured['metadata'][k]) for k in ('query', 'memory'))
            fields = data['fields0'].to(device)
            one = SimpleNamespace(layers=[head.layers[0]])
            decoder = lambda: reference.JointSchemaHead._single_field_decoder(one, fields, memory)
            check('decoder_stock_vs_actual', data['fields1'].to(device), decoder(), exact=True)
            qkv = projections(attn, query, memory, 'stock')
            original_attention = cross(attn, query, memory, 'stock')
            check('cross_stock_vs_actual', captured['output'].to(device), original_attention, exact=True)
            result['layout'] = {}
            for mode in ('view', 'contiguous'):
                kv = projections(attn, query, memory, mode)
                result['layout'][mode] = [metadata(x) for x in kv]
                for label, a, b in zip(('q', 'k', 'v'), qkv, kv): check(mode+'_'+label, a, b)
                check(mode+'_cross', original_attention, cross(attn, query, memory, mode))
                selection.set(mode, {id(attn)})
                check(mode+'_decoder', data['fields1'].to(device), decoder())
                cand = head(hidden, lexical, schema)
                check(mode+'_one_place_logits', stock_logits, cand)
                c = check(mode+'_one_place_prob', probabilities(stock_logits,schema), probabilities(cand,schema))
                assert c['max_abs'] <= .02
                check(mode+'_one_place_repeat', cand, head(hidden, lexical, schema), exact=True)
            selection.set()
            if args.phase == 'single':
                paired('decoder0', decoder, {id(attn)}, 16)
            elif args.phase == 'full':
                for mode in ('view', 'contiguous'):
                    selection.set(mode)
                    cand = head(hidden, lexical, schema)
                    check(mode+'_six_logits', stock_logits, cand)
                    c = check(mode+'_six_prob', probabilities(stock_logits,schema), probabilities(cand,schema))
                    assert c['max_abs'] <= .02
                    check(mode+'_six_repeat', cand, head(hidden,lexical,schema), exact=True)
                paired('full_head_L967_O4', lambda: probabilities(head(hidden,lexical,schema),schema), None, 8)
                for label in ('od-0105', 'od-0677', 'options2', 'options7'):
                    d, h, l, s = payload(label)
                    selection.set();a=head(h,l,s)
                    check(label+'_stock_actual', d['logits'].to(device), a, exact=True)
                    for mode in ('view','contiguous'):
                        selection.set(mode);b=head(h,l,s)
                        check(label+'_'+mode+'_logits',a,b)
                        c=check(label+'_'+mode+'_prob',probabilities(a,s),probabilities(b,s))
                        assert c['max_abs']<=.02
                    paired(label,lambda: probabilities(head(h,l,s),s),None,4)
            else:
                config = torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                    data_simplification=False, record_op_args=False,
                    export_type=[torch_npu.profiler.ExportType.Text])
                def samples(phase, pairs):
                    for pair in range(pairs):
                        arms = ['stock', 'view', 'stock_repeat', 'contiguous']
                        if pair % 2: arms.reverse()
                        for arm in arms:
                            selection.set('stock' if arm == 'stock_repeat' else arm, {id(attn)})
                            with torch.profiler.record_function(f'headkv/{phase}/{pair}/{arm}'):
                                row = timed(decoder, 1)
                            result['measurements'].append(dict(scope='decoder0_trace_'+phase,
                                pair=pair, arm=arm, calls=1, **row))
                    save()
                for mode in ('stock', 'view', 'contiguous'):
                    selection.set(mode, {id(attn)})
                    result['warmup'].append(dict(scope='decoder0_trace', arm=mode,
                                                  calls=8, **timed(decoder, 8)))
                samples('pre', 2)
                traces = args.output.parent/f'traces-r{args.rank}'
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False, profile_memory=False, with_stack=False,
                    with_modules=False, experimental_config=config,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(traces), worker_name=f'headkv-r{args.rank}', analyse_flag=False)):
                    samples('profile', 3)
                samples('post', 2)
                result.update(trace_directory=str(traces),
                    trace_scope='representative rank serial; 1 call/sample vs paired batch16',
                    profiler='CPU+NPU Level1/AiCoreNone, offline export after restore')
            result['status']='completed'
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        result['elapsed_s']=time.monotonic()-start;save()
        print(json.dumps({k:result[k] for k in ('rank','phase','status','elapsed_s')}),flush=True)


if __name__ == '__main__':
    main()
