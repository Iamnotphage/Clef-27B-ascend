#!/usr/bin/env python3
"""Owner27-only GroupedMMAR + native AddRMSNorm cell; no service lifecycle.

--preflight performs CPU-only identity checks. --timing and --profile run only
after BOTH ranks pass the unchanged gate against the service NZ reference.
GEMM is part of every timed arm; input/layout preparation and barriers are outside.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
import socket
from pathlib import Path
import time
import traceback
from contract import (OWNER, WEIGHT_KEY, check_owner, cpu_inputs, load_gate,
                      record_tensor, sha256, tensor_bytes, verify, N1_CELL, N1_GROUP, n1_inputs)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', choices=[OWNER], required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/workspace/yangchen/models/Clef-27B'))
    p.add_argument('--rows', type=int, choices=[967], default=967)
    p.add_argument('--seed', type=int, choices=[20261118], default=20261118)
    p.add_argument('--n1-cell', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--timing', action='store_true')
    p.add_argument('--profile', action='store_true')
    return p.parse_args()


def main():
    a = args()
    identity = json.loads(a.identity.read_text())
    verify(identity, a.adapter, a.gate)
    if a.preflight:
        print(json.dumps({'preflight': 'pass', 'device_execution': False,
                          'identity_sha256': sha256(a.identity)}))
        return 0
    rank, local = check_owner(a.owner)
    if os.getpgrp() != int(os.environ.get('CLEF_GROUPED29_PROBE_PGID', '-1')):
        raise ValueError('Rank escaped supervised probe process group')
    out = a.output / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    meta = {'schema': 'grouped29-probe/1', 'owner': a.owner, 'rank': rank,
            'rows': a.rows, 'seed': a.seed, 'eps': 1e-6, 'stage': 'prechecked',
            'identity_sha256': sha256(a.identity), 'identity': identity,
            'inputs': {}, 'files': {}, 'performance_valid': False,
            'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'clock': vars(time.get_clock_info('perf_counter')),
            'pid': os.getpid(), 'pgid': os.getpgrp(),
            'n1_cell': str(a.n1_cell),
            'parameters': {'splitItem': 0, 'groupList': None, 'reduceOp': 'sum',
                           'commTurn': 0, 'streamMode': 1, 'groups': 1, 'bias': 'FP32 zero [5120]'},
            'boundary': 'GroupedMatMulAllReduce followed by native AddRMSNorm',
            'reference': 'service-format BF16 NZ F.linear -> BF16 dist.all_reduce -> native AddRMSNorm',
            'native_grouped_rstd': None, 'rstd_source': 'actual output of subsequent native AddRMSNorm'}

    def flush():
        (out / 'result.json').write_text(json.dumps(meta, indent=2) + '\n')
    flush()
    try:
        import torch
        import torch_npu
        import torch.distributed as dist
        from safetensors import safe_open
        evaluate = load_gate(a.gate)
        torch.set_num_threads(1)
        if torch.__version__ != '2.10.0+cpu' or torch_npu.__version__ != '2.10.0.post7':
            raise RuntimeError('Fixed torch/torch_npu versions required')
        meta['versions'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__}
        shard = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map'][WEIGHT_KEY]
        with safe_open(a.model / shard, framework='pt', device='cpu') as f:
            full_w = f.get_tensor(WEIGHT_KEY)
        if tuple(full_w.shape) != (5120, 17408) or full_w.dtype != torch.bfloat16:
            raise RuntimeError('Actual model weight shape/dtype differs from frozen TP2 contract')
        layout_metrics = {}
        source = cpu_inputs(torch, full_w, a.rows, a.seed, rank, layout_metrics)
        meta['inputs'] = {name: record_tensor(torch, t) for name, t in source.items()}
        meta['weight'] = {'key': WEIGHT_KEY, 'shard': shard, 'slice': [rank * 8704, (rank + 1) * 8704]}
        for name in ('x', 'r', 'gamma'):
            (out / (name + '.bin')).write_bytes(tensor_bytes(torch, source[name]))
        if meta['inputs'] != n1_inputs(rank, a.n1_cell):
            raise RuntimeError('N1 input identity mismatch')
        meta['n1_input_match'] = True
        torch.npu.set_device(local)
        torch.npu.config.allow_internal_format = True
        dist.init_process_group('hccl', timeout=timedelta(seconds=90))
        group = dist.group.WORLD._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        if group != N1_GROUP:
            raise RuntimeError('HCCL commgroup differs from fixed N1 group')
        meta['group'] = group
        both = [None, None]
        dist.all_gather_object(both, meta['inputs'])
        for name in ('x', 'r', 'gamma'):
            if both[0][name] != both[1][name]:
                raise RuntimeError('Ranks differ in shared input: ' + name)
        meta['both_rank_inputs'] = both
        torch.npu.synchronize()
        layout_begin = time.perf_counter_ns()
        w = source['w'].npu()
        torch.npu.synchronize()
        layout_metrics['candidate_weight_upload_wall_ms'] = (time.perf_counter_ns() - layout_begin) / 1e6
        tensors = {name: source[name].npu() for name in ('x', 'r', 'gamma')}
        tensors['w'] = w
        x, residual, gamma = (tensors[n] for n in ('x', 'r', 'gamma'))
        bias = torch.zeros(5120, dtype=torch.float32, device=x.device)
        torch.npu.synchronize()
        layout_begin = time.perf_counter_ns()
        # NZ is the actual service reference. Candidate accepts only ND.
        service_w = torch_npu.npu_format_cast(w.t().contiguous(), 29)
        torch.npu.synchronize()
        meta['layout'] = {**layout_metrics, 'candidate_weight_format': torch_npu.get_npu_format(w),
                          'service_weight_format': torch_npu.get_npu_format(service_w),
                          'service_nz_setup_wall_ms': (time.perf_counter_ns() - layout_begin) / 1e6,
                          'candidate_static_weight_bytes': w.numel() * w.element_size(),
                          'scope': 'separate CPU ND layout, ND weight upload, and device transpose/NZ reference setup; excluded from timing'}
        if meta['layout']['candidate_weight_format'] != 2 or meta['layout']['service_weight_format'] != 29:
            raise RuntimeError('Unexpected ND/NZ storage formats')
        if not torch.equal(service_w.cpu(), source['w'].t()):
            raise RuntimeError('Static format conversion changed weight values')
        torch.ops.load_library(str(a.adapter.resolve()))
        meta['provider'] = torch.ops.clef_grouped29.provider()
        meta['stage'] = 'loaded_and_inputs_fixed'; flush()

        def invoke(arm, poison=False):
            if arm == 'candidate':
                reduced, workspace = torch.ops.clef_grouped29.run(x, w, bias, group, poison)
                meta['workspace_bytes'] = workspace
            else:
                reduced = (torch.nn.functional.linear(x, service_w) if arm == 'baseline'
                           else torch.matmul(x, w))
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            y, rstd, res = torch_npu.npu_add_rms_norm(reduced, residual, gamma, 1e-6)
            return {'y': y, 'residual': res, 'rstd': rstd, 'reduced': reduced}

        def save(arm, values, prefix=''):
            for name, tensor in values.items():
                key = prefix + name + '.' + arm
                raw = tensor_bytes(torch, tensor)
                (out / (key + '.bin')).write_bytes(raw)
                meta['files'][key] = record_tensor(torch, tensor)

        def gate(candidate, reference):
            return evaluate({k: candidate[k].cpu() for k in ('y', 'residual', 'rstd')},
                            {k: reference[k].cpu() for k in ('y', 'residual', 'rstd')})

        def both_pass(result):
            values = [None, None]
            dist.all_gather_object(values, bool(result))
            return all(values)

        first = {}
        for arm in ('baseline', 'nd_native', 'candidate'):
            dist.barrier(); torch.npu.synchronize()
            begin = time.perf_counter_ns()
            first[arm] = invoke(arm, poison=arm == 'candidate')
            torch.npu.synchronize()
            meta.setdefault('first_call_wall_ms', {})[arm] = (time.perf_counter_ns() - begin) / 1e6
            save(arm, first[arm])
        meta['fixed_gate'] = gate(first['candidate'], first['baseline'])
        meta['nd_diagnostic_gate'] = gate(first['nd_native'], first['baseline'])
        cr, rr = first['candidate']['reduced'].float(), first['baseline']['reduced'].float()
        meta['reduced_diagnostic'] = {'max_abs': float((cr - rr).abs().max().cpu()),
                                     'bit_equal': bool(torch.equal(cr, rr)),
                                     'extra_acceptance_gate': False}
        meta['reduced_preconditions'] = {
            'shape_ok': tuple(first['candidate']['reduced'].shape) == (a.rows, 5120),
            'dtype_ok': first['candidate']['reduced'].dtype == torch.bfloat16,
            'finite': bool(torch.isfinite(cr).all().cpu()) and bool(torch.isfinite(rr).all().cpu())}
        meta['inputs_unchanged'] = all(record_tensor(torch, t) == meta['inputs'][name]
                                       for name, t in tensors.items())
        meta['output_preconditions'] = all(
            tuple(values[n].shape) == ((a.rows, 1) if n == 'rstd' else (a.rows, 5120)) and
            values[n].dtype == (torch.float32 if n == 'rstd' else torch.bfloat16)
            for values in first.values() for n in ('y', 'residual', 'rstd', 'reduced'))
        meta['pass'] = both_pass(meta['output_preconditions'] and meta['fixed_gate']['pass'] and meta['inputs_unchanged']
                                 and all(meta['reduced_preconditions'].values()))
        meta['stage'] = 'gate_complete'; flush()
        if not meta['pass']:
            dist.destroy_process_group()
            return 1

        if a.timing:
            # Input-ready complete chain, all allocations/launches included.
            for _ in range(2):
                for arm in ('baseline', 'candidate'):
                    invoke(arm)
            torch.npu.synchronize()
            meta['timing'] = {'rounds': [], 'reps': 10, 'warmup_per_arm': 2,
                              'includes': 'GEMM, AllReduce, norm, allocator and host dispatch',
                              'excludes': 'weight layout initialization, input upload, barriers, event setup'}
            with (out / 'samples.jsonl').open('x') as f:
                for round_id, order in enumerate((('baseline', 'candidate'), ('candidate', 'baseline')) * 2):
                    last = {}
                    for rep in range(10):
                        for arm in order:
                            dist.barrier(); torch.npu.synchronize()
                            start = torch.npu.Event(enable_timing=True)
                            end = torch.npu.Event(enable_timing=True)
                            start.record(); end.record(); torch.npu.synchronize()
                            begin = time.perf_counter_ns()
                            start.record(); last[arm] = invoke(arm); end.record()
                            torch.npu.synchronize()
                            stop = time.perf_counter_ns()
                            f.write(json.dumps({'round': round_id, 'rep': rep, 'arm': arm, 'rank': rank,
                                                'start_ns': begin, 'end_ns': stop, 'wall_ms': (stop - begin) / 1e6,
                                                'event_ms': start.elapsed_time(end)}) + '\n'); f.flush()
                    result = gate(last['candidate'], last['baseline'])
                    stable = gate(last['baseline'], first['baseline'])
                    ok = both_pass(result['pass'] and stable['pass'])
                    meta['timing']['rounds'].append({'round': round_id, 'gate': result, 'baseline_stable': stable, 'pass': ok})
                    save('candidate', last['candidate'], f'round{round_id}.')
                    save('baseline', last['baseline'], f'round{round_id}.')
                    flush()
                    if not ok:
                        raise RuntimeError('Timing output gate failed; performance invalid')
            meta['samples_sha256'] = sha256(out / 'samples.jsonl')
            meta['performance_valid'] = True

        if a.profile:
            profile_dir = out / 'profile'
            profile_dir.mkdir()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, with_stack=False, profile_memory=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))) as prof:
                profiled = {}
                for arm in ('baseline', 'candidate'):
                    dist.barrier(); torch.npu.synchronize()
                    with torch.profiler.record_function('grouped29.' + arm):
                        profiled[arm] = invoke(arm)
                        torch.npu.synchronize()
                    prof.step()
            result = gate(profiled['candidate'], profiled['baseline'])
            meta['profile_gate'] = result
            for arm, values in profiled.items():
                save(arm, values, 'profile.')
            if not both_pass(result['pass']):
                meta['performance_valid'] = False
                raise RuntimeError('Profile output gate failed')
            meta['profile'] = str(profile_dir)
            meta['profile_files'] = {str(p.relative_to(profile_dir)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                                     for p in sorted(profile_dir.rglob('*')) if p.is_file()}
            if not meta['profile_files']:
                raise RuntimeError('Profiler produced no raw files')
        meta['inputs_unchanged_after'] = all(record_tensor(torch, t) == meta['inputs'][name]
                                             for name, t in tensors.items())
        if not meta['inputs_unchanged_after']:
            raise RuntimeError('Input mutation after timing/profile')
        meta['stage'] = 'complete'; flush()
        dist.destroy_process_group()
        return 0
    except BaseException:
        meta['performance_valid'] = False
        meta['error'] = traceback.format_exc(); flush()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
