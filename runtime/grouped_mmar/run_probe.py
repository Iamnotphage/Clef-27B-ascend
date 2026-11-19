#!/usr/bin/env python3
"""Dual-rank Grouped MMAR probe. Plan-only by default; device only with
``--execute`` and explicit temporary paths.

First matrix (frozen): A8 plain all_reduce, A0 old bridge at M=128, H1 group
string keepalive created before prepare, H2 workspace keepalive until after
sync.  A12 is a separate official-sample binary (fixed source supplied by 33).
Every arm records thread/context/stream/group/workspace/descriptor/executor
addresses, loaded library maps, rc and whether sync returned.  Success requires
API return, both ranks observing sync return, and finite output; rc alone is
not success.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
KEEP_GROUP = 1
KEEP_WORKSPACE = 2
KEEP_DESCRIPTORS = 4
KEEP_ALL = 8
DIRECT_LAUNCH = 16
REFETCH_STREAM = 32
MEMSET_WORKSPACE = 64
FIRST_MATRIX = ('A8', 'A12', 'A0', 'H1', 'H2')
ARMS = {
    'A8': {'flags': 0, 'native': False, 'rows': 128,
           'title': 'plain matmul + dist.all_reduce + sync (comm health)'},
    'A0': {'flags': 0, 'native': True, 'rows': 128,
           'title': 'old bridge style: local group string + immediate descriptor close'},
    'H1': {'flags': KEEP_GROUP, 'native': True, 'rows': 128,
           'title': 'same std::string object created before prepare, c_str passed, kept past sync'},
    'H2': {'flags': KEEP_WORKSPACE, 'native': True, 'rows': 128,
           'title': 'workspace tensor kept until sync returns'},
    'H3': {'flags': KEEP_DESCRIPTORS, 'native': True, 'rows': 128,
           'title': 'descriptor lists kept until sync returns (stage 2)'},
    'H11': {'flags': KEEP_ALL, 'native': True, 'rows': 128,
            'title': 'full captured state kept (stage-2 diagnostic; executor is never destroyed)'},
    'H4': {'flags': DIRECT_LAUNCH, 'native': True, 'rows': 128,
           'title': 'calling-thread launch + aclrtSynchronizeStream (stage 2)'},
    'H5': {'flags': REFETCH_STREAM, 'native': True, 'rows': 128,
           'title': 're-fetch current stream inside handler (stage 2)'},
    'H4K': {'flags': KEEP_ALL | DIRECT_LAUNCH, 'native': True, 'rows': 128,
            'title': 'DIRECT_LAUNCH with KEEP_ALL (calling-thread body plus real handler stream sync; retention matches H11)'},
    'H5K': {'flags': KEEP_ALL | REFETCH_STREAM, 'native': True, 'rows': 128,
            'title': 'REFETCH_STREAM with KEEP_ALL (handler-thread current stream; retention matches H11; single stream-source variable)'},
    'H6': {'flags': MEMSET_WORKSPACE, 'native': True, 'rows': 128,
           'title': 'aclrtMemsetAsync workspace before launch (stage 2)'},
    'H7': {'flags': 0, 'native': True, 'rows': None,
           'title': 'M sweep 128/256/1024/967 (stage 2)'},
    'A13': {'flags': 0, 'native': True, 'rows': 967,
            'title': 'MatMulAllReduce (non-grouped MC2) control with torch multi-process comm'},
    'A14': {'flags': KEEP_ALL, 'native': True, 'rows': 128,
            'title': 'MatmulReduceScatter V1 AICPU MC2 control (A3-registered; full state kept until sync)'},
    'A15': {'flags': KEEP_ALL, 'native': True, 'rows': 967,
            'title': 'MRS V1 + all_gather reconstruction (bias-free, tail pad/crop, KEEP_ALL as the A14 baseline)'},
}
A12 = {
    'title': 'official A3 sample positive control (separate binary, fixed source from 33)',
    'native': 'external',
    'source': 'a12/test_aclnn_grouped_mat_mul_all_reduce.cpp',
    'source_sha256': '6c1a90aba84501a901c413eaa5391889d60beff48b766bed9865ab5bf26eeb9f',
    'source_commit': 'f5703e415b9be16a2af7ba85564c37414592fc03',
    'build': 'derived CMake glue; official CMakeLists not supplied -> not an as-is official build',
    'compile': 'bash a12/compile_a12.sh <absolute-build-dir>',
    'run': 'bash a12/run_a12.sh <absolute-build-dir>/build/bin/main',
    'argv': ['2', '1', '0,1'],
    'a12_expected': ['API rc0', 'both rank threads return from the timed sync',
                     'no hang inside 240 s (liveness only; no numeric reference)'],
    'note': ('FP16 original bytes; BF16 changes the bias dtype and is a derived arm, '
             'not the frozen A12 source'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard_path(value, name, *, exists=False):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f'{name} must be an absolute path: {value}')
    resolved = str(path.resolve())
    if exists and not path.exists():
        raise ValueError(f'{name} missing: {resolved}')
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=sorted(ARMS) + ['A12'], default='A0')
    p.add_argument('--matrix', action='store_true', help='print the first matrix plan')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--rows', type=int, default=None)
    p.add_argument('--k', type=int, default=8704)
    p.add_argument('--n', type=int, default=5120)
    p.add_argument('--seed', type=int, default=20261119)
    p.add_argument('--temp-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--weight-bin', type=Path)
    p.add_argument('--weight-sha')
    p.add_argument('--arm-timeout', type=float, default=240.0)
    p.add_argument('--comm-timeout', type=float, default=90.0)
    p.add_argument('--norm', action='store_true', help='stage 3: add native AddRMSNorm')
    p.add_argument('--watchdog-selftest', type=float, default=None,
                   help='CPU-only watchdog check; exits 124 after the delay')
    return p.parse_args(argv)


def arm_plan(a):
    if a.arm == 'A12':
        return dict(arm='A12', **A12, external=True)
    spec = ARMS[a.arm]
    rows = a.rows if a.rows is not None else (spec['rows'] or 128)
    return dict(arm=a.arm, title=spec['title'], flags=spec['flags'],
                flag_names=[name for name, bit in (
                    ('keep_group', KEEP_GROUP), ('keep_workspace', KEEP_WORKSPACE),
                    ('keep_descriptors', KEEP_DESCRIPTORS), ('keep_all', KEEP_ALL),
                    ('direct_launch', DIRECT_LAUNCH), ('refetch_stream', REFETCH_STREAM),
                    ('memset_workspace', MEMSET_WORKSPACE)) if spec['flags'] & bit],
                native=spec['native'], rows=rows, k=a.k, n=a.n, seed=a.seed,
                norm=a.norm,
                bias=('nullptr' if a.arm in ('A14', 'A15')
                      else ('bf16_zero' if a.arm == 'A13' else 'fp32_zero')),
                external=False)


def build_plan(a):
    for value, name in ((a.temp_root, 'temp-root'), (a.out, 'out')):
        guard_path(value, name)
    if a.adapter is not None:
        guard_path(a.adapter, 'adapter')
    if a.weight_bin is not None:
        guard_path(a.weight_bin, 'weight-bin', exists=True)
        if not a.weight_sha:
            raise ValueError('--weight-bin requires --weight-sha')
    plan = arm_plan(a)
    plan.update(
        schema='grouped-mmar-probe-plan/1',
        first_matrix=list(FIRST_MATRIX),
        first_matrix_note=('A12 is a separate single-process two-thread official sample; '
                           'A8/A0/H1/H2 run one fresh process per arm under a two-rank launcher'),
        temp_root=str(a.temp_root.resolve()), out=str(a.out.resolve()),
        adapter=str(a.adapter.resolve()) if a.adapter else None,
        adapter_sha256=sha256(a.adapter) if a.adapter and a.adapter.is_file() else None,
        weight={'path': str(a.weight_bin.resolve()) if a.weight_bin else None,
                'sha256': a.weight_sha},
        timeout={'arm_seconds': a.arm_timeout, 'comm_seconds': a.comm_timeout},
        expected=['API returns', 'both ranks observe sync return', 'output finite',
                  'trace has prepare/handler thread/context/stream/group pointers'],
        failure_branches=['prepare rc!=0', 'launch rc!=0', 'sync does not return within arm timeout',
                          'rank group names differ', 'output non-finite', 'release rc!=0'],
        recovery='stop probe processes; restore aa3e/s1 with the official start.sh; '
                 'normal success requires branch == aa3e-noop; f810->fused is rescue-only and never success',
        no_service_lifecycle=True, no_f810_revert=True,
    )
    return plan


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(temp, path)


def start_watchdog(out_dir, seconds, rank):
    def fire():
        record = {'schema': 'grouped-mmar-probe-timeout/1', 'rank': rank,
                  'timeout_seconds': seconds, 'monotonic_ns': time.monotonic_ns()}
        try:
            data = (json.dumps(record) + '\n').encode()
            fd = os.open(Path(out_dir) / 'timeout.json',
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
            os.write(fd, data)
            os.close(fd)
        finally:
            os._exit(124)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def sync_and_release(sync, release, run_id, keep):
    """Release is allowed only after sync returned; no keep means no release."""
    sync()
    if keep and run_id:
        return release(run_id)
    return 0


def call_release(release, run_id, keep):
    """Release is cleanup, not the measurement: record a failure instead of aborting."""
    if not (keep and run_id):
        return 0, None
    try:
        return release(run_id), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'[:512]


def library_evidence(adapter):
    wanted = ('libhccl', 'libopapi', 'libnnopbase', 'libascendcl', 'libtorch_npu')
    names = set(wanted)
    if adapter is not None:
        names.add(Path(adapter).name)
    found = {}
    maps = Path('/proc/self/maps')
    if maps.is_file():
        for line in maps.read_text(errors='replace').splitlines():
            parts = line.split()
            if not parts:
                continue
            path = parts[-1]
            if not path.startswith('/'):
                continue
            if any(name in Path(path).name for name in names) and Path(path).is_file():
                found[path] = True
    result = []
    for path in sorted(found):
        entry = {'path': path, 'sha256': None, 'soname': None}
        try:
            entry['sha256'] = sha256(path)
        except OSError:
            pass
        try:
            text = subprocess.run(['readelf', '-d', path], capture_output=True, text=True,
                                  timeout=10).stdout
            match = re.search(r'\(SONAME\).*\[(.+?)\]', text)
            entry['soname'] = match.group(1) if match else None
        except (OSError, subprocess.SubprocessError):
            pass
        result.append(entry)
    return result


def prepare_out(path):
    """Both ranks share the parent output dir; rank subdirs are per-rank."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def padded_rows(rows, world):
    """Tail-pad M to a multiple of world for MatmulReduceScatter."""
    if rows < 1 or world < 2:
        raise ValueError('padded_rows requires rows >= 1 and world >= 2')
    rem = rows % world
    return rows if rem == 0 else rows + (world - rem)


def row_block(rank, rows, world):
    """Row block [start, end) that rank owns after the reduce-scatter."""
    if rank < 0 or rank >= world:
        raise ValueError('row_block rank out of range')
    per = padded_rows(rows, world) // world
    return rank * per, (rank + 1) * per


def arm_output(arm, y, full=None, rows=None):
    """A14 keeps the MRS output; A15 crops the all-gathered full matrix."""
    if arm == 'A15':
        return full[:rows]
    return y


def all_reduce_sum(dist, tensor):
    """torch.distributed.all_reduce returns None; keep the reduced tensor."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def a15_flow(rows, k, n, world, rank, x, w, *, ones, empty, run_mrs, sync,
             all_gather, final_sync, matmul, all_reduce, flags, trace=None):
    """One real A15 pass: tail pad, MRS, sync, all-gather, crop, full-row reference."""
    m_pad = padded_rows(rows, world)
    if m_pad != rows:
        x_mrs = ones((m_pad, k))
        x_mrs[:rows] = x
    else:
        x_mrs = x
    start, end = row_block(rank, rows, world)
    if trace is not None:
        trace.emit('pad_ready', rows=rows, padded_rows=m_pad)
    y, workspace_bytes, run_id = run_mrs(x_mrs, w, world, flags)
    sync()
    full = empty((m_pad, n), y)
    all_gather(full, y)
    reference = all_reduce(matmul(x, w))
    final_sync()
    output = full[:rows]
    info = {'pad': {'rows': rows, 'padded_rows': m_pad,
                    'pad_value': 1.0 if m_pad != rows else None},
            'row_block': {'rank': rank, 'start': start, 'end': end,
                          'padded_end': m_pad},
            'workspace_bytes': workspace_bytes, 'run_id': run_id}
    return output, reference, info


def execute(a, plan):
    rank = int(os.environ.get('RANK', '-1'))
    world = int(os.environ.get('WORLD_SIZE', '-1'))
    local = int(os.environ.get('LOCAL_RANK', '-1'))
    if world != 2 or rank not in (0, 1) or local != rank:
        raise ValueError('requires two ranks with RANK=LOCAL_RANK in 0,1')
    out = prepare_out(plan['out'])
    rank_dir = out / f'rank{rank}'
    rank_dir.mkdir()
    from probe_trace import ProbeTrace
    trace = ProbeTrace(rank_dir / 'trace.jsonl', rank)
    timer = start_watchdog(rank_dir, a.arm_timeout, rank)
    meta = {'schema': 'grouped-mmar-probe-result/1', 'arm': a.arm, 'plan': plan,
            'rank': rank, 'world': world, 'local_rank': local,
            'hostname': os.uname().nodename, 'pid': os.getpid(), 'pgid': os.getpgrp(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'clock': vars(time.get_clock_info('monotonic')),
            'api_returned': False, 'sync_returned': False, 'finite': False,
            'timeout': False, 'pass': False, 'libraries': [], 'trace': str(trace.path)}
    flush = lambda: write_json_atomic(rank_dir / 'result.json', meta)
    flush()
    dist = None
    try:
        import torch
        import torch_npu
        import torch.distributed as dist
        trace.emit('process_start', torch=torch.__version__, torch_npu=torch_npu.__version__)
        torch.set_num_threads(1)
        torch.npu.set_device(local)
        torch.npu.config.allow_internal_format = True
        meta['versions'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__}
        dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=a.comm_timeout))
        group = dist.group.WORLD._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        names = [None, None]
        dist.all_gather_object(names, group)
        if not group or names[0] != names[1] or not names[0]:
            raise RuntimeError(f'HCCL group names are not symmetric: {names}')
        meta['group'] = group
        meta['group_names'] = names
        trace.emit('dist_ready', group=group)
        rows = plan['rows']
        gen = torch.Generator(device='cpu').manual_seed(a.seed)
        x = torch.randn(rows, a.k, generator=gen).to(torch.bfloat16)
        residual = torch.randn(rows, a.n, generator=gen).to(torch.bfloat16)
        gamma = torch.rand(a.n, generator=gen).to(torch.bfloat16)
        if a.weight_bin is not None:
            raw = a.weight_bin.read_bytes()
            if hashlib.sha256(raw).hexdigest() != a.weight_sha:
                raise RuntimeError('weight-bin SHA mismatch')
            weight = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(
                torch.bfloat16).reshape(a.k, a.n).clone()
        else:
            gen_r = torch.Generator(device='cpu').manual_seed(a.seed + rank)
            weight = torch.randn(a.k, a.n, generator=gen_r).to(torch.bfloat16)
        bias = None
        if a.arm not in ('A14', 'A15'):
            bias_dtype = torch.bfloat16 if a.arm == 'A13' else torch.float32
            bias = torch.zeros(a.n, dtype=bias_dtype)
        meta['inputs'] = {
            'x': {'shape': list(x.shape), 'dtype': str(x.dtype),
                  'sha256': hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()},
            'weight': {'shape': list(weight.shape), 'dtype': str(weight.dtype),
                       'sha256': hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()},
        }
        if bias is not None:
            meta['inputs']['bias'] = {'shape': list(bias.shape), 'dtype': str(bias.dtype),
                                      'value': 'zero'}
        if a.adapter is not None:
            if not a.adapter.is_file():
                raise RuntimeError(f'adapter missing: {a.adapter}')
            meta['adapter'] = {'path': str(a.adapter), 'sha256': sha256(a.adapter)}
            torch.ops.load_library(str(a.adapter))
            meta['provider'] = torch.ops.clef_mmar_probe.provider()
        x_n, w_n = x.npu(), weight.npu()
        bias_n = bias.npu() if bias is not None else None
        residual_n, gamma_n = residual.npu(), gamma.npu()
        torch.npu.synchronize()
        meta['libraries'] = library_evidence(a.adapter)
        trace.emit('inputs_ready')
        begin = time.monotonic_ns()
        if a.arm == 'A8':
            reduced = x_n.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            trace.emit('api_return', arm='A8')
            meta['api_returned'] = True
            sync_begin = time.monotonic_ns()
            torch.npu.synchronize()
            trace.emit('sync_return', sync_ms=(time.monotonic_ns() - sync_begin) / 1e6)
            meta['sync_returned'] = True
            output = reduced
            reference = None
            run_id = 0
        else:
            trace.emit('arm_enter', flags=plan['flags'])
            if a.arm == 'A15':
                def a15_sync():
                    sync_begin = time.monotonic_ns()
                    trace.emit('sync_enter')
                    torch.npu.synchronize()
                    trace.emit('sync_return',
                               sync_ms=(time.monotonic_ns() - sync_begin) / 1e6)
                    meta['sync_returned'] = True

                def a15_run_mrs(xm, wm, world_, flags_):
                    y, ws, rid = torch.ops.clef_mmar_probe.run_mrs(
                        xm, wm, group, world_, flags_)
                    trace.emit('api_return', workspace_bytes=ws, run_id=rid)
                    meta['api_returned'] = True
                    meta['workspace_bytes'] = ws
                    meta['run_id'] = rid
                    return y, ws, rid

                def a15_gather(out, inp):
                    begin = time.monotonic_ns()
                    trace.emit('gather_enter')
                    dist.all_gather_into_tensor(out, inp)
                    torch.npu.synchronize()
                    meta['gather_ms'] = (time.monotonic_ns() - begin) / 1e6
                    trace.emit('gather_return', gather_ms=meta['gather_ms'])

                output, reference, info = a15_flow(
                    rows, a.k, a.n, world, rank, x_n, w_n,
                    ones=lambda shape: torch.ones(shape, dtype=x_n.dtype,
                                                  device=x_n.device),
                    empty=lambda shape, like: torch.empty(
                        shape, dtype=like.dtype, device=like.device),
                    run_mrs=a15_run_mrs, sync=a15_sync, all_gather=a15_gather,
                    final_sync=torch.npu.synchronize,
                    matmul=torch.matmul,
                    all_reduce=lambda t: all_reduce_sum(dist, t),
                    flags=plan['flags'], trace=trace)
                meta.update(info)
                run_id = info['run_id']
            else:
                if a.arm == 'A14':
                    y, workspace_bytes, run_id = torch.ops.clef_mmar_probe.run_mrs(
                        x_n, w_n, group, world, plan['flags'])
                else:
                    op = (torch.ops.clef_mmar_probe.run_mmar if a.arm == 'A13'
                          else torch.ops.clef_mmar_probe.run)
                    y, workspace_bytes, run_id = op(x_n, w_n, bias_n, group,
                                                    plan['flags'])
                trace.emit('api_return', workspace_bytes=workspace_bytes, run_id=run_id)
                meta['api_returned'] = True
                meta['workspace_bytes'] = workspace_bytes
                meta['run_id'] = run_id
                sync_begin = time.monotonic_ns()
                trace.emit('sync_enter')
                torch.npu.synchronize()
                trace.emit('sync_return', sync_ms=(time.monotonic_ns() - sync_begin) / 1e6)
                meta['sync_returned'] = True
                reduced = torch.matmul(x_n, w_n)
                if a.arm == 'A14':
                    reference = torch.empty((rows // world, a.n), dtype=reduced.dtype,
                                            device=reduced.device)
                    dist.reduce_scatter_tensor(reference, reduced, op=dist.ReduceOp.SUM)
                else:
                    reference = reduced
                    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
                torch.npu.synchronize()
                output = arm_output(a.arm, y, None, rows)
        if a.norm:
            from fixed_gate import evaluate
            cand = dict(zip(('y', 'residual', 'rstd'),
                            torch_npu.npu_add_rms_norm(output, residual_n, gamma_n, 1e-6)))
            ref = dict(zip(('y', 'residual', 'rstd'),
                           torch_npu.npu_add_rms_norm(reference, residual_n, gamma_n, 1e-6)))
            meta['gate'] = evaluate({k: v.cpu() for k, v in cand.items()},
                                    {k: v.cpu() for k, v in ref.items()})
            finite = bool(torch.isfinite(cand['y'].float()).all())
        else:
            finite = bool(torch.isfinite(output.float()).all())
            if reference is not None:
                diff = (output.float() - reference.float()).abs()
                item = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                        'bit_equal': bool(torch.equal(output, reference))}
                meta['reduced'] = item
                if a.arm == 'A15':
                    meta['equivalence'] = {'reference': 'matmul+all_reduce full rows',
                                           'shape': [rows, a.n], **item}
                    meta['diagnostic_gate'] = {
                        'name': 'mrs_allgather_equivalence',
                        'max_abs_le': 0.03125, 'mean_abs_le': 1e-4,
                        'within_bounds': bool(item['max_abs'] <= 0.03125 and
                                              item['mean_abs'] <= 1e-4),
                        'provenance': ('MC2 RS probe 0.03125; Dense SP probe '
                                       '0.015625 / 2e-5; not the fixed_gate'),
                        'status': 'diagnostic_only_not_an_official_gate'}
        meta['finite'] = finite
        meta['wall_ms'] = (time.monotonic_ns() - begin) / 1e6
        trace.emit('result', finite=finite)
        flush()
        release_rc, release_error = call_release(
            torch.ops.clef_mmar_probe.release, run_id,
            plan['flags'] & (KEEP_GROUP | KEEP_WORKSPACE | KEEP_DESCRIPTORS | KEEP_ALL))
        meta['release_rc'] = release_rc
        meta['harness_release_ok'] = release_error is None and release_rc == 0
        if release_error is not None:
            meta['release_error'] = release_error
            trace.emit('release_error', error=release_error)
        else:
            trace.emit('release_return', rc=release_rc)
        flush()
        both = [None, None]
        dist.all_gather_object(both, bool(finite))
        meta['both_finite'] = all(both)
        # The numeric gate is api/sync/finite/reference; a release failure is a harness
        # record (harness_release_ok=False), never a reason to discard completed numbers.
        numeric_pass = bool(meta['api_returned'] and meta['sync_returned'] and finite and
                            all(both) and (not a.norm or meta.get('gate', {}).get('pass')))
        meta['numeric_pass'] = numeric_pass
        # pass/exit follow the numeric gate so a known cleanup defect cannot mask valid
        # numbers; harness_pass additionally requires clean cleanup and is reported apart.
        meta['harness_pass'] = bool(numeric_pass and meta['harness_release_ok'])
        meta['pass'] = numeric_pass
        trace.emit('done', passed=meta['pass'],
                   harness_release_ok=meta['harness_release_ok'])
        timer.cancel()
        trace.close()
        flush()
        return 0 if numeric_pass else 1
    except BaseException as exc:
        timer.cancel()
        meta['error'] = traceback.format_exc()
        meta['timeout'] = isinstance(exc, TimeoutError)
        try:
            trace.emit('python_exception', error=f'{type(exc).__name__}: {exc}'[:512])
            trace.close()
        except BaseException:
            pass
        flush()
        return 1
    finally:
        # Both ranks always attempt the existing teardown, on success and failure paths.
        # No new collective/barrier; a cleanup failure is a harness record, and the
        # already-flushed numeric result stays intact. The 240 s window remains the backstop.
        try:
            if dist is not None and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            meta['harness_cleanup_error'] = f'{type(exc).__name__}: {exc}'[:512]
            try:
                flush()
            except Exception:
                pass


def watchdog_selftest(seconds, temp_root, out):
    root = guard_path(temp_root, 'temp-root', exists=True)
    target = guard_path(out, 'out')
    target.mkdir(parents=True, exist_ok=False)
    start_watchdog(target, seconds, 0)
    time.sleep(seconds + 5)
    (root / 'unreachable.txt').write_text('watchdog did not exit\n')
    return 0


def main(argv=None):
    a = parse_args(argv)
    if a.watchdog_selftest is not None:
        return watchdog_selftest(a.watchdog_selftest, a.temp_root, a.out)
    if a.matrix:
        print(json.dumps({'schema': 'grouped-mmar-probe-matrix/1',
                          'first_matrix': list(FIRST_MATRIX),
                          'arms': {name: {**ARMS.get(name, A12)} for name in FIRST_MATRIX},
                          'note': 'run one arm per fresh two-rank process; A12 is a separate binary'},
                         ensure_ascii=False, indent=2))
        return 0
    plan = build_plan(a)
    if not a.execute or a.arm == 'A12':
        plan['status'] = 'plan_not_executed' if a.arm != 'A12' else 'external_positive_control'
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    return execute(a, plan)


if __name__ == '__main__':
    raise SystemExit(main())
