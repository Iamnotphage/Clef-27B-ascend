#!/usr/bin/env python3
"""Fresh-process dispatcher smoke; only release(1) in an empty state registry.

31 runs this against an isolated AOT build. This script neither builds nor
initializes a device. Loading errors are terminal; no torch_npu import or
device-initialization fallback is attempted. Native library constructors are
outside this Python-level check and require the owner's CPU-only load gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


NAMESPACE = 'clef_mmar_probe::'
SCHEMA = 'clef_mmar_probe::release(int id) -> int'
ABSENT_ID = 1  # Positive, int32-safe; no run is called in this fresh process.
TIMEOUT_SECONDS = 30


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--adapter-sha', required=True)
    parser.add_argument('--expect', choices=('zero', 'not-implemented'), required=True)
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.adapter.is_absolute():
        parser.error('--adapter must be an absolute path')
    if (len(args.adapter_sha) != 64 or
            any(c not in '0123456789abcdef' for c in args.adapter_sha)):
        parser.error('--adapter-sha must be a lowercase SHA-256')
    return args


def pinned_adapter(args):
    path = args.adapter.resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != args.adapter_sha:
        raise RuntimeError(f'adapter SHA mismatch: {digest}')
    return path


def require_no_python_npu_modules():
    modules = sorted(name for name in sys.modules
                     if name == 'torch_npu' or name.startswith('torch_npu.'))
    if modules:
        raise RuntimeError(f'torch_npu Python modules already imported: {modules}')


def child(args):
    record = {
        'schema': 'grouped-release-dispatcher-smoke/1',
        'scope': 'dispatcher_and_invalid_id_only',
        'pid': os.getpid(), 'ppid': os.getppid(),
        'adapter': str(args.adapter), 'adapter_sha256': args.adapter_sha,
        'expected': args.expect, 'id': ABSENT_ID, 'passed': False,
        'valid_state_cleanup_verified': False,
        'sdk_resource_release_verified': False, 'task13_fix_verified': False,
        'stage': 'pre_import',
    }
    try:
        if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
            raise RuntimeError('TORCH_DEVICE_BACKEND_AUTOLOAD must be 0 before startup')
        adapter = pinned_adapter(args)
        require_no_python_npu_modules()
        import torch
        require_no_python_npu_modules()
        record['torch_version'] = str(torch.__version__)
        before = sorted(name for name in torch._C._dispatch_get_all_op_names()
                        if name.startswith(NAMESPACE))
        record['namespace_ops_before_load'] = before
        if before:
            raise RuntimeError('namespace must be empty before loading one adapter')
        record['stage'] = 'load_library'
        torch.ops.load_library(str(adapter))
        require_no_python_npu_modules()
        pinned_adapter(args)
        schema = str(torch._C._dispatch_find_schema_or_throw(
            NAMESPACE + 'release', '').schema())
        record['release_schema'] = schema
        if schema != SCHEMA:
            raise RuntimeError(f'release schema mismatch: {schema}')
        record['stage'] = 'release_invalid_id'
        try:
            value = torch.ops.clef_mmar_probe.release(ABSENT_ID)
        except NotImplementedError as exc:
            record['observed'] = 'not-implemented'
            record['exception_type'] = type(exc).__name__
            record['exception'] = str(exc)
            record['passed'] = (args.expect == 'not-implemented' and
                                NAMESPACE + 'release' in str(exc))
        else:
            record['observed'] = 'returned'
            record['return_value'] = value
            record['passed'] = (args.expect == 'zero' and
                                type(value) is int and value == 0)
        record['stage'] = 'complete'
    except Exception:
        record['error'] = traceback.format_exc()
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record['passed'] else 1


def main():
    args = arguments()
    if args._child:
        return child(args)
    # The parent uses only stdlib; each invocation starts one isolated child.
    # Old control and candidate are invoked separately and never share a registry.
    env = dict(os.environ)
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    argv = [sys.executable, '-I', str(Path(__file__).resolve()), '--_child',
            '--adapter', str(args.adapter), '--adapter-sha', args.adapter_sha,
            '--expect', args.expect]
    result = {'schema': 'grouped-release-dispatcher-parent/1',
              'parent_pid': os.getpid(), 'argv': argv,
              'timeout_seconds': TIMEOUT_SECONDS, 'passed': False}
    try:
        proc = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        result.update(returncode=proc.returncode, stdout=proc.stdout,
                      stderr=proc.stderr, passed=proc.returncode == 0)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value
        result.update(timed_out=True, stdout=decoded(exc.stdout),
                      stderr=decoded(exc.stderr))
    except Exception:
        result['error'] = traceback.format_exc()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
