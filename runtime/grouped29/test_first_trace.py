"""CPU host-logger tests; no Torch, ACL, extension loading or device execution."""
import ast
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from first_trace import FD_ENV, FirstCallTrace

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class PythonFirstTraceTests(unittest.TestCase):
    def test_disabled_logger_has_no_file_or_environment_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {FD_ENV: 'existing'}):
            path = Path(tmp) / 'disabled.jsonl'
            trace = FirstCallTrace(path, 0, enabled=False)
            trace.emit('must_not_appear', rc=161002)
            trace.close()
            self.assertFalse(path.exists())
            self.assertEqual(os.environ[FD_ENV], 'existing')

    def test_rank_files_append_independently_and_close_restores_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {FD_ENV: 'prior'}):
            for rank in (0, 1):
                path = Path(tmp) / f'rank{rank}.jsonl'
                trace = FirstCallTrace(path, rank)
                self.assertEqual(os.environ[FD_ENV], str(trace.fd))
                trace.emit('enter')
                # Force the shared offset backwards; O_APPEND still appends.
                os.lseek(trace.fd, 0, os.SEEK_SET)
                trace.emit('return', rc=-9)
                observed = rows(path)  # Visible before close, without stdout.
                self.assertEqual([r['stage'] for r in observed], ['enter', 'return'])
                self.assertEqual([r['rank'] for r in observed], [rank, rank])
                self.assertEqual(observed[1]['rc'], -9)
                self.assertTrue(all(r['source'] == 'python' for r in observed))
                self.assertLessEqual(observed[0]['monotonic_ns'], observed[1]['monotonic_ns'])
                trace.close()
                self.assertEqual(os.environ[FD_ENV], 'prior')
                trace.emit('closed_does_not_log')
                self.assertEqual(len(rows(path)), 2)

    def test_close_removes_previously_absent_fd_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop(FD_ENV, None)
            trace = FirstCallTrace(Path(tmp) / 'trace.jsonl', 0)
            trace.close()
            self.assertNotIn(FD_ENV, os.environ)
            trace.close()  # Idempotent, with no double-close of unrelated fds.

    def test_sigkill_preserves_already_emitted_line(self):
        code = '''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from first_trace import FirstCallTrace
trace = FirstCallTrace(sys.argv[2], 1)
trace.emit('before_block', rc=161002)
os.write(1, b'R')
time.sleep(30)
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'killed.jsonl'
            proc = subprocess.Popen([sys.executable, '-c', code, str(ROOT), str(path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertTrue(select.select([proc.stdout], [], [], 5)[0], 'child did not emit readiness')
                self.assertEqual(proc.stdout.read(1), b'R')
                before = path.read_bytes()
                self.assertEqual(rows(path)[0]['stage'], 'before_block')
                proc.kill()
                proc.communicate(timeout=5)
                self.assertEqual(proc.returncode, -signal.SIGKILL)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(rows(path)[0]['rc'], 161002)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=5)


class CppFirstTraceTests(unittest.TestCase):
    """Run the exact production header as a standalone POSIX CPU program."""
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler unavailable for standalone host-logger tests')
        cls.temp = tempfile.TemporaryDirectory(prefix='grouped29-host-trace-')
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        source = root / 'trace_test.cpp'
        source.write_text(r'''
#include "first_trace.h"
#include <cstring>
int main(int argc, char** argv) {
    if (argc != 2) return 99;
    try {
        bool enabled = std::strcmp(argv[1], "disabled") != 0;
        grouped29_trace::FirstCallTrace trace(enabled);
        if (std::strcmp(argv[1], "retained") == 0) {
            trace.mark("retained_before_close");
            ::close(std::atoi(std::getenv("CLEF_GROUPED29_TRACE_FD")));
            if (::write(STDOUT_FILENO, "R", 1) != 1) return 90;
            char release = 0;
            if (::read(STDIN_FILENO, &release, 1) != 1) return 91;
            trace.rc("retained_after_close", -17);
            return 0;
        }
        // These values exercise logger serialization only, without any ACL call.
        trace.mark("workspace_enter");
        trace.workspace(161002, 4096, false);
        trace.rc("launch_return", -7);
        trace.rc("launch_cleanup_return", 23);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 77;
    }
}
''')
        cls.binary = root / 'trace_test'
        result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
                                 '-I', str(ROOT), str(source), '-o', str(cls.binary)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError('Standalone host trace build failed:\n' + result.stdout + result.stderr)

    def test_actual_return_codes_are_serialized_without_alteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cpp.jsonl'
            trace = FirstCallTrace(path, 0)
            try:
                env = {**os.environ, 'RANK': '0', FD_ENV: str(trace.fd)}
                result = subprocess.run([str(self.binary), 'codes'], env=env,
                                        pass_fds=(trace.fd,), capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                got = rows(path)
                self.assertEqual([r['stage'] for r in got],
                                 ['workspace_enter', 'workspace_return', 'launch_return', 'launch_cleanup_return'])
                self.assertNotIn('rc', got[0])
                self.assertEqual([r['rc'] for r in got[1:]], [161002, -7, 23])
                self.assertEqual(got[1]['workspace_bytes'], 4096)
                self.assertIs(got[1]['executor_nonnull'], False)
                self.assertTrue(all(r['rank'] == 0 and r['source'] == 'cpp' for r in got))
            finally:
                trace.close()

    def test_disabled_cpp_logger_does_not_read_invalid_environment_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'disabled-cpp.jsonl'
            trace = FirstCallTrace(path, 1)
            try:
                env = {**os.environ, 'RANK': 'invalid', FD_ENV: 'invalid-fd'}
                result = subprocess.run([str(self.binary), 'disabled'], env=env,
                                        pass_fds=(trace.fd,), capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(path.read_bytes(), b'')
            finally:
                trace.close()

    def test_cpp_logger_rejects_nonappend_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not-append.jsonl'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            try:
                env = {**os.environ, 'RANK': '1', FD_ENV: str(fd)}
                result = subprocess.run([str(self.binary), 'codes'], env=env,
                                        pass_fds=(fd,), capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 77)
                self.assertIn(b'append regular file', result.stderr)
                self.assertEqual(path.read_bytes(), b'')
            finally:
                os.close(fd)

    def test_cpp_retained_descriptor_survives_original_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'retained.jsonl'
            trace = FirstCallTrace(path, 1)
            env = {**os.environ, 'RANK': '1', FD_ENV: str(trace.fd)}
            proc = subprocess.Popen([str(self.binary), 'retained'], env=env,
                                    pass_fds=(trace.fd,), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertTrue(select.select([proc.stdout], [], [], 5)[0], 'child did not retain fd')
                self.assertEqual(proc.stdout.read(1), b'R')
                trace.close()  # Child has closed its original; only its duplicate remains.
                _, stderr = proc.communicate(b'G', timeout=5)
                self.assertEqual(proc.returncode, 0, stderr)
                got = rows(path)
                self.assertEqual([r['stage'] for r in got],
                                 ['retained_before_close', 'retained_after_close'])
                self.assertEqual(got[1]['rc'], -17)
                self.assertTrue(all(r['rank'] == 1 for r in got))
            finally:
                trace.close()
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=5)


class InvokeHostControlFlowTests(unittest.TestCase):
    """Execute the production Python invoke body with CPU host-control stubs.

    No Torch/torch_npu module is imported, and no ACL/device behavior is claimed.
    """
    def invocation(self, binding):
        events, norm_calls = [], []
        values = {name: object() for name in
                  ('x', 'w', 'bias', 'residual', 'gamma', 'service_w', 'y', 'rstd', 'res')}

        def forbidden(*args, **kwargs):
            raise AssertionError('Unexpected non-candidate operator')

        def norm(reduced, residual, gamma, epsilon):
            norm_calls.append((reduced, residual, gamma, epsilon))
            return values['y'], values['rstd'], values['res']

        env = {
            **values, 'group': 'cpu-stub-group', 'meta': {},
            'trace': SimpleNamespace(emit=lambda stage, **fields: events.append((stage, fields))),
            'torch': SimpleNamespace(
                ops=SimpleNamespace(clef_grouped29=SimpleNamespace(run=binding)),
                nn=SimpleNamespace(functional=SimpleNamespace(linear=forbidden)),
                matmul=forbidden),
            'torch_npu': SimpleNamespace(npu_add_rms_norm=norm),
            'dist': SimpleNamespace(all_reduce=forbidden, ReduceOp=SimpleNamespace(SUM='sum')),
        }
        source = ROOT / 'run_probe.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'invoke']
        self.assertEqual(len(definitions), 1)
        module = ast.Module(body=[definitions[0]], type_ignores=[])
        exec(compile(module, str(source), 'exec'), env)
        return env, events, norm_calls

    def test_actual_invoke_traces_first_only_and_preserves_poison_argument(self):
        poison_values, binding_args = [], []
        reduced = object()

        def binding(x, w, bias, group, poison):
            binding_args.append((x, w, bias, group))
            poison_values.append(poison)
            return reduced, 4096

        env, events, norm_calls = self.invocation(binding)
        invoke = env['invoke']
        expected = {'y': env['y'], 'residual': env['res'], 'rstd': env['rstd'], 'reduced': reduced}
        self.assertEqual(invoke('candidate', poison=True), expected)
        self.assertEqual([stage for stage, fields in events],
                         ['binding_call_enter', 'binding_call_return', 'norm_enter', 'norm_return'])
        self.assertEqual(events[1][1], {'workspace_bytes': 4096})
        self.assertEqual(env['meta'], {'workspace_bytes': 4096})
        frozen_events = list(events)
        env['trace'] = None  # Any later trace access would fail this production body.
        for _ in range(3):
            self.assertEqual(invoke('candidate'), expected)
        self.assertEqual(events, frozen_events)
        self.assertEqual(poison_values, [True, False, False, False])
        self.assertEqual(binding_args, [(env['x'], env['w'], env['bias'], env['group'])] * 4)
        self.assertEqual(norm_calls, [(reduced, env['residual'], env['gamma'], 1e-6)] * 4)

    def test_actual_invoke_propagates_prepare_exception_without_fabricated_return(self):
        failure = RuntimeError('prepare rc=161002')

        def binding(*args):
            raise failure

        env, events, norm_calls = self.invocation(binding)
        with self.assertRaisesRegex(RuntimeError, 'prepare rc=161002') as raised:
            env['invoke']('candidate', poison=True)
        self.assertIs(raised.exception, failure)
        self.assertEqual(events, [('binding_call_enter', {})])
        self.assertEqual(norm_calls, [])
        self.assertEqual(env['meta'], {})


class ExceptionHandlerHostControlFlowTests(unittest.TestCase):
    """Run only the real outer exception handler and flush on CPU fixtures."""
    def run_handler_case(self, emit_fails, close_fails):
        source = ROOT / 'run_probe.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        outer_tries = [node for node in main.body if isinstance(node, ast.Try)]
        self.assertEqual(len(outer_tries), 1)
        handler = outer_tries[0].handlers[0]
        self.assertIsInstance(handler.type, ast.Name)
        self.assertEqual(handler.type.id, 'BaseException')
        flushes = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'flush']
        self.assertEqual(len(flushes), 1)
        # Include no main-body imports or device setup: only raise the original
        # CPU error, then execute the exact production handler and disk flush.
        raise_original = ast.Raise(exc=ast.Name(id='original', ctx=ast.Load()), cause=None)
        fixture = ast.Try(body=[raise_original], handlers=[handler], orelse=[], finalbody=[])
        module = ast.fix_missing_locations(ast.Module(body=[flushes[0], fixture], type_ignores=[]))
        attempts = []

        class TraceStub:
            def emit(self, stage, **fields):
                attempts.append(('emit', stage, fields))
                if emit_fails:
                    raise OSError('trace emit failed')

            def close(self):
                attempts.append(('close',))
                if close_fails:
                    raise OSError('trace close failed')

        original = RuntimeError('prepare rc=161002')
        meta = {'performance_valid': True, 'files': {}}
        with tempfile.TemporaryDirectory() as tmp:
            env = {'original': original, 'trace': TraceStub(), 'meta': meta,
                   'out': Path(tmp), 'json': json, 'traceback': traceback}
            with self.assertRaisesRegex(RuntimeError, 'prepare rc=161002') as raised:
                exec(compile(module, str(source), 'exec'), env)
            self.assertIs(raised.exception, original)
            result = Path(tmp) / 'result.json'
            self.assertTrue(result.is_file())
            persisted = json.loads(result.read_text())
            self.assertEqual(persisted, meta)
            self.assertIs(persisted['performance_valid'], False)
            self.assertTrue(persisted['error'].rstrip().endswith('RuntimeError: prepare rc=161002'))
            self.assertNotIn('trace emit failed', persisted['error'])
            self.assertNotIn('trace close failed', persisted['error'])
            self.assertEqual(persisted['files'], {})
            expected_keys = {'performance_valid', 'files', 'error'}
            if emit_fails:
                expected_keys.add('trace_error')
                self.assertEqual(persisted['trace_error'], "OSError('trace emit failed')")
            if close_fails:
                expected_keys.add('trace_close_error')
                self.assertEqual(persisted['trace_close_error'], "OSError('trace close failed')")
            self.assertEqual(set(persisted), expected_keys)
            self.assertEqual(attempts,
                             [('emit', 'python_exception',
                               {'exception_type': 'RuntimeError', 'message': 'prepare rc=161002'}),
                              ('close',)])

    def test_actual_handler_flushes_original_error_when_emit_fails(self):
        self.run_handler_case(emit_fails=True, close_fails=False)

    def test_actual_handler_flushes_original_error_when_close_fails(self):
        self.run_handler_case(emit_fails=False, close_fails=True)

    def test_actual_handler_flushes_original_error_when_emit_and_close_fail(self):
        self.run_handler_case(emit_fails=True, close_fails=True)

    def test_actual_handler_flushes_original_error_when_trace_succeeds(self):
        self.run_handler_case(emit_fails=False, close_fails=False)


if __name__ == '__main__':
    unittest.main()
