"""CPU-only query deadline boundaries; no live npu-smi, child or signal calls."""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import test_vision_cache_window as prior
import vision_cache_window as v


class EndOfRound(Exception):
    pass


class SamplerTimeoutCPU(unittest.TestCase):
    setUp = prior.VisionCPU.setUp

    @classmethod
    def setUpClass(cls):
        cls.f = v.load_frozen(Path(__file__).with_name('grouped_mmar') / 'supervised_probe_window.py')

    def sampler_round(self, *, ready=True, durations=(0.1, 0.1, 0.1, 0.1),
                      fault=None, partials=(b'partial\xff stdout', b'partial\xfe stderr'), scan_delay=0):
        directory = self.root / str(len(list(self.root.iterdir())))
        directory.mkdir()
        table = {9000001: prior.ident(), 9000002: prior.ident(9000002, ppid=9000001),
                 9000003: prior.ident(9000003, ppid=9000001)}
        if ready:
            self.f.write_json(directory/'ready.json', {'workers': {'0': table[9000002], '1': table[9000003]}})
        clock = SimpleNamespace(now=100.0)
        calls, sleeps = [], []

        def scan():
            clock.now += scan_delay
            return table

        def command(argv, **kwargs):
            index = len(calls)
            calls.append({'argv': list(argv), 'timeout': kwargs['timeout']})
            clock.now += durations[index]
            kind = fault[1] if fault and fault[0] == index else None
            if kind == 'timeout':
                raise v.subprocess.TimeoutExpired(argv, kwargs['timeout'],
                                                  output=partials[0], stderr=partials[1])
            if argv == ['/usr/local/bin/npu-smi', 'info']:
                stdout = 'Process id in container\nNo running processes found in NPU 7'
                if kind == 'wrong-card':
                    stdout = 'Process id in container\n|8 0|1|VLLMWorker_TP|0|9000002|'
                if kind == 'missing-header':
                    stdout = 'No running processes found in NPU 7'
            elif argv[3] == 'usages':
                stdout = 'HBM Capacity(MB): 65536\nHBM Usage Rate(%): 91'
                if kind == 'parse':
                    stdout = 'HBM Capacity(MB): 65536\n'
            else:
                chip = int(argv[-1])
                stdout = f'Process id:{9000002+chip} Process name:VLLMWorker_TP Process memory(MB):56754\n'
                if chip == 0:
                    stdout += 'Process id:9000001 Process name:python Process memory(MB):106\n'
                if kind == 'unknown-pid':
                    stdout += 'Process id:9000099 Process name:python Process memory(MB):106\n'
            return SimpleNamespace(returncode=7 if kind == 'nonzero' else 0,
                                   stdout=stdout, stderr='query stderr')

        def sleep(seconds):
            sleeps.append(seconds)
            raise EndOfRound()

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(v.subprocess, 'run', side_effect=command))
            stack.enter_context(mock.patch.object(self.f, 'scan', side_effect=scan))
            stack.enter_context(mock.patch.object(self.f, 'proc_identity', side_effect=table.get))
            stack.enter_context(mock.patch.object(v.time, 'monotonic', side_effect=lambda: clock.now))
            stack.enter_context(mock.patch.object(v.time, 'sleep', side_effect=sleep))
            try:
                rc = v.sample_loop(self.f, directory, prior.ident())
            except EndOfRound:
                rc = 0
        rows = v.json_lines(directory/'hbm.jsonl')
        self.assertEqual(len(rows), 1)
        return rc, rows[0], calls, sleeps

    def test_startup_and_ready_allow_query_over_one_second_within_bounds(self):
        for ready, durations, slow_index in ((False, (1.2, 0.1, 0.1), 0),
                                              (True, (0.1, 0.1, 1.2, 0.1), 2)):
            with self.subTest(ready=ready):
                rc, row, calls, sleeps = self.sampler_round(ready=ready, durations=durations)
                self.assertEqual(rc, 0)
                self.assertTrue(row['ok'])
                self.assertEqual(len(calls), len(durations))
                self.assertEqual(calls[slow_index]['timeout'], 3)
                query = row['queries'][slow_index]
                self.assertGreater(query['finished']-query['started'], 1)
                self.assertLess(query['finished'], query['deadline'])
                self.assertAlmostEqual(sleeps[0], 5-sum(durations))
                self.assertTrue(all(value > 2*v.GIB for value in v.hbm_gate([row], row['finished'])))

    def test_ready_queries_use_remaining_absolute_round_budget(self):
        rc, row, calls, sleeps = self.sampler_round(durations=(2.5, 0.2, 0.2, 0.2))
        self.assertEqual(rc, 0)
        self.assertTrue(row['ok'])
        for call, query, expected in zip(calls, row['queries'], (3, 1.5, 1.3, 1.1)):
            self.assertAlmostEqual(call['timeout'], expected)
            self.assertEqual(query['timeout_s'], call['timeout'])
        self.assertEqual([query['deadline'] for query in row['queries']], [103, 104, 104, 104])
        self.assertAlmostEqual(sleeps[0], 1.9)

    def test_command_and_round_deadline_equal_or_late_reject_success_return(self):
        for durations, expected_calls in (((3,), 1), ((3.01,), 1),
                                           ((2.5, 1.5), 2), ((2.5, 1.51), 2)):
            with self.subTest(durations=durations):
                rc, row, calls, sleeps = self.sampler_round(durations=durations)
                self.assertEqual(rc, 1)
                self.assertFalse(row['ok'])
                self.assertIn('query deadline', row['error'])
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(row['queries'][-1]['rc'], 0)
                self.assertGreaterEqual(row['queries'][-1]['finished'], row['queries'][-1]['deadline'])
                self.assertEqual(sleeps, [])
                with self.assertRaises(RuntimeError):
                    v.hbm_gate([row], row['finished'])

    def test_exhausted_round_does_not_spawn_a_query(self):
        for ready in (False, True):
            for scan_delay in (4, 4.1):
                with self.subTest(ready=ready, scan_delay=scan_delay):
                    rc, row, calls, sleeps = self.sampler_round(ready=ready, scan_delay=scan_delay)
                    self.assertEqual(rc, 1)
                    self.assertFalse(row['ok'])
                    self.assertEqual(calls, [])
                    self.assertEqual(sleeps, [])
                    self.assertLessEqual(row['queries'][0]['timeout_s'], 0)
                    self.assertIsNone(row['queries'][0]['rc'])
                    self.assertIn('query round deadline', row['error'])

    def test_timeout_preserves_partial_bytes_and_stops_without_retry(self):
        for partials in ((b'partial\xff stdout', b'partial\xfe stderr'), (None, None), ('text', 'error')):
            with self.subTest(partials=partials):
                rc, row, calls, sleeps = self.sampler_round(
                    durations=(0.1, 0.1, 3), fault=(2, 'timeout'), partials=partials)
                self.assertEqual(rc, 1)
                self.assertFalse(row['ok'])
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(row['queries']), 3)
                self.assertEqual(calls[-1]['argv'][3:], ['usages', '-i', '7', '-c', '1'])
                self.assertEqual(sleeps, [])
                query = row['queries'][-1]
                self.assertIsNone(query['rc'])
                self.assertIn('TimeoutExpired', query['error'])
                self.assertIn('TimeoutExpired', row['error'])
                for key, partial in zip(('stdout', 'stderr'), partials):
                    self.assertEqual(query['partial_'+key+'_available'], partial is not None)
                    if isinstance(partial, bytes):
                        self.assertEqual(query[key], partial.decode(errors='replace'))
                        self.assertEqual(bytes.fromhex(query[key+'_hex']), partial)
                    else:
                        self.assertEqual(query[key], partial)
                        self.assertNotIn(key+'_hex', query)

    def test_nonzero_parse_and_process_evidence_failures_still_abort(self):
        for ready, fault in ((True, (0, 'nonzero')), (True, (0, 'parse')),
                             (True, (1, 'unknown-pid')), (False, (0, 'wrong-card')),
                             (False, (0, 'missing-header'))):
            with self.subTest(ready=ready, fault=fault):
                rc, row, calls, sleeps = self.sampler_round(ready=ready, fault=fault)
                self.assertEqual(rc, 1)
                self.assertFalse(row['ok'])
                self.assertEqual(len(calls), fault[0]+1)
                self.assertEqual(len(row['queries']), fault[0]+1)
                self.assertEqual(sleeps, [])
                self.assertIn('error', row)
                self.assertEqual(row['queries'][-1]['rc'], 7 if fault[1]=='nonzero' else 0)

    def test_existing_hbm_round_cadence_stale_card_and_reserve_contract(self):
        samples = [prior.sample(100), prior.sample(105)]
        self.assertEqual(v.hbm_gate(samples, 106), [4*v.GIB, 4*v.GIB])
        mutations = (
            lambda rows: rows[0].update(finished=104),
            lambda rows: rows[1].update(started=106.01, finished=106.5),
            lambda rows: rows[0]['chips'][0].update(free_lower_bytes=2*v.GIB),
            lambda rows: rows[0]['chips'][0].update(card=8),
            lambda rows: rows[0].update(ok=False),
        )
        for index, mutate in enumerate(mutations):
            rows = copy.deepcopy(samples); mutate(rows)
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                v.hbm_gate(rows, 107)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            v.hbm_gate(samples, 115.01)

    def test_plan_requires_three_second_query_and_unchanged_monitor_contract(self):
        plan = prior.VisionCPU.order_plan(self)
        with self.assertRaisesRegex(RuntimeError, 'sampler policy'):
            v.validate_plan(plan)
        plan['memory_guard']['query_timeout_s'] = 3
        v.validate_plan(plan)
        for key, value in (('query_timeout_s', 1), ('query_timeout_s', 4), ('sample_interval_s', 6),
                           ('sample_round_timeout_s', 5), ('max_sample_age_s', 11)):
            bad = copy.deepcopy(plan); bad['memory_guard'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(RuntimeError, 'sampler policy'):
                v.validate_plan(bad)


if __name__ == '__main__':
    unittest.main()
