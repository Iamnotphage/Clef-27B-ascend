"""CPU mocks for vision/c80 composition only; never spawn, signal or query a device."""
import copy
import contextlib
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import vision_cache_window as v


def ident(pid=9000001, **overrides):
    return {**dict(pid=pid, starttime='401718579', boot_id='boot', pgid=pid, sid=pid,
                   ppid=9000000, state='S', cmdline=['python']), **overrides}


def constructor(cap=16384):
    return dict(async_scheduling=False, balance_scheduling=False, compute_config=16384,
                compute_effective=16384, cache_config_before=16384, cache_requested=cap,
                cache_effective=cap, unit='encoder_embeddings',
                parent='vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler',
                mro=['clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler',
                     'vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler',
                     'vllm.v1.core.sched.scheduler.Scheduler', 'builtins.object'])


def log(row):
    return 'Clef encoder cache candidate initialized: ' + json.dumps(row) + '\n'


def sample(start=0, free=4*v.GIB):
    return dict(started=start, finished=start+0.5, ok=True,
                chips=[dict(card=7, chip=c, free_lower_bytes=free) for c in (0, 1)])


QUALITY = {'baseline_correct': 219, 'total': 400, 'max_drop_pp': 2, 'min_correct': 211}


class VisionCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = v.load_frozen(Path(__file__).with_name('grouped_mmar') / 'supervised_probe_window.py')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # A regression must fail the test before touching any live child or signal.
        for name in ('Popen', 'run'):
            p = mock.patch.object(v.subprocess, name, side_effect=AssertionError('unmocked subprocess'))
            p.start(); self.addCleanup(p.stop)
        for name in ('kill', 'killpg'):
            p = mock.patch.object(v.os, name, side_effect=AssertionError('unmocked signal'))
            p.start(); self.addCleanup(p.stop)

    def test_frozen_loader_checks_entire_original(self):
        wrong = self.root / 'frozen.py'; wrong.write_text('raise AssertionError("executed")')
        with self.assertRaisesRegex(RuntimeError, 'SHA'):
            v.load_frozen(wrong)
        self.assertEqual(self.f.settle_attributed.__module__, 'vision_frozen_c80')

    def test_exact_raw_argv_json_quotes_empty_tail_and_d0_conversion(self):
        raw = b'python\0--json\0{"x": "a b", "q": "\\\""}\0\0'
        expected = ['python', '--json', '{"x": "a b", "q": "\\\""}', '']
        with mock.patch.object(self.f, 'proc_identity', return_value=ident()):
            row = v.bound_root(self.f, ident(), expected, lambda _: raw)
        self.assertEqual(row['argv'], expected)
        self.assertEqual(row['cmdline'], raw.replace(b'\0', b' ').decode(errors='replace').strip())
        self.assertEqual(bytes.fromhex(row['raw_hex']), raw)
        self.assertEqual(row['ppid'], 9000000)  # non-PID1 is precisely bound

    def test_full_root_identity_drift_and_missing_fields_fail_before_stop(self):
        for key in self.f.FIELDS:
            with self.subTest(key=key):
                saved = ident(); changed = dict(saved); changed[key] = 'drift'
                with mock.patch.object(self.f, 'proc_identity', return_value=changed):
                    with self.assertRaises(RuntimeError):
                        v.bound_root(self.f, saved, ['python'], lambda _: b'python\0')
                del saved[key]
                with self.assertRaises(RuntimeError):
                    v.bound_root(self.f, saved, ['python'], lambda _: b'python\0')

    def test_argv_types_truncation_and_post_read_parent_drift_rejected(self):
        with mock.patch.object(self.f, 'proc_identity', return_value=ident()):
            for argv in ('python', ['python'], ['python', 1]):
                with self.subTest(argv=argv), self.assertRaises(RuntimeError):
                    v.bound_root(self.f, ident(), argv, lambda _: b'python\0\0')
        second = ident(); second['ppid'] += 1
        with mock.patch.object(self.f, 'proc_identity', side_effect=[ident(), second]):
            with self.assertRaisesRegex(RuntimeError, 'after argv'):
                v.bound_root(self.f, ident(), ['python'], lambda _: b'python\0')

    def arm(self):
        reg = SimpleNamespace(snapshot=lambda: [], stop=mock.Mock(), record_now=mock.Mock())
        with mock.patch.object(self.f, 'DescendantRegistry') as factory:
            factory.return_value.start.return_value = reg
            arm = v.OwnedArm(self.f, self.root, {1})
        return arm

    def close_mocks(self, stack, outcome=None, settled=None):
        outcome = outcome if outcome is not None else dict(members=[], unknown=[], drift=[])
        settled = settled if settled is not None else dict(clean=True, deadline_hit=False, residual=[], table={})
        clean = stack.enter_context(mock.patch.object(self.f, 'cleanup_attributed', return_value=outcome))
        settle = stack.enter_context(mock.patch.object(self.f, 'settle_attributed', return_value=settled))
        stack.enter_context(mock.patch.object(self.f, 'port_open', return_value=111))
        stack.enter_context(mock.patch.object(v, 'strict_npu', return_value={'rows': []}))
        return clean, settle

    def test_immediate_empty_cleanup_still_settles_once_and_records_before_signal(self):
        arm = self.arm()
        with ExitStack() as s:
            cleanup, settle = self.close_mocks(s)
            cleanup.side_effect = lambda *a, **k: (arm.registry.stop.assert_called_once() or
                                                  dict(members=[], unknown=[], drift=[]))
            self.assertTrue(arm.close(18143)); self.assertTrue(arm.close(18143))
        self.assertEqual(cleanup.call_count, 1); self.assertEqual(settle.call_count, 1)
        self.assertEqual(settle.call_args.kwargs['min_empty'], 2)
        self.assertLessEqual(settle.call_args.kwargs['deadline'], 300)

    def test_unclean_empty_and_deadline_flags_latch_failure_without_new_budget(self):
        for clean, hit in ((False, False), (True, True), (False, True)):
            arm = self.arm()
            with self.subTest(clean=clean, hit=hit), ExitStack() as s:
                _, settle = self.close_mocks(s, settled=dict(clean=clean, deadline_hit=hit, residual=[], table={}))
                self.assertFalse(arm.close(18143)); deadline = arm.cleanup_deadline
                self.assertFalse(arm.close(18143))
                self.assertEqual(arm.cleanup_deadline, deadline); self.assertEqual(settle.call_count, 1)

    def test_frozen_two_empty_at_deadline_is_rejected(self):
        now = [0.0]
        result = self.f.settle_attributed([], None, deadline=300, interval=300,
                    min_empty=2, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0]+n),
                    table_reader=lambda: {})
        self.assertEqual(result['empty_rounds'], 2)
        self.assertTrue(result['deadline_hit']); self.assertFalse(result['clean'])

    def test_cleanup_time_is_charged_to_same_deadline(self):
        now = [10.0]; arm = self.arm()
        with ExitStack() as s:
            cleanup, settle = self.close_mocks(s)
            s.enter_context(mock.patch.object(v.time, 'monotonic', side_effect=lambda: now[0]))
            def cost(*a, **kw):
                now[0] += 20
                return dict(members=[], unknown=[], drift=[])
            cleanup.side_effect = cost
            self.assertTrue(arm.close(18143))
            self.assertEqual(settle.call_args.kwargs['deadline'], 280)
            self.assertEqual(arm.cleanup_deadline, 310)

    def test_protected_unknown_and_pid_reuse_block_even_if_later_empty(self):
        for outcome in (dict(reason='protected_overlap'), dict(unknown=[9000002]),
                        dict(drift=[{'saved': ident(), 'live': ident()}])):
            arm = self.arm()
            with self.subTest(outcome=outcome), ExitStack() as s:
                self.close_mocks(s, outcome=outcome)
                self.assertFalse(arm.close(18143))

    def test_unknown_same_pgid_never_authorizes_group_signal(self):
        root, other = ident(), ident(9000002)
        other['pgid'] = root['pid']
        table = {r['pid']: r for r in (root, other)}
        targets = self.f._attributed_targets([root], table=table, reader=table.get)
        self.assertEqual(targets['groups'], {}); self.assertEqual(targets['singles'], {root['pid']})
        self.assertEqual(targets['unknown'], [other['pid']])

    def test_adopted_and_escaped_descendants_remain_in_fresh_restore_gate(self):
        for change in ('ppid', 'pgid', 'starttime'):
            saved = ident(9000002); live = dict(saved)
            live[change] = 1 if change != 'starttime' else 'new'
            registry = SimpleNamespace(snapshot=lambda: [saved])
            survivors = self.f.attributed_survivors(registry, table={live['pid']: live}, reader=lambda _: live)
            self.assertEqual(survivors, [live['pid']])

    def test_running_client_sampler_open_port_or_busy_card_blocks_close(self):
        for mode in ('client', 'port', 'card', 'unbound'):
            arm = self.arm()
            if mode in ('client', 'unbound'):
                arm.processes = [SimpleNamespace(poll=lambda: None)]
                arm.identities = [ident()] if mode == 'client' else []
            with self.subTest(mode=mode), ExitStack() as s:
                self.close_mocks(s)
                if mode == 'port': s.enter_context(mock.patch.object(self.f, 'port_open', return_value=0))
                if mode == 'card': s.enter_context(mock.patch.object(v, 'strict_npu', side_effect=RuntimeError('busy')))
                self.assertFalse(arm.close(18143))

    def test_fresh_gate_does_not_reuse_settle_table(self):
        arm = self.arm(); arm.closed = arm.clean = True; arm.identities = [ident(9000002)]
        table = {9000002: ident(9000002)}
        with mock.patch.object(self.f, 'scan', return_value=table):
            with self.assertRaisesRegex(RuntimeError, 'fresh attributed'):
                v.final_restore_gate(self.f, {'port':18143}, ident(), [arm], True, {})

    def test_frozen_recovery_and_subreaper_once_are_consumed(self):
        for allowed in (True, False):
            result = {}
            with mock.patch.object(self.f, 'begin_restore', return_value=allowed) as begin:
                if allowed: v.restore_once(self.f, {'run_dir': str(self.root)}, result)
                else:
                    with self.assertRaises(RuntimeError): v.restore_once(self.f, {'run_dir': str(self.root)}, result)
                with self.assertRaises(RuntimeError): v.restore_once(self.f, {'run_dir': str(self.root)}, result)
            self.assertEqual(begin.call_count, 1); self.assertEqual(result['restore_attempts'], 1)

    def test_startup_three_caps_and_every_drift_fail_closed(self):
        for cap in v.CAPACITIES: self.assertEqual(v.startup_record(log(constructor(cap)), cap)['cache_effective'], cap)
        changes = dict(async_scheduling=0, balance_scheduling=None, cache_config_before=4096,
                       cache_effective=327680, cache_requested=0, compute_config=4096,
                       compute_effective=20000, parent='wrong', unit='tokens', mro=[])
        for key, value in changes.items():
            row = constructor(); row[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): v.startup_record(log(row), 16384)
        for text in ('', log(constructor()) * 2, 'Clef encoder cache candidate initialized: {broken'):
            with self.assertRaises((RuntimeError, ValueError)): v.startup_record(text, 16384)

    def readme_erqa_args(self):
        text = (Path(__file__).parents[1]/'Clef-27B'/'README.md').read_text()
        text = text.split('ERQA 优选启动配置（2026-11-20 已采用）',1)[1].split('```bash\n',1)[1].split('```',1)[0]
        args = shlex.split(text.replace('\\\n', ''))
        self.assertEqual(args[:2], ['python','Clef-27B/inference.py'])
        args = args[2:]
        self.assertEqual(v.option(args,'--config'),'s1')
        for flag, placeholder in (('--model-dir','/path/to/Clef-27B'),
                                  ('--output-dir','/path/to/new-erqa-run')):
            self.assertEqual(v.option(args,flag),placeholder)
            i = args.index(flag)
            del args[i:i+2]  # LaunchTests supplies isolated temporary model/output paths.
        return args

    def startup_fixture(self, utilization):
        from test_inference_launch import LaunchTests
        case = LaunchTests(); case.setUp(); self.addCleanup(case.doCleanups)
        args = self.readme_erqa_args()
        args[args.index('--gpu-memory-utilization')+1] = str(utilization)
        args[args.index('--encoder-cache-tokens')+1] = '131072'  # Preserve this CLI fixture.
        manifest = case.launch(*args)
        module = Path(__file__).parents[1]/'Clef-27B'/'clef_encoder_cache_scheduler.py'
        digest = manifest['effective']['encoder_cache_scheduler_sha256']
        plan = {'cleanup_restore': {'source_target': 'aa3e.py'}}
        phase = dict(cache_cli=131072, service_argv=list(manifest['argv']),
                     startup_gate=dict(module_path=str(module),module_sha256=digest,
                         scheduler_cls='clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler'))
        text = '(EngineCore_DP0 pid=9000002) ' + log(constructor(131072)) + (
            '(EngineCore_DP0 pid=9000002) ObservabilityConfig(enable_mm_processor_stats=False, '
            'enable_logging_iteration_details=False)\n')
        stack = ExitStack()
        stack.enter_context(mock.patch.object(self.f,'scan',return_value={
            9000001:ident(),9000002:ident(9000002,ppid=9000001)}))
        stack.enter_context(mock.patch.object(self.f,'sha256',side_effect=lambda p:
            v.AA3E if str(p)=='aa3e.py' else digest))
        stack.enter_context(mock.patch.object(self.f,'read_root_env',return_value={
            'PYTHONPATH':str(module.parent).encode(),'CLEF_ENCODER_CACHE_TOKENS':b'131072'}))
        stack.enter_context(mock.patch.object(self.f,'joint_env',return_value={'ok':True}))
        stack.enter_context(mock.patch.object(self.f,'read_json',return_value={'branch':'aa3e-noop'}))
        return stack, plan, phase, manifest, text

    def test_startup_utilization_three_way_match_with_actual_launcher_manifests(self):
        for util in (0.85, 0.9):
            stack, plan, phase, manifest, text = self.startup_fixture(util)
            with self.subTest(util=util), stack:
                result = v.verify_startup(self.f,plan,phase,manifest,ident(),text)
                self.assertEqual(result['manifest']['effective']['gpu_memory_utilization'],util)

    def test_startup_utilization_drift_missing_duplicate_and_nonfinite_rejected(self):
        stack, plan, original, manifest, text = self.startup_fixture(0.85)
        with stack:
            for target in ('plan','argv','effective'):
                for value in ('missing','duplicate','nan','inf','bad',float('nan'),float('inf'),True,0.9):
                    if target=='effective' and value=='duplicate': continue
                    phase, current = copy.deepcopy(original), copy.deepcopy(manifest)
                    if target=='effective':
                        if value=='missing': current['effective'].pop('gpu_memory_utilization')
                        else: current['effective']['gpu_memory_utilization'] = value
                    else:
                        argv = phase['service_argv'] if target=='plan' else current['argv']
                        i = argv.index('--gpu-memory-utilization')
                        if value=='missing': del argv[i:i+2]
                        elif value=='duplicate': argv.extend(argv[i:i+2])
                        else: argv[i+1] = value if value is True else str(value)
                    with self.subTest(target=target,value=value), self.assertRaises((RuntimeError,ValueError)):
                        v.verify_startup(self.f,plan,phase,current,ident(),text)

    def test_hbm_conservative_reserve_and_large_arm_boundary(self):
        value = v.hbm_values('HBM Capacity(MB): 65536\nHBM Usage Rate(%): 91\n')
        self.assertEqual(value['free_lower_bytes'], 8 * 65536 * 1048576 // 100)
        self.assertEqual(v.ADMISSION_BYTES, 3672279040)
        self.assertEqual(v.hbm_gate([sample(free=v.ADMISSION_BYTES)], 1), [v.ADMISSION_BYTES] * 2)
        with self.assertRaises(RuntimeError): v.hbm_gate([sample(free=2*v.GIB)], 1)

    def test_hbm_stale_missing_error_wrong_card_chip_and_cadence_rejected(self):
        cases = [([], 1), ([sample()], 11), ([dict(sample(), ok=False)], 1),
                 ([sample(), sample(12)], 13)]
        for key, value in [('card', 8), ('chip', 2)]:
            row = sample(); row['chips'][0][key] = value; cases.append(([row], 1))
        for rows, now in cases:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError): v.hbm_gate(rows, now)
        for text in ('', 'HBM Capacity(MB): 32768\nHBM Usage Rate(%): 90',
                     'HBM Capacity(MB): 65536\nHBM Usage Rate(%): 101'):
            with self.assertRaises(RuntimeError): v.hbm_values(text)

    def test_npu_card_gate_rejects_wrong_card_even_with_correct_pids_and_chips(self):
        rows = [dict(card=8, chip=i, pid=9000001+i, name='VLLMWorker_TP') for i in (0, 1)]
        with mock.patch.object(v.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='x', stderr='')), \
                mock.patch.object(self.f, 'npu_rows', return_value=rows):
            with self.assertRaisesRegex(RuntimeError, 'card/chip'): v.strict_npu(self.f, {9000001,9000002})

    def test_partial_jsonl_tail_is_deferred_but_malformed_completed_record_fails(self):
        path = self.root / 'progress'; path.write_text('{"status":"ok"}\n{"stat')
        self.assertEqual(v.json_lines(path, complete=False), [{'status':'ok'}])
        path.write_text('{broken}\n')
        with self.assertRaises(ValueError): v.json_lines(path, complete=False)

    def window(self, *, fail_arm=None, fail_close=None, stop_rc=0, reserve=4*v.GIB,
               subreaper=True, order=None, phase_check=None):
        directory = self.root / ('window-' + str(len(list(self.root.iterdir()))))
        events = []
        class Arm:
            def __init__(this, frozen, path, protected):
                this.directory = path; this.closed = this.clean = False; this.extra_entries = []
            def entries(this): return this.extra_entries
            def spawn(this, name, argv):
                events.append(name)
                return SimpleNamespace(wait=lambda **kw: stop_rc)
            def close(this, port):
                if not this.closed: events.append('close:' + this.directory.name)
                this.closed = True; this.clean = this.directory.name != fail_close
                return this.clean
        plan = dict(run_root=str(directory), current_service={'root':9000001},
                    restore_environment_expect={}, budgets_s={'overall_outer':5760},
                    cleanup_restore=dict(restore_output=str(directory/'restore'), source_target='aa3e.py',
                                         restore_source={'selected': {'argv':['python']}}, stop_argv_template=['stop']),
                    phases=[dict(cache_cli=cap, name=str(cap), max_arm_active_s=1020)
                            for cap in (v.CAPACITIES if order is None else order)])
        if order is not None:
            plan['arm_order'] = list(order)
        def phase(frozen, plan, spec, arm, expected, reference, deadline, path):
            events.append(spec['cache_cli'])
            if spec['cache_cli'] == fail_arm: raise RuntimeError('phase failure')
            if phase_check: phase_check(spec['cache_cli'])
            free = reserve[spec['cache_cli']] if isinstance(reserve, dict) else reserve
            return ['reference'], [free, free]
        def gate(frozen, plan, saved, arms, stopped, result):
            v.require(stopped and all(a.closed and a.clean for a in arms), 'closed gate')
        with ExitStack() as s:
            for name, value in [('get_subreaper',0), ('enable_subreaper',subreaper),
                                ('restore_subreaper',True), ('scan',{9000001:ident()}),
                                ('pending_clients',[]), ('port_open',0), ('begin_restore',True)]:
                s.enter_context(mock.patch.object(self.f, name, return_value=value))
            for name in ('check_pins','bound_root','strict_npu','check_environment','verify_restore'):
                s.enter_context(mock.patch.object(v, name, return_value={}))
            s.enter_context(mock.patch.object(v, 'read_reference', return_value=['reference']))
            s.enter_context(mock.patch.object(v, 'OwnedArm', Arm))
            s.enter_context(mock.patch.object(v, 'run_phase', side_effect=phase))
            s.enter_context(mock.patch.object(v, 'final_restore_gate', side_effect=gate))
            begin = self.f.begin_restore
            if not subreaper:
                with self.assertRaisesRegex(RuntimeError, 'subreaper'): v.execute(self.f, plan, ident(), [], 'frozen')
                self.assertEqual(begin.call_count, 0); return None, events
            rc = v.execute(self.f, plan, ident(), [], 'frozen')
            result = json.loads((directory/'window.json').read_text())
            self.assertEqual(begin.call_count, result['restore_attempts'])
            self.assertEqual(rc, 0 if result['ok'] else 1)
        return result, events

    def test_full_window_stops_once_runs_three_arms_then_restores_once(self):
        result, events = self.window()
        self.assertTrue(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertEqual(events, ['stop', 'close:initial-stop', 16384, 'close:16384',
                                  131072, 'close:131072', 262144, 'close:262144'])

    def test_window_failure_stops_progress_and_finally_does_not_repeat_cleanup(self):
        result, events = self.window(fail_arm=131072)
        self.assertFalse(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertNotIn(262144, events); self.assertEqual(events.count('close:131072'), 1)

    def test_window_unclean_and_subreaper_failure_block_restore(self):
        result, events = self.window(fail_close='131072')
        self.assertEqual(result['restore_attempts'], 0); self.assertNotIn(262144, events)
        self.window(subreaper=False)

    def test_stop_nonzero_gone_and_clean_allows_one_final_recovery(self):
        result, events = self.window(stop_rc=1)
        self.assertEqual(result['restore_attempts'], 1); self.assertFalse(result['ok'])
        self.assertNotIn(16384, events)

    def test_hbm_peak_minimum_admission_skips_only_large_arm(self):
        result, events = self.window(reserve=v.ADMISSION_BYTES-1)
        self.assertTrue(result['ok']); self.assertNotIn(262144, events)
        self.assertIn('skipped', result['arms'][-1])
        rows = [sample(0, free=v.ADMISSION_BYTES-1), sample(5, free=5*v.GIB)]
        self.assertEqual(v.hbm_gate(rows, 6), [v.ADMISSION_BYTES-1]*2)

    def test_reverse_window_runs_131072_then_16384_with_capacity_keyed_results(self):
        result, events = self.window(order=(131072, 16384), reserve={131072:4*v.GIB, 16384:5*v.GIB})
        self.assertTrue(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertEqual(events, ['stop','close:initial-stop',131072,'close:131072',16384,'close:16384'])
        self.assertEqual(result['arm_order'], [131072,16384])
        self.assertEqual(set(result['by_capacity']), {'131072','16384'})
        self.assertEqual(result['by_capacity']['131072']['min_free_lower_bytes'], [4*v.GIB]*2)
        self.assertEqual(result['by_capacity']['16384']['min_free_lower_bytes'], [5*v.GIB]*2)

    def test_reverse_first_arm_failure_blocks_control_and_restores_once(self):
        result, events = self.window(order=(131072, 16384), fail_arm=131072)
        self.assertFalse(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertNotIn(16384, events); self.assertNotIn(262144, events)

    def test_legacy_large_arm_admission_uses_131072_capacity_evidence(self):
        result, events = self.window(reserve={16384:v.ADMISSION_BYTES-1, 131072:4*v.GIB, 262144:3*v.GIB})
        self.assertTrue(result['ok']); self.assertIn(262144, events)
        result, events = self.window(reserve={16384:5*v.GIB, 131072:v.ADMISSION_BYTES-1})
        self.assertNotIn(262144, events)
        self.assertEqual(result['by_capacity']['262144']['prior_capacity'], 131072)
        self.assertEqual(result['by_capacity']['262144']['prior_min_free_bytes'], [v.ADMISSION_BYTES-1]*2)

    def test_fullset_order_runs_131072_then_262144_at_admission_threshold(self):
        result, events = self.window(order=(131072, 262144),
                                     reserve={131072:v.ADMISSION_BYTES, 262144:3*v.GIB})
        self.assertTrue(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertEqual(events, ['stop','close:initial-stop',131072,'close:131072',262144,'close:262144'])
        self.assertEqual(result['arm_order'], [131072,262144])
        self.assertEqual(set(result['by_capacity']), {'131072','262144'})
        self.assertEqual(result['by_capacity']['131072']['min_free_lower_bytes'], [v.ADMISSION_BYTES]*2)
        self.assertEqual(result['by_capacity']['262144']['min_free_lower_bytes'], [3*v.GIB]*2)

    def test_fullset_order_skips_large_arm_below_131072_admission_threshold(self):
        result, events = self.window(order=(131072, 262144), reserve={131072:v.ADMISSION_BYTES-1})
        self.assertTrue(result['ok']); self.assertEqual(result['restore_attempts'], 1)
        self.assertNotIn(262144, events); self.assertNotIn(16384, events)
        skipped = result['by_capacity']['262144']
        self.assertEqual(skipped['prior_capacity'], 131072)
        self.assertEqual(skipped['prior_min_free_bytes'], [v.ADMISSION_BYTES-1]*2)
        self.assertEqual(skipped['required'], v.ADMISSION_BYTES)

    def test_fullset_first_arm_failure_or_unclean_blocks_large_arm(self):
        for kw in ({'fail_arm':131072}, {'fail_close':'131072'}):
            with self.subTest(kw=kw):
                result, events = self.window(order=(131072, 262144), **kw)
                self.assertFalse(result['ok']); self.assertNotIn(262144, events)
                self.assertEqual(result['restore_attempts'], 0 if 'fail_close' in kw else 1)

    def order_plan(self, caps=v.CAPACITIES):
        return dict(schema='clef-vision-cache-plan/2', candidate_commit='3bf466d913ea269d4f4bb37b560a6ee7a3fdbf47',
                    memory_guard=dict(sample_interval_s=5,query_timeout_s=3,sample_round_timeout_s=4,max_sample_age_s=10),
                    cleanup_restore=dict(max_restore_attempts=1),
                    phases=[dict(name=str(cap),cache_cli=cap,
                                 ready=dict(method='GET',image_warmup=0,text_warmup=0),
                                 passes=[dict(label=label,argv=['eval','--retries','1','--scope','full'],
                                              score_argv=['score','--verify-requests'])
                                         for label in ('cold-first','natural-second')]) for cap in caps])

    def test_explicit_reverse_and_legacy_default_plan_only_report_actual_order(self):
        for caps in (v.CAPACITIES, (131072,16384), (131072,262144)):
            plan = self.order_plan(caps)
            if caps != v.CAPACITIES: plan['arm_order'] = list(caps)
            path = self.root/'plan.json'; path.write_text(json.dumps(plan))
            out = io.StringIO()
            with mock.patch.object(v.sys,'argv',['vision_cache_window.py','--plan',str(path),
                 '--plan-sha',self.f.sha256(path),'--frozen',self.f.__file__]), contextlib.redirect_stdout(out):
                self.assertEqual(v.main(),0)
            self.assertEqual(json.loads(out.getvalue())['capacities'], list(caps))

    def test_bad_arm_order_or_duplicate_phase_rejected_before_any_preflight(self):
        bad=[]
        for order in ([], None, '131072,16384', [0,16384], [16384,131072], [131072,131072],
                      [262144,131072,16384], [131072.0,16384], [True,16384],
                      [262144,131072], [131072,524288], [131072,262144,262144]):
            plan = self.order_plan((131072,16384)); plan['arm_order']=order; bad.append(plan)
        bad.append(self.order_plan((131072,16384)))  # reverse must be explicit
        bad.append(self.order_plan((131072,262144)))  # fullset must be explicit
        plan=self.order_plan((262144,131072)); plan['arm_order']=[131072,262144]; bad.append(plan)
        plan=self.order_plan(); plan['arm_order']=[131072,16384]; bad.append(plan)
        plan=self.order_plan(); plan['phases'][1]['name']=plan['phases'][0]['name']; bad.append(plan)
        plan=self.order_plan(); plan['phases'][1]['cache_cli']=131072.0; bad.append(plan)
        with mock.patch.object(v,'check_pins') as pins, mock.patch.object(v,'bound_root') as bound:
            for plan in bad:
                with self.subTest(plan=plan), self.assertRaises(RuntimeError):
                    v.execute(self.f,plan,ident(),[],'frozen')
            pins.assert_not_called(); bound.assert_not_called()

    def test_optional_quality_policy_is_fixed_and_legacy_absence_remains_valid(self):
        plan = self.order_plan()
        v.validate_plan(plan)
        plan['quality_gate'] = dict(QUALITY); v.validate_plan(plan)
        for key, value in (('baseline_correct',216),('total',399),('max_drop_pp',3),
                           ('min_correct',210),('min_correct',211.0)):
            bad = copy.deepcopy(plan); bad['quality_gate'][key]=value
            with self.subTest(key=key,value=value), self.assertRaisesRegex(RuntimeError,'quality policy'):
                v.validate_plan(bad)
        for quality in (None, {}, {'enabled':False}):
            plan['quality_gate']=quality
            with self.assertRaisesRegex(RuntimeError,'quality policy'): v.validate_plan(plan)

    def test_erqa_adopted_flags_and_explicit_zero_reuse_existing_launcher(self):
        from test_inference_launch import LaunchTests
        case = LaunchTests(); case.setUp(); self.addCleanup(case.doCleanups)
        environment = dict(v.os.environ)
        def launch(*args):
            # Match each real env-i launch; the launcher prepends vendor paths.
            with mock.patch.dict(v.os.environ, environment, clear=True):
                return case.launch(*args)
        args=self.readme_erqa_args()
        chosen=launch(*args)
        effective=chosen['effective']
        self.assertEqual(effective['encoder_cache_tokens'],262144)
        self.assertEqual(effective['gpu_memory_utilization'],0.85)
        self.assertEqual(effective['max_model_len'],16384)
        self.assertEqual(effective['max_num_batched_tokens'],16384)
        self.assertEqual(effective['tensor_parallel_size'],2)
        self.assertEqual(json.loads(effective['limit_mm_per_prompt']),{'image':16,'video':0})
        self.assertEqual(json.loads(effective['mm_processor_kwargs'])['images_kwargs']['size'],
                         {'shortest_edge':65536,'longest_edge':1048576})
        self.assertEqual(v.option(chosen['argv'],'--scheduler-cls'),
                         'clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler')
        i=args.index('--encoder-cache-tokens')
        args[i+1]='0'
        args[args.index('--gpu-memory-utilization')+1]='0.9'
        fallback=launch(*args)
        self.assertNotIn('encoder_cache_tokens',fallback['effective'])
        self.assertNotIn('--scheduler-cls',fallback['argv'])
        self.assertEqual(fallback['effective']['gpu_memory_utilization'],0.9)
        self.assertEqual(fallback,launch(*args[:i],*args[i+2:]))
        for config in ('s1','s2'):
            self.assertEqual(launch('--config',config),launch('--config',config,'--encoder-cache-tokens','0'))

    def predictions_fixture(self, accuracy=None, missing_score=False):
        expected = [dict(id=f'erqa-{i:04}', record_index=i, images=[{'sha256': 'image'}], image_count=1)
                    for i in range(400)]
        rows = [dict(id=s['id'], record_index=s['record_index'], image_sha256=['image'], image_count=1,
                     status='ok', probabilities={'A':0.4, 'B':0.6}, option_count=2, prediction='B',
                     latency_ms=100.0, started_at='2026-11-20T00:00:00', finished_at='2026-11-20T00:00:01',
                     request_wire_sha256='sha', request_structural_sha256='struct', image_order=[0]) for s in expected]
        attempts = [dict(attempt=1, status_code=200, response_sha256='sha')]
        summary = dict(coverage=dict(fixed_total=400, ok=400, failed=0, retry_count=0),
                       run=dict(skipped_resumed=0, retries=1, scope='full'))
        score = dict(request_verification=dict(checked=True, passed=True, mismatches=[]), summary_comparison={'equal':True})
        if accuracy is not None: score['accuracy'] = summary['accuracy'] = accuracy
        def reader(path):
            path = Path(path)
            if path.name == 'record.json': return {'attempts':attempts}
            if path.name == 'response.json': return {'answers': {'answer': {'probabilities': {'A':0.4,'B':0.6}}}}
            if path.name == 'summary.json': return summary
            if path.name == 'erqa-score.json':
                if missing_score: raise FileNotFoundError(path)
                return score
            raise AssertionError(str(path))
        stack = ExitStack()
        stack.enter_context(mock.patch.object(v, 'json_lines', side_effect=lambda p: rows if Path(p).name=='predictions.jsonl' else attempts))
        stack.enter_context(mock.patch.object(self.f, 'read_json', side_effect=reader))
        stack.enter_context(mock.patch.object(self.f, 'sha256', return_value='sha'))
        stack.enter_context(mock.patch.object(Path, 'read_bytes', return_value=b'raw'))
        return stack, expected, rows, attempts, summary

    def test_full_erqa_originals_cold_warm_and_input_drift(self):
        stack, expected, rows, attempts, summary = self.predictions_fixture()
        reference = copy.deepcopy(rows)
        with stack:
            hashes = v.verify_predictions(self.f, self.root, expected, reference)
            self.assertEqual(v.verify_predictions(self.f, self.root, expected, hashes), hashes)
            rows[0]['request_structural_sha256'] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'fixed request identity drift'):
                v.verify_predictions(self.f, self.root, expected, hashes)

    def test_erqa_missing_duplicate_order_error_retry_probability_and_raw_failures(self):
        for mode in ('missing','duplicate','order','failed','retry','probability','raw','resume'):
            stack, expected, rows, attempts, summary = self.predictions_fixture()
            reference = copy.deepcopy(rows)
            with self.subTest(mode=mode), stack:
                if mode == 'missing': rows.pop()
                elif mode == 'duplicate': rows[1] = rows[0]
                elif mode == 'order': rows[1], rows[2] = rows[2], rows[1]
                elif mode == 'failed': rows[0]['status'] = 'failed'
                elif mode == 'retry': attempts.append(attempts[0])
                elif mode == 'probability': rows[0]['probabilities']['A'] = float('nan')
                elif mode == 'raw': rows[0]['request_wire_sha256'] = 'bad'
                else: summary['run']['skipped_resumed'] = 1
                with self.assertRaises(RuntimeError): v.verify_predictions(self.f, self.root, expected, reference)

    def test_quality_211_boundary_210_failure_and_bad_score_with_persisted_reason(self):
        for correct, total in ((211,400),(210,400),(216,400),(211,399)):
            stack, expected, rows, _, _ = self.predictions_fixture(
                dict(correct=correct,total=total,value=correct/total))
            with self.subTest(correct=correct,total=total), stack, mock.patch.object(self.f,'write_json') as write:
                if total==400 and correct>=211:
                    v.verify_predictions(self.f,self.root,expected,copy.deepcopy(rows),QUALITY)
                else:
                    with self.assertRaisesRegex(RuntimeError,'ERQA quality gate'):
                        v.verify_predictions(self.f,self.root,expected,copy.deepcopy(rows),QUALITY)
                record=write.call_args.args[1]
                self.assertEqual(record['baseline_correct'],219); self.assertEqual(record['min_correct'],211)
                self.assertEqual(record['actual_correct'],correct)
                self.assertEqual(record['passed'],total==400 and correct>=211)
                self.assertIn('reason',record)
        stack, expected, rows, _, _ = self.predictions_fixture(missing_score=True)
        with stack, self.assertRaises(FileNotFoundError):
            v.verify_predictions(self.f,self.root,expected,copy.deepcopy(rows),QUALITY)

    def test_quality_cold_or_natural_failure_blocks_C_and_restores_once(self):
        for failed_pass in ('cold-first','natural-second'):
            visited=[]; proof=[]
            def check(cap):
                for label in ('cold-first','natural-second'):
                    visited.append((cap,label)); correct=210 if label==failed_pass else 211
                    stack, expected, rows, _, _ = self.predictions_fixture(
                        dict(correct=correct,total=400,value=correct/400))
                    with stack, mock.patch.object(self.f,'write_json',side_effect=lambda p,r:proof.append(r)):
                        v.verify_predictions(self.f,self.root/label,expected,copy.deepcopy(rows),QUALITY)
            with self.subTest(failed_pass=failed_pass):
                result, events=self.window(order=(131072,262144),phase_check=check)
                self.assertFalse(result['ok']); self.assertNotIn(262144,events)
                self.assertEqual(result['restore_attempts'],1)
                self.assertEqual(visited[-1],(131072,failed_pass))
                self.assertFalse(proof[-1]['passed']); self.assertEqual(proof[-1]['actual_correct'],210)

    def test_get_only_then_cold_then_natural_second_in_same_process(self):
        for cold_failure in ('ok', 'cold-failure', 'score-equal', 'score-after', 'quality-cold', 'quality-natural'):
            directory = self.root / str(cold_failure); directory.mkdir()
            service_dir = directory/'service'
            events = []; alive = SimpleNamespace(pid=9000001, poll=lambda:None)
            phase = dict(cache_cli=16384, service_argv=['python','--output-dir',str(service_dir)],
                         launch_argv=['launch'], launch_cwd=str(directory), startup_timeout_s=300,
                         ready={'expected_body':{'status':'ready'}},
                         passes=[dict(label=label, argv=[label], score_argv=[label+'-score'],
                                      output_dir=str(directory/label), outer_wall_deadline_s=300, score_timeout_s=60)
                                 for label in ('cold-first','natural-second')])
            def spawn(name, argv, **kw):
                events.append(name)
                if name == 'launch':
                    service_dir.mkdir(); (service_dir/'launch-manifest.json').write_text('{}')
                    (service_dir/'service.log').write_text('fresh log')
                if name in ('cold-first','natural-second'): (directory/name).mkdir()
                return alive if name in ('launch','sampler') else SimpleNamespace(poll=lambda:0, returncode=0)
            arm = SimpleNamespace(directory=directory, spawn=spawn)
            def http(*args, **kw): events.append('GET'); return {'status':'ready'}
            actual_monitor = v.monitor
            def monitored(*args):
                if events[-1].endswith('-score') and cold_failure.startswith('score-'):
                    with mock.patch.object(v.time, 'monotonic', return_value=args[-1] + (0.1 if cold_failure=='score-after' else 0)):
                        return actual_monitor(*args)
                return [sample(1)]
            with self.subTest(failure=cold_failure), ExitStack() as s:
                s.enter_context(mock.patch.object(v.time, 'monotonic', return_value=1))
                s.enter_context(mock.patch.object(v, 'service_identity', return_value=ident()))
                s.enter_context(mock.patch.object(v, 'monitor', side_effect=monitored))
                s.enter_context(mock.patch.object(v, 'verify_startup', return_value={'constructor':constructor()}))
                s.enter_context(mock.patch.object(v, 'check_environment'))
                s.enter_context(mock.patch.object(v, 'strict_npu', return_value={'rows':[
                    dict(chip=c,pid=9000002+c,name='VLLMWorker_TP') for c in (0,1)]}))
                s.enter_context(mock.patch.object(self.f, 'http_json', side_effect=http))
                s.enter_context(mock.patch.object(self.f, 'scan', return_value={9000001:ident()}))
                s.enter_context(mock.patch.object(self.f, 'proc_identity', return_value=ident()))
                post = s.enter_context(mock.patch.object(self.f, 'post_json', side_effect=AssertionError('early POST')))
                verify = s.enter_context(mock.patch.object(v, 'verify_predictions', return_value=['inputs']))
                if cold_failure == 'cold-failure': verify.side_effect = RuntimeError('bad cold originals')
                if cold_failure.startswith('quality-'):
                    failed = 'cold-first' if cold_failure=='quality-cold' else 'natural-second'
                    def checked(frozen, output, expected, reference, quality):
                        self.assertEqual(quality,QUALITY)
                        if output.name==failed: raise RuntimeError('ERQA quality gate')
                        return ['inputs']
                    verify.side_effect=checked
                    with self.assertRaisesRegex(RuntimeError,'quality gate'):
                        v.run_phase(self.f,{'candidate_environment_expect':{},'quality_gate':QUALITY},
                                    phase,arm,[],None,1000,'frozen')
                    self.assertEqual(verify.call_count,1 if failed=='cold-first' else 2)
                    if failed=='cold-first': self.assertNotIn('natural-second',events)
                elif cold_failure == 'cold-failure':
                    with self.assertRaisesRegex(RuntimeError, 'bad cold'): v.run_phase(self.f, {'candidate_environment_expect':{}}, phase, arm, [], None, 1000, 'frozen')
                    self.assertNotIn('natural-second', events)
                elif cold_failure.startswith('score-'):
                    with self.assertRaisesRegex(RuntimeError, 'deadline'): v.run_phase(self.f, {'candidate_environment_expect':{}}, phase, arm, [], None, 1000, 'frozen')
                    self.assertNotIn('natural-second', events)
                else:
                    v.run_phase(self.f, {'candidate_environment_expect':{}}, phase, arm, [], None, 1000, 'frozen')
                    self.assertLess(events.index('GET'), events.index('cold-first'))
                    self.assertLess(events.index('cold-first-score'), events.index('natural-second'))
                post.assert_not_called(); self.assertEqual(events.count('launch'), 1)

    def sampler_round(self, mode):
        directory = self.root / mode; directory.mkdir()
        table = {9000001:ident(), 9000002:ident(9000002, ppid=9000001),
                 9000003:ident(9000003, ppid=9000001)}
        ready = not mode.startswith('startup')
        if ready:
            (directory/'ready.json').write_text(json.dumps({'workers':{'0':table[9000002],'1':table[9000003]}}))
        def command(argv, **kwargs):
            if argv == ['/usr/local/bin/npu-smi','info']:
                text = 'Process id in container\nNo running processes found in NPU 7'
                if mode == 'startup-no-header': text = 'No running processes found in NPU 7'
                if mode == 'startup-empty-stdout': text = ''
                if mode == 'startup-wrong-card': text = 'Process id in container\n|8 0|1|VLLMWorker_TP|0|9000002|'
                if mode == 'startup-unknown': text = 'Process id in container\n|7 0|1|VLLMWorker_TP|0|9000099|'
            elif argv[3] == 'usages': text = 'HBM Capacity(MB): 65536\nHBM Usage Rate(%): 91'
            else:
                chip = int(argv[-1]); pid = 9000002+chip
                text = f'Process id:{pid} Process name:VLLMWorker_TP Process memory(MB):56754\n'
                if chip == 0: text += 'Process id:9000001 Process name:python Process memory(MB):106\n'
                if mode == 'ready-missing-worker' and chip == 1: text = 'Process id:9000001 Process name:python Process memory(MB):106\n'
                if mode == 'ready-unknown': text += 'Process id:9000099 Process name:python Process memory(MB):106\n'
            return SimpleNamespace(returncode=0, stdout=text, stderr='')
        with mock.patch.object(v.subprocess, 'run', side_effect=command), \
                mock.patch.object(self.f, 'scan', return_value=table), \
                mock.patch.object(self.f, 'proc_identity', side_effect=table.get), \
                mock.patch.object(v.time, 'monotonic', return_value=100), \
                mock.patch.object(v.time, 'sleep', side_effect=RuntimeError('end of mock round')):
            try: rc = v.sample_loop(self.f, directory, ident())
            except RuntimeError as exc:
                self.assertEqual(str(exc), 'end of mock round'); rc = 0
        return rc, v.json_lines(directory/'hbm.jsonl')

    def test_startup_sampling_requires_proven_empty_header_and_card_and_pid(self):
        rc, rows = self.sampler_round('startup-empty')
        self.assertEqual(rc, 0); self.assertTrue(rows[0]['ok']); self.assertEqual(len(rows[0]['queries']), 3)
        for mode in ('startup-no-header','startup-empty-stdout','startup-wrong-card','startup-unknown'):
            with self.subTest(mode=mode):
                rc, rows = self.sampler_round(mode)
                self.assertEqual(rc, 1); self.assertFalse(rows[0]['ok'])

    def test_ready_sampling_accepts_owned_root_plus_one_worker_per_chip(self):
        rc, rows = self.sampler_round('ready-valid')
        self.assertEqual(rc, 0); self.assertEqual(len(rows[0]['chips'][0]['processes']), 2)
        self.assertEqual(len(rows[0]['queries']), 4)
        for mode in ('ready-missing-worker','ready-unknown'):
            with self.subTest(mode=mode):
                rc, rows = self.sampler_round(mode)
                self.assertEqual(rc, 1); self.assertFalse(rows[0]['ok'])

    def test_first_pass_wire_order_option_drift_fails_even_with_matching_local_hashes(self):
        for field, value in [('request_wire_sha256','changed'), ('image_order',[1]), ('option_count',3)]:
            stack, expected, rows, attempts, summary = self.predictions_fixture()
            reference = copy.deepcopy(rows)
            with self.subTest(field=field), stack:
                rows[0][field] = value
                with mock.patch.object(self.f, 'sha256', return_value=rows[0]['request_wire_sha256']):
                    with self.assertRaisesRegex(RuntimeError, 'fixed request identity drift'):
                        v.verify_predictions(self.f, self.root, expected, reference)

    def test_reference_remote_sha_size_400_order_required_without_local_fallback(self):
        path = self.root/'reference.jsonl'
        reference = [dict(id=f'erqa-{i:04}', record_index=i, image_count=1, option_count=2,
                          image_sha256=['image'], image_order=[0],
                          request_structural_sha256='struct', request_wire_sha256='wire') for i in range(400)]
        payload = ''.join(json.dumps(r)+'\n' for r in reference).encode()
        path.write_bytes(payload)
        import hashlib
        spec = dict(remote_path=str(path), local_path=str(path), bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest())
        plan = {'evaluation_frozen':{'request_reference':spec}}
        self.assertEqual(v.read_reference(self.f, plan), reference)
        spec['sha256'] = 'bad'
        with self.assertRaisesRegex(RuntimeError, 'SHA/bytes'): v.read_reference(self.f, plan)
        spec['sha256'] = hashlib.sha256(payload).hexdigest()
        spec['remote_path'] = str(self.root/'missing')
        with self.assertRaises(FileNotFoundError): v.read_reference(self.f, plan)
        spec['remote_path'] = str(path)
        reference.reverse(); payload = ''.join(json.dumps(r)+'\n' for r in reference).encode()
        path.write_bytes(payload); spec['sha256'] = hashlib.sha256(payload).hexdigest()
        with self.assertRaisesRegex(RuntimeError, 'coverage/order'): v.read_reference(self.f, plan)

    def test_external_erqa_runners_block_shared_gates_without_signal(self):
        for name in ('erqa_baseline.py','erqa_score.py'):
            row = ident(9000009, cmdline=['python','/outside/'+name])
            with self.subTest(name=name), mock.patch.object(self.f, 'scan', return_value={row['pid']:row}):
                self.assertEqual(v.pending_runners(self.f, {row['pid']:row}), [row['pid']])
                with self.assertRaisesRegex(RuntimeError, 'pending vision runner'):
                    v.final_restore_gate(self.f, {'port':18143}, ident(), [], True, {})

    def test_cleanup_npu_proof_at_or_after_original_deadline_cannot_mark_clean(self):
        for finished in (299.95, 300.0, 300.1):
            arm = self.arm(); now = [0.0]
            with self.subTest(finished=finished), ExitStack() as s:
                _, settle = self.close_mocks(s)
                s.enter_context(mock.patch.object(v.time, 'monotonic', side_effect=lambda:now[0]))
                def settled(*a, **kw):
                    now[0] = 299.9
                    return dict(clean=True, deadline_hit=False, residual=[], table={})
                settle.side_effect = settled
                def npu(*a, **kw):
                    self.assertEqual(kw['deadline'], 300)
                    now[0] = finished
                    return {'rows':[]}
                s.enter_context(mock.patch.object(v, 'strict_npu', side_effect=npu))
                self.assertEqual(arm.close(18143), finished < 300)
                self.assertEqual(arm.cleanup_deadline, 300)
                if finished >= 300:
                    with self.assertRaisesRegex(RuntimeError, 'unclean arm'):
                        v.final_restore_gate(self.f, {}, ident(), [arm], True, {})

    def test_npu_command_timeout_uses_remaining_budget_and_rechecks_return_time(self):
        for finished in (299.95, 300.0, 300.1):
            with self.subTest(finished=finished), \
                    mock.patch.object(v.time, 'monotonic', side_effect=[299.9, finished]), \
                    mock.patch.object(v.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=0, stdout='Process id in container\nNo running processes found in NPU 7', stderr='')) as run:
                if finished < 300: v.strict_npu(self.f, deadline=300)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'deadline'): v.strict_npu(self.f, deadline=300)
                self.assertAlmostEqual(run.call_args.kwargs['timeout'], 0.1)

    def test_final_restore_complete_at_deadline_cannot_pass_or_retry(self):
        for finished in (99.9, 100.0, 100.1):
            output = self.root/str(finished); output.mkdir(); path=output/'launch-manifest.json'; path.write_text('{}')
            live=ident(9000008, ppid=1)
            manifest=dict(output_dir=str(output), config='s1', argv=['python'], effective={}, pid=9000008)
            restore=dict(restore_output=str(output), restore_source={'selected':{'argv':['python']}},
                         clean_env_expect='original.json', text_body=str(path), image_body=str(path))
            plan=dict(cleanup_restore=restore,current_service={'root':9000001},restore_environment_expect={})
            result={'restore':{'wrapper_pid':99},'restore_attempts':1}; now=[5.0]
            with self.subTest(finished=finished), ExitStack() as s:
                s.enter_context(mock.patch.object(v.time,'monotonic',side_effect=lambda:now[0]))
                s.enter_context(mock.patch.object(self.f,'http_json',side_effect=lambda b,p,**kw:
                    {'status':'ready','model':'clef'} if p=='/health' else {'config':'s1'}))
                s.enter_context(mock.patch.object(self.f,'read_json',side_effect=lambda p:
                    manifest if Path(p).name=='launch-manifest.json' else {'branch':'aa3e-noop'}))
                for name, value in [('wait_restored_root',live),('proc_identity',live),('read_root_env',{}),
                                    ('joint_env',{'ok':True}),('env_evidence',{'match':True}),
                                    ('scan',{9000008:live}),('pending_clients',[]),('post_json',(200,{}))]:
                    s.enter_context(mock.patch.object(self.f,name,return_value=value))
                for name in ('bound_root','check_environment','strict_npu'):
                    s.enter_context(mock.patch.object(v,name,return_value={}))
                s.enter_context(mock.patch.object(v,'check_pins',side_effect=lambda *a:now.__setitem__(0,finished)))
                if finished < 100:
                    v.verify_restore(self.f,plan,result,0,100); self.assertIn('restore_checks',result)
                else:
                    with self.assertRaisesRegex(RuntimeError,'completion deadline'): v.verify_restore(self.f,plan,result,0,100)
                    self.assertNotIn('restore_checks',result)
                    with self.assertRaisesRegex(RuntimeError,'already consumed'): v.restore_once(self.f,{},result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
