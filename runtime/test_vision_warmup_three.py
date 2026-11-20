"""CPU boundaries for native baseline + one warmup and three measured ERQA passes."""
import contextlib
import copy
from contextlib import ExitStack
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import test_vision_cache_window as prior
import vision_cache_window as v


class WarmupThreeCPU(unittest.TestCase):
    setUp = prior.VisionCPU.setUp  # Blocks every unmocked process launch and signal.
    readme_erqa_args = prior.VisionCPU.readme_erqa_args
    predictions_fixture = prior.VisionCPU.predictions_fixture

    @classmethod
    def setUpClass(cls):
        cls.f = v.load_frozen(Path(__file__).with_name('grouped_mmar') / 'supervised_probe_window.py')

    def plan(self):
        plan = prior.VisionCPU.order_plan(self, (0, 262144))
        plan.update(evaluation_mode='warmup-three', arm_order=[0, 262144],
                    quality_gate=dict(prior.QUALITY))
        plan['memory_guard'].update(runtime_min_free_bytes=2*v.GIB,
                                   repeat_admission=copy.deepcopy(v.REPEAT_ADMISSION))
        plan['budgets_s'] = dict(initial_stop=120, initial_settle_total=300, startup_each_arm=300,
            each_400_client_outer=300, original_per_http_timeout=600, each_score=60, active_each_arm=1740,
            cleanup_each_arm=120, settle_each_arm_total=300, each_arm_including_cleanup=2160,
            all_arms_child_total_including_cleanup=4320, final_restore_total_reserved=900,
            outer_recording_margin=120, overall_outer=5760, unused_removed_third_arm_budget=0)
        for phase in plan['phases']:
            cap = phase['cache_cli']
            phase.update(max_arm_active_s=1740, startup_timeout_s=300, cleanup_timeout_s=120,
                settle_timeout_s=300, max_arm_with_cleanup_s=2160,
                startup_gate={'cache_mode': 'native' if cap == 0 else 'wrapper'},
                service_argv=['launch', '--encoder-cache-tokens', str(cap),
                              '--gpu-memory-utilization', '0.9' if cap == 0 else '0.85'])
            phase['passes'] = []
            for label in v.REPEAT_LABELS:
                output = str(self.root / str(cap) / label)
                phase['passes'].append(dict(label=label, expected_count=400, output_dir=output,
                    outer_wall_deadline_s=300, score_timeout_s=60,
                    argv=['eval', '--retries', '1', '--scope', 'full', '--output-dir', output],
                    score_argv=['score', '--verify-requests', '--run-dir', output,
                                '--output', str(Path(output)/'erqa-score.json')]))
        return plan

    def test_repeat_plan_only_and_legacy_orders(self):
        plans = [self.plan()]
        for caps in (v.CAPACITIES, (131072, 16384), (131072, 262144)):
            plan = prior.VisionCPU.order_plan(self, caps)
            if caps != v.CAPACITIES:
                plan['arm_order'] = list(caps)
            plans.append(plan)
        for plan in plans:
            with self.subTest(order=plan.get('arm_order')):
                path = self.root/'plan.json'; self.f.write_json(path, plan)
                out = io.StringIO()
                with mock.patch.object(v.sys, 'argv', ['vision_cache_window.py', '--plan', str(path),
                     '--plan-sha', self.f.sha256(path), '--frozen', self.f.__file__]), \
                     mock.patch.object(v, 'execute') as execute, contextlib.redirect_stdout(out):
                    self.assertEqual(v.main(), 0)
                execute.assert_not_called()
                self.assertEqual(json.loads(out.getvalue())['capacities'], list(v.planned_arm_order(plan)))

    def test_bad_repeat_contract_rejected(self):
        mutations = {
            'implicit-mode': lambda p: p.pop('evaluation_mode'),
            'unknown-mode': lambda p: p.update(evaluation_mode='warmup-four'),
            'implicit-order': lambda p: p.pop('arm_order'),
            'wrong-order': lambda p: p.update(arm_order=[262144, 0]),
            'missing-quality': lambda p: p.pop('quality_gate'),
            'weak-quality': lambda p: p['quality_gate'].update(min_correct=210),
            'missing-admission': lambda p: p['memory_guard'].pop('repeat_admission'),
            'false-admission': lambda p: p['memory_guard']['repeat_admission'].update(capacity=131072),
            'lower-reserve': lambda p: p['memory_guard'].update(runtime_min_free_bytes=v.GIB),
            'missing-warmup': lambda p: p['phases'][0]['passes'].pop(0),
            'swapped-passes': lambda p: p['phases'][0]['passes'].reverse(),
            'duplicate-output': lambda p: p['phases'][1]['passes'][0].update(
                output_dir=p['phases'][0]['passes'][0]['output_dir']),
            'bad-mode': lambda p: p['phases'][0]['startup_gate'].update(cache_mode='wrapper'),
            'bad-util': lambda p: p['phases'][0]['service_argv'].__setitem__(-1, '0.85'),
            'bad-cache': lambda p: p['phases'][0]['service_argv'].__setitem__(2, '16384'),
            'short-pass': lambda p: p['phases'][0]['passes'][0].update(expected_count=399),
            'foreign-eval-output': lambda p: p['phases'][0]['passes'][0]['argv'].__setitem__(-1, '/other'),
            'foreign-score-output': lambda p: p['phases'][0]['passes'][0]['score_argv'].__setitem__(-1, '/other'),
            'global-budget': lambda p: p['budgets_s'].update(overall_outer=6000),
            'arm-budget': lambda p: p['phases'][0].update(max_arm_active_s=1800),
            'score-budget': lambda p: p['phases'][0]['passes'][0].update(score_timeout_s=61),
            'retry': lambda p: p['phases'][0]['passes'][0]['argv'].__setitem__(2, '2'),
        }
        for label, mutate in mutations.items():
            plan = self.plan(); mutate(plan)
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                v.validate_plan(plan)

    def startup_fixture(self, cap=0):
        from test_inference_launch import LaunchTests
        case = LaunchTests(); case.setUp(); self.addCleanup(case.doCleanups)
        args = self.readme_erqa_args()
        args[args.index('--encoder-cache-tokens')+1] = str(cap)
        args[args.index('--gpu-memory-utilization')+1] = '0.9' if cap == 0 else '0.85'
        manifest = case.launch(*args)
        plan = dict(evaluation_mode='warmup-three', cleanup_restore={'source_target':'aa3e.py'})
        phase = dict(cache_cli=cap, startup_gate={'cache_mode':'native' if cap == 0 else 'wrapper'},
                     service_argv=list(manifest['argv']))
        digest = manifest['effective'].get('encoder_cache_scheduler_sha256')
        if cap:
            phase['startup_gate'].update(module_path=str(Path(__file__).parents[1]/'Clef-27B'/'clef_encoder_cache_scheduler.py'),
                module_sha256=digest, scheduler_cls='clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler')
        text = ('(EngineCore_DP0 pid=9000002) Initializing a V1 LLM engine with config: '
                'ObservabilityConfig(enable_mm_processor_stats=False, enable_logging_iteration_details=False)\n')
        if cap:
            text += '(EngineCore_DP0 pid=9000002) ' + prior.log(prior.constructor(cap))
        live = {key: value.encode() for key, value in manifest['service_environment'].items()}
        stack = ExitStack()
        stack.enter_context(mock.patch.object(self.f, 'scan', return_value={
            9000001: prior.ident(), 9000002: prior.ident(9000002, ppid=9000001)}))
        stack.enter_context(mock.patch.object(self.f, 'sha256', side_effect=lambda path:
            v.AA3E if str(path) == 'aa3e.py' else digest))
        stack.enter_context(mock.patch.object(self.f, 'read_root_env', return_value=live))
        stack.enter_context(mock.patch.object(self.f, 'joint_env', return_value={'ok':True}))
        stack.enter_context(mock.patch.object(self.f, 'read_json', return_value={'branch':'aa3e-noop'}))
        return stack, plan, phase, manifest, live, text

    def test_native_actual_launcher_manifest_accepts_unknown_internal_cache(self):
        stack, plan, phase, manifest, _, text = self.startup_fixture()
        with stack:
            result = v.verify_startup(self.f, plan, phase, manifest, prior.ident(), text)
        self.assertEqual(result['cache_mode'], 'native')
        self.assertIsNone(result['constructor'])
        self.assertIsNone(result['module_sha256'])
        self.assertIsNone(result['native_cache']['cache_capacity'])
        self.assertIsNone(result['native_cache']['observed_class'])
        self.assertIsNone(result['native_cache']['compute_effective'])
        self.assertEqual(result['manifest']['effective']['gpu_memory_utilization'], 0.9)
        self.assertNotIn('--scheduler-cls', manifest['argv'])
        self.assertNotIn('encoder_cache_tokens', manifest['effective'])

    def test_adopted_repeat_actual_wrapper_and_effective_image_processor(self):
        stack, plan, phase, manifest, _, text = self.startup_fixture(262144)
        with stack:
            record = v.verify_startup(self.f, plan, phase, manifest, prior.ident(), text)
            self.assertEqual(record['cache_mode'], 'wrapper')
            self.assertEqual(record['constructor']['cache_effective'], 262144)
            self.assertEqual(record['constructor']['compute_effective'], 16384)
            self.assertEqual(record['manifest']['effective']['gpu_memory_utilization'], 0.85)
            for field, value in (('limit_mm_per_prompt', '{"image":1,"video":0}'),
                                 ('mm_processor_kwargs', '{"images_kwargs":{"size":{"shortest_edge":32768,"longest_edge":1048576}}}')):
                changed = copy.deepcopy(manifest); changed['effective'][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'effective image/processor'):
                    v.verify_startup(self.f, plan, phase, changed, prior.ident(), text)

    def test_native_rejects_wrapper_util_shape_and_foreign_engine_evidence(self):
        stack, plan, phase, original, live, text = self.startup_fixture()
        cases = ('scheduler', 'scheduler-equals', 'effective-cap', 'effective-sha', 'live-env',
                 'manifest-env', 'manifest-pythonpath', 'wrapper-log', 'missing-init', 'duplicate-init',
                 'timing-on', 'foreign-engine', 'util-argv', 'util-effective', 'compute-effective')
        with stack:
            for label in cases:
                manifest = copy.deepcopy(original); current_text = text; live.pop('CLEF_ENCODER_CACHE_TOKENS', None)
                if label == 'scheduler': manifest['argv'].extend(['--scheduler-cls', 'foreign.Scheduler'])
                elif label == 'scheduler-equals': manifest['argv'].append('--scheduler-cls=foreign.Scheduler')
                elif label == 'effective-cap': manifest['effective']['encoder_cache_tokens'] = 0
                elif label == 'effective-sha': manifest['effective']['encoder_cache_scheduler_sha256'] = 'sha'
                elif label == 'live-env': live['CLEF_ENCODER_CACHE_TOKENS'] = b'0'
                elif label == 'manifest-env': manifest['service_environment']['CLEF_ENCODER_CACHE_TOKENS'] = '0'
                elif label == 'manifest-pythonpath': manifest['service_environment']['PYTHONPATH'] = '/candidate'
                elif label == 'wrapper-log': current_text += prior.log(prior.constructor())
                elif label == 'missing-init': current_text = text.replace('Initializing a V1 LLM engine', 'old log')
                elif label == 'duplicate-init': current_text *= 2
                elif label == 'timing-on': current_text = text.replace('enable_mm_processor_stats=False', 'enable_mm_processor_stats=True')
                elif label == 'foreign-engine': current_text = text.replace('9000002', '9000099')
                elif label == 'util-argv': manifest['argv'][manifest['argv'].index('--gpu-memory-utilization')+1] = '0.85'
                elif label == 'util-effective': manifest['effective']['gpu_memory_utilization'] = 0.85
                elif label == 'compute-effective': manifest['effective']['max_num_batched_tokens'] = 8192
                with self.subTest(label=label), self.assertRaises(RuntimeError):
                    v.verify_startup(self.f, plan, phase, manifest, prior.ident(), current_text)

    def summaries(self, directory):
        phase = {'passes': []}
        for label, mean, p95, wall in zip(v.REPEAT_LABELS, (9999, 10, 30, 20), (9999, 80, 60, 100), (9999, 3, 1, 2)):
            output = directory/label; output.mkdir(parents=True)
            phase['passes'].append(dict(label=label, output_dir=str(output)))
            self.f.write_json(output/'summary.json', dict(latency_ms={'mean':mean, 'p95':p95},
                run={'wall_seconds':wall}, accuracy={'correct':216}))
        return phase

    def test_summary_excludes_warmup_and_uses_three_per_pass_metrics(self):
        phase = self.summaries(self.root/'summaries')
        record = v.summarize_repeats(self.f, phase)
        self.assertEqual(record['warmup']['mean_ms'], 9999)
        self.assertEqual([r['label'] for r in record['measurements']], list(v.REPEAT_LABELS[1:]))
        self.assertEqual(record['aggregate'], {'mean_ms': {'median':20, 'range':[10,30]},
            'p95_ms': {'median':80, 'range':[60,100]}, 'wall_s': {'median':2, 'range':[1,3]}})

    def test_summary_rejects_incomplete_order_and_invalid_metrics(self):
        phase = self.summaries(self.root/'summaries')
        bad = copy.deepcopy(phase); bad['passes'].pop()
        with self.assertRaises(RuntimeError): v.summarize_repeats(self.f, bad)
        for value in (float('nan'), float('inf'), 0, -1, True, '10'):
            path = Path(phase['passes'][2]['output_dir'])/'summary.json'
            summary = self.f.read_json(path); summary['latency_ms']['mean'] = value
            self.f.write_json(path, summary)
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                v.summarize_repeats(self.f, phase)
        path.unlink()
        with self.assertRaises(FileNotFoundError): v.summarize_repeats(self.f, phase)

    def phase_run(self, failed_label=None):
        directory = self.root/('phase-'+str(failed_label)); directory.mkdir()
        service_dir = directory/'service'; events = []; checks = []
        alive = SimpleNamespace(pid=9000001, poll=lambda:None)
        phase = dict(cache_cli=0, service_argv=['launch','--output-dir',str(service_dir)],
            launch_argv=['launch'], launch_cwd=str(directory), startup_timeout_s=300,
            ready={'expected_body':{'status':'ready'}}, passes=[
                dict(label=label, argv=[label], score_argv=[label+'-score'],
                     output_dir=str(directory/label), outer_wall_deadline_s=300, score_timeout_s=60)
                for label in v.REPEAT_LABELS])
        def spawn(name, argv, **kwargs):
            events.append(name)
            if name == 'launch':
                service_dir.mkdir(); self.f.write_json(service_dir/'launch-manifest.json', {})
                (service_dir/'service.log').write_text('fresh log')
            if name in v.REPEAT_LABELS:
                (directory/name).mkdir()
                self.f.write_json(directory/name/'summary.json', dict(latency_ms={'mean':10,'p95':20},
                    run={'wall_seconds':4}, accuracy={'correct':211}))
            return alive if name in ('launch','sampler') else SimpleNamespace(poll=lambda:0, returncode=0)
        def checked(frozen, output, expected, reference, quality):
            self.assertEqual(quality, prior.QUALITY); checks.append(output.name)
            if output.name == failed_label: raise RuntimeError('ERQA quality gate')
            return ['fixed']
        def http(*args, **kwargs): events.append('GET'); return {'status':'ready'}
        with ExitStack() as s:
            s.enter_context(mock.patch.object(v.time, 'monotonic', return_value=1))
            s.enter_context(mock.patch.object(v, 'service_identity', return_value=prior.ident()))
            s.enter_context(mock.patch.object(v, 'monitor', return_value=[prior.sample(1)]))
            s.enter_context(mock.patch.object(v, 'verify_startup', return_value={'cache_mode':'native','constructor':None}))
            s.enter_context(mock.patch.object(v, 'verify_predictions', side_effect=checked))
            s.enter_context(mock.patch.object(v, 'check_environment'))
            s.enter_context(mock.patch.object(v, 'strict_npu', return_value={'rows':[
                dict(chip=c,pid=9000002+c,name='VLLMWorker_TP') for c in (0,1)]}))
            s.enter_context(mock.patch.object(self.f, 'http_json', side_effect=http))
            s.enter_context(mock.patch.object(self.f, 'scan', return_value={9000001:prior.ident()}))
            s.enter_context(mock.patch.object(self.f, 'proc_identity', return_value=prior.ident()))
            post = s.enter_context(mock.patch.object(self.f, 'post_json', side_effect=AssertionError('early POST')))
            args = (self.f, {'evaluation_mode':'warmup-three', 'candidate_environment_expect':{},
                    'quality_gate':prior.QUALITY}, phase, SimpleNamespace(directory=directory,spawn=spawn),
                    [], ['fixed'], 1741, 'frozen')
            if failed_label:
                with self.assertRaisesRegex(RuntimeError, 'quality gate'): v.run_phase(*args)
                self.assertFalse((directory/'evaluation-summary.json').exists())
            else:
                v.run_phase(*args)
                self.assertTrue((directory/'evaluation-summary.json').exists())
            post.assert_not_called()
        self.assertEqual(events.count('launch'), 1)
        self.assertLess(events.index('GET'), events.index('warmup'))
        return events, checks

    def test_get_only_warmup_then_three_measurements_share_one_service(self):
        events, checks = self.phase_run()
        self.assertEqual(checks, list(v.REPEAT_LABELS))
        for earlier, later in zip(v.REPEAT_LABELS, v.REPEAT_LABELS[1:]):
            self.assertLess(events.index(earlier+'-score'), events.index(later))

    def test_warmup_or_any_measurement_failure_stops_remaining_passes(self):
        for i, label in enumerate(v.REPEAT_LABELS):
            with self.subTest(label=label):
                events, checks = self.phase_run(label)
                self.assertEqual(checks, list(v.REPEAT_LABELS[:i+1]))
                for later in v.REPEAT_LABELS[i+1:]: self.assertNotIn(later, events)

    def window(self, fail=None, fail_close=None):
        directory = self.root/('window-'+str(len(list(self.root.iterdir())))); events = []
        plan = self.plan(); plan.update(run_root=str(directory), current_service={'root':9000001},
                                      restore_environment_expect={})
        plan['cleanup_restore'].update(restore_output=str(directory/'restore'), source_target='aa3e.py',
            restore_source={'selected':{'argv':['python']}}, stop_argv_template=['stop'])
        class Arm:
            def __init__(this, frozen, path, protected):
                this.directory=path; this.closed=this.clean=False; this.extra_entries=[]
            def entries(this): return this.extra_entries
            def spawn(this, name, argv):
                events.append(name); return SimpleNamespace(wait=lambda **kw:0)
            def close(this, port):
                if not this.closed: events.append('close:'+this.directory.name)
                this.closed=True; this.clean=this.directory.name != fail_close; return this.clean
        def run(frozen, plan, phase, arm, expected, reference, deadline, path):
            cap = phase['cache_cli']; events.append(cap)
            for label in v.REPEAT_LABELS:
                events.append((cap,label)); correct=210 if fail==(cap,label) else 211
                stack, expected, rows, _, _ = self.predictions_fixture(
                    dict(correct=correct,total=400,value=correct/400))
                output = arm.directory/label; output.mkdir()
                with stack:
                    v.verify_predictions(self.f, output, expected, copy.deepcopy(rows), prior.QUALITY)
            self.f.write_json(arm.directory/'evaluation-summary.json', {'aggregate':{
                key:dict(median=100 if cap==0 else 80, range=[75,105])
                for key in ('mean_ms','p95_ms','wall_s')}})
            # Native baseline margin cannot stand in for prior131k evidence; repeated adopted is independently authorized.
            return reference, [2*v.GIB+1, 2*v.GIB+1]
        def gate(frozen, plan, saved, arms, stopped, result):
            v.require(stopped and all(arm.closed and arm.clean for arm in arms), 'closed gate')
        with ExitStack() as s:
            for name, value in [('get_subreaper',0), ('enable_subreaper',True), ('restore_subreaper',True),
                                ('scan',{9000001:prior.ident()}), ('pending_clients',[]), ('port_open',0),
                                ('begin_restore',True)]:
                s.enter_context(mock.patch.object(self.f,name,return_value=value))
            for name in ('check_pins','bound_root','strict_npu','check_environment','verify_restore'):
                s.enter_context(mock.patch.object(v,name,return_value={}))
            s.enter_context(mock.patch.object(v,'read_reference',return_value=['fixed']))
            s.enter_context(mock.patch.object(v,'OwnedArm',Arm))
            s.enter_context(mock.patch.object(v,'run_phase',side_effect=run))
            s.enter_context(mock.patch.object(v,'final_restore_gate',side_effect=gate))
            rc = v.execute(self.f,plan,prior.ident(),[],'frozen')
            result=self.f.read_json(directory/'window.json')
            self.assertEqual(rc,0 if result['ok'] else 1)
            self.assertEqual(self.f.begin_restore.call_count,result['restore_attempts'])
        return result, events

    def test_repeat_adopted_admission_is_independent_of_native_minimum(self):
        result, events = self.window()
        self.assertTrue(result['ok']); self.assertEqual(result['arm_order'],[0,262144])
        self.assertEqual([e for e in events if type(e) is int],[0,262144])
        self.assertEqual(result['restore_attempts'],1)
        self.assertEqual(events.count('stop'),1)
        self.assertEqual(set(result['by_capacity']),{'0','262144'})
        for value in result['comparison']['median_delta_pct'].values(): self.assertAlmostEqual(value,-20)
        self.assertEqual(events.count('close:0'),1); self.assertEqual(events.count('close:262144'),1)

    def test_quality_210_in_any_pass_aborts_next_work_and_restores_at_most_once(self):
        for cap in (0,262144):
            for label in v.REPEAT_LABELS:
                with self.subTest(cap=cap,label=label):
                    result, events=self.window(fail=(cap,label))
                    self.assertFalse(result['ok']); self.assertNotIn('comparison',result)
                    self.assertEqual(result['restore_attempts'],1)
                    evaluated=[e for e in events if type(e) is tuple]
                    self.assertEqual(evaluated[-1],(cap,label))
                    if cap == 0: self.assertNotIn(262144,events)
        result, events=self.window(fail_close='0')
        self.assertFalse(result['ok']); self.assertEqual(result['restore_attempts'],0)
        self.assertNotIn(262144,events)

    def test_native_monitor_keeps_hbm_identity_and_no_wrapper_gates(self):
        service_dir=self.root/'service'; service_dir.mkdir()
        (service_dir/'service.log').write_text('native service')
        self.f.write_json(self.root/'ready.json', dict(cache_mode='native', constructor=None,
            engine_identity=prior.ident(9000002),workers={'0':prior.ident(9000003),'1':prior.ident(9000004)}))
        alive=SimpleNamespace(poll=lambda:None); arm=SimpleNamespace(directory=self.root)
        with mock.patch.object(v.time,'monotonic',return_value=2), \
             mock.patch.object(self.f,'proc_identity',side_effect=lambda pid:prior.ident(pid)), \
             mock.patch.object(v,'json_lines',return_value=[prior.sample(1,free=2*v.GIB+1)]):
            self.assertEqual(len(v.monitor(self.f,arm,alive,alive,prior.ident(),service_dir,1,10)),1)
            (service_dir/'service.log').write_text(prior.log(prior.constructor()))
            with self.assertRaisesRegex(RuntimeError,'native wrapper'):
                v.monitor(self.f,arm,alive,alive,prior.ident(),service_dir,1,10)
            (service_dir/'service.log').write_text('native service')
            with mock.patch.object(v,'json_lines',return_value=[prior.sample(1,free=2*v.GIB-1)]):
                with self.assertRaises(RuntimeError): v.monitor(self.f,arm,alive,alive,prior.ident(),service_dir,1,10)


if __name__ == '__main__':
    unittest.main()
