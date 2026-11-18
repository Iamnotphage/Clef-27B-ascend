#!/usr/bin/env python3
"""Run frozen acceptance phases against an owner-managed, already-ready service.

No service lifecycle, source install, package import or device code. Default is
plan-only. --execute is reserved for the device owner after its resource gate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from fractions import Fraction
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import urllib.request

CODE = Path(__file__).resolve().parents[1]
FIELDS = ('pid', 'ppid', 'pgid', 'starttime', 'boot_id', 'cmdline')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def verify_file(item):
    p = Path(item['path'])
    require(p.is_file() and sha(p) == item['sha256'], f'file identity mismatch: {p}')


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().split(') ', 1)[1].split()
    argv = (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]),
            'starttime': fields[19], 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'state': fields[0], 'cmdline': ' '.join(argv), 'argv': argv}


def scan_proc():
    found = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                found[int(p.name)] = identity(int(p.name))
            except FileNotFoundError:
                # A process that exited during enumeration has no live identity.
                continue
    return found  # Permission and parse failures propagate; no partial success.


def descendants(all_procs, root):
    result, frontier = set(), {root}
    while frontier:
        children = {p for p, v in all_procs.items() if v['ppid'] in frontier} - result
        result |= children
        frontier = children
    return result


def get_json(base, suffix):
    with urllib.request.urlopen(base + suffix, timeout=10) as response:
        return json.loads(response.read())


def service_check(m, phase, identities, launch_path):
    saved = read(identities)
    if isinstance(saved, dict):
        saved = [saved['root'], *saved['descendants']]
    require(isinstance(saved, list) and len(saved) >= 3, 'ready root + descendants required')
    require(all(all(k in s for k in FIELDS) for s in saved), 'incomplete saved identity')
    root = saved[0]
    require(type(root['pid']) is int and type(root['pgid']) is int
            and type(root['starttime']) is str, 'invalid root identity types')
    all_procs = scan_proc()
    for s in saved:
        live = all_procs.get(s['pid'])
        require(live and live['state'] != 'Z' and all(live[k] == s[k] for k in FIELDS),
                f'live identity changed: {s["pid"]}')
    require(descendants(all_procs, root['pid']) == {s['pid'] for s in saved[1:]},
            'descendant set changed; owner must capture fresh ready identities')
    live_root = all_procs[root['pid']]
    expected = m['phases'][phase]
    require(live_root['argv'] == expected['expected_argv'], 'runtime argv differs from frozen phase')
    launch = read(launch_path)
    require(launch['pid'] == root['pid'] and launch['argv'] == live_root['argv'], 'launch/root mismatch')
    require(launch['effective'] == expected['effective'] and launch['config'] == 's1', 'effective config mismatch')
    label_dir = Path(launch['output_dir'])
    require(label_dir.name != 'final-default-s1', 'retired label forbidden')
    require(int((label_dir / 'api.pid').read_text()) == root['pid'], 'api.pid mismatch')
    raw_env = (Path('/proc') / str(root['pid']) / 'environ').read_bytes()
    env = dict(part.decode().split('=', 1) for part in raw_env.split(b'\0') if b'=' in part)
    for key, value in expected['service_environment'].items():
        require(env.get(key) == value, f'service environment mismatch: {key}')
    forbidden = m.get('forbidden_environment', [])
    require(not any(env.get(k) not in (None, '', '0') for k in forbidden), 'unfrozen candidate/profile/cache environment')
    with socket.socket() as s:
        s.settimeout(5)
        code = s.connect_ex(('127.0.0.1', 18143))
    require(code == 0, f'port connect_ex={code}; 111=closed, other nonzero=unknown')
    health = get_json(m['base_url'], '/health')
    info = get_json(m['base_url'], '/v1/decide/info')
    require(health.get('status') == 'ready' and health.get('model') == 'clef', 'health not ready')
    require(info['revision'] == launch['revision'] == health['revision'], 'model revision mismatch')
    require(info['config'] == 's1' and info['tensor_parallel_size'] == 2
            and info['max_model_len'] == expected['effective']['max_model_len'], 'info config mismatch')
    require(info['cache'] == {'prefix_caching': False, 'chunked_prefill': False,
                             'schema_token_cache': False, 'schema_token_cache_version': 0}, 'cache mismatch')
    return {'captured_utc': now(), 'root': live_root, 'descendants': saved[1:],
            'full_proc_scan_count': len(all_procs), 'port_connect_ex': code,
            'health': health, 'info': info, 'launch_manifest_sha256': sha(launch_path)}


def frozen_check(m):
    for name, version in m['distributions'].items():
        require(importlib.metadata.version(name) == version, f'distribution mismatch: {name}')
    for source in m['software_repositories']:
        head = subprocess.check_output(['git', '-C', source['path'], 'rev-parse', 'HEAD'], text=True).strip()
        require(head == source['head'], f'software git HEAD mismatch: {source["path"]}')
    for item in m['data'].values():
        if isinstance(item, dict) and 'sha256' in item:
            verify_file(item)
    for item in m['source_files']:
        verify_file(item)
    verify_file(m['model_manifest'])
    for rel, item in m['runner_files'].items():
        verify_file({'path': str(CODE / rel), 'sha256': item['sha256']})
    found = set()
    for row in m['model_expected_files']:
        name = row.get('Path', '')
        if name in m['required_model_files']:
            found.add(name)
            p = Path(m['model_dir']) / name
            require(p.is_file() and p.stat().st_size == row['Size'], f'model size mismatch: {name}')
            if p.suffix == '.json':
                require(sha(p) == row['Sha256'], f'model configuration hash mismatch: {name}')
    require(found == set(m['required_model_files']), 'model manifest lacks required files')
    # Frozen request/template alignment is checked before the first HTTP POST.
    data = read(m['data']['od_samples']['path'])
    samples = data['samples'] if isinstance(data, dict) else data
    with Path(m['data']['od_template']['path']).open() as f:
        templates = [json.loads(line) for line in f if line.strip()]
    require(len(samples) == len(templates) == 2352, 'OD count mismatch')
    require(all(s['request'] == t['model_output']['metadata']['choice_request']
                for s, t in zip(samples, templates)), 'OD template mismatch')


def commands(m, phase, out):
    py, d, p = m['python'], m['data'], m['phases'][phase]
    steps = []
    if phase == 'erqa':
        dest = out / 'erqa'
        steps.append(('erqa', [py, str(CODE / 'scripts/official_eval/erqa_baseline.py'),
            '--data-dir', d['erqa_dir'], '--output-dir', str(dest), '--api-url', m['base_url']+'/v1',
            '--model', 'clef', '--scope', 'full', '--retries', '3', '--retry-interval', '5', '--timeout', '600']))
        steps.append(('erqa-score', [py, str(CODE / 'scripts/official_eval/erqa_score.py'),
            '--run-dir', str(dest), '--data-dir', d['erqa_dir'], '--verify-requests',
            '--output', str(out / 'erqa-score.json')]))
    else:
        for c in p['concurrencies']:
            for n in range(1, p['rounds']+1):
                name = f'c{c}-r{n}'
                report = out / (name+'.json')
                steps.append((name, [py, str(CODE / 'runtime/od_full_benchmark.py'),
                    '--sample-file', d['od_samples']['path'], '--url', m['base_url']+'/v1/systemone',
                    '--concurrency', str(c), '--warmup', '0', '--timeout', '300',
                    '--report', str(report), '--journal', str(out / (name+'.jsonl'))]))
                steps.append((name+'-quality', [py, str(CODE / 'runtime/od_raw_acceptance.py'),
                    '--data', d['od_data']['path'], '--scorer', d['od_scorer']['path'],
                    '--template', d['od_template']['path'], '--run', 'candidate='+str(report),
                    '--reference', 'candidate', '--output', str(out / (name+'-quality.json'))]))
    return steps


def verify_od(path, m):
    d = read(path)
    rows = d['requests']
    require(len(rows) == 2352 and sorted(r['index'] for r in rows) == list(range(2352)), 'OD incomplete coverage')
    require(d['summary']['successful'] == 2352 and d['summary']['failed'] == 0, 'OD execution failures retained; run incomplete')
    samples = read(m['data']['od_samples']['path'])
    samples = samples['samples'] if isinstance(samples, dict) else samples
    for row in rows:
        require(row['status'] == 'ok' and row['request'] == samples[row['index']]['request'], 'OD response/request mismatch')
        probs = row['probabilities']
        require(probs and all(isinstance(v, (int,float)) and math.isfinite(v) and 0 <= v <= 1
                             for v in probs.values()), 'invalid raw probability')
    return d['summary']


def summarize(m, phase, out):
    if phase == 'erqa':
        score = read(out / 'erqa-score.json')
        run = read(out / 'erqa/summary.json')
        require(score['coverage']['fixed_total'] == score['coverage']['ok'] == 400
                and score['coverage']['failed'] == 0, 'ERQA incomplete coverage')
        require(score['request_verification']['checked'] and score['request_verification']['passed']
                and score['summary_comparison']['equal'], 'ERQA verification failed')
        require(score['accuracy']['total'] == 400, 'ERQA accuracy denominator mismatch')
        exact = Fraction(score['accuracy']['correct'], score['accuracy']['total'])
        baseline = Fraction(m['baseline_metrics']['erqa_correct'], m['baseline_metrics']['erqa_total'])
        delta = exact - baseline
        return {'accuracy': score['accuracy'], 'coverage': score['coverage'], 'breakdown': score['breakdown'],
                'retries': run['coverage']['retry_count'], 'delta_accuracy': float(delta),
                'quality_passed': abs(delta) <= Fraction(str(m['quality_gate']['erqa_absolute_accuracy']))}
    points=[]
    for c in m['phases'][phase]['concurrencies']:
        for n in range(1, m['phases'][phase]['rounds']+1):
            name=f'c{c}-r{n}'; perf=verify_od(out/(name+'.json'), m)
            quality=read(out/(name+'-quality.json'))['runs']['candidate']
            require(quality['order_mismatch']==0 and quality['protocol_failures']==0
                    and quality['accuracy_denominator']==2352 and quality['score_mae_denominator']==588,
                    'quality alignment/denominator mismatch')
            deltas={k:quality['scoring_result'][k]-m['baseline_metrics'][k]
                    for k in ('calibration','intelligence','capability')}
            points.append({'name':name,'concurrency':c,'performance':perf,'quality':quality,
                           'delta_points':deltas,'quality_passed':all(abs(v)<=2 for v in deltas.values())})
    result={'points':points,'quality_passed':all(p['quality_passed'] for p in points)}
    if phase=='s1':
        require(len(points)>=3, 'S1 requires >=3 complete rounds')
        values=[p['performance']['latency_ms']['p95'] for p in points]
        result.update(p95_median_ms=statistics.median(values),p95_range_ms=[min(values),max(values)])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=CODE/'runtime/final_acceptance_manifest.json')
    ap.add_argument('--phase',choices=('s1','s2_b8','s2_b16','erqa'),required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--identities',type=Path)
    ap.add_argument('--launch-manifest',type=Path)
    ap.add_argument('--owner-evidence',type=Path,help='27 resource/restore gate evidence; preserved verbatim, interpreted by owner')
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();m=read(a.manifest);out=a.output_dir.resolve()
    steps=commands(m,a.phase,out)
    if not a.execute:
        print(json.dumps({'status':'plan_not_executed','phase':a.phase,'steps':steps},ensure_ascii=False,indent=2));return 0
    require(a.identities and a.launch_manifest and a.owner_evidence,'execute requires fresh owner identity, launch and resource evidence')
    require(not out.exists(),'output directory already exists; choose a new run')
    require(all(p.is_file() for p in (a.identities,a.launch_manifest,a.owner_evidence)), 'owner input file missing')
    require(m['phases']['s1']['rounds']>=3,'manifest S1 minimum is 3')
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'running','started_utc':now(),'manifest_sha256':sha(a.manifest),'phase':a.phase,'steps':[]}
    state_path=out/'execution.json'
    write(out/'frozen-manifest.json',m)
    for name,path in [('identities-input.json',a.identities),('launch-input.json',a.launch_manifest),('owner-evidence.json',a.owner_evidence)]:
        (out/name).write_bytes(path.read_bytes())
    write(state_path,state)
    try:
        frozen_check(m)
        write(out/'service-before.json',service_check(m,a.phase,a.identities,a.launch_manifest))
        for name,argv in steps:
            step={'name':name,'argv':argv,'started_utc':now(),'status':'running'}
            state['steps'].append(step);write(state_path,state)
            with (out/(name+'.log')).open('xb') as log:
                proc=subprocess.run(argv,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=False)
            step.update(returncode=proc.returncode,finished_utc=now(),status='complete' if proc.returncode==0 else 'failed')
            write(state_path,state)
            require(proc.returncode==0,f'{name} failed; raw evidence retained')
            if name.startswith('c') and not name.endswith('-quality'):
                verify_od(out/(name+'.json'),m)
        frozen_check(m)
        write(out/'service-after.json',service_check(m,a.phase,a.identities,a.launch_manifest))
        summary=summarize(m,a.phase,out);write(out/'acceptance.json',summary)
        require(summary['quality_passed'],'2pp quality gate failed; evidence retained')
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_utc']=now();write(state_path,state)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
