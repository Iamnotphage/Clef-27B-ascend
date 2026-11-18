#!/usr/bin/env python3
"""Recover the exact first 8b46 service arm after guarded cmdline refusal."""
import argparse,datetime,hashlib,importlib.util,json,os,pathlib,subprocess,traceback,uuid
P=pathlib.Path
HELPER=P('/tmp/clef27-recover-f0b679.py')
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!='f0b679f65b9fcaef0a59aa9fa0a84d9ed978393c55738c9e5716a74aa4decff5':raise RuntimeError('helper hash changed')
spec=importlib.util.spec_from_file_location('frozen_recovery27',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
out=P('/workspace/yangchen/logs/handoff27-ab-recover-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);h.out=out
AB=P('/workspace/yangchen/logs/joint-fused-service-ab-27-8b46-first')
LABEL='d0-ab-20261118T183933Z-0-0-ccef63'
IDENTITY=P('/tmp/clef27-ab-live-identity.json')
IDENTITY_SHA='d436fef70394227cd9309caec02a48a2dd373f0b2f825d2c2a898d6b6c0f5fd5'

def main():
 h.require(h.sha(IDENTITY)==IDENTITY_SHA,'expected identity hash mismatch')
 expected=json.loads(IDENTITY.read_text())
 h.require(h.sha(h.STOP)=='d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886','stop script hash')
 h.require(h.sha(h.WRAPPER)=='9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5','wrapper hash')
 h.require(h.sha(h.TARGET)==h.PRE,'head not f810')
 guard=P('/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py')
 h.require(h.sha(guard)=='76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9','guard not original')
 h.require(not (h.TARGET.parent/'clef_smallq_kernel.py').exists(),'unexpected kernel')
 prior=json.loads((AB/'stop-cleanup-arm.json').read_text())['stop']
 h.require(prior['error']=='saved cmdline mismatch','unexpected previous refusal')
 for key in ['pid','ppid','pgid','starttime','boot_id']:
  h.require(prior['saved'][key]==expected[key] and prior['live'][key]==expected[key],'exec continuity mismatch '+key)
 h.require(prior['live']['cmdline']==expected['cmdline'],'runtime cmdline changed since original refusal')
 h.require(int((P('/workspace/yangchen/logs/perf-wave2-20261118')/LABEL/'api.pid').read_text())==expected['pid'],'label pid changed')
 rows=h.scan();root=rows.get(expected['pid']);h.require(root is not None,'target absent; stop and reassess')
 for key,value in expected.items():h.require(root[key]==value,'fresh identity mismatch '+key)
 env=dict(x.split('=',1) for x in P('/proc/%s/environ'%root['pid']).read_bytes().decode().split('\0') if '=' in x)
 for k,v in {'CLEF_JOINT_FUSED_ATTN':'0','CLEF_JOINT_ALLOWED_POSTIMAGE_SHA':h.POST,'CLEF_LAUNCH_CONFIG':'s1'}.items():h.require(env.get(k)==v,'runtime candidate attribution mismatch '+k)
 targets={root['pid']}
 while True:
  found={pid for pid,p in rows.items() if p['ppid'] in targets}
  if found.issubset(targets):break
  targets|=found
 h.require(all(rows[pid]['pgid']==root['pgid'] for pid in targets),'descendant group changed')
 h.dump('bound-runtime-tree.json',[rows[pid] for pid in sorted(targets)]);h.dump('runtime-identity.json',expected);h.dump('original-stop-refusal.json',prior)
 h.run('stop',[h.PYTHON,str(h.STOP),'--api-pid',str(root['pid']),'--saved-identity',str(out/'runtime-identity.json'),'--out',str(out/'stop-record.json')],45)
 after=h.scan();h.require(all(pid not in after or after[pid]['state']=='Z' for pid in targets),'surviving owned process')
 h.inactive('pre-launch')
 spec=importlib.util.spec_from_file_location('window27',h.WRAPPER);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 for path,expected_sha in [(w.ACTIVATE_DEFAULT,w.ACTIVATE_SHA),(w.LAUNCH_DEFAULT,w.LAUNCH_SHA),(w.WAIT_DEFAULT,w.WAIT_SHA)]:h.require(h.sha(path)==expected_sha,'launch contract changed')
 clean={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG') if k in os.environ};clean['PATH']='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
 result=subprocess.run(['bash','-c','set -a; source "$1" >/dev/null 2>&1 || exit 3; env -0','bash',w.ACTIVATE_DEFAULT],capture_output=True,env=clean,timeout=60)
 h.require(result.returncode==0,'activate failed');prod=w.parse_environ(result.stdout)
 for k in w.CANDIDATE_KEYS | {'CLEF_JOINT_ALLOWED_POSTIMAGE_SHA'}:prod.pop(k,None)
 for k in w.CLEANUP_KEYS:prod.pop(k,None)
 prod['CLEF_LAUNCH_CONFIG']='s1';h.require(prod.get('PYTHON_BIN')==h.PYTHON,'activate python mismatch')
 args=argparse.Namespace(run_dir=str(out),repo=str(h.REPO),tree=h.TREE,python=h.PYTHON,activate=w.ACTIVATE_DEFAULT,launch=w.LAUNCH_DEFAULT,wait=w.WAIT_DEFAULT,wait_timeout=900)
 window=w.Window(args);window.state['preflight']={};window._stage_d0(h.REPO);window.prod_env=prod;window.source_env=prod;window._loader_check(prod)
 result=window.step('restore');h.require(result.get('ok') and result.get('verify_ok'),'restore four checks failed '+repr(result))
 final=h.scan();h.dump('service-identities.json',[final[pid] for pid in result['service_pids']]);h.require(h.sha(h.TARGET)==h.PRE and h.sha(guard)=='76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9','restored sources changed')
 h.dump('complete.json',{'ok':True,'source_sha256':h.sha(h.TARGET),'guard_sha256':h.sha(guard),'restore':result,'port':h.port()});print(json.dumps({'ok':True,'run_dir':str(out),'restore':result}),flush=True)

if __name__=='__main__':
 out.mkdir(parents=True,exist_ok=False)
 try:main()
 except BaseException:
  (out/'error.txt').write_text(traceback.format_exc());print(json.dumps({'ok':False,'run_dir':str(out),'error':traceback.format_exc()}),flush=True);raise
