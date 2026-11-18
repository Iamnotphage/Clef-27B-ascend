#!/usr/bin/env python3
"""One-shot owner27 baseline recovery; fail closed, no candidate execution."""
import argparse, datetime, hashlib, importlib.util, json, os, pathlib, shutil, socket, errno, subprocess, sys, traceback, uuid
P=pathlib.Path
PRE='f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863'
POST='aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'
BOOT='393ecc15-721f-46f7-b119-21fa327a2b3b'
REPO=P('/workspace/yangchen/work/scratch-24/joint-fused-5c045eb')
WRAPPER=REPO/'runtime/joint_fused_attention_window.py'
STOP=P('/tmp/d0cc/d0_stop.py')
OLD=P('/workspace/yangchen/logs/joint-fused-service-ab-20261118T181847Z')
INSTALLER=OLD/'stage/runtime/apply_joint_fused_attention.py'
TREE=P('/workspace/yangchen/work/src/vllm')
TARGET=TREE/'vllm/model_executor/models/clef_joint_schema.py'
BACKUP=P('/workspace/yangchen/logs/joint-fused-attention-20261119/phase0')
PYTHON='/workspace/yangchen/venvs/clef-py312/bin/python'
EXPECTED_SHELL_CMDLINE='bash -c P=$(pgrep -f "joint_fused_service_ab.py" | head -1); echo "runner_pid=$P"; PG=$(ps -o pgid= -p $P 2>/dev/null | tr -d " "); echo "pgid=$PG"; kill -TERM $P 2>/dev/null; for i in $(seq 1 12); do pgrep -f joint_fused_service_ab >/dev/null || { echo "runner exited"; break; }; sleep 5; done; pgrep -cf joint_fused_service_ab; echo "=== restore ==="; /workspace/yangchen/logs/perf-wave2-20261118/launch.sh final-default-s1 2>&1 | tail -1; for i in $(seq 1 40); do s=$(curl -s -m 3 http://127.0.0.1:18143/health 2>/dev/null); if echo "$s" | grep -q ready; then echo "READY $(date -u +%H:%M:%SZ) workers=$(npu-smi info | grep -cE VLLMWorker_TP)"; break; fi; sleep 10; done'
def sha(p): return hashlib.sha256(P(p).read_bytes()).hexdigest()
def require(v,msg):
 if not v: raise RuntimeError(msg)
def dump(name,obj): (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def run(name,cmd,timeout,env=None):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,env=env)
 dump(name+'.json',{'cmd':cmd,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 require(r.returncode==0,name+' rc='+str(r.returncode));return r

def scan():
 rows={}; errors=[]; boot=P('/proc/sys/kernel/random/boot_id').read_text().strip()
 require(boot==BOOT,'boot changed')
 for path in P('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   raw=(path/'stat').read_text();f=raw[raw.rfind(')')+2:].split();argv=[s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s]
   rows[int(path.name)]={'pid':int(path.name),'ppid':int(f[1]),'pgid':int(f[2]),'sid':int(f[3]),'starttime':f[19],'boot_id':boot,'state':f[0],'comm':raw[raw.find('(')+1:raw.rfind(')')],'cmdline':' '.join(argv),'argv':argv}
  except FileNotFoundError:
   if path.exists():errors.append({'pid':path.name,'error':'partial proc read'})
  except Exception as e:errors.append({'pid':path.name,'error':repr(e)})
 require(not errors,'unknown proc scan '+repr(errors)); return rows

def port():
 s=socket.socket();s.settimeout(3)
 try:rc=s.connect_ex(('127.0.0.1',18143))
 finally:s.close()
 require(rc in (0,errno.ECONNREFUSED),'unknown port '+str(rc));return {'rc':rc,'state':'open' if rc==0 else 'closed'}

def inactive(name):
 rows=scan(); bad=[]
 for p in rows.values():
  if p['state']=='Z':continue
  if p['comm'].startswith('VLLM') or any(P(a).name in ('joint_fused_service_ab.py','joint_fused_attention_window.py','supervised_window.py') for a in p['argv']) or 'vllm.entrypoints.cli.main' in p['argv']:bad.append(p)
 status=port();r=run(name+'-npu',['npu-smi','info'],30)
 dump(name+'.json',{'port':status,'resource_processes':bad,'scan_count':len(rows),'known_old_pids':{str(x):rows.get(x) for x in (1188313,1190200,1190202,1190564)}})
 require(not bad,'resource process still running');require(status['state']=='closed','port not closed')
 require('No running processes found in NPU 7' in r.stdout,'NPU query did not prove empty card7')
 return rows

def main():
 for path,expected in [(WRAPPER,'9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5'),(STOP,'d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886'),(INSTALLER,'b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406'),(OLD/'apply.json','96bf48321c9b986907d63079d9229cd3de5c49b420e514f8edbb84cdb394165d'),(BACKUP/'clef_joint_schema.py.preimage',PRE)]:require(sha(path)==expected,'hash mismatch '+str(path))
 require(sha(TARGET)==POST,'expected installed postimage')
 spec=importlib.util.spec_from_file_location('window27',WRAPPER);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 for path,expected in [(w.ACTIVATE_DEFAULT,w.ACTIVATE_SHA),(w.LAUNCH_DEFAULT,w.LAUNCH_SHA),(w.WAIT_DEFAULT,w.WAIT_SHA)]:require(sha(path)==expected,'launch contract mismatch '+str(path))
 for name,expected in w.D0_MODULES.items():require(sha(REPO/'runtime/d0'/name)==expected,'d0 module mismatch '+name)
 rows=scan();root=rows.get(1190564);dump('old-shell-fresh.json',{'root':root})
 if root is not None:
  require(root['ppid']==879777 and root['pgid']==1190564 and root['sid']==1190564 and root['starttime']=='394308827' and root['cmdline']==EXPECTED_SHELL_CMDLINE,'old shell identity changed')
  targets={1190564}
  while True:
   found={pid for pid,p in rows.items() if p['ppid'] in targets}
   if found.issubset(targets):break
   targets|=found
  for pid in targets-{1190564}:
   a=rows[pid]['argv'];require(a in (['sleep','10'],['curl','-s','-m','3','http://127.0.0.1:18143/health'],['grep','-q','ready']),'unexpected old shell child '+repr(rows[pid]))
  dump('old-shell-tree.json',[rows[x] for x in sorted(targets)])
  dump('old-shell-identity.json',{k:root[k] for k in ('pid','ppid','pgid','starttime','boot_id','cmdline')})
  if root['state']!='Z':run('old-shell-stop',[PYTHON,str(STOP),'--api-pid','1190564','--saved-identity',str(out/'old-shell-identity.json'),'--out',str(out/'old-shell-stop-record.json')],45)
  after=scan();require(all(pid not in after or after[pid]['state']=='Z' for pid in targets),'old shell tree survived')
 else:dump('old-shell-already-gone.json',{'pid':1190564,'absence_confirmed_by_successful_proc_scan':True})
 inactive('pre-revert')
 shutil.copy2(TARGET,out/'head-before.py');shutil.copy2(OLD/'apply.json',out/'original-apply.json');shutil.copy2(BACKUP/'clef_joint_schema.py.preimage',out/'owned-preimage.py')
 apply=json.loads((OLD/'apply.json').read_text());require(apply['ready'] and apply['action']=='apply' and apply['sha256_before']==PRE and apply['postimage_sha256']==POST and apply['file']==str(TARGET),'apply ownership contract mismatch')
 run('revert',[PYTHON,str(INSTALLER),'--tree',str(TREE),'--backup-dir',str(BACKUP),'--revert','--manifest',str(out/'revert-manifest.json')],120)
 require(sha(TARGET)==PRE,'source failed to revert');require(not (TARGET.parent/'clef_smallq_kernel.py').exists(),'unexpected smallq kernel')
 inactive('pre-launch')
 # Use an explicit clean environment; preserve the real account home only.
 env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG') if k in os.environ}
 env['PATH']='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
 r=subprocess.run(['bash','-c','set -a; source "$1" >/dev/null 2>&1 || exit 3; env -0','bash',w.ACTIVATE_DEFAULT],capture_output=True,env=env,timeout=60)
 require(r.returncode==0,'activate failed rc='+str(r.returncode))
 # Do not persist the full sourced environment: only keys necessary to validate runtime.
 prod=w.parse_environ(r.stdout)
 for k in w.CANDIDATE_KEYS:prod.pop(k,None)
 for k in w.CLEANUP_KEYS:prod.pop(k,None)
 prod['CLEF_LAUNCH_CONFIG']='s1'
 require(prod.get('PYTHON_BIN')==PYTHON,'activate python mismatch')
 dump('activate.json',{'rc':r.returncode,'selected_env':{k:prod.get(k) for k in ('PYTHON_BIN','ASCEND_HOME_PATH','CLEF_CANN_ENV','CLEF_ATB_ENV','CLEF_VENV','CLEF_LAUNCH_CONFIG')},'candidate_keys_absent':all(k not in prod for k in w.CANDIDATE_KEYS)})
 args=argparse.Namespace(run_dir=str(out),repo=str(REPO),tree=TREE,python=PYTHON,activate=w.ACTIVATE_DEFAULT,launch=w.LAUNCH_DEFAULT,wait=w.WAIT_DEFAULT,wait_timeout=900)
 window=w.Window(args);window.state['preflight']={};window._stage_d0(REPO);window.prod_env=prod;window.source_env=prod;window._loader_check(prod)
 result=window.step('restore');require(result.get('ok') and result.get('verify_ok'),'restore four checks failed '+repr(result))
 final=scan();pids=result['service_pids'];dump('service-identities.json',[final[pid] for pid in pids]);require(sha(TARGET)==PRE,'source changed after restore')
 dump('complete.json',{'ok':True,'source_sha256':sha(TARGET),'restore':result,'port':port(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'ok':True,'run_dir':str(out),'restore':result}),flush=True)

out=P('/workspace/yangchen/logs/handoff27-restore-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
if __name__=='__main__':
 out.mkdir(parents=True,exist_ok=False)
 try:main()
 except BaseException:
  (out/'error.txt').write_text(traceback.format_exc());print(json.dumps({'ok':False,'run_dir':str(out),'error':traceback.format_exc()}),flush=True);raise
