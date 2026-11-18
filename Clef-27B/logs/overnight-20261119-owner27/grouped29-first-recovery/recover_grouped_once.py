#!/usr/bin/env python3
"""Single authorized recovery after Grouped v3 timeout; no signal path."""
import argparse,datetime,hashlib,importlib.util,json,pathlib,shutil,socket,subprocess,traceback,uuid
P=pathlib.Path
REPO=P('/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769')
OLD=P('/workspace/yangchen/logs/grouped29-first967-owner27-20261119-v3')
EXPECTED=P('/workspace/yangchen/logs/joint-fused-source-control-27-ebca-gate/clean-env-expected.frozen.json')
STRICT=P('/workspace/yangchen/work/final-owner27-ab2903db592dd1e690f0f9ac9ad0b437ef96365e/final_acceptance_supervisor.py')
TARGET=P('/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py')
GUARD=P('/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py')
LAUNCH='/workspace/yangchen/logs/perf-wave2-20261118/launch.sh'
WAIT='/workspace/yangchen/logs/perf-wave2-20261118/wait.sh'
LOG=P('/workspace/yangchen/logs/perf-wave2-20261118')
PIDS={1221671,1221672,1221680,1221681}
BOOT='393ecc15-721f-46f7-b119-21fa327a2b3b'
HASHES={
 str(REPO/'runtime/joint_fused_source_control.py'):'a1c502a3a709b836d31dc994a156f702d6a826ce5f2854beeec9ac50701f7299',
 str(REPO/'runtime/joint_fused_service_ab.py'):'127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648',
 str(REPO/'runtime/joint_fused_attention_window.py'):'700cf10cdc3fda18d1ae7f113158342e1ef6ec114569e52b15b134949bfe8473',
 str(STRICT):'03409c78b0e1d39f44299785247586a545c171a020ed78d886c441559c59db41',
 str(OLD/'window.json'):'6b55c358a04de315e22332635f0f64d434e42bc2b61fa663d603fb66b8f190e0',
 str(OLD/'probe/record.json'):'a48242773b5a69c2a97d51c93266d28a8032f986b1f4d459b44ab05df2376ccd',
 '/tmp/clef27-grouped-timeout-inventory-20261118T194239Z.json':'972d157d578ba816b22f992d99d1bb4d58b010db84e925a85df74683ae61833b',
 str(EXPECTED):'fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230',
 str(TARGET):'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863',
 str(GUARD):'76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9',
 LAUNCH:'a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321',
 WAIT:'e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0',
}
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def dump(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def port():
 with socket.socket() as s:s.settimeout(3);rc=s.connect_ex(('127.0.0.1',18143))
 require(rc in (0,111),'unknown port '+str(rc));return rc
def empty_gate(strict,name):
 require(P('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'boot changed')
 table=strict.scan();require(not(PIDS & set(table)),'old tracked PID present');require(not any(v['pgid']==1221671 for v in table.values()),'old group present')
 markers={'joint_fused_service_ab.py','joint_fused_source_control.py','joint_fused_attention_window.py','supervised_window.py','window_probe.py','run_cell.py','bench_ready.py','final_acceptance.py','final_acceptance_supervisor.py'}
 active=[]
 for pid,row in table.items():
  if row['state']=='Z':continue
  p=P('/proc')/str(pid)
  try:argv=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
  except FileNotFoundError:
   require(not p.exists(),'partial proc argv read');continue
  if any(P(a).name in markers for a in argv) or 'vllm.entrypoints.cli.main' in argv:active.append(row)
 rc=port();n=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
 dump(name+'.json',{'boot_id':BOOT,'scan_count':len(table),'tracked':{str(p):table.get(p) for p in sorted(PIDS)},'active':active,'port_connect_ex':rc,'npu':{'rc':n.returncode,'stdout':n.stdout,'stderr':n.stderr}})
 require(not active and rc==111,'active runner/service or port not closed');require(n.returncode==0 and 'No running processes found in NPU 7' in n.stdout,'card7 empty unavailable')
 require(sha(TARGET)==HASHES[str(TARGET)] and sha(GUARD)==HASHES[str(GUARD)],'source/guard changed');require(not (TARGET.parent/'clef_smallq_kernel.py').exists(),'smallq still installed')
 return table

def main():
 for p,h in HASHES.items():require(sha(p)==h,'hash mismatch '+p)
 require(EXPECTED.stat().st_size==12933,'expected size changed')
 e=json.loads(EXPECTED.read_text())['service_environment']
 for key,count,h in [('ASCEND_CUSTOM_OPP_PATH',9,'7b79c5dbfac4fa09692cf902c3cb6f4dc6b9db4ace099144e4df488515869ef1'),('LD_LIBRARY_PATH',12,'ca27dbb7cd79489e2e2c0fc25185f5aa35b83ee3d05a424c482c0a5758be6b6c')]:
  require(len(e[key].split(':'))==count and hashlib.sha256(e[key].encode()).hexdigest()==h,'expected env changed '+key)
 strict=load(STRICT,'strict_recovery');sc=load(REPO/'runtime/joint_fused_source_control.py','reviewed_clean_recovery')
 shutil.copy2(OLD/'window.json',out/'original-window.json');shutil.copy2(OLD/'probe/record.json',out/'original-probe-record.json');shutil.copy2(EXPECTED,out/'clean-env-expected.frozen.json')
 empty_gate(strict,'fresh-gone')
 args=argparse.Namespace(run_dir=str(out),repo=str(REPO),tree='/workspace/yangchen/work/src/vllm',python='/workspace/yangchen/venvs/clef-py312/bin/python',launch=LAUNCH,wait=WAIT,wait_timeout=600,port=18143,service_log_root=str(LOG),clean_env_expect=str(out/'clean-env-expected.frozen.json'))
 runner=sc.SourceControl(args);runner.state={'steps':{}};runner.prod_env={};runner.stage_d0()
 original_run=subprocess.run
 def guarded_run(argv,*a,**kw):
  if len(argv)>=2 and argv[:2]==['/bin/bash',LAUNCH]:
   require(len(argv)==3 and argv[2].startswith('d0-restore-'),'unexpected restore argv');require(not (LOG/argv[2]).exists(),'restore label already exists');require(kw.get('env')=={'PATH':sc.CLEAN_PATH,'CLEF_LAUNCH_CONFIG':'s1'},'restore parent is not exact clean seed')
   table=strict.scan();require(not(PIDS & set(table)) and not any(v['pgid']==1221671 for v in table.values()),'old PID/group reappeared');require(port()==111,'fresh port not closed before launch');dump('launch-boundary.json',{'argv':argv,'env':kw['env'],'old_pids_absent':True,'port_connect_ex':111})
  return original_run(argv,*a,**kw)
 subprocess.run=guarded_run
 try:result=runner.restore_prod()
 finally:subprocess.run=original_run
 require(result['verify_ok'] and result['restore_env']['match'],'restore gates failed')
 table=strict.scan();pids=result['service_pids'];require(all(p in table and table[p]['state']!='Z' for p in pids),'restored tree missing')
 root=result['restore_env']['root_pid'];require(all(table[p]['pgid']==root for p in pids),'restored PGID mismatch')
 identities=[]
 for pid in pids:
  row=table[pid];fields=(P('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split();row['sid']=int(fields[3]);identities.append(row)
 dump('service-identities.json',identities);require(sha(TARGET)==HASHES[str(TARGET)] and sha(GUARD)==HASHES[str(GUARD)] and port()==0,'final source/port mismatch')
 dump('complete.json',{'ok':True,'old_timeout_remains_failed':True,'signals_sent_by_recovery':[],'source_sha':sha(TARGET),'guard_sha':sha(GUARD),'restore':result,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps({'ok':True,'run_dir':str(out),'restore':result}),flush=True)

out=P('/workspace/yangchen/logs/handoff27-grouped-recover-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
if __name__=='__main__':
 out.mkdir(parents=True,exist_ok=False)
 try:main()
 except BaseException:
  (out/'error.txt').write_text(traceback.format_exc());print(json.dumps({'ok':False,'run_dir':str(out),'error':traceback.format_exc()}),flush=True);raise
