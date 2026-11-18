#!/usr/bin/env python3
"""CPU tests at the real CLI/HTTP/file boundaries; no service/NPU imports."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('final_acceptance', ROOT/'runtime/final_acceptance.py')
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


class Handler(BaseHTTPRequestHandler):
    seen = []
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"model":"clef"}')
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.seen.append(payload)
        mode = payload.get('mode', 'ok')
        if payload.get('state', {}).get('question', '').startswith('ERROR'):
            mode = 'error'
        if mode == 'error':
            self.send_response(503); self.end_headers(); self.wfile.write(b'full failure evidence'*1000); return
        self.send_response(200); self.end_headers()
        if mode == 'malformed':
            self.wfile.write(b'not-json-original'); return
        if mode in ('NaN','Infinity','-Infinity','1e999'):
            self.wfile.write(('{"answers":{"answer":{"probabilities":{"A":'+mode+'}}}}').encode());return
        self.wfile.write(json.dumps({'answers':{'answer':{'type':'choice','choice':'A',
            'probabilities':{'A':0.75,'B':0.25}}},'usage':{'input_tokens':10,'output_tokens':1}}).encode())


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def run_cli(self, script, *args):
        return subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],capture_output=True,text=True,timeout=20)
    def test_od_real_cli_journal_and_failure_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);samples=[{'name':str(n),'request':{'mode':mode}} for n,mode in enumerate(['ok','error','malformed'])]
            (p/'samples.json').write_text(json.dumps(samples))
            args=['--sample-file',p/'samples.json','--url',self.base+'/v1/systemone','--concurrency','1',
                  '--warmup','0','--timeout','2','--report',p/'report.json','--journal',p/'journal.jsonl']
            proc=self.run_cli('runtime/od_full_benchmark.py',*args)
            self.assertEqual(proc.returncode,0,proc.stderr)
            d=json.loads((p/'report.json').read_text());journal=[json.loads(l) for l in (p/'journal.jsonl').read_text().splitlines()]
            self.assertEqual(d['summary']['failed'],2);self.assertEqual(len(journal),3)
            self.assertEqual(d['requests'],journal)
            self.assertEqual(d['requests'][1]['response_text'],'full failure evidence'*1000)
            self.assertEqual(d['requests'][2]['response_text'],'not-json-original')
            self.assertTrue(all(r['finished_at']>=r['started_at'] for r in journal))
            before=(p/'report.json').read_bytes();count=len(Handler.seen)
            proc=self.run_cli('runtime/od_full_benchmark.py',*args)
            self.assertNotEqual(proc.returncode,0);self.assertEqual(len(Handler.seen),count)
            self.assertEqual((p/'report.json').read_bytes(),before)
    def test_erqa_actual_protocol_retry_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=p/'data';data.mkdir();blob=b'test-image-fixture';(data/'image.jpg').write_bytes(blob)
            sample={'id':'erqa-0000','record_index':0,'question':'Choices: A. Yes B. No','answer':'A',
                    'question_type':'Other','image_count':1,'visual_index_pattern':'empty','visual_indices':[],
                    'images':[{'path':'image.jpg','format':'jpeg','sha256':hashlib.sha256(blob).hexdigest()}]}
            (data/'samples.jsonl').write_text(''.join(json.dumps({**sample, 'id': f'erqa-{i:04d}',
                'record_index': i})+'\n' for i in range(400)))
            manifest={'source':{'sha256':'1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2'},
                      'records':400,'images':{'total':400},'question_types':{'Other':400}}
            (data/'manifest.json').write_text(json.dumps(manifest))
            proc=self.run_cli('scripts/official_eval/erqa_baseline.py','--data-dir',data,'--output-dir',p/'run',
                              '--api-url',self.base+'/v1','--scope','full','--limit','1','--retries','3','--timeout','2')
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads((p/'run/summary.json').read_text())['coverage']['ok'],1)
            sample_dir=p/'run/samples/erqa-0000'
            self.assertEqual((sample_dir/'response-attempt-1.bin').read_bytes(),(sample_dir/'response.json').read_bytes())
            proc=self.run_cli('scripts/official_eval/erqa_score.py','--data-dir',data,'--run-dir',p/'run',
                              '--allow-subset','--verify-requests','--output',p/'score.json')
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertTrue(json.loads((p/'score.json').read_text())['request_verification']['passed'])
            samples=(data/'samples.jsonl').read_text()
            (data/'samples.jsonl').write_text(samples.replace('Choices:', 'ERROR Choices:'))
            proc=self.run_cli('scripts/official_eval/erqa_baseline.py','--data-dir',data,'--output-dir',p/'failed',
                              '--api-url',self.base+'/v1','--scope','full','--limit','1','--retries','2',
                              '--retry-interval','0','--timeout','2')
            self.assertEqual(proc.returncode,0,proc.stderr)  # legacy runner reports failures in coverage
            result=json.loads((p/'failed/summary.json').read_text())
            self.assertEqual(result['coverage']['failed'],1);self.assertEqual(result['coverage']['retry_count'],1)
            for n in (1,2):
                self.assertEqual((p/f'failed/samples/erqa-0000/response-attempt-{n}.bin').read_bytes(),
                                 b'full failure evidence'*1000)
    def test_plan_and_existing_directory_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'new'
            proc=self.run_cli('runtime/final_acceptance.py','--phase','s1','--output-dir',out)
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertFalse(out.exists())
            steps=json.loads(proc.stdout)['steps'];self.assertEqual(len(steps),6)
            out.mkdir();(out/'preserve').write_text('user evidence')
            proc=self.run_cli('runtime/final_acceptance.py','--phase','s1','--output-dir',out,
                              '--execute','--identities','missing','--launch-manifest','missing','--owner-evidence','missing')
            self.assertNotEqual(proc.returncode,0);self.assertIn('already exists',proc.stderr)
            self.assertEqual((out/'preserve').read_text(),'user evidence')
    def test_source_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_text('original');item={'path':str(p),'sha256':accept.sha(p)}
            accept.verify_file(item);p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):accept.verify_file(item)
    def test_nonfinite_http_json_persists_every_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);values=['NaN','Infinity','-Infinity','1e999']
            (p/'samples.json').write_text(json.dumps([{'request':{'mode':v}} for v in values]))
            proc=self.run_cli('runtime/od_full_benchmark.py','--sample-file',p/'samples.json',
                              '--url',self.base+'/v1/systemone','--report',p/'report.json','--journal',p/'journal.jsonl')
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows=[json.loads(line) for line in (p/'journal.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            for row,value in zip(rows,values):
                self.assertEqual(row['status'],'error');self.assertFalse(row['passed'])
                self.assertIn('"A":'+value,row['response_text']);self.assertNotIn('response',row)
            self.assertEqual(json.loads((p/'report.json').read_text())['summary']['failed'],4)
    def test_phase_commands_use_real_scanner_flags(self):
        m=accept.read(ROOT/'runtime/final_acceptance_manifest.json')
        self.assertEqual(m['phases']['s2_b8']['concurrencies'],[8,16])
        self.assertEqual(m['phases']['s2_b16']['concurrencies'],[32,64])
        for phase in m['phases']:
            for name,argv in accept.commands(m,phase,Path('/tmp/unused')):
                help_result=self.run_cli(Path(argv[1]).relative_to(ROOT),'--help')
                self.assertEqual(help_result.returncode,0,help_result.stderr)
                for token in argv[2:]:
                    if token.startswith('--'):self.assertIn(token,help_result.stdout)
    def test_erqa_gate_and_incomplete_denominator(self):
        m=accept.read(ROOT/'runtime/final_acceptance_manifest.json')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'erqa').mkdir()
            (out/'erqa/summary.json').write_text(json.dumps({'coverage':{'retry_count':0}}))
            report={'coverage':{'fixed_total':400,'ok':400,'failed':0},
                    'request_verification':{'checked':True,'passed':True},
                    'summary_comparison':{'equal':True},'accuracy':{'value':211/400,'correct':211,'total':400},'breakdown':{}}
            for correct,passed in [(211,True),(227,True),(210,False),(228,False)]:
                report['accuracy']['value']=correct/400;report['accuracy']['correct']=correct
                (out/'erqa-score.json').write_text(json.dumps(report))
                self.assertEqual(accept.summarize(m,'erqa',out)['quality_passed'],passed)
            report['coverage']['ok']=399;report['coverage']['failed']=1
            (out/'erqa-score.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'incomplete'):accept.summarize(m,'erqa',out)


if __name__=='__main__':unittest.main()
