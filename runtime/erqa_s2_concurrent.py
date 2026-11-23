
#!/usr/bin/env python3
"""Bounded ERQA concurrency around the frozen single-attempt runner."""
from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock, local
from time import monotonic, monotonic_ns, time_ns

import erqa_baseline as base

FROZEN="8d0632825ca5bf58e33427510f5d297b3080b00cce84959f736c72e974b2755a"
PREDICTION_KEYS=(
    "id","record_index","question_type","gold","prediction","probabilities","confidence",
    "correct","status","image_count","option_count","visual_index_pattern","image_sha256",
    "image_order","request_structural_sha256","request_wire_sha256","body_bytes",
    "input_tokens","output_tokens","latency_ms","started_at","finished_at",
)

class Inflight:
    def __init__(self,limit):
        self.limit=limit
        self.inflight=0
        self.peak=0
        self.lock=Lock()
        self.context=local()

    def track(self,send):
        def post(url,body,timeout):
            digest=sha256(body).hexdigest()
            with self.lock:
                self.inflight+=1
                self.peak=max(self.peak,self.inflight)
                assert self.inflight<=self.limit
            event={"http_begin_ns":monotonic_ns(),"http_begin_wall_ns":time_ns(),
                   "wire_sha256":digest}
            try:
                code,data=send(url,body,timeout)
                event["http_status"]=code
                return code,data
            except Exception as exc:
                event["exception"]=type(exc).__name__
                if hasattr(exc,"code"):
                    event["http_status"]=exc.code
                raise
            finally:
                event["http_end_ns"]=monotonic_ns()
                event["http_end_wall_ns"]=time_ns()
                self.context.event=event
                with self.lock:
                    self.inflight-=1
        return post

def bounded(work,ordered,concurrency):
    inputs=iter(ordered)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending=set()
        for _ in range(concurrency):
            item=next(inputs,None)
            if item is None: break
            pending.add(pool.submit(work,item))
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for job in finished:
                yield job.result()
                item=next(inputs,None)
                if item is not None:
                    pending.add(pool.submit(work,item))

def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--api-url",required=True)
    p.add_argument("--concurrency",type=int,choices=(8,16),required=True)
    p.add_argument("--timeout",type=float,default=600)
    args=p.parse_args()
    args.started_at=base.now()
    args.scope="full"
    args.model="clef"
    args.retries=1
    args.retry_interval=0
    assert sha256(Path(base.__file__).read_bytes()).hexdigest()==FROZEN
    samples,manifest=base.load_samples(args.data_dir)
    assert manifest["source"]["sha256"]==base.OFFICIAL_TFRECORD_SHA256
    assert len(samples)==400 and [x["record_index"] for x in samples]==list(range(400))
    assert len({x["id"] for x in samples})==400
    result=args.output_dir.expanduser().resolve()
    result.mkdir(parents=False,exist_ok=False)
    (result/"samples").mkdir()
    service=base.service_info(args.api_url)
    assert service.get("status_code")==200
    visible=Inflight(args.concurrency)
    base.post=visible.track(base.post)
    def perform(sample):
        visible.context.event=None
        start=monotonic_ns()
        sample_dir=result/"samples"/sample["id"]
        got=base.run_one(sample,args.data_dir,sample_dir,args)
        assert len(got["attempts"])==1
        event=visible.context.event
        assert event and event["wire_sha256"]==got["request_wire_sha256"]
        return {"record":got,"http":event,"client_begin_ns":start,"client_end_ns":monotonic_ns()}
    records=[]
    with (result/"journal.jsonl").open("x",encoding="utf-8") as journal:
        start_ns=monotonic_ns()
        for completed in bounded(perform,samples,args.concurrency):
            journal.write(json.dumps(completed,ensure_ascii=False,allow_nan=False)+"\n")
            journal.flush()
            records.append(completed["record"])
        end_ns=monotonic_ns()
    assert visible.inflight==0 and visible.peak<=args.concurrency
    records.sort(key=lambda r:r["record_index"])
    assert [x["record_index"] for x in records]==list(range(400))
    elapsed=(end_ns-start_ns)/1e9
    summary=base.summarize(records,elapsed,args,service,manifest)
    summary["run"].update({"skipped_resumed":0,"concurrency":args.concurrency,
        "max_http_inflight":visible.peak,"wall_start_ns":start_ns,"wall_end_ns":end_ns,
        "wall_scope":"From first worker submission through last flushed journal entry; includes request preparation, HTTP and original per-item writes. Setup GET and final summaries excluded."})
    summary["throughput"]["basis"]="valid successful decisions / complete bounded-run wall clock"
    (result/"predictions.jsonl").write_text("".join(json.dumps({k:r.get(k) for k in PREDICTION_KEYS},
        ensure_ascii=False,allow_nan=False)+"\n" for r in records))
    (result/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"summary":str(result/"summary.json"),"coverage":summary["coverage"],
        "accuracy":summary["accuracy"],"throughput":summary["throughput"],"latency_ms":summary["latency_ms"]}),flush=True)

if __name__=="__main__":
    main()
