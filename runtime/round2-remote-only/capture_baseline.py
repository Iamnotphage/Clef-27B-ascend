#!/usr/bin/env python3
"""Capture baseline responses for fixed text, per-type, multi-field and multimodal samples."""
import hashlib, json, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, "/workspace/yangchen/work/profiling-tools-20261113/scripts")
from verify_systemone import cases

BASE_URL = "http://127.0.0.1:8000"
WORKLOADS = Path("/workspace/yangchen/logs/profiling/20261112-223732-systemone/workloads")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def post(path, body, timeout=300):
    data = json.dumps(body, ensure_ascii=False).encode()
    digest = hashlib.sha256(data).hexdigest()
    start = time.monotonic()
    req = Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
        status = resp.status
    elapsed_ms = (time.monotonic() - start) * 1000
    return status, digest, json.loads(payload), elapsed_ms


record = {}
# 1) verify_systemone cases: choice / noul / score / mixed(fixed307) / choice_reordered
for kind, request in cases("clef"):
    status, digest, response, ms = post("/v1/systemone", request)
    record[f"verify_{kind}"] = {"status": status, "request_sha256": digest, "elapsed_ms": ms,
                                "request": request, "response": response}
    print(f"verify_{kind}: status={status} sha={digest[:12]} ms={ms:.1f}")

# 2) workload samples: fields-1(pure choice), fields-8, multimodal
for name in ["fields-1", "fields-8", "mm-red-png", "controlled-length-3868" ]:
    sample_file = WORKLOADS / f"{name}.json"
    if not sample_file.exists():
        print(f"skip {name}: no file")
        continue
    doc = json.loads(sample_file.read_text())
    sample = doc["samples"][0]
    request = sample["request"]
    status, digest, response, ms = post("/v1/systemone", request)
    record[f"workload_{name}"] = {"status": status, "request_sha256": digest, "elapsed_ms": ms,
                                  "request": request, "response": response,
                                  "metadata": sample.get("metadata", {})}
    print(f"workload_{name}: status={status} sha={digest[:12]} ms={ms:.1f}")

(OUT / "baseline-responses.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
print("saved:", OUT / "baseline-responses.json")
