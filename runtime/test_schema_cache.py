#!/usr/bin/env python3
"""Offline equivalence + CPU A/B for the production schema token cache.

Uses the real ServingEncoder construction (pinned vLLM HEAD enforced), the
production _SchemaTokenCache class, and the real /v1/systemone request set.
Cache-off is the strategy as constructed (disabled by default).
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/yangchen/work/profiling-tools-20261113/scripts")

MODEL_DIR = Path("/workspace/yangchen/models/Clef-27B")
VLLM_SRC = Path("/workspace/yangchen/work/src/vllm")

from prepare_systemone_profile_workloads import ServingEncoder

encoder = ServingEncoder(MODEL_DIR, VLLM_SRC, 4096)
strategy = encoder.strategies[4096]
print("strategy:", type(strategy).__name__)

from vllm.entrypoints.openai.decisions.strategies import _SchemaTokenCache

# the request set: five smoke cases + fields-8 + a long-state variant
import hashlib
smoke = json.load(open("/workspace/yangchen/logs/tuning/sample-smoke5.json"))["samples"]
f8 = json.load(open("/workspace/yangchen/logs/profiling/20261112-223732-systemone/workloads/fields-8.json"))["samples"][0]
state_a = f8["request"]["state"]
state_b = state_a + "\nAppended tail: caller credit balance updated mid-call, different length."
state_c = "Short, completely different plain-text state."
requests = [s["request"] for s in smoke] + [f8["request"]]
variants = []
for r in requests:
    variants.append((r, r["state"]))
    variants.append((r, state_b))
    variants.append((r, state_c))
print("variants:", len(variants))


def parse(request):
    return encoder.parse_questions(request, strategy.limits())


def encode_all():
    out = []
    for request, state in variants:
        questions = parse(request)
        rendered = strategy.render_state(state)
        out.append(strategy.encode(questions, rendered))
    return out


# cache-off reference
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, False, 4096, 4_000_000)
ref = encode_all()

# cache-on, cold then warm
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000)
cold = encode_all()
warm = encode_all()
warm2 = encode_all()

failures = []
if cold != ref:
    failures.append("cold-cache encodings differ from reference")
if warm != ref:
    failures.append("warm-cache encodings differ from reference")
if warm2 != ref:
    failures.append("warm2-cache encodings differ from reference")

# mutation protection on prompt lists
probe = encode_all()
probe[0][0][0] = -1
probe[0][1]["question_spans"][0][0] = -999
after = encode_all()
if after != ref:
    failures.append("mutation protection failed")

stats = strategy._schema_cache.stats()
print("cache stats:", json.dumps(stats))
if stats["hits"] == 0:
    failures.append("no hits recorded in warm rounds")

# eviction check with tiny budget
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 8, 100_000)
for i in range(120):
    q = parse(requests[0])
    strategy.encode(q, strategy.render_state(f'{{"round": {i}}}'))
st = strategy._schema_cache.stats()
if st["entries"] > 8:
    failures.append(f"eviction failed: entries={st['entries']}")
print("tiny-budget stats:", json.dumps(st))

# CPU A/B on the mixed workload
def bench(rounds=400):
    times = []
    n = len(variants)
    for i in range(rounds):
        request, state = variants[i % n]
        questions = parse(request)
        rendered = strategy.render_state(state)
        t0 = time.perf_counter()
        strategy.encode(questions, rendered)
        times.append((time.perf_counter() - t0) * 1000)
    return times

strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, False, 4096, 4_000_000)
off_times = bench()
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000)
on_cold = bench(40)
on_times = bench()
print("encode CPU ms off: mean=%.3f p50=%.3f" % (statistics.mean(off_times), statistics.median(off_times)))
print("encode CPU ms on(cold): mean=%.3f" % statistics.mean(on_cold))
print("encode CPU ms on(warm): mean=%.3f p50=%.3f" % (statistics.mean(on_times), statistics.median(on_times)))
print("failures:", failures if failures else "none")
sys.exit(1 if failures else 0)
