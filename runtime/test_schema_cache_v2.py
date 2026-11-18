#!/usr/bin/env python3
"""Offline equivalence + budget tests for the v2 schema token cache.

Extends the round-2 suite with the v2 fixes: oversized-fragment bypass,
text-key byte budget, guard-free eviction, budget invariants.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, "/workspace/yangchen/work/profiling-tools-20261113/scripts")

MODEL_DIR = Path("/workspace/yangchen/models/Clef-27B")
VLLM_SRC = Path("/workspace/yangchen/work/src/vllm")

from prepare_systemone_profile_workloads import ServingEncoder

encoder = ServingEncoder(MODEL_DIR, VLLM_SRC, 4096)
strategy = encoder.strategies[4096]
print("strategy:", type(strategy).__name__)

from vllm.entrypoints.openai.decisions.strategies import _SchemaTokenCache

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


request_class = encoder.request_class


def parse(request):
    return encoder.parse_questions(request_class.model_validate(request), strategy.limits())


def encode_all():
    out = []
    for request, state in variants:
        questions = parse(request)
        rendered = strategy.render_state(state)
        out.append(strategy.encode(questions, rendered))
    return out


failures = []

# cache-off reference
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, False, 4096, 4_000_000)
ref = encode_all()

# cache-on equivalence: cold/warm/warm2
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000)
cold = encode_all()
warm = encode_all()
warm2 = encode_all()
if cold != ref:
    failures.append("cold-cache encodings differ from reference")
if warm != ref:
    failures.append("warm-cache encodings differ from reference")
if warm2 != ref:
    failures.append("warm2-cache encodings differ from reference")

# mutation protection
probe = encode_all()
probe[0][0][0] = -1
probe[0][1]["question_spans"][0][0] = -999
if encode_all() != ref:
    failures.append("mutation protection failed")

stats = strategy._schema_cache.stats()
print("cache stats:", json.dumps(stats))
if stats["hits"] == 0:
    failures.append("no hits recorded in warm rounds")
if stats.get("oversized_skips", 0) != 0:
    failures.append("unexpected oversized skips in normal budgets")

# v2: token-budget invariant under churn with a small budget
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 8, 100_000)
for i in range(120):
    q = parse(requests[0])
    strategy.encode(q, strategy.render_state(f'{{"round": {i}}}'))
    st = strategy._schema_cache.stats()
    if st["entries"] > 8 or st["token_total"] > 100_000:
        failures.append(f"budget invariant violated: {st}")
        break
st = strategy._schema_cache.stats()
print("tiny-entry-budget stats:", json.dumps(st))
if st["entries"] > 8:
    failures.append(f"eviction failed: entries={st['entries']}")

# v2: oversized fragment bypass (tokens > max_tokens never cached)
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 8)
outs = encode_all()
st = strategy._schema_cache.stats()
if st["token_total"] > 8 or st["entries"] > 8:
    failures.append(f"token budget violated: {st}")
if st["oversized_skips"] == 0:
    failures.append("oversized bypass not counted")
if outs != ref:
    failures.append("oversized-bypass encodings differ from reference")
print("oversized-token stats:", json.dumps(st))

# v2: long text-key bypass via max_key_bytes (direct unit check + suite)
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000, max_key_bytes=1024)
outs = encode_all()
st = strategy._schema_cache.stats()
if outs != ref:
    failures.append("long-key-bypass encodings differ from reference")
long_text = "schema fragment padding " * 200  # ~4.6 KB key
ids_long = strategy._schema_cache.tokens(strategy.tokenizer, long_text)
st2 = strategy._schema_cache.stats()
if st2["entries"] == 0 and len(ids_long) == 0:
    failures.append("long-key bypass produced no ids")
if st2["oversized_skips"] == 0:
    failures.append(f"long key not skipped: {st2}")
if ids_long != list(strategy.tokenizer.encode(long_text, add_special_tokens=False)):
    failures.append("long-key bypass ids differ from direct encode")
print("key-byte-budget stats:", json.dumps(st2))

# v2: cache may empty completely under pressure
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4, 64)
outs = encode_all()
st = strategy._schema_cache.stats()
if outs != ref:
    failures.append("empty-cache encodings differ from reference")
print("hard-budget stats:", json.dumps(st))

# v2: budget config changes respected
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000)
encode_all()
st = strategy._schema_cache.stats()
if st["entries"] == 0:
    failures.append("normal budget cached nothing")

# v2: the request state never enters the schema cache.
strategy._schema_cache = _SchemaTokenCache(strategy.tokenizer, True, 4096, 4_000_000)
seen = []
original_tokens = strategy._schema_cache.tokens


def record_tokens(tokenizer, text):
    seen.append(text)
    return original_tokens(tokenizer, text)


strategy._schema_cache.tokens = record_tokens
marker = "CLEF-STATE-BYPASS-MARKER-7f3a"
state_request = dict(requests[0])
state_request["state"] = marker + " " + state_request["state"]
state_questions = parse(state_request)
strategy.encode(state_questions, strategy.render_state(state_request["state"]))
if any(marker in text for text in seen):
    failures.append("request state entered the schema cache")
print("state-bypass schema keys:", len(seen))

print("failures:", failures if failures else "none")
sys.exit(1 if failures else 0)
