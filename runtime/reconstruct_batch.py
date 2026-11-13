#!/usr/bin/env python3
"""Reconstruct the actual scheduler batch distribution from two Prometheus scrapes.

Reads vllm:iteration_tokens_total {bucket,count,sum} increments between
metrics-before.txt and metrics-after.txt, deducts the sequential warmup
iterations (each B1), and solves the integer batch combination that matches
the formal iteration count, the formal token sum, and the bucket deltas for
fixed-size requests.
"""
import argparse
import json
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--before", required=True)
parser.add_argument("--after", required=True)
parser.add_argument("--warmup", type=int, default=10, help="sequential warmup requests (each B1)")
parser.add_argument("--tokens-per-request", type=int, default=307)
parser.add_argument("--max-batch", type=int, default=16)
parser.add_argument("--output", required=True)
args = parser.parse_args()

LINE = re.compile(
    r"^vllm:iteration_tokens_total(_bucket|_count|_sum)?\{([^}]*)\}\s+([0-9.eE+]+)(?:\s+(\d+))?$",
    re.MULTILINE,
)


def parse(path):
    buckets = {}
    count = 0.0
    total = 0.0
    created = None
    for kind, labels, value, ts in LINE.findall(open(path).read()):
        value = float(value)
        label_map = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        if label_map.get("model_name", "clef") != "clef":
            continue
        if kind == "_bucket":
            le = label_map.get("le")
            if le and le != "+Inf":
                buckets[float(le)] = buckets.get(float(le), 0.0) + value
        elif kind in (None, "_count"):
            count += value
            if ts:
                created = int(ts)
        elif kind == "_sum":
            total += value
    return {"count": count, "sum": total, "buckets": buckets, "created": created}


before = parse(args.before)
after = parse(args.after)
d_count = after["count"] - before["count"]
d_sum = after["sum"] - before["sum"]
bucket_deltas = {le: after["buckets"].get(le, 0.0) - before["buckets"].get(le, 0.0)
                 for le in sorted(set(before["buckets"]) | set(after["buckets"]))}

warmup_iters = args.warmup
warmup_tokens = warmup_iters * args.tokens_per_request
formal_iters = int(round(d_count - warmup_iters))
formal_tokens = int(round(d_sum - warmup_tokens))

les = sorted(le for le in bucket_deltas if bucket_deltas[le] >= 0)
# cumulative formal iterations with tokens <= le (each warmup iteration is B1,
# i.e. tokens_per_request tokens, so it lands in buckets with le >= tokens_per_request)
cum_actual = {}
for le in les:
    warmup_below = warmup_iters if args.tokens_per_request <= le else 0
    cum_actual[le] = bucket_deltas[le] - warmup_below


def batch_regions(b):
    tokens = b * args.tokens_per_request
    regions = []
    prev = 0.0
    for le in les:
        if prev < tokens <= le:
            regions.append(le)
        prev = le
    if les and tokens > les[-1]:
        regions.append("inf")
    return regions


solutions = []
max_b = args.max_batch


def search(b, iters_left, tokens_left, acc):
    if b > max_b:
        if iters_left == 0 and tokens_left == 0:
            solutions.append(dict(acc))
        return
    per_iter = b * args.tokens_per_request
    max_c = min(iters_left, tokens_left // per_iter)
    # bucket boundaries closed by batches up to b (next batch cannot add below them)
    for c in range(max_c + 1):
        if c:
            acc[b] = c
        closed = [le for le in les
                  if args.tokens_per_request <= le < (b + 1) * args.tokens_per_request]
        ok = all(
            abs(sum(n for bb, n in acc.items() if bb * args.tokens_per_request <= le)
                - cum_actual[le]) <= 1e-6
            for le in closed
        )
        if ok:
            search(b + 1, iters_left - c, tokens_left - c * per_iter, acc)
        if c:
            acc.pop(b, None)


search(1, formal_iters, formal_tokens, {})


def bucket_error(sol):
    err = 0.0
    for le in les:
        pred = sum(n for b, n in sol.items() if b * args.tokens_per_request <= le)
        err += abs(pred - cum_actual[le])
    return err


solutions.sort(key=bucket_error)

result = {
    "delta_iterations": d_count,
    "delta_tokens": d_sum,
    "bucket_deltas": {str(le): v for le, v in bucket_deltas.items()},
    "warmup_deduction": {"iterations": warmup_iters, "tokens": warmup_tokens},
    "formal_iterations": formal_iters,
    "formal_tokens": formal_tokens,
    "tokens_per_request": args.tokens_per_request,
    "created_before": before["created"],
    "created_after": after["created"],
}
if formal_iters > 0:
    result["mean_batch_per_iteration"] = formal_tokens / formal_iters / args.tokens_per_request
if solutions:
    result["batch_solution"] = {str(k): v for k, v in sorted(solutions[0].items())}
    result["bucket_check_error_iterations"] = bucket_error(solutions[0])
    result["alternative_solutions"] = len(solutions) - 1
else:
    result["batch_solution"] = None
    result["note"] = "no integer combination matched iterations and tokens exactly"
json.dump(result, open(args.output, "w"), indent=1)
print(json.dumps(result, indent=1))
