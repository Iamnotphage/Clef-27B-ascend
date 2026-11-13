#!/usr/bin/env python3
"""Extract effective config and memory-budget lines from each p2 point service log."""
import json, re, sys, glob, os

run = sys.argv[1]
out = {}
for d in sorted(glob.glob(os.path.join(run, "p2-s*"))):
    label = os.path.basename(d)
    log = os.path.join(d, "service.log")
    if not os.path.exists(log):
        continue
    text = open(log, errors="replace").read()
    entry = {}
    m = re.search(r"non-default args: (\{.*?\})\n", text)
    if m:
        try:
            entry["non_default_args"] = ast_literal = eval(m.group(1))  # vLLM prints python dict repr
        except Exception:
            entry["non_default_args_raw"] = m.group(1)[:400]
    for rank in re.finditer(r"\[(Worker_TP\d+)[^\]]*\] ([^\n]*(?:KV cache memory|Actual usage|Model loading)[^\n]*)", text):
        entry.setdefault(rank.group(1), []).append(rank.group(2))
    out[label] = entry
print(json.dumps(out, indent=1))
