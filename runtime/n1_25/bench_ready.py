#!/usr/bin/env python3
"""Owner21-only input-ready AB/BA, gated by a matching successful AR-on cell.

Measures native AR+AddRmsNorm versus N1 on explicit native GEMM partial bytes.
GEMM is outside this subchain measurement. Copies and barriers are outside timing.
No installation, compilation, or service lifecycle actions occur here.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time
from contract import parse_diag, sha256, verify_identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True, choices=["clef-27b-ascend-21"])
    ap.add_argument("--passed-cell", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--vendor-root", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=20)
    a = ap.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or rank not in (0, 1) or local != rank or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("requires owner21 two-rank logical 0,1 window")
    if not (1 <= a.warmup <= 20 and 1 <= a.pairs <= 100):
        raise ValueError("benchmark bounds exceeded")
    identity = json.loads(a.identity.read_text())
    api = verify_identity(identity, a.vendor_root, a.adapter)
    metas = []
    for i in (0, 1):
        root = a.passed_cell / f"rank{i}"
        meta = json.loads((root / "meta.json").read_text())
        gate = json.loads((root / "result.json").read_text())
        if (not gate.get("pass") or not gate["fixed_gate"]["pass"] or
            not gate["reduced_bit_equal"] or not gate["diag"]["pass"] or
            meta["use_hccl"] != 1 or meta["identity_sha256"] != sha256(a.identity) or
            meta["adapter_sha256"] != identity["adapter_sha256"]):
            raise ValueError(f"rank{i}: matching ARon fixed gate required")
        metas.append(meta)
    if any(metas[0][k] != metas[1][k] for k in ("rows", "cores", "seed", "eps")):
        raise ValueError("gate cells disagree")
    m = metas[rank]; source = a.passed_cell / f"rank{rank}"
    raw_inputs = {}
    for name in ("partial", "r", "gamma"):
        path = source / f"{name}.bin"; spec = m["files"][name]
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"input identity mismatch: {path}")
        raw_inputs[name] = bytearray(path.read_bytes())
    out = a.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu
    import torch.distributed as dist
    from fixed_gate import evaluate
    torch.set_num_threads(1)
    torch.npu.set_device(local)
    dist.init_process_group("hccl", timeout=timedelta(seconds=90))
    group = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.ops.load_library(str(a.adapter.resolve()))
    tensors = {n: torch.frombuffer(raw, dtype=torch.bfloat16).reshape(m["files"][n]["shape"]).npu()
               for n, raw in raw_inputs.items()}
    partial, r, g = (tensors[n] for n in ("partial", "r", "gamma"))
    scratch = {arm: torch.empty_like(partial) for arm in ("baseline", "candidate")}

    def prepare(arm):
        scratch[arm].copy_(partial)
        dist.barrier()
        torch.npu.synchronize()

    def run(arm):
        if arm == "baseline":
            dist.all_reduce(scratch[arm])
            y, rs, res = torch_npu.npu_add_rms_norm(scratch[arm], r, g, m["eps"])
            return y, res, rs.reshape(-1), scratch[arm], None
        return torch.ops.clef_n1_25.run(scratch[arm], r, g, m["eps"], group,
                                      1, m["cores"], str(api), False)

    for _ in range(a.warmup):
        for arm in ("baseline", "candidate"):
            prepare(arm); run(arm)
    torch.npu.synchronize()
    last = {}
    with (out / "samples.jsonl").open("x") as f:
        for pair in range(a.pairs):
            order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
            for arm in order:
                prepare(arm)
                start = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
                begin_ns = time.perf_counter_ns()
                start.record(); outputs = run(arm); end.record()
                torch.npu.synchronize()
                end_ns = time.perf_counter_ns()
                last[arm] = outputs
                f.write(json.dumps({"pair": pair, "arm": arm, "rank": rank,
                                    "start_ns": begin_ns, "end_ns": end_ns,
                                    "wall_ms": (end_ns-begin_ns)/1e6,
                                    "event_ms": start.elapsed_time(end)}) + "\n")
                f.flush()
    cand = {k: last["candidate"][i].cpu() for i, k in enumerate(("y", "residual", "rstd"))}
    ref = {k: last["baseline"][i].cpu() for i, k in enumerate(("y", "residual", "rstd"))}
    gate = evaluate(cand, ref)
    equal = torch.equal(last["candidate"][3], last["baseline"][3])
    diag = parse_diag(last["candidate"][4].cpu().tolist(), m["rows"], m["cores"], 1, rank)
    valid = gate["pass"] and equal and diag["pass"]
    result = {"performance_valid": valid, "final_fixed_gate": gate, "reduced_bit_equal": equal,
              "diag": diag, "rows": m["rows"], "cores": m["cores"], "rank": rank,
              "hostname": socket.gethostname(), "clock": vars(time.get_clock_info("perf_counter")),
              "source_cell": str(a.passed_cell.resolve()), "identity_sha256": sha256(a.identity),
              "script_sha256": sha256(__file__), "warmup": a.warmup, "pairs": a.pairs,
              "scope": "input-ready AR+AddRmsNorm; GEMM/copies/barriers outside timing; no service claim"}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
