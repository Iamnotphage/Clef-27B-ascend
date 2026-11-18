#!/usr/bin/env python3
"""Owner27 input-ready timing after the same window's cores24 ARon gate.

GEMM/input restoration/barriers stay outside timing. AOT compilation is recorded
separately; first-use initialization has its own samples, never steady statistics.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time
import traceback
from contract import parse_diag, sha256, verify_identity
from window_common import CELL, IDENTITY_SHA
from timing_contract import ROUND_ORDERS, REPS, WARMUP, schedule


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True, choices=["clef-27b-ascend-27"])
    ap.add_argument("--passed-cell", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--vendor-root", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or rank not in (0, 1) or local != rank or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("requires two ranks on logical devices 0,1")
    if os.environ.get("CLEF_N1_EXECUTION_OWNER") != a.owner or CELL != {"rows": 967, "seed": 20261118, "cores": 24, "use_hccl": 1}:
        raise ValueError("owner27 / cores24 ARon contract required")
    if a.passed_cell.resolve() != a.output.resolve().parent / "cell":
        raise ValueError("timing must follow the gate in the same supervised probe directory")
    if sha256(a.identity) != IDENTITY_SHA:
        raise ValueError("frozen v3 rstd8 identity required")
    identity = json.loads(a.identity.read_text())
    api = verify_identity(identity, a.vendor_root, a.adapter)
    # CPU-only: validates BOTH complete cores24 gates, identities and all raw bytes.
    from window_probe import aggregate, observed_diag
    prior = aggregate(a.passed_cell, identity)
    actual_diag = observed_diag(a.passed_cell)
    source = a.passed_cell / f"rank{rank}"
    meta = json.loads((source / "meta.json").read_text())
    raw_inputs = {name: bytearray((source / f"{name}.bin").read_bytes()) for name in ("partial", "r", "gamma")}
    out = a.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "n1-timing-rank/1", "complete": False, "performance_valid": False,
              "owner": a.owner, "rank": rank, "cell": CELL, "identity_sha256": IDENTITY_SHA,
              "adapter_sha256": sha256(a.adapter), "script_sha256": sha256(__file__),
              "source_cell": str(a.passed_cell.resolve()), "source_gate_records": prior,
              "source_observed_diag": actual_diag, "hostname": socket.gethostname(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "clock": vars(time.get_clock_info("perf_counter")),
              "orders": ROUND_ORDERS, "reps": REPS, "warmup": WARMUP,
              "compilation": {"mode": "prebuilt AOT", "package_sha256": identity["package_sha256"],
                  "build_executed_in_window": False, "runtime_compile_ms": None,
                  "note": "No compiler was invoked. Any uninstrumented lazy setup is included in first_call samples."},
              "round_gates": [], "files": {},
              "scope": "input-ready native all_reduce+AddRMSNorm versus N1; GEMM, input copies, barriers excluded; adapter allocations and diag are included; no service claim"}

    def flush():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    flush()
    try:
        setup_start_ns = time.perf_counter_ns()
        import torch
        import torch_npu
        import torch.distributed as dist
        from fixed_gate import evaluate
        torch.set_num_threads(1)
        torch.npu.set_device(local)
        dist.init_process_group("hccl", timeout=timedelta(seconds=90))
        group = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        torch.ops.load_library(str(a.adapter.resolve()))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        if report["versions"] != {"torch": "2.10.0+cpu", "torch_npu": "2.10.0.post7"}:
            raise RuntimeError("fixed torch/torch_npu versions required")
        tensors = {n: torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["files"][n]["shape"]).npu()
                   for n, raw in raw_inputs.items()}
        partial, r, g = (tensors[n] for n in ("partial", "r", "gamma"))
        scratch = {arm: torch.empty_like(partial) for arm in ("baseline", "candidate")}
        torch.npu.synchronize()
        setup_end_ns = time.perf_counter_ns()
        report["setup"] = {"start_ns": setup_start_ns, "end_ns": setup_end_ns,
                           "wall_ms": (setup_end_ns - setup_start_ns) / 1e6,
                           "scope": "backend import, group/library setup and raw input upload; excluded from call samples"}
        flush()
        last = {}

        def prepare(arm):
            scratch[arm].copy_(partial)
            dist.barrier()
            torch.npu.synchronize()

        def invoke(arm):
            if arm == "baseline":
                dist.all_reduce(scratch[arm], op=dist.ReduceOp.SUM)
                y, rs, res = torch_npu.npu_add_rms_norm(scratch[arm], r, g, meta["eps"])
                return y, res, rs.reshape(-1), scratch[arm], None
            return torch.ops.clef_n1_25_rstd8.run(scratch[arm], r, g, meta["eps"], group,
                                          1, 24, str(api), False)

        def check(round_id, save_raw):
            cand = {k: last["candidate"][i].cpu() for i, k in enumerate(("y", "residual", "rstd"))}
            ref = {k: last["baseline"][i].cpu() for i, k in enumerate(("y", "residual", "rstd"))}
            gate = evaluate(cand, ref)
            cr, rr = last["candidate"][3].cpu(), last["baseline"][3].cpu()
            equal = (cr.shape == rr.shape and cr.dtype == rr.dtype and
                     bool(torch.isfinite(cr.float()).all()) and bool(torch.isfinite(rr.float()).all()) and torch.equal(cr, rr))
            diag_tensor = last["candidate"][4].cpu()
            diag = parse_diag(diag_tensor.tolist(), 967, 24, 1, rank)
            # The baseline must also agree with this window's pre-timing native reference.
            source_ref = {}
            for key in ("y", "residual", "rstd"):
                spec = meta["files"][key + ".reference"]
                dtype = torch.float32 if key == "rstd" else torch.bfloat16
                source_ref[key] = torch.frombuffer(bytearray((source / (key + ".reference.bin")).read_bytes()), dtype=dtype).reshape(spec["shape"])
            source_gate = evaluate(ref, source_ref)
            rr_bytes = rr.contiguous().view(torch.uint8).numpy().tobytes()
            import hashlib
            source_reduced_equal = hashlib.sha256(rr_bytes).hexdigest() == meta["files"]["reduced.reference"]["sha256"]
            ok = gate["pass"] and equal and diag["pass"] and source_gate["pass"] and source_reduced_equal
            entry = {"round": round_id, "pass": ok, "fixed_gate": gate, "reduced_bit_equal": equal,
                     "diag": diag, "baseline_source_gate": source_gate, "baseline_source_reduced_equal": source_reduced_equal}
            report["round_gates"].append(entry)
            if save_raw:
                outputs = {**{k + ".candidate": v for k, v in cand.items()},
                           **{k + ".reference": v for k, v in ref.items()},
                           "reduced.candidate": cr, "reduced.reference": rr, "diag": diag_tensor}
                for name, tensor in outputs.items():
                    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    (out / (name + ".bin")).write_bytes(raw)
                    report["files"][name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                             "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            flush()
            if not ok:
                raise RuntimeError(f"post-timing output gate failed: round {round_id}")

        samples = schedule()
        with (out / "samples.jsonl").open("x") as f:
            for index, spec in enumerate(samples):
                arm = spec["arm"]
                prepare(arm)
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                # Establish events before this sample; all input restoration is complete.
                start.record(); end.record(); torch.npu.synchronize()
                begin_ns = time.perf_counter_ns()
                start.record(); outputs = invoke(arm); end.record()
                torch.npu.synchronize()
                end_ns = time.perf_counter_ns()
                last[arm] = outputs
                sample = {**spec, "rank": rank, "start_ns": begin_ns, "end_ns": end_ns,
                          "wall_ms": (end_ns - begin_ns) / 1e6, "event_ms": start.elapsed_time(end)}
                f.write(json.dumps(sample) + "\n"); f.flush()
                if spec["phase"] == "measure" and (index + 1 == len(samples) or samples[index + 1]["round"] != spec["round"]):
                    check(spec["round"], save_raw=index + 1 == len(samples))
        report["samples_sha256"] = sha256(out / "samples.jsonl")
        report["sample_count"] = len(samples)
        report["complete"] = True
        report["performance_valid"] = len(report["round_gates"]) == 4 and all(r["pass"] for r in report["round_gates"])
        flush()
        dist.destroy_process_group()
        return 0 if report["performance_valid"] else 1
    except BaseException:
        report["performance_valid"] = False
        report["error"] = traceback.format_exc()
        flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
