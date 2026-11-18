#!/usr/bin/env python3
"""N1 adapter for --tree/--device/--output in worker24's supervised window."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import traceback

from window_common import CELL, IDENTITY_SHA, VENDOR, check_head, digest, frozen, staged_paths


def command(stage, run, python):
    return [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0", "--run-path", "--start-method=spawn",
            str(stage / "run_cell.py"), "--owner", "clef-27b-ascend-21",
            "--identity", str(stage / "build_identity.json"),
            "--vendor-root", str(run / "isolated-opp/vendors" / VENDOR),
            "--adapter", str(stage / "n1_adapter.so"), "--output", str(run / "probe/cell"),
            "--rows", "967", "--seed", "20261118", "--cores", "24", "--use-hccl", "1"]


def bench_command(stage, run, python):
    return [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0", "--run-path", "--start-method=spawn",
            str(stage / "bench_ready.py"), "--owner", "clef-27b-ascend-27",
            "--passed-cell", str(run / "probe/cell"),
            "--identity", str(stage / "build_identity.json"),
            "--vendor-root", str(run / "isolated-opp/vendors" / VENDOR),
            "--adapter", str(stage / "n1_adapter.so"), "--output", str(run / "probe/timing")]


def observed_diag(cell):
    from contract import parse_diag
    result = []
    for rank in (0, 1):
        raw = (cell / f"rank{rank}/diag.bin").read_bytes()
        words = struct.unpack("<1536I", raw)
        rows = [list(words[i:i + 32]) for i in range(0, 1536, 32)]
        parsed = parse_diag(rows, CELL["rows"], CELL["cores"], CELL["use_hccl"], rank)
        if parsed["pass"] is not True:
            raise ValueError(f"rank{rank} raw diagnostic failed: {parsed['errors']}")
        slots = [{"slot": i, "block_id": r[2], "recorded_block_num": r[3],
                  "row_start": r[4], "row_end": r[5], "words": r}
                 for i, r in enumerate(rows) if any(r)]
        result.append({"rank": rank, "requested_cores": CELL["cores"],
                       "observed_nonzero_slots": slots, "observed_nonzero_slot_count": len(slots),
                       "note": "actual terminal diagnostic bytes; requested core budget is recorded separately"})
    return result


def aggregate_timing(folder, stage, gate_records):
    from timing_contract import aggregate_samples
    samples, metas = [], []
    required_files = {"y.candidate", "y.reference", "residual.candidate", "residual.reference",
                      "rstd.candidate", "rstd.reference", "reduced.candidate", "reduced.reference", "diag"}
    for rank in (0, 1):
        root = folder / f"rank{rank}"
        report = json.loads((root / "result.json").read_text())
        if (report.get("schema") != "n1-timing-rank/1" or report.get("rank") != rank or
                report.get("owner") != "clef-27b-ascend-27" or report.get("cell") != CELL or
                report.get("complete") is not True or report.get("performance_valid") is not True or
                report.get("error") is not None or report.get("identity_sha256") != IDENTITY_SHA or
                report.get("adapter_sha256") != digest(stage / "n1_adapter.so") or
                report.get("script_sha256") != digest(stage / "bench_ready.py") or
                report.get("source_gate_records") != gate_records):
            raise ValueError(f"rank{rank} timing result identity/gate mismatch")
        gates = report.get("round_gates", [])
        if len(gates) != 4 or any(g.get("round") != i or g.get("pass") is not True or
                g.get("fixed_gate", {}).get("pass") is not True or
                g.get("baseline_source_gate", {}).get("pass") is not True or
                g.get("diag", {}).get("pass") is not True or g.get("reduced_bit_equal") is not True or
                g.get("baseline_source_reduced_equal") is not True for i, g in enumerate(gates)):
            raise ValueError(f"rank{rank} timing post-output gate failed")
        if set(report.get("files", {})) != required_files:
            raise ValueError(f"rank{rank} incomplete timing raw outputs")
        for name, spec in report["files"].items():
            path = root / (name + ".bin")
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError(f"rank{rank} timing raw output hash mismatch")
        if digest(root / "samples.jsonl") != report.get("samples_sha256"):
            raise ValueError(f"rank{rank} timing sample hash mismatch")
        samples.append([json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()])
        metas.append(report)
    summary = aggregate_samples(samples, metas)
    summary["post_observed_diag"] = observed_diag(folder)
    summary["rank_result_sha256"] = [digest(folder / f"rank{r}/result.json") for r in (0, 1)]
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"pass": True, "summary_path": str(folder / "summary.json"),
            "summary_sha256": digest(folder / "summary.json"), "measure": summary["measure"]}


def aggregate(cell, identity):
    ranks = []
    for rank in (0, 1):
        root = cell / f"rank{rank}"
        meta = json.loads((root / "meta.json").read_text())
        result = json.loads((root / "result.json").read_text())
        expected = {**CELL, "rank": rank, "owner": "clef-27b-ascend-21", "stage": "complete",
                    "identity_sha256": IDENTITY_SHA, "adapter_sha256": identity["adapter_sha256"],
                    "api_sha256": identity["vendor_files"]["op_api/lib/libcust_opapi.so"], "eps": 1e-6}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError(f"rank{rank} meta/cell identity mismatch")
        if (meta.get("pass") is not True or result.get("pass") is not True or
                result.get("fixed_gate", {}).get("pass") is not True or
                result.get("reduced_bit_equal") is not True or
                result.get("diag", {}).get("pass") is not True or meta.get("error") is not None):
            raise ValueError(f"rank{rank} gate/diag/reduced failed")
        required = {"partial", "r", "gamma", "diag", "reduced.reference", "reduced.candidate",
                    "y.reference", "y.candidate", "residual.reference", "residual.candidate",
                    "rstd.reference", "rstd.candidate"}
        if set(meta.get("files", {})) != required or set(meta.get("inputs", {})) != {"x", "w", "r", "gamma", "partial"}:
            raise ValueError(f"rank{rank} missing raw/input evidence")
        for name in ("partial", "r", "gamma"):
            if meta["inputs"][name] != meta["files"][name]:
                raise ValueError(f"rank{rank} input/raw identity mismatch: {name}")
        for name, item in meta["files"].items():
            path = root / (name + ".bin")
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"rank{rank} raw byte identity mismatch: {name}")
        ranks.append({"rank": rank, "meta_sha256": digest(root / "meta.json"),
                      "result_sha256": digest(root / "result.json"), "inputs": meta["inputs"]})
    for name in ("x", "r", "gamma"):
        if ranks[0]["inputs"][name] != ranks[1]["inputs"][name]:
            raise ValueError(f"cross-rank input mismatch: {name}")
    return ranks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = {"schema": "n1-probe/1", "gate_passed": False, "cell": CELL, "error": None}
    try:
        stage, run = staged_paths(__file__, a.output.resolve().parent.parent)
        if a.device != 0 or a.output.resolve().parent != run / "probe":
            raise ValueError("N1 cell requires device=0 and supervised probe output")
        check_head(a.tree)
        identity = frozen(stage)
        from contract import verify_identity
        vendor = run / "isolated-opp/vendors" / VENDOR
        verify_identity(identity, vendor, stage / "n1_adapter.so")
        env = dict(os.environ)
        if env.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1":
            raise ValueError("child visible devices must be 0,1")
        if env.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")[0] != str(vendor):
            raise ValueError("child OPP prepend mismatch")
        if env.get("LD_LIBRARY_PATH", "").split(":")[0] != str(vendor / "op_api/lib"):
            raise ValueError("child opapi prepend mismatch")
        env["PYTHONPATH"] = str(stage) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # --run-path selects MultiprocessContext. Both spawned ranks inherit this
        # process group; plain torchrun uses per-rank start_new_session=True.
        cmd = command(stage, run, sys.executable)
        report.update(argv=cmd, pgid=os.getpgrp(), identity_sha256=IDENTITY_SHA,
                      child_env={k: env[k] for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH")})
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        rc = subprocess.call(cmd, env=env, start_new_session=False)
        report["child_rc"] = rc
        if rc != 0:
            raise RuntimeError(f"N1 two-rank child failed: {rc}")
        report["ranks"] = aggregate(run / "probe/cell", identity)
        report["observed_diag"] = observed_diag(run / "probe/cell")
        report["correctness_gate_passed"] = True
        report["timing_argv"] = bench_command(stage, run, sys.executable)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        timing_rc = subprocess.call(report["timing_argv"], env=env, start_new_session=False)
        report["timing_child_rc"] = timing_rc
        if timing_rc != 0:
            raise RuntimeError(f"N1 timing child failed: {timing_rc}")
        report["timing"] = aggregate_timing(run / "probe/timing", stage, report["ranks"])
        report["gate_passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
