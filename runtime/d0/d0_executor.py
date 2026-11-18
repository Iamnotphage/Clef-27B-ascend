"""Real D0 executor. Imported by d0_run.py; never imported by the parser.

This module is the only place that would touch the device, and it does so
only through the standalone ClefD0Trace op. It is written to be reviewable:
every step records what it did, and the failure paths are explicit.

The sequence is fixed:
    clear trace on the explicit stream -> launch -> synchronize -> read back
Values read back are normalised with & 0xFFFFFFFF because the container is
int32 on this stack while the tensor view is uint32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TRACE_SLOTS = 72
TRACE_FIELDS = 8
U32 = 0xFFFFFFFF


def tensor_hash(t) -> str:
    return hashlib.sha256(t.view(-1).cpu().numpy().tobytes()).hexdigest()


def describe(t, name: str) -> dict:
    return {"name": name, "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device),
            "contiguous": bool(t.is_contiguous()),
            "sha256": tensor_hash(t)}


def normalize_trace(rows: list[list[int]]) -> list[int]:
    """Flatten a [72, 8] container readback to unsigned 32-bit values."""
    flat = [v for row in rows for v in row]
    return [int(v) & U32 for v in flat]


def run(torch, torch_npu, args, out_dir: Path) -> dict:
    """Execute one D0 cell. Caller has already validated exclusivity."""
    record = {"cell": {"mm_dim": args.mm_dim, "use_hccl": args.use_hccl,
                       "rows": args.rows, "seed": args.seed}}

    torch.npu.config.allow_internal_format = True
    from safetensors import safe_open  # local import: keeps the parser light
    import json as _json

    model = Path("/workspace/yangchen/models/Clef-27B")
    idx = _json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.mlp.down_proj.weight"
    with safe_open(model / idx[key], framework="pt", device="cpu") as h:
        w_cpu = h.get_tensor(key)
    per = w_cpu.shape[1] // 2
    w_nd = w_cpu[:, 0:per].t().contiguous().npu()
    w_nz = torch_npu.npu_format_cast(w_nd, 29)
    record["w_format_after_cast"] = torch_npu.get_npu_format(w_nz)
    record["w_nd_format"] = torch_npu.get_npu_format(w_nd)

    k, n = w_nd.shape
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows, k, generator=g).to(torch.bfloat16).npu()
    r = torch.randn(args.rows, n, generator=g).to(torch.bfloat16).npu()
    gamma = torch.rand(n, generator=g).to(torch.bfloat16).npu()
    record["inputs"] = [describe(x, "x"), describe(w_nz, "w_nz"),
                        describe(r, "residual"), describe(gamma, "gamma")]

    out, res_out, rstd, diag, trace = torch.ops.clef_d0.d0_trace(
        x, w_nz, r, gamma, 1e-6, 1, "hccl_world", args.use_hccl, -1, 1,
        args.mm_dim, 2, 0, 1, 0, 0, 0, 0)
    torch.npu.synchronize()                    # explicit sync before readback

    rows = trace.cpu().tolist()
    record["trace_bytes"] = TRACE_SLOTS * TRACE_FIELDS * 4
    record["trace_shape"] = list(trace.shape)
    record["raw_trace"] = rows
    record["flat_u32"] = normalize_trace(rows)
    (out_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def summarize_failure(exc: BaseException) -> dict:
    return {"failure": type(exc).__name__, "message": str(exc)[:2000],
            "trace_state": "unavailable (launch or readback did not complete)"}
