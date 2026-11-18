"""Real D0 executor: one cell, one rank, explicit adapter load.

Nothing here is executed from a CPU-only review. The pieces that would touch
the device are the explicit load of a hash-checked prebuilt extension and the
single op call; both are parameterised so the caller supplies the artefacts
rather than discovering them at call time.

Sequence per rank:
    metadata flushed -> clear on the explicit stream -> launch -> sync -> read back
Readback is normalised with & 0xFFFFFFFF because the container is int32 while
the aclTensor view is uint32.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

TRACE_SLOTS = 72
TRACE_FIELDS = 8
U32 = 0xFFFFFFFF


def raw_bytes_hash(t) -> str:
    """Hash a BF16 tensor through a uint16 view.

    `t.numpy()` refuses bfloat16, so the bytes are taken from a uint16 view of
    the contiguous CPU copy instead.
    """
    import torch
    # Host-side bytes only: copy out first, then view as uint16. Never view an
    # NPU or NZ tensor as another dtype.
    host = t.detach().cpu().contiguous()
    return hashlib.sha256(host.view(torch.uint16).numpy().tobytes()).hexdigest()


def describe(t, name: str, extra: dict | None = None) -> dict:
    import torch
    d = {"name": name, "shape": list(t.shape), "stride": list(t.stride()),
         "dtype": str(t.dtype), "device": str(t.device),
         "contiguous": bool(t.is_contiguous())}
    if t.dtype == torch.bfloat16:
        d["sha256_uint16"] = raw_bytes_hash(t)
    elif t.is_floating_point() or t.dtype == torch.int32:
        d["sha256_bytes"] = hashlib.sha256(
            t.contiguous().cpu().numpy().tobytes()).hexdigest()
    if extra:
        d.update(extra)
    return d


def load_adapter(so_path: Path, expected_sha: str):
    """Load the prebuilt, hash-checked extension. No on-the-fly compilation."""
    import torch
    actual = hashlib.sha256(so_path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(f"adapter sha mismatch: {actual} != {expected_sha}")
    torch.ops.load_library(str(so_path))
    return actual


def normalize_trace(rows: list[list[int]]) -> list[int]:
    return [int(v) & U32 for row in rows for v in row]


def precheck(env, args, out_dir: Path) -> dict:
    """Everything that must be true before the device is touched.

    No set_device, no process-group init, no adapter load happens until this
    returns; an illegal mapping therefore cannot reach any of them.
    """
    world = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", str(rank)))
    if world != 2:
        raise RuntimeError(f"WORLD_SIZE must be 2 for this cell, got {world}")
    if rank not in (0, 1):
        raise RuntimeError(f"RANK must be 0 or 1, got {rank}")
    if local_rank not in (0, 1):
        raise RuntimeError(f"LOCAL_RANK must be 0 or 1, got {local_rank}")
    if local_rank != rank:
        # One card, two logical devices: this cell binds rank r to device r.
        raise RuntimeError(f"LOCAL_RANK {local_rank} does not match RANK {rank} for this cell")
    if args.mm_dim != 1 or args.use_hccl != 2:
        raise RuntimeError(f"cell must be mm_dim=1/use_hccl=2, got "
                           f"{args.mm_dim}/{args.use_hccl}")
    if not Path(args.adapter).is_file():
        raise RuntimeError(f"adapter not found: {args.adapter}")
    actual = hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest()
    if actual != args.adapter_sha:
        raise RuntimeError(f"adapter sha mismatch: {actual} != {args.adapter_sha}")
    meta = {"rank": rank, "local_rank": local_rank, "world_size": world,
            "group": None,                       # filled only after real acquisition
            "cell": {"mm_dim": args.mm_dim, "use_hccl": args.use_hccl,
                     "rows": args.rows, "seed": args.seed},
            "adapter": {"path": str(args.adapter), "expected_sha": args.adapter_sha,
                        "actual_sha": actual},
            "precheck": "passed before any device call"}
    (out_dir / f"meta-rank{rank}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n")   # flush #1: pre-device
    return meta


def flush(out_dir: Path, meta: dict) -> None:
    (out_dir / f"meta-rank{meta['rank']}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n")


def run(torch, torch_npu, args, out_dir: Path, dist=None, env=None) -> dict:
    env = dict(os.environ if env is None else env)
    meta = precheck(env, args, out_dir)          # nothing device-side yet
    rank, local_rank = meta["rank"], meta["local_rank"]

    torch.npu.set_device(local_rank)                      # one card, logical 0/1
    if dist is not None and not dist.is_initialized():
        dist.init_process_group("hccl", init_method="env://")
    group_name = (dist.group.WORLD._get_backend(torch.device("npu"))
                  .get_hccl_comm_name(rank)) if dist is not None else None
    meta["group"] = group_name
    flush(out_dir, meta)                                   # flush #2: real group known

    # Explicit, hash-checked load: the op stays unregistered without this call.
    loaded_sha = load_adapter(Path(args.adapter), args.adapter_sha)
    meta["adapter"]["loaded_sha"] = loaded_sha
    meta["adapter"]["load_stage"] = "loaded after precheck, before op call"
    flush(out_dir, meta)

    torch.npu.config.allow_internal_format = True
    from safetensors import safe_open

    model = Path("/workspace/yangchen/models/Clef-27B")
    idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.mlp.down_proj.weight"
    shard = idx[key]
    with safe_open(model / shard, framework="pt", device="cpu") as h:
        w_cpu = h.get_tensor(key)
    per = w_cpu.shape[1] // 2
    w_nd = w_cpu[:, rank * per:(rank + 1) * per].t().contiguous().npu()
    w_nz = torch_npu.npu_format_cast(w_nd, 29)
    meta["weight"] = {"key": key, "shard": shard, "tp": 2,
                      "rank_slice": [rank * per, (rank + 1) * per],
                      "nd_format": torch_npu.get_npu_format(w_nd),
                      "nz_format": torch_npu.get_npu_format(w_nz)}

    k, n = w_nd.shape
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows, k, generator=g).to(torch.bfloat16).npu()
    r = torch.randn(args.rows, n, generator=g).to(torch.bfloat16).npu()
    gamma = torch.rand(n, generator=g).to(torch.bfloat16).npu()

    dev = x.device
    for t, nm in ((w_nz, "w"), (r, "residual"), (gamma, "gamma")):
        if t.device != dev:
            raise RuntimeError(f"{nm} on {t.device}, expected {dev}")
    if x.shape[1] != w_nz.shape[0]:
        raise RuntimeError(f"x K={x.shape[1]} != w K={w_nz.shape[0]}")
    meta["inputs"] = [describe(x, "x"), describe(w_nz, "w_nz"),
                      describe(r, "residual"), describe(gamma, "gamma"),
                      {"name": "trace", "shape": [TRACE_SLOTS, TRACE_FIELDS],
                       "container_dtype": "int32", "acl_view": "uint32",
                       "bytes": TRACE_SLOTS * TRACE_FIELDS * 4}]
    meta["pre_call"] = {"device": str(dev), "k": int(k), "n": int(n),
                        "rows": int(args.rows), "use_hccl": args.use_hccl,
                        "mm_dim": args.mm_dim}
    flush(out_dir, meta)                                   # flush #3: complete, pre-call

    out, res_out, rstd, diag, trace = torch.ops.clef_d0.d0_trace(
        x, w_nz, r, gamma, 1e-6, 1, group_name, args.use_hccl, -1, 1,
        args.mm_dim, 2, 0, 1, 0, 0, 0, 0)
    torch.npu.synchronize()

    if tuple(trace.shape) != (TRACE_SLOTS, TRACE_FIELDS):
        raise RuntimeError(f"trace shape {tuple(trace.shape)} != {(TRACE_SLOTS, TRACE_FIELDS)}")
    if int(trace.numel()) != TRACE_SLOTS * TRACE_FIELDS:
        raise RuntimeError(f"trace numel {int(trace.numel())} != {TRACE_SLOTS * TRACE_FIELDS}")
    rows = trace.cpu().tolist()
    record = {"rank": rank, "group": group_name, "raw_trace": rows,
              "flat_u32": normalize_trace(rows),
              "diag": [float(v) for v in diag.cpu()],
              "trace_shape": list(trace.shape),
              "trace_bytes": TRACE_SLOTS * TRACE_FIELDS * 4}
    (out_dir / f"result-rank{rank}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def summarize_failure(exc: BaseException) -> dict:
    return {"failure": type(exc).__name__, "message": str(exc)[:4000],
            "trace_state": "unavailable (launch or readback did not complete)"}
