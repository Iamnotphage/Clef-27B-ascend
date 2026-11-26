"""Tier 2 engine hooks: B=1 continuous-KV / GDN / joint-memory reuse.

This module is a probe-grade implementation installed by
``probe_engine/sitecustomize.py``.  It patches four points:

- ``NPUModelRunner.execute_model``: build the prefix split/key, run the TP2
  consensus vote, and bind/publish HostStore state around the model call;
- ``ClefFlashForJointSchema.forward``: one full-length tap pass on populate,
  or the suffix-only resume pass on a hit;
- ``AscendAttentionBackendImpl.forward``: tap post-RoPE K/V; resume with
  ``cat([K_prefix, K_suffix])`` + FIA ``TND``/``sparse_mode=3``;
- ``QwenGatedDeltaNetAttention._forward_core`` / ``DeviceOperator.fla_gdn_prefill``:
  tap conv/ssm at ``L``; resume through the reference conv and
  ``fla_gdn_prefill(initial_state=...)``;
- ``JointSchemaPooler``/``JointSchemaHead``: tap ``prefix_memory`` while the
  stock pooler runs; consume it on hits without the ``AllPool`` truncation.

Populate is a **single** full-length pass: the whole prompt runs the stock
decoupled path, so the warm-up round is bitwise equal to the OFF service, and
the layer hooks only slice the states that are already materialised.  The one
extra cost is the GDN state at ``L``, which the fused prefill kernel can only
return for a whole call, so each GDN layer re-runs its prefix slice once.

The captured package travels a two-slot pinned D2H ring into the pageable
HostStore; hits restore through the pinned ``Prefetcher`` staging pool.  The
only admission of ``miss`` (no populate) is the aged keep-longest rule.

Only one fresh request per step is admitted in this milestone; anything else
falls back to the stock path.  It never touches vLLM's prefix-caching or KV
cache manager.
"""

from __future__ import annotations

import os
import queue
import re
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 - workers already have the backend loaded

from runtime.tiered_prefix_cache.async_transfer import (
    AsyncCapture,
    PinnedRing,
    Prefetcher,
)
from runtime.tiered_prefix_cache.host_store import (
    HostStore,
    tp_vote_result,
    tp_vote_vector,
)
from runtime.tiered_prefix_cache.prefix_key import (
    MmItem,
    REPRESENTATION_BOUNDARY,
    compute_prefix_key,
    compute_representation_key,
    locate_state_prefix,
)
from runtime.tiered_prefix_cache.representation_plan import (
    MODE_HIT as REPR_MODE_HIT,
    MODE_MISS as REPR_MODE_MISS,
    MODE_POPULATE as REPR_MODE_POPULATE,
    PATH_REPRESENTATION,
    PATH_STATE_RESUME,
    choose_single_path,
    plan_batch,
    should_capture,
)
from runtime.tiered_prefix_cache.weight_pages import (
    drop_safetensors_page_cache,
)

DELIMITER_TEXT = "\n\nSCHEMA FIELDS:\n"
ALIGN = 128
TEXT_MIN_PREFIX = 512
MM_MIN_PREFIX = 128
#: Tier 2A stores the continuous ``(hidden, residual)`` pair that enters this
#: layer.  Both tensors are BF16 ``[T, hidden]`` on every TP rank.
REPR_NAMESPACE = "clef-tier2a"
REPRESENTATION_TOKEN_BYTES = 2 * 2 * 5120  # fallback when hidden_size is absent
#: One full dataset round observes at most 418 (OD) / 400 (ERQA) eligible
#: prefixes, so 500 sightings keep a live dataset active while a previous
#: dataset's entries age out within the next dataset's first measured pass.
#: Larger windows (the former 3000) let a departed 25 GB working set linger
#: for many rounds and force the newcomer into eviction churn.
ACTIVE_WINDOW = 500
#: Tier 2B per-rank state package.  The multi-granularity split reserves most
#: of the host budget for Tier 2A representations and keeps 10 GiB for the
#: aged prefix-state pool.
TIER2B_RANK_BYTES_DEFAULT = 10 << 30
TIER2A_RANK_BYTES_DEFAULT = 28 << 30
#: TP2 state package per rank: 16 full-attention layers x K/V x 2 heads x
#: 256 elements x 2 bytes = 32 KiB/token, plus the rank-0 2 KiB/token memory
#: projection.  Both ranks use the same estimate so the admission decision
#: is byte-identical across TP workers.
PER_TOKEN_BYTES = 34 * 1024
STATE_TAIL_BYTES = 40 << 20
_STATE: dict = {
    "store": None,
    "store2a": None,
    "tokenizer": None,
    "delimiter_ids": None,
    "mask": None,
    "ctx": None,
    "pool_ctx": None,
    "pool_only": False,
    "lock": threading.RLock(),
    "capture_layer": None,
    "capture": None,
    "capture2a": None,
    "capture2a_ring": None,
    "capture2a_pending": None,
    "capture2a_stock": None,
    "repr_boundary": None,
    "prefetcher": None,
    "batch": None,
    "batch_tap": None,
    "batch_pool": None,
    "batch_layer": None,
    "sched": None,
    "runner": None,
    "packed_req_ids": None,
    "sub": None,
}


def _flag(name: str) -> bool:
    return Path(f"/tmp/{name}").is_file()


def _env_on() -> bool:
    value = os.environ.get("CLEF_PREFIX_STATE_CACHE")
    if value is not None:
        return value.strip().lower() not in ("0", "false", "no", "off")
    return _flag("clef_tier2_engine") or os.environ.get(
        "CLEF_TIER2_ENGINE", "0").strip().lower() in ("1", "true", "yes", "on")


def _trace_on() -> bool:
    return _flag("clef_tier2_trace") or os.environ.get(
        "CLEF_TIER2_TRACE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _debug(msg: str) -> None:
    if not os.path.exists("/tmp/clef_tier2_debug"):
        return
    try:
        with open("/tmp/clef_tier2_engine.log", "a") as handle:
            handle.write(f"{os.getpid()} {time.time():.3f} {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _trace(msg: str) -> None:
    if not _trace_on():
        return
    # One write per line: two TP workers share this stdout file, and print()
    # can split the payload from its newline and interleave the lines.
    line = f"[tier2-timing] pid={os.getpid()} {msg}\n".encode()
    try:
        os.write(1, line)
    except OSError:
        pass


def _device() -> str:
    return f"npu:{torch.npu.current_device()}"


def _mask() -> torch.Tensor:
    if _STATE["mask"] is None:
        tril = torch.ones((2048, 2048), dtype=torch.bool, device=_device()).tril_()
        _STATE["mask"] = ~tril
    return _STATE["mask"]


def _store() -> HostStore:
    if _STATE["store"] is None:
        # Deterministic per-rank budget: both TP workers must agree byte for
        # byte, so never read /proc/meminfo here.
        budget = int(os.environ.get(
            "CLEF_PREFIX_CACHE_RANK_BYTES", str(TIER2B_RANK_BYTES_DEFAULT)))
        _STATE["store"] = HostStore(
            budget_bytes=budget, admission="always", use_shm=False,
            eviction_policy="aged_keep_longest",
            active_window=int(
                os.environ.get("CLEF_TIER2_ACTIVE_WINDOW")
                or os.environ.get("CLEF_PREFIX_ACTIVE_WINDOW",
                                  str(ACTIVE_WINDOW))),
            eviction_watermark=0.95)
    return _STATE["store"]


def _repr_enabled() -> bool:
    """Tier 2A is part of the multi-granularity cache and defaults on."""
    value = os.environ.get("CLEF_TIER2A")
    if value is not None:
        return value.strip().lower() not in ("0", "false", "no", "off")
    budget = os.environ.get("CLEF_TIER2A_RANK_BYTES")
    if budget is not None:
        try:
            return int(budget) > 0
        except ValueError:
            return True
    return True


def _repr_boundary() -> int:
    if _STATE["repr_boundary"] is None:
        _STATE["repr_boundary"] = int(os.environ.get(
            "CLEF_TIER2A_BOUNDARY", str(REPRESENTATION_BOUNDARY)))
    return _STATE["repr_boundary"]


def _store2a() -> HostStore:
    if _STATE["store2a"] is None:
        # Pageable, LRU: the full OD + ERQA working set is expected to fit, so
        # a warm round sees zero evictions; pressure evicts the least-recently
        # used prompt instead of rejecting short prompts.
        budget = int(os.environ.get(
            "CLEF_TIER2A_RANK_BYTES", str(TIER2A_RANK_BYTES_DEFAULT)))
        _STATE["store2a"] = HostStore(
            budget_bytes=budget, admission="always", use_shm=False,
            eviction_policy="lru", eviction_watermark=0.98)
    return _STATE["store2a"]


def _capture2a() -> AsyncCapture:
    if _STATE["capture2a_ring"] is None:
        slot_bytes = int(os.environ.get(
            "CLEF_TIER2A_RING_BYTES", str(640 << 20)))
        slots = int(os.environ.get("CLEF_TIER2A_RING_SLOTS", "4"))
        _STATE["capture2a_ring"] = AsyncCapture(
            PinnedRing(slots=slots, slot_bytes=slot_bytes))
    return _STATE["capture2a_ring"]


def _hidden_size(runner=None) -> int:
    try:
        config = getattr(runner.model_config, "hf_text_config", None)
        if config is None:
            config = runner.model_config.hf_config
        value = int(getattr(config, "hidden_size", 0))
        if value > 0:
            return value
    except Exception:  # noqa: BLE001
        pass
    try:
        from transformers import AutoConfig  # noqa: PLC0415

        path = os.environ.get("CLEF_MODEL_DIR")
        if path:
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            value = int(getattr(config, "hidden_size", 0))
            if value > 0:
                return value
    except Exception:  # noqa: BLE001
        pass
    return 5120


def _repr_estimate(length: int, runner=None) -> int:
    return int(length) * _hidden_size(runner) * 2 * 2


def _tp_size(runner=None) -> int:
    try:
        return int(runner.model_config.parallel_config.tensor_parallel_size)
    except Exception:  # noqa: BLE001
        return int(os.environ.get("CLEF_TENSOR_PARALLEL_SIZE", "2"))


def _request_mm_items(req) -> tuple[MmItem, ...]:
    return tuple(
        MmItem(
            mm_hash=str(f.identifier or f.mm_hash or ""),
            placeholder_offset=int(f.mm_position.offset),
            placeholder_length=int(f.mm_position.length),
            expanded_offset=int(f.mm_position.offset),
            expanded_length=int(f.mm_position.length),
            modality=str(f.modality),
        )
        for f in (getattr(req, "mm_features", ()) or ())
    )


def _repr_key(tokens, mm_items, runner) -> bytes:
    return compute_representation_key(
        REPR_NAMESPACE, tuple(tokens), tuple(mm_items),
        model_revision=str(getattr(runner.model_config, "revision", "") or ""),
        tp_size=_tp_size(runner), boundary=_repr_boundary())


def _capture() -> AsyncCapture:
    if _STATE["capture"] is None:
        slot_bytes = int(os.environ.get(
            "CLEF_TIER2_RING_BYTES", str(640 << 20)))
        slots = int(os.environ.get("CLEF_TIER2_RING_SLOTS", "8"))
        _STATE["capture"] = AsyncCapture(
            PinnedRing(slots=slots, slot_bytes=slot_bytes))
    return _STATE["capture"]


def _prefetcher() -> Prefetcher:
    if _STATE["prefetcher"] is None:
        _STATE["prefetcher"] = Prefetcher(
            staging_slots=2,
            staging_bytes=int(os.environ.get(
                "CLEF_TIER2_STAGING_BYTES", str(384 << 20))))
    return _STATE["prefetcher"]


def _estimate_package_bytes(length: int) -> int:
    return int(length) * PER_TOKEN_BYTES + STATE_TAIL_BYTES


def _delimiter_ids(model_config) -> list[int]:
    if _STATE["delimiter_ids"] is None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        path = model_config.tokenizer or model_config.model
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        _STATE["tokenizer"] = tokenizer
        _STATE["delimiter_ids"] = tokenizer.encode(
            DELIMITER_TEXT, add_special_tokens=False)
    return _STATE["delimiter_ids"]


def _joint_rank0_only() -> bool:
    return os.environ.get(
        "CLEF_JOINT_RANK0_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on")


class Context:
    def __init__(self, key, split, mode, rank):
        self.key = key
        self.split = split
        self.mode = mode
        self.rank = rank
        self.T = split.total_tokens
        self.L = split.l_aligned
        self.S = split.l_suffix
        self.phase = "tap" if mode == "populate" else "resume"
        self.prefix_memory = None
        self.model_done = False
        self.pool_done = False
        self.captured_kv = {}
        self.captured_conv = {}
        self.captured_ssm = {}
        self.restored = None
        self.restored_kv = {}
        self.restored_conv = {}
        self.restored_ssm = {}
        self.lease = None
        self.ssm_group = 0
        self.ssm_pending: list[tuple[int, int]] = []
        self.ssm_packed = None
        self.ssm_scale = None
        self.ssm_fused = None
        self.ssm_original = None


class RepresentationContext:
    """A B1 Tier 2A hit: restore and resume at the layer boundary."""

    def __init__(self, key, total, rank):
        self.key = key
        self.T = total
        self.rank = rank
        self.mode = "hit2a"
        self.phase = "resume"
        self.lease = None
        self.restored = None
        self.hidden = None
        self.residual = None
        self.model_done = False
        self.pool_done = False


class ReprCapture:
    """One B1 pending representation capture."""

    __slots__ = ("key", "length", "rank")

    def __init__(self, key, length, rank):
        self.key = key
        self.length = int(length)
        self.rank = int(rank)


def current():
    return _STATE["ctx"]


def _retire_pool_ctx(ctx) -> None:
    if ctx is None:
        return
    if ctx.lease is not None:
        try:
            ctx.lease.release()
        except Exception:  # noqa: BLE001
            pass
        ctx.lease = None
    ctx.pool_done = True
    ctx.restored = None
    _abort_ssm(ctx)
    _STATE["ctx"] = None
    _STATE["pool_ctx"] = None


def _layer_index(name: str) -> int | None:
    match = re.search(r"layers\.(\d+)", name or "")
    return int(match.group(1)) if match else None


def _layer_name(layer) -> str:
    return str(getattr(layer, "layer_name", ""))


def _fresh_b1(scheduler_output):
    if len(scheduler_output.scheduled_new_reqs) != 1:
        return None
    req = scheduler_output.scheduled_new_reqs[0]
    if (req.prompt_token_ids is None
            or req.num_computed_tokens != 0
            or req.pooling_params is None
            or getattr(req.pooling_params, "task", None) != "token_classify"):
        return None
    tokens = list(req.prompt_token_ids)
    if not tokens or scheduler_output.total_num_scheduled_tokens != len(tokens):
        return None
    if scheduler_output.num_scheduled_tokens != {req.req_id: len(tokens)}:
        return None
    if (scheduler_output.scheduled_cached_reqs.req_ids
            or scheduler_output.preempted_req_ids
            or scheduler_output.scheduled_spec_decode_tokens):
        return None
    return req, tokens


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def _log_store(decision: str, ctx: Context | None, key: bytes | None = None,
               total: int | None = None, length: int | None = None) -> None:
    if not _trace_on():
        return
    snapshot = _store().snapshot()
    counters = snapshot["counters"]
    if ctx is not None:
        key, total, length = ctx.key, ctx.T, ctx.L
    _trace(
        f"decision={decision} key={None if key is None else key.hex()[:12]} "
        f"T={total} L={length} "
        f"obs={snapshot['obs_seq']} entries={snapshot['entries']} "
        f"bytes={snapshot['bytes']} stale={snapshot['stale_entries']} "
        f"populates={counters['populates']} "
        f"publications={counters['publications']} "
        f"evictions={counters['evictions']} "
        f"rejections={counters['rejections']} hits={counters['hits']} "
        f"misses={counters['misses']}")


def _log_store2a(decision: str, key: bytes, total: int) -> None:
    if not _trace_on():
        return
    snapshot = _store2a().snapshot()
    counters = snapshot["counters"]
    _trace(
        f"tier2a decision={decision} key={key.hex()[:12]} T={total} "
        f"obs={snapshot['obs_seq']} entries={snapshot['entries']} "
        f"bytes={snapshot['bytes']} populates={counters['populates']} "
        f"publications={counters['publications']} "
        f"evictions={counters['evictions']} "
        f"rejections={counters['rejections']} hits={counters['hits']} "
        f"misses={counters['misses']}")


def prepare_execute(scheduler_output, runner) -> None:
    """Set the Tier 2 context (B1 or batch) before the model runs."""
    _STATE["ctx"] = None
    _STATE["capture2a_pending"] = None
    _STATE["capture2a_stock"] = None
    _clear_capture(clear_pending=False)
    old_batch = _STATE.get("batch")
    if old_batch is not None:
        _STATE["batch"] = None
        _STATE["batch_pool"] = None
        old_batch.release()
    _STATE["batch"] = None
    _STATE["packed_req_ids"] = None
    _STATE["pool_only"] = False
    if not _env_on():
        return
    if scheduler_output.total_num_scheduled_tokens == 0:
        pool_ctx = _STATE.get("pool_ctx")
        if pool_ctx is not None:
            _STATE["ctx"] = pool_ctx
            _STATE["pool_only"] = True
        return
    old = _STATE.get("pool_ctx")
    if old is not None:
        if old.lease is not None:
            old.lease.release()
        _STATE["pool_ctx"] = None
    try:
        # The input batch is re-ordered/compacted inside execute_model, so a
        # multi-request step is admitted from the model forward, where
        # ``runner.input_batch.req_ids`` is the real packed order.
        if len(scheduler_output.num_scheduled_tokens) > 1:
            _STATE["sched"] = scheduler_output
            _STATE["runner"] = runner
            return
    except Exception:  # noqa: BLE001 - never poison the stock path
        import traceback
        _debug("batch prepare error:\n" + traceback.format_exc())
        _STATE["sched"] = None
        _STATE["runner"] = None
        _STATE["batch"] = None
        return
    lease = None
    lease2a = None
    try:
        fresh = _fresh_b1(scheduler_output)
        if fresh is None:
            _trace(
                f"bypass fresh new={len(scheduler_output.scheduled_new_reqs)} "
                f"total={scheduler_output.total_num_scheduled_tokens} "
                f"cached={len(scheduler_output.scheduled_cached_reqs.req_ids)}")
            return
        req, tokens = fresh
        mm_items = _request_mm_items(req)
        rank = _tp_rank()
        revision = str(getattr(runner.model_config, "revision", "") or "")
        # Tier 2A is checked first: a complete-prompt representation hit
        # resumes at the boundary and never enters the Tier 2B suffix hooks.
        if _repr_enabled() and not any(not item.mm_hash for item in mm_items):
            try:
                key2a = _repr_key(tokens, mm_items, runner)
                decision2a = _store2a().observe(
                    key2a, len(tokens), _repr_estimate(len(tokens), runner))
                load_ok2a = True
                if decision2a == "hit":
                    lease2a = _store2a().acquire(key2a)
                    entry = lease2a.entry if lease2a is not None else None
                    problem = (None if entry is not None
                               else "entry vanished after acquire")
                    if entry is not None:
                        problem = _repr_shard_problem(entry, rank)
                    load_ok2a = problem is None
                    if not load_ok2a:
                        _trace(f"repr hit unusable rank={rank} "
                               f"problem={problem}")
                if _vote_decision(key2a, decision2a, load_ok2a):
                    if decision2a == "hit":
                        ctx = RepresentationContext(
                            key2a, len(tokens), rank)
                        entry = (lease2a.entry if lease2a is not None
                                 else None)
                        if entry is not None and _bind_repr(ctx, entry, rank):
                            ctx.lease = lease2a
                            lease2a = None
                            _STATE["ctx"] = ctx
                            _log_store2a("hit2a", key2a, len(tokens))
                            return
                        _trace("repr bind failed; falling back to Tier 2B")
                        if lease2a is not None:
                            lease2a.release()
                            lease2a = None
                    else:
                        _STATE["capture2a_pending"] = ReprCapture(
                            key2a, len(tokens), rank)
                else:
                    _trace(f"repr vote failed decision={decision2a} "
                           f"load_ok={load_ok2a}")
                    if lease2a is not None:
                        lease2a.release()
                        lease2a = None
                    if decision2a != "hit":
                        _STATE["capture2a_pending"] = ReprCapture(
                            key2a, len(tokens), rank)
            except Exception:  # noqa: BLE001 - never poison Tier 2B
                import traceback
                _debug("repr prepare error:\n" + traceback.format_exc())
                if lease2a is not None:
                    lease2a.release()
                    lease2a = None
        if any(not item.mm_hash for item in mm_items):
            return
        delimiter = _delimiter_ids(runner.model_config)
        min_prefix = MM_MIN_PREFIX if mm_items else TEXT_MIN_PREFIX
        split = locate_state_prefix(
            tokens, delimiter, align=ALIGN, min_prefix=min_prefix,
            mm_items=mm_items)
        if split is None:
            _trace(
                f"bypass split T={len(tokens)} "
                f"mm={[(item.placeholder_offset, item.placeholder_length) for item in mm_items]}")
            return
        key = compute_prefix_key(
            "clef-tier2", split, model_revision=revision, align=ALIGN,
            tp_size=_tp_size(runner))
        store = _store()
        decision = store.observe(
            key, split.l_aligned, _estimate_package_bytes(split.l_aligned))
        load_ok = True
        if decision == "hit":
            lease = store.acquire(key)
            entry = lease.entry if lease is not None else None
            problem = (None if entry is not None
                       else "entry vanished after acquire")
            if entry is not None:
                problem = _shard_problem(entry, rank)
            load_ok = problem is None
            if not load_ok:
                _trace(f"hit unusable rank={rank} problem={problem}")
        if not _vote_decision(key, decision, load_ok):
            _trace(f"vote failed decision={decision} load_ok={load_ok}")
            return
        if decision == "miss":
            _log_store("miss", None, key, split.total_tokens, split.l_aligned)
            return
        ctx = Context(key, split, decision, rank)
        if decision == "hit":
            entry = lease.entry if lease is not None else None
            if entry is None or not _bind_restored(ctx, entry, rank):
                return
            ctx.lease = lease
            lease = None
        _STATE["ctx"] = ctx
        _log_store(decision, ctx)
    except Exception:  # noqa: BLE001 - never poison the stock fallback
        import traceback
        _debug("prepare error:\n" + traceback.format_exc())
        _STATE["ctx"] = None
    finally:
        if lease is not None:
            lease.release()
        if lease2a is not None:
            lease2a.release()


def _shard_problem(entry, rank: int) -> str | None:
    shard = entry.shards.get(rank) or {}
    missing = [name for name in ("kv_k", "kv_v", "conv_all", "ssm_all")
               if name not in shard]
    if missing:
        return f"missing={missing} keys={sorted(shard)}"
    # The probe store keeps prefix_memory as a pageable tensor
    # (use_shm=False); an SHM-only entry cannot be restored by the
    # Prefetcher, so it must fail closed instead of resuming without memory.
    if (not _joint_rank0_only() or rank == 0) and entry.prefix_memory is None:
        return f"prefix_memory missing keys={sorted(shard)}"
    return None


def _bind_restored(ctx: Context, entry, rank: int) -> bool:
    """Queue the package H2D on the Prefetcher stream and bind layer views."""
    shard = entry.shards.get(rank) or {}
    data = {name: shard[name]
            for name in ("kv_k", "kv_v", "conv_all", "ssm_all")}
    if entry.prefix_memory is not None:
        data["prefix_memory"] = entry.prefix_memory
    try:
        restored = _prefetcher().load(data, device=_device())
    except Exception:  # noqa: BLE001 - fail closed to the stock path
        import traceback
        _debug("prefetch error:\n" + traceback.format_exc())
        return False
    ctx.restored = restored
    kv_layers = [int(value) for value in shard["kv_layers"].tolist()]
    ctx.restored_kv = {
        layer: (restored.tensors["kv_k"][index],
                restored.tensors["kv_v"][index])
        for index, layer in enumerate(kv_layers)}
    conv_layers = [int(value) for value in shard["conv_layers"].tolist()]
    ctx.restored_conv = {
        layer: restored.tensors["conv_all"][index]
        for index, layer in enumerate(conv_layers)}
    ssm_layers = [int(value) for value in shard["ssm_layers"].tolist()]
    ctx.restored_ssm = {
        layer: restored.tensors["ssm_all"][index].unsqueeze(0)
        for index, layer in enumerate(ssm_layers)}
    ctx.prefix_memory = restored.tensors.get("prefix_memory")
    return True


def _wait_restored(ctx: Context) -> None:
    restored = ctx.restored
    if restored is None:
        return
    groups = [name for name in
              ("kv_k", "kv_v", "conv_all", "ssm_all", "prefix_memory")
              if name in restored.tensors]
    restored.wait(groups)


# ----------------------------------------------------------------------
# Tier 2A: continuous representation at the layer boundary
def _repr_shard_problem(entry, rank: int) -> str | None:
    shard = entry.shards.get(rank) or {}
    missing = [name for name in ("hidden", "residual") if name not in shard]
    if missing:
        return f"missing={missing} keys={sorted(shard)}"
    return None


def _bind_repr(ctx: RepresentationContext, entry, rank: int) -> bool:
    shard = entry.shards.get(rank) or {}
    if _repr_shard_problem(entry, rank) is not None:
        return False
    try:
        restored = _prefetcher().load(
            {"hidden": shard["hidden"], "residual": shard["residual"]},
            device=_device())
    except Exception:  # noqa: BLE001 - fail closed to the stock path
        import traceback
        _debug("repr prefetch error:\n" + traceback.format_exc())
        return False
    ctx.restored = restored
    ctx.hidden = restored.tensors["hidden"]
    ctx.residual = restored.tensors["residual"]
    return True


def _bind_repr_batch(request, entry, rank: int) -> bool:
    shard = entry.shards.get(rank) or {}
    if _repr_shard_problem(entry, rank) is not None:
        return False
    try:
        restored = _prefetcher().load(
            {"hidden": shard["hidden"], "residual": shard["residual"]},
            device=_device())
    except Exception:  # noqa: BLE001 - fail closed to the cascade
        import traceback
        _debug("batch repr prefetch error:\n" + traceback.format_exc())
        return False
    request.restored2a = restored
    return True


def _wait_repr(ctx: RepresentationContext) -> None:
    if ctx.restored is not None:
        ctx.restored.wait(("hidden", "residual"))


def _submit_publish_repr(key: bytes, rank: int, length: int,
                         hidden, residual) -> None:
    """Record the boundary event, then hand the package to a publisher.

    The D2H ring is intentionally small; a C8/C16 step submits every row of
    the batch at layer 55 in one burst.  A dedicated thread waits for a slot
    instead of dropping the tail, while the model thread continues into
    layers 56..63.
    """
    ready = torch.npu.Event()
    ready.record(torch.npu.current_stream())
    payload = {"hidden": hidden, "residual": residual}
    if not _repr_queue().submit(ready, key, rank, length, payload):
        _trace(f"repr queue full T={length} rank={rank}")


def _publish_repr_now(ready, key, rank, length, payload) -> None:
    def on_ready(host: dict) -> None:
        try:
            ok = _store2a().publish(
                key, {rank: host}, length=length)
            _trace(
                f"repr published ok={ok} T={length} rank={rank} "
                f"keys={sorted(host)}")
        except Exception:  # noqa: BLE001
            import traceback
            _debug("repr publish error:\n" + traceback.format_exc())

    started = time.perf_counter()
    ticket = _capture2a().submit(
        payload, on_ready, ready_event=ready, ring_timeout=None)
    _trace(
        f"repr submitted T={length} rank={rank} dropped={ticket is None} "
        f"wait_ms={(time.perf_counter() - started) * 1000.0:.2f}")


class _ReprPublishQueue:
    """Serialize Tier 2A publishes off both the model and D2H ring threads."""

    def __init__(self, capacity: int = 512):
        self._queue: "queue.Queue" = queue.Queue(maxsize=max(1, capacity))
        self._thread = threading.Thread(
            target=self._run, name="clef-tier2a-publish", daemon=True)
        self._thread.start()

    def submit(self, ready, key, rank, length, payload) -> bool:
        try:
            self._queue.put_nowait((ready, key, rank, length, payload))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                _publish_repr_now(*item)
            except Exception:  # noqa: BLE001
                import traceback
                _debug("repr queue error:\n" + traceback.format_exc())
            finally:
                self._queue.task_done()


_REPR_QUEUE: "_ReprPublishQueue | None" = None


def _repr_queue() -> "_ReprPublishQueue":
    global _REPR_QUEUE
    if _REPR_QUEUE is None:
        _REPR_QUEUE = _ReprPublishQueue(
            capacity=int(os.environ.get("CLEF_TIER2A_QUEUE", "512")))
    return _REPR_QUEUE


def _plan_capture(entries, offsets, runner=None) -> dict | None:
    """Build a layer-55 capture plan from ``(request, begin)`` pairs."""
    items = []
    boundary = _repr_boundary()
    for request, begin in zip(entries, offsets):
        key = getattr(request, "key2a", None)
        if key is None or getattr(request, "decision2a", "") != "populate":
            continue
        length = int(getattr(request, "T", 0))
        if length <= 0:
            continue
        items.append({
            "begin": int(begin),
            "end": int(begin) + length,
            "key": key,
            "length": length,
            "rank": _tp_rank(),
        })
    if not items:
        return None
    return {"after_layer": boundary - 1, "items": items}


def _pending_capture_plan(pending: ReprCapture | None) -> dict | None:
    if pending is None:
        return None
    return {"after_layer": _repr_boundary() - 1,
            "items": [{"begin": 0, "end": pending.length, "key": pending.key,
                       "length": pending.length, "rank": pending.rank}]}


def _clear_capture(clear_pending: bool = True) -> None:
    _STATE["capture2a"] = None
    if clear_pending:
        _STATE["capture2a_pending"] = None


def _vote_modes(rows) -> list[int | None]:
    """One CPU all-reduce that agrees on key, bucket and mode per row."""
    local_rows = [list(tp_vote_vector(key, bool(load_ok))) + [int(mode)]
                  for key, load_ok, mode in rows]
    if not local_rows:
        return []
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tensor = torch.tensor(local_rows, dtype=torch.int32, device="cpu")
        get_tp_group().cpu_all_reduce_(tensor)
        summed_rows = [[int(value) for value in row] for row in tensor.tolist()]
    except Exception:  # noqa: BLE001
        import traceback
        _debug("repr vote error:\n" + traceback.format_exc())
        return [None] * len(local_rows)
    results: list[int | None] = []
    for local, summed in zip(local_rows, summed_rows):
        eligible, resume = tp_vote_result(tuple(local[:34]), tuple(summed[:34]))
        ok = bool(eligible and resume and summed[34] == 2 * local[34])
        results.append(int(summed[34] // 2) if ok else None)
    return results


class BatchRequest:
    """One request of a multi-request step (M2)."""

    def __init__(self, req_id, index, start, tokens, split=None, key=None,
                 decision="bypass", lease=None):
        self.req_id = req_id
        self.index = index
        self.start = start
        self.T = len(tokens)
        self.L = int(split.l_aligned) if split is not None else 0
        self.S = self.T - self.L
        self.split = split
        self.key = key
        self.decision = decision
        self.lease = lease
        self.key2a = None
        self.decision2a = "bypass"
        self.lease2a = None
        self.restored2a = None
        self.captured_kv = {}
        self.captured_conv = {}
        self.captured_ssm = {}
        self.prefix_memory = None
        self.restored = None
        self.restored_kv = {}
        self.restored_conv = {}
        self.restored_ssm = {}
        self.output_start = 0
        self.output_len = 0


class BatchContext:
    """The agreed per-request plan of one multi-request step."""

    def __init__(self, req_ids, requests, total_tokens):
        self.req_ids = req_ids
        self.requests = requests
        self.T_total = total_tokens
        self.path = "cascade"
        self.capture2a: list[BatchRequest] = []
        self.fresh = [r for r in requests
                      if r.decision in ("bypass", "populate")]
        self.hits = [r for r in requests if r.decision == "hit"]
        self.populates = [r for r in requests if r.decision == "populate"]

    def release(self) -> None:
        for request in self.requests:
            if request.lease is not None:
                try:
                    request.lease.release()
                except Exception:  # noqa: BLE001
                    pass
                request.lease = None
            if request.lease2a is not None:
                try:
                    request.lease2a.release()
                except Exception:  # noqa: BLE001
                    pass
                request.lease2a = None
            request.restored = None
            request.restored2a = None


def _vote_batch(rows) -> list[bool]:
    """One ``[B,35]`` CPU all-reduce over the whole step."""
    local_rows = [list(tp_vote_vector(key, bool(load_ok))) + [int(mode)]
                  for key, load_ok, mode in rows]
    if not local_rows:
        return []
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tensor = torch.tensor(local_rows, dtype=torch.int32, device="cpu")
        get_tp_group().cpu_all_reduce_(tensor)
        summed_rows = [[int(value) for value in row] for row in tensor.tolist()]
    except Exception:  # noqa: BLE001
        import traceback
        _debug("batch vote error:\n" + traceback.format_exc())
        return [False] * len(local_rows)
    results = []
    for local, summed in zip(local_rows, summed_rows):
        eligible, resume = tp_vote_result(tuple(local[:34]), tuple(summed[:34]))
        results.append(bool(eligible and resume
                            and summed[34] == 2 * local[34]))
    return results


def _prepare_batch(scheduler_output, runner, store) -> BatchContext | None:
    """Admit a multi-request step by ``input_batch.req_ids`` order."""
    packed = _STATE.get("packed_req_ids")
    if packed is not None:
        req_ids = list(packed)
    else:
        req_ids = list(getattr(getattr(runner, "input_batch", None),
                               "req_ids", ()) or ())
    _trace(
        f"batch probe ids={len(req_ids)} "
        f"sched={len(scheduler_output.num_scheduled_tokens)} "
        f"total={scheduler_output.total_num_scheduled_tokens} "
        f"new={len(scheduler_output.scheduled_new_reqs)} "
        f"cached={len(scheduler_output.scheduled_cached_reqs.req_ids)}")
    if len(req_ids) < 2:
        return None
    if (scheduler_output.scheduled_cached_reqs.req_ids
            or scheduler_output.preempted_req_ids
            or scheduler_output.scheduled_spec_decode_tokens):
        _trace(f"batch bypass cached={len(scheduler_output.scheduled_cached_reqs.req_ids)}")
        return None
    new_by_id = {req.req_id: req for req in scheduler_output.scheduled_new_reqs}
    offsets = []
    total = 0
    for req_id in req_ids:
        req = new_by_id.get(req_id)
        num = int(scheduler_output.num_scheduled_tokens.get(req_id, 0))
        if (req is None or req.prompt_token_ids is None
                or req.num_computed_tokens != 0
                or num != len(req.prompt_token_ids)):
            _trace(f"batch bypass nonfresh req={req_id} num={num}")
            return None
        offsets.append(total)
        total += num
    if total != int(scheduler_output.total_num_scheduled_tokens):
        _trace(
            f"batch bypass total {total} != "
            f"{scheduler_output.total_num_scheduled_tokens}")
        return None
    revision = str(getattr(runner.model_config, "revision", "") or "")
    rank = _tp_rank()
    requests = []
    for index, req_id in enumerate(req_ids):
        req = new_by_id[req_id]
        tokens = list(req.prompt_token_ids)
        mm_items = _request_mm_items(req)
        request = BatchRequest(req_id, index, offsets[index], tokens)
        requests.append(request)
        if getattr(req.pooling_params, "task", None) != "token_classify":
            continue
        if any(not item.mm_hash for item in mm_items):
            continue
        if not _repr_enabled():
            continue
        try:
            key2a = _repr_key(tokens, mm_items, runner)
            decision2a = _store2a().observe(
                key2a, len(tokens), _repr_estimate(len(tokens), runner))
            lease2a = None
            if decision2a == "hit":
                lease2a = _store2a().acquire(key2a)
                entry = lease2a.entry if lease2a is not None else None
                if entry is None or _repr_shard_problem(entry, rank) is not None:
                    if lease2a is not None:
                        lease2a.release()
                        lease2a = None
                    decision2a = "miss"
            request.key2a = key2a
            request.decision2a = decision2a
            request.lease2a = lease2a
        except Exception:  # noqa: BLE001 - never poison the cascade
            import traceback
            _debug("batch repr prepare error:\n" + traceback.format_exc())
            request.key2a = None
            request.decision2a = "bypass"
            request.lease2a = None

    # Tier 2A vote and whole-batch resume.  A single miss demotes the step to
    # the Tier 2B cascade; locally-acquired hits are released, never trusted.
    capture_candidates: list[BatchRequest] = []
    if _repr_enabled() and any(request.key2a is not None for request in requests):
        repr_rows = []
        repr_requests = []
        for request in requests:
            if request.key2a is None:
                continue
            mode = {"miss": 0, "populate": 1, "hit": 2}[request.decision2a]
            load_ok = (request.decision2a != "hit"
                       or request.lease2a is not None)
            repr_rows.append((request.key2a, load_ok, mode))
            repr_requests.append(request)
        modes = _vote_modes(repr_rows)
        repr_modes: list[int | None] = [None] * len(requests)
        for request, mode in zip(repr_requests, modes):
            repr_modes[request.index] = mode
            if mode == 2:
                request.decision2a = "hit"
            elif mode in (0, 1):
                request.decision2a = "populate"
            else:
                request.decision2a = "bypass"
            if mode != 2 and request.lease2a is not None:
                request.lease2a.release()
                request.lease2a = None
        plan = plan_batch(repr_modes)
        if plan.path == PATH_REPRESENTATION:
            bound = True
            for request in requests:
                entry = (request.lease2a.entry
                         if request.lease2a is not None else None)
                if entry is None or not _bind_repr_batch(request, entry, rank):
                    bound = False
                    break
            if bound:
                context = BatchContext(req_ids, requests, total)
                context.path = PATH_REPRESENTATION
                context.fresh = []
                context.hits = []
                context.populates = []
                _trace(
                    f"batch repr admit B={len(req_ids)} T={total} "
                    f"hit={len(req_ids)}")
                return context
            _trace("batch repr bind failed; demoting to Tier 2B cascade")
            for request in requests:
                if request.lease2a is not None:
                    request.lease2a.release()
                    request.lease2a = None
                request.restored2a = None
                request.decision2a = "populate"
            capture_candidates = list(requests)
        else:
            for request in requests:
                if request.decision2a == "populate":
                    capture_candidates.append(request)
    if capture_candidates:
        _trace(f"batch repr capture candidates={len(capture_candidates)}")
    delimiter = _delimiter_ids(runner.model_config)
    for request in requests:
        req = new_by_id[request.req_id]
        tokens = list(req.prompt_token_ids)
        mm_items = _request_mm_items(req)
        if getattr(req.pooling_params, "task", None) != "token_classify":
            continue
        if any(not item.mm_hash for item in mm_items):
            continue
        min_prefix = MM_MIN_PREFIX if mm_items else TEXT_MIN_PREFIX
        split = locate_state_prefix(
            tokens, delimiter, align=ALIGN, min_prefix=min_prefix,
            mm_items=mm_items)
        if split is None:
            continue
        key = compute_prefix_key(
            "clef-tier2", split, model_revision=revision, align=ALIGN,
            tp_size=_tp_size(runner))
        decision = store.observe(
            key, split.l_aligned, _estimate_package_bytes(split.l_aligned))
        lease = None
        if decision == "hit":
            lease = store.acquire(key)
            entry = lease.entry if lease is not None else None
            problem = (None if entry is not None
                       else "entry vanished after acquire")
            if entry is not None:
                problem = _shard_problem(entry, rank)
            if problem is not None:
                if lease is not None:
                    lease.release()
                    lease = None
                decision = "bypass"
        request.split = split
        request.key = key
        request.L = split.l_aligned
        request.S = request.T - request.L
        request.decision = decision
        request.lease = lease
    if not any(request.key is not None for request in requests):
        # No request in this step is Tier 2B eligible: zero-collective bypass.
        # Tier 2A candidates still capture from the stock full-length pass.
        if capture_candidates:
            _STATE["capture2a_stock"] = _plan_capture(
                capture_candidates,
                [request.start for request in capture_candidates], runner)
        return None
    rows = []
    for request in requests:
        mode = {"bypass": 0, "miss": 0, "populate": 1,
                "hit": 2}[request.decision]
        load_ok = request.decision != "hit" or request.lease is not None
        rows.append((request.key, load_ok, mode))
    votes = _vote_batch(rows)
    for request, ok in zip(requests, votes):
        if not ok and request.decision != "bypass":
            if request.lease is not None:
                request.lease.release()
                request.lease = None
            request.decision = "bypass"
    seen: set[bytes] = set()
    for request in requests:
        if request.decision != "populate":
            continue
        if request.key in seen:
            request.decision = "bypass"
        else:
            seen.add(request.key)
    for request in requests:
        if request.decision != "hit":
            continue
        entry = request.lease.entry if request.lease is not None else None
        if entry is None or not _bind_restored(request, entry, rank):
            if request.lease is not None:
                request.lease.release()
                request.lease = None
            request.decision = "bypass"
    for request in requests:
        if request.decision == "hit":
            request.decision2a = "bypass"
    live = [request for request in requests
            if request.decision in ("populate", "hit")]
    if not live:
        for request in requests:
            if request.lease is not None:
                request.lease.release()
                request.lease = None
        capture_candidates = [
            request for request in capture_candidates
            if request.decision != "hit"]
        if capture_candidates:
            _STATE["capture2a_stock"] = _plan_capture(
                capture_candidates,
                [request.start for request in capture_candidates], runner)
        return None
    context = BatchContext(req_ids, requests, total)
    context.capture2a = [
        request for request in capture_candidates
        if request.decision != "hit"]
    _trace(
        f"batch admit B={len(req_ids)} T={total} "
        f"populate={len(context.populates)} hit={len(context.hits)} "
        f"bypass={len([r for r in requests if r.decision == 'bypass'])} "
        f"repr_capture={len(context.capture2a)}")
    return context


def finish_execute() -> None:
    _STATE["batch_tap"] = None
    _STATE["batch_layer"] = None
    _STATE["sched"] = None
    _STATE["runner"] = None
    _STATE["capture2a"] = None
    _STATE["capture2a_pending"] = None
    _STATE["capture2a_stock"] = None
    batch = _STATE.get("batch")
    if batch is not None:
        if _STATE.get("batch_pool") is batch:
            # v2 pooling calls ``pool()`` after ``execute_model`` returns;
            # keep the context (and its leases/captures) for that call.
            _STATE["ctx"] = None
            return
        _STATE["batch"] = None
        _STATE["batch_pool"] = None
        batch.release()
    _STATE["packed_req_ids"] = None
    ctx = _STATE["ctx"]
    _STATE["ctx"] = None
    if ctx is None:
        return
    try:
        if _STATE.get("pool_only"):
            _STATE["pool_only"] = False
            _STATE["pool_ctx"] = None
            if ctx.lease is not None:
                ctx.lease.release()
                ctx.lease = None
            ctx.pool_done = True
            return
        if getattr(ctx, "model_done", False) and not getattr(
                ctx, "pool_done", False):
            # The pooler runs in this same step; keep the context (and its
            # lease) alive only if it is genuinely deferred.
            _STATE["pool_ctx"] = ctx
            _STATE["ctx"] = ctx
            return
        if ctx.lease is not None:
            ctx.lease.release()
            ctx.lease = None
    except Exception:  # noqa: BLE001
        pass


def _vote_decision(key: bytes, decision: str, load_ok: bool) -> bool:
    """One blocking CPU all-reduce: both ranks agree on key, bucket and load.

    ``cpu_all_reduce_`` runs on the gloo CPU group, so the vote never touches
    the NPU stream.  ``miss`` votes too: a rank that would skip without a
    collective must not silently diverge from its peer.
    """
    mode = {"miss": 0, "populate": 1, "hit": 2}.get(decision)
    if mode is None:
        return False
    local = list(tp_vote_vector(key, load_ok)) + [mode]
    try:
        from vllm.distributed.parallel_state import get_tp_group

        vector = torch.tensor(local, dtype=torch.int32, device="cpu")
        get_tp_group().cpu_all_reduce_(vector)
        summed = tuple(int(value) for value in vector.tolist())
    except Exception:  # noqa: BLE001 - collective unavailable: fail closed
        import traceback
        _debug("vote error:\n" + traceback.format_exc())
        return False
    eligible, resume = tp_vote_result(tuple(local[:34]), summed[:34])
    if not eligible or not resume:
        return False
    return summed[34] == 2 * mode


def _gdn_chunk_size() -> int:
    try:
        from vllm_ascend.ops.gdn_attn_builder import _GDN_CHUNK_SIZE

        return int(_GDN_CHUNK_SIZE)
    except Exception:  # noqa: BLE001
        return 128


def _capture_meta(tokens: int):
    chunk = _gdn_chunk_size()
    return SimpleNamespace(
        cu_seqlens_host=(0, tokens),
        chunk_indices_chunk64_host=tuple(
            value for index in range((tokens + chunk - 1) // chunk)
            for value in (0, index)),
        keep_meta=None,
    )


def _pack_capture_device(ctx: Context) -> dict:
    """One contiguous device tensor per state family, plus CPU layer ids."""
    payload: dict = {}
    if ctx.captured_kv:
        layers = sorted(ctx.captured_kv)
        payload["kv_k"] = torch.stack(
            [ctx.captured_kv[layer][0] for layer in layers])
        payload["kv_v"] = torch.stack(
            [ctx.captured_kv[layer][1] for layer in layers])
        payload["kv_layers"] = torch.tensor(layers, dtype=torch.int64)
    if ctx.captured_conv:
        layers = sorted(ctx.captured_conv)
        payload["conv_all"] = torch.stack(
            [ctx.captured_conv[layer] for layer in layers])
        payload["conv_layers"] = torch.tensor(layers, dtype=torch.int64)
    if ctx.captured_ssm:
        layers = sorted(ctx.captured_ssm)
        payload["ssm_all"] = torch.stack(
            [ctx.captured_ssm[layer].squeeze(0) for layer in layers])
        payload["ssm_layers"] = torch.tensor(layers, dtype=torch.int64)
    return payload


def _capture_counts_ok(captured_kv, captured_conv, captured_ssm) -> bool:
    if not (captured_kv and captured_conv and captured_ssm):
        return False
    if set(captured_conv) != set(captured_ssm):
        return False
    counts = (len(captured_kv), len(captured_conv), len(captured_ssm))
    expected = _STATE.get("layer_counts")
    if expected is None:
        _STATE["layer_counts"] = counts
        return True
    return expected == counts


def _submit_publish_now(store, key, rank, length, payload,
                        ready_event=None) -> None:
    """Hand one package to the async D2H ring; publish on completion."""

    def on_ready(host: dict) -> None:
        try:
            memory = host.pop("prefix_memory", None)
            ok = store.publish(
                key, {rank: host}, prefix_memory=memory, length=length)
            _trace(
                f"populate published ok={ok} L={length} rank={rank} "
                f"memory={'yes' if memory is not None else 'no'} "
                f"keys={sorted(host)}")
        except Exception:  # noqa: BLE001
            import traceback
            _debug("publish error:\n" + traceback.format_exc())

    started = time.perf_counter()
    try:
        ticket = _capture().submit(
            payload, on_ready, ready_event=ready_event)
    except Exception:  # noqa: BLE001
        import traceback
        _debug("capture submit error:\n" + traceback.format_exc())
        return
    _trace(
        f"populate submitted L={length} rank={rank} dropped="
        f"{ticket is None} "
        f"submit_ms={(time.perf_counter() - started) * 1000.0:.2f}")


class _PublishQueue:
    """Serialize D2H publishes off the request path.

    A C16 populate batch can produce more packages than the two-slot pinned
    ring holds; submitting inline makes the main thread wait the full acquire
    timeout and drops packages.  The queue absorbs the burst and keeps the
    capture stream ordering by entering the capture's stream explicitly.
    """

    def __init__(self, capacity: int = 256):
        self._queue: "queue.Queue" = queue.Queue(maxsize=max(1, capacity))
        self._thread = threading.Thread(
            target=self._run, name="clef-tiered-publish", daemon=True)
        self._thread.start()

    def submit(self, ready, store, key, rank, length, payload) -> bool:
        try:
            self._queue.put_nowait(
                (ready, store, key, rank, length, payload))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                ready, store, key, rank, length, payload = item
                _submit_publish_now(
                    store, key, rank, length, payload,
                    ready_event=ready)
            except Exception:  # noqa: BLE001
                import traceback
                _debug("publish queue error:\n" + traceback.format_exc())
            finally:
                self._queue.task_done()


_PUBLISH_QUEUE: "_PublishQueue | None" = None


def _publish_queue() -> "_PublishQueue":
    global _PUBLISH_QUEUE
    if _PUBLISH_QUEUE is None:
        _PUBLISH_QUEUE = _PublishQueue()
    return _PUBLISH_QUEUE


def _submit_publish(store, key, rank, length, payload) -> None:
    """Enqueue one package; the queue thread owns the ring wait."""
    ready = torch.npu.Event()
    ready.record(torch.npu.current_stream())
    if not _publish_queue().submit(
            ready, store, key, rank, length, payload):
        _trace(f"populate queue full L={length} rank={rank}")


def _publish_populate(ctx: Context) -> None:
    """Package, submit and clear one B1 request's populate capture."""
    if not _capture_counts_ok(
            ctx.captured_kv, ctx.captured_conv, ctx.captured_ssm):
        _trace(
            f"populate skipped: incomplete capture "
            f"kv={len(ctx.captured_kv)} conv={len(ctx.captured_conv)} "
            f"ssm={len(ctx.captured_ssm)}")
        return
    try:
        payload = _pack_capture_device(ctx)
    except Exception:  # noqa: BLE001
        import traceback
        _debug("pack error:\n" + traceback.format_exc())
        return
    if ctx.prefix_memory is not None:
        payload["prefix_memory"] = ctx.prefix_memory.detach()
    _submit_publish(_store(), ctx.key, ctx.rank, ctx.L, payload)
    ctx.captured_kv.clear()
    ctx.captured_conv.clear()
    ctx.captured_ssm.clear()


# ----------------------------------------------------------------------
# M2: batch taps during the stock full-batch pass
def _tap_batch_kv(batch: BatchContext, layer_index: int, key, value) -> None:
    tokens = key.shape[0]
    for request in batch.populates:
        end = request.start + request.L
        if request.L > 0 and end <= tokens:
            request.captured_kv[layer_index] = (
                key[request.start:end].detach().clone(),
                value[request.start:end].detach().clone())


def _tap_batch_conv(batch: BatchContext, layer_index: int, mixed_qkv) -> None:
    tokens = mixed_qkv.shape[0]
    for request in batch.populates:
        end = request.start + request.L
        if request.L >= 3 and end <= tokens:
            request.captured_conv[layer_index] = mixed_qkv[
                end - 3:end, :5120].detach().clone()


def _tap_batch_ssm(batch: BatchContext, layer: int, q, k, v, g, beta, scale,
                   original, fused_fwd, runtime, qkv_pre_normalized) -> None:
    tokens = int(q.shape[1])
    populate = [request for request in batch.populates
                if request.L > 0 and request.start + request.L <= tokens]
    if not populate:
        return
    q_parts, k_parts, v_parts, g_parts, b_parts = [], [], [], [], []
    cu = [0]
    chunk = _gdn_chunk_size()
    chunks = []
    for request in populate:
        begin, end = request.start, request.start + request.L
        q_parts.append(q[0, begin:end])
        k_parts.append(k[0, begin:end])
        v_parts.append(v[0, begin:end])
        g_parts.append(g[0, begin:end])
        b_parts.append(beta[0, begin:end])
        cu.append(cu[-1] + request.L)
        chunks.append((request.L + chunk - 1) // chunk)
    q_cat = torch.cat(q_parts, dim=0).unsqueeze(0)
    k_cat = torch.cat(k_parts, dim=0).unsqueeze(0)
    v_cat = torch.cat(v_parts, dim=0).unsqueeze(0)
    g_cat = torch.cat(g_parts, dim=0).unsqueeze(0)
    b_cat = torch.cat(b_parts, dim=0).unsqueeze(0)
    meta = SimpleNamespace(
        cu_seqlens_host=tuple(cu),
        chunk_indices_chunk64_host=tuple(
            value for seq in range(len(populate))
            for index in range(chunks[seq])
            for value in (seq, index)),
        keep_meta=None,
    )
    zeros = torch.zeros((len(populate), 24, 128, 128), dtype=torch.bfloat16,
                        device=q.device)
    runtime_stub = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=lambda: runtime_stub)
    _, states = original(
        q_cat, k_cat, v_cat, g_cat, b_cat, zeros, scale, meta, fused_fwd,
        decision_only=False, runtime=runtime_stub,
        qkv_pre_normalized=qkv_pre_normalized)
    for index, request in enumerate(populate):
        request.captured_ssm[layer] = states[index].unsqueeze(0).clone()


def _publish_batch(batch: BatchContext) -> None:
    store = _store()
    rank = _tp_rank()
    for request in batch.populates:
        if not _capture_counts_ok(
                request.captured_kv, request.captured_conv,
                request.captured_ssm):
            _trace(
                f"batch populate skipped key={request.key.hex()[:12]} "
                f"kv={len(request.captured_kv)} "
                f"conv={len(request.captured_conv)} "
                f"ssm={len(request.captured_ssm)}")
            continue
        try:
            payload = _pack_capture_device(request)
        except Exception:  # noqa: BLE001
            import traceback
            _debug("batch pack error:\n" + traceback.format_exc())
            continue
        if request.prefix_memory is not None:
            payload["prefix_memory"] = request.prefix_memory.detach()
        _submit_publish(store, request.key, rank, request.L, payload)
        request.captured_kv.clear()
        request.captured_conv.clear()
        request.captured_ssm.clear()


class SubBatch:
    """One explicit varlen pass over a subset of the batch."""

    def __init__(self, kind: str, requests):
        self.kind = kind
        self.requests = requests
        self.offsets: list[int] = []
        self.cu: list[int] = [0]
        self.embeds = None
        self.positions = None
        self.keep_idx = None


def _conv_input(x):
    """Expose the first 5120 channels of a strided packed QKVZ row."""
    if x.is_contiguous():
        return x
    if (x.dim() == 2 and x.stride(1) == 1 and x.stride(0) == 8192):
        end = x.storage_offset() + x.shape[0] * 8192
        if end * x.element_size() <= x.untyped_storage().nbytes():
            return x.as_strided((x.shape[0], 8192), (8192, 1))
    raise RuntimeError("tier2: unsupported GDN input layout")


_VARLEN_RUNTIME = None
_VARLEN_VENDOR = None
_ORIGINAL_FLA = None


def _gdn_stub():
    runtime = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=None)
    runtime.state_free = lambda: runtime
    return runtime


def _varlen_qkv(x, weight, cu):
    """Run ``aclnnClefGdnConvQkvVarlen`` for 2..64 packed sequences."""
    global _VARLEN_RUNTIME, _VARLEN_VENDOR
    import ctypes
    from fla_npu.ops.ascendc import _runtime as aclnn_runtime

    vendor = os.environ.get("CLEF_CONV_QKV_VARLEN_VENDOR")
    if not vendor:
        raise RuntimeError("CLEF_CONV_QKV_VARLEN_VENDOR is required")
    if _VARLEN_RUNTIME is None:
        library = ctypes.CDLL(
            f"{vendor}/op_api/lib/libcust_opapi.so",
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0))
        _VARLEN_RUNTIME = aclnn_runtime.runtime()
        _VARLEN_RUNTIME._libraries.insert(0, library)
        _VARLEN_VENDOR = vendor
    elif vendor != _VARLEN_VENDOR:
        raise RuntimeError("CLEF_CONV_QKV_VARLEN_VENDOR changed")
    tokens = x.shape[0]
    tile = int(os.environ.get("CLEF_CONV_QKV_VARLEN_TILE", "64"))
    q = torch.empty((1, tokens, 8, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, 24, 128), dtype=x.dtype, device=x.device)
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnConvQkvVarlen",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctx.tensor(cu, "cuSeqlens"), ctypes.c_double(1e-6),
                     ctypes.c_int64(tile), ctx.tensor(q, "qOut"),
                     ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        (q, k, v),
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_double,
                                ctypes.c_int64, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return q, k, v


def _sub_meta(lengths):
    chunk = _gdn_chunk_size()
    cu = [0]
    chunks = []
    for length in lengths:
        cu.append(cu[-1] + length)
        chunks.append((length + chunk - 1) // chunk)
    return SimpleNamespace(
        cu_seqlens_host=tuple(cu),
        chunk_indices_chunk64_host=tuple(
            value for seq in range(len(lengths))
            for index in range(chunks[seq])
            for value in (seq, index)),
        keep_meta=None,
    )


def _build_sub_batch(kind, requests, embeds, positions):
    sub = SubBatch(kind, requests)
    parts_embeds = []
    parts_positions = []
    total = 0
    for request in requests:
        if kind == "fresh":
            begin = request.start
        else:
            begin = request.start + request.L
        end = request.start + request.T
        embed = embeds[begin:end]
        position = positions[..., begin:end]
        sub.offsets.append(total)
        total += embed.shape[0]
        parts_embeds.append(embed)
        parts_positions.append(position)
        sub.cu.append(total)
    sub.embeds = torch.cat(parts_embeds, dim=0)
    sub.positions = torch.cat(parts_positions, dim=-1)
    if kind == "resume":
        keep = []
        augmented = 0
        for request in requests:
            keep.extend(range(augmented + 3,
                              augmented + 3 + request.S))
            augmented += 3 + request.S
        sub.keep_idx = torch.tensor(
            keep, dtype=torch.int64, device=sub.embeds.device)
    return sub


def _sub_fia(self, query, key, value, q_cu, kv_cu, attn_metadata):
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.npu_fused_infer_attention_score(
        query=query,
        key=key.contiguous(),
        value=value.contiguous(),
        attn_metadata=attn_metadata,
        key_cache=None,
        value_cache=None,
        current_key=key,
        current_value=value,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        head_size=self.head_size,
        scale=self.scale,
        is_prefill_no_cache=True,
        atten_mask=_mask(),
        input_layout="TND",
        actual_seq_lengths=list(q_cu),
        actual_seq_lengths_kv=list(kv_cu),
        block_table=None,
        block_size=0,
        sparse_mode=3,
    )[0]


def _tap_sub_ssm(sub, layer, q, k, v, g, beta, scale):
    pairs = [(request, begin)
             for request, begin in zip(sub.requests, sub.offsets)
             if request.decision == "populate"]
    if not pairs:
        return
    q_parts, k_parts, v_parts, g_parts, b_parts = [], [], [], [], []
    lengths = []
    for request, begin in pairs:
        end = begin + request.L
        q_parts.append(q[0, begin:end])
        k_parts.append(k[0, begin:end])
        v_parts.append(v[0, begin:end])
        g_parts.append(g[0, begin:end])
        b_parts.append(beta[0, begin:end])
        lengths.append(request.L)
    q_cat = torch.cat(q_parts, dim=0).unsqueeze(0)
    k_cat = torch.cat(k_parts, dim=0).unsqueeze(0)
    v_cat = torch.cat(v_parts, dim=0).unsqueeze(0)
    g_cat = torch.cat(g_parts, dim=0).unsqueeze(0)
    b_cat = torch.cat(b_parts, dim=0).unsqueeze(0)
    zeros = torch.zeros((len(pairs), 24, 128, 128),
                        dtype=torch.bfloat16, device=q.device)
    runtime_stub = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=lambda: runtime_stub)
    _, states = _ORIGINAL_FLA(
        q_cat, k_cat, v_cat, g_cat, b_cat, zeros, scale, _sub_meta(lengths),
        _get_fla_op(), decision_only=False, runtime=runtime_stub,
        qkv_pre_normalized=True)
    for index, (request, _begin) in enumerate(pairs):
        request.captured_ssm[layer] = states[index].unsqueeze(0).clone()


def _get_fla_op():
    from vllm_ascend.ops.gdn import _get_fla_gdn_prefill_op

    return _get_fla_gdn_prefill_op()


def _fresh_gdn_sub(self, sub, layer, mixed_qkv, b, a, core_attn_out):
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.gdn import _get_packed_conv_weights

    tokens = mixed_qkv.shape[0]
    weight = _get_packed_conv_weights(self)
    x = _conv_input(mixed_qkv)
    if len(sub.requests) == 1:
        q, k, v = _v4_qkv(x, weight)
    else:
        cu = torch.tensor(
            sub.cu, dtype=torch.int32, device=x.device)
        q, k, v = _varlen_qkv(x, weight, cu)
    for request, begin in zip(sub.requests, sub.offsets):
        if request.decision != "populate":
            continue
        end = begin + request.L
        request.captured_conv[layer] = mixed_qkv[
            end - 3:end, :5120].detach().clone()
    g, beta = DeviceOperator.fused_gdn_gating(
        self.A_log, a[:tokens], b[:tokens], self.dt_bias)
    scale = self.head_k_dim ** -0.5
    output, _ = DeviceOperator.fla_gdn_prefill(
        q, k, v, g, beta, None, scale,
        _sub_meta([request.T for request in sub.requests]), _get_fla_op(),
        decision_only=True, runtime=_gdn_stub(),
        qkv_pre_normalized=True)
    core_attn_out[:tokens] = output.reshape(tokens, 24, 128).to(
        core_attn_out.dtype)
    _tap_sub_ssm(sub, layer, q, k, v, g, beta, scale)


def _resume_gdn_sub(self, sub, layer, mixed_qkv, b, a, core_attn_out):
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.gdn import _get_packed_conv_weights

    tokens = mixed_qkv.shape[0]
    weight = _get_packed_conv_weights(self)
    x = mixed_qkv[:, :5120].contiguous()
    parts = []
    augmented_cu = [0]
    for request, begin in zip(sub.requests, sub.offsets):
        conv_state = request.restored_conv.get(layer)
        if conv_state is None:
            raise RuntimeError(
                f"tier2: missing conv state for layer {layer}")
        parts.append(conv_state)
        parts.append(x[begin:begin + request.S])
        augmented_cu.append(augmented_cu[-1] + 3 + request.S)
    x_aug = torch.cat(parts, dim=0).contiguous()
    if len(sub.requests) == 1:
        q, k, v = _v4_qkv(x_aug, weight)
        q = q[:, 3:].contiguous()
        k = k[:, 3:].contiguous()
        v = v[:, 3:].contiguous()
    else:
        cu = torch.tensor(
            augmented_cu, dtype=torch.int32, device=x_aug.device)
        q, k, v = _varlen_qkv(x_aug, weight, cu)
        q = q.index_select(1, sub.keep_idx).contiguous()
        k = k.index_select(1, sub.keep_idx).contiguous()
        v = v.index_select(1, sub.keep_idx).contiguous()
    g, beta = DeviceOperator.fused_gdn_gating(
        self.A_log, a[:tokens], b[:tokens], self.dt_bias)
    states = []
    for request in sub.requests:
        state = request.restored_ssm.get(layer)
        if state is None:
            raise RuntimeError(
                f"tier2: missing ssm state for layer {layer}")
        states.append(state.squeeze(0))
    ssm_batch = torch.stack(states, dim=0)
    scale = self.head_k_dim ** -0.5
    output, _ = DeviceOperator.fla_gdn_prefill(
        q, k, v, g, beta, ssm_batch, scale,
        _sub_meta([request.S for request in sub.requests]), _get_fla_op(),
        decision_only=True, runtime=_gdn_stub(),
        qkv_pre_normalized=True)
    core_attn_out[:tokens] = output.reshape(tokens, 24, 128).to(
        core_attn_out.dtype)


def _wait_hits(batch: BatchContext) -> None:
    for request in batch.hits:
        if request.restored is not None:
            request.restored.wait(
                [name for name in ("kv_k", "kv_v", "conv_all", "ssm_all",
                                   "prefix_memory")
                 if name in request.restored.tensors])


def _run_batch_model(self, input_ids, positions, inputs_embeds, batch,
                     original):
    """Pass 1 (bypass+populate full) then Pass 2 (hit suffix), if needed."""
    core = self.language_model.model
    if inputs_embeds is None:
        embeds = core.embed_input_ids(input_ids)
    else:
        embeds = inputs_embeds
    if batch.hits:
        _wait_hits(batch)
    if batch.fresh:
        sub = _build_sub_batch("fresh", batch.fresh, embeds, positions)
        _STATE["sub"] = sub
        by_request = {id(request): offset
                      for request, offset in zip(sub.requests, sub.offsets)}
        capture = [request for request in batch.capture2a
                   if id(request) in by_request]
        plan = _plan_capture(
            capture, [by_request[id(request)] for request in capture], self)
        if plan is not None:
            _STATE["capture2a"] = plan
        try:
            hidden = _run_layers(core, sub.embeds, sub.positions)
        finally:
            _STATE["sub"] = None
            _STATE["capture2a"] = None
        for index, request in enumerate(sub.requests):
            start = sub.offsets[index]
            request.out = hidden[start:start + request.T]
    if batch.hits:
        sub = _build_sub_batch("resume", batch.hits, embeds, positions)
        _STATE["sub"] = sub
        try:
            hidden = _run_layers(core, sub.embeds, sub.positions)
        finally:
            _STATE["sub"] = None
        for index, request in enumerate(sub.requests):
            start = sub.offsets[index]
            request.out = hidden[start:start + request.S]
    total = 0
    parts = []
    for request in batch.requests:
        request.output_start = total
        request.output_len = int(request.out.shape[0])
        total += request.output_len
        parts.append(request.out)
    packed = torch.cat(parts, dim=0) if parts else embeds[:0]
    _STATE["batch_pool"] = batch
    return packed


def _batch_pooler(self, hidden_states, pooling_metadata,
                  batch: BatchContext):
    """One collective lexical embed, per-request head, batched publish."""
    from vllm.model_executor.models import clef_joint_schema as schema_mod
    from vllm.utils.torch_utils import async_tensor_h2d

    params_list = pooling_metadata.pooling_params
    token_lists = pooling_metadata.get_prompt_token_ids_cpu()
    if (len(params_list) != len(batch.requests)
            or len(token_lists) != len(batch.requests)):
        _trace("batch pooler fallback: length mismatch")
        return None
    device = hidden_states.device
    entries = []
    flat_tokens: list[int] = []
    for index, request in enumerate(batch.requests):
        hidden = hidden_states[
            request.output_start:request.output_start + request.output_len]
        params = params_list[index]
        parsed = schema_mod.parse_joint_schema(params.extra_kwargs, request.T)
        if parsed is None:
            entries.append((request, hidden, params, None, None, None, 0))
            continue
        schema, lexical_ids, lexical_span_ids = self._indices(
            *parsed, device)
        if request.decision == "hit":
            schema = schema_mod.SchemaIndices(
                positions=schema.positions - request.L,
                span_ids=schema.span_ids,
                span_lengths=schema.span_lengths,
                question_types=schema.question_types,
                option_question=schema.option_question,
                option_slots=schema.option_slots,
                option_mask=schema.option_mask,
            )
        token_ids = token_lists[index]
        tokens = [int(token_ids[position]) for position in
                  lexical_ids.tolist()]
        flat_tokens.extend(tokens)
        entries.append((request, hidden, params, parsed, schema,
                        lexical_span_ids, len(tokens)))
    embeddings = None
    if flat_tokens:
        embeddings = self.embed_lexical(
            async_tensor_h2d(flat_tokens, device))
    if _joint_rank0_only() and _tp_rank() != 0:
        outputs = []
        for request, hidden, params, parsed, _schema, _span, _count in entries:
            if parsed is None:
                outputs.append(hidden.new_empty(0, dtype=torch.float32))
            else:
                outputs.append(hidden.new_zeros(
                    sum(int(count) for count in parsed[2]),
                    dtype=torch.float32))
        _publish_batch(batch)
        batch.release()
        _STATE["batch_pool"] = None
        _STATE["batch"] = None
        return outputs
    outputs = []
    offset = 0
    for request, hidden, params, parsed, schema, span_ids, count in entries:
        if parsed is None:
            outputs.append(hidden.new_empty(0, dtype=torch.float32))
            continue
        lexical = schema_mod.segment_mean(
            embeddings[offset:offset + count], span_ids,
            schema.span_lengths[len(parsed[0]):])
        offset += count
        if request.decision == "populate":
            head = self.head
            projection = head.memory_projection
            captured: dict = {}

            def tap(hidden_norm, _projection=projection,
                    _captured=captured, _limit=request.L):
                memory = _projection(hidden_norm)
                _captured["memory"] = memory[:_limit].clone()
                return memory

            object.__setattr__(head, "memory_projection", tap)
            try:
                logits = self.head(hidden, lexical, schema).float()
            finally:
                object.__setattr__(
                    head, "memory_projection", projection)
            request.prefix_memory = captured.get("memory")
            logits = self.head(hidden, lexical, schema).float()
        elif request.decision == "hit":
            logits = self.head(
                hidden, lexical, schema,
                prefix_memory=request.prefix_memory).float()
        else:
            logits = self.head(hidden, lexical, schema).float()
        if params.use_activation is not False:
            mask = schema.option_mask
            grid = logits.new_full((mask.numel(),), float("-inf"))
            grid[schema.option_slots] = logits
            probs = grid.view(mask.shape).softmax(dim=-1)
            logits = probs.view(-1)[schema.option_slots]
        outputs.append(logits)
    _publish_batch(batch)
    batch.release()
    _STATE["batch_pool"] = None
    _STATE["batch"] = None
    return outputs


# ----------------------------------------------------------------------
# model forward: one full tap pass, or the suffix-only resume pass
def _run_layers(core, embeds, positions):
    hidden = embeds
    residual = None
    for layer in core.layers:
        hidden, residual = layer(
            positions=positions, hidden_states=hidden, residual=residual)
    return core.norm(hidden, residual)[0]


def _run_layers_range(core, hidden, residual, positions, start):
    """Run ``core.layers[start:]`` then the final norm.

    ``hidden``/``residual`` are the exact state the stock loop would carry
    into layer ``start``; every kernel and the runner's varlen metadata are the
    native ones, so a Tier 2A resume is bitwise equal to the full pass.
    """
    for index, layer in enumerate(core.layers):
        if index < start:
            continue
        hidden, residual = layer(
            positions=positions, hidden_states=hidden, residual=residual)
    return core.norm(hidden, residual)[0]


def _run_batch_2a(self, positions, batch):
    """Whole-batch representation resume, then the native pooler."""
    hidden_parts = []
    residual_parts = []
    for request in batch.requests:
        restored = request.restored2a
        if restored is None:
            raise RuntimeError("tier2a: missing representation package")
        restored.wait(("hidden", "residual"))
        hidden_parts.append(restored.tensors["hidden"])
        residual_parts.append(restored.tensors["residual"])
    hidden = torch.cat(hidden_parts, dim=0)
    residual = torch.cat(residual_parts, dim=0)
    for request in batch.requests:
        request.restored2a = None
    if (hidden.shape[0] != batch.T_total
            or residual.shape[0] != batch.T_total):
        raise RuntimeError(
            f"tier2a: representation rows {hidden.shape[0]}/"
            f"{residual.shape[0]} != {batch.T_total}")
    core = self.language_model.model
    output = _run_layers_range(
        core, hidden, residual, positions, start=_repr_boundary())
    batch.release()
    _STATE["batch"] = None
    return output


def _stock_forward(original, module, input_ids, positions,
                   intermediate_tensors, inputs_embeds, kwargs):
    """Run the stock full pass, consuming a pending Tier 2A capture plan."""
    plan = _STATE.pop("capture2a_stock", None)
    if plan is None:
        plan = _pending_capture_plan(_STATE.get("capture2a_pending"))
    if plan is not None:
        _STATE["capture2a"] = plan
    try:
        return original(module, input_ids, positions, intermediate_tensors,
                        inputs_embeds, **kwargs)
    finally:
        _STATE["capture2a"] = None
        _STATE["capture2a_pending"] = None


def _model_forward(original):
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        sched = _STATE.pop("sched", None)
        if sched is not None:
            _trace(
                f"model sched total={sched.total_num_scheduled_tokens} "
                f"inter={intermediate_tensors is not None}")
        if sched is not None and intermediate_tensors is None:
            runner = _STATE.get("runner")
            _STATE["runner"] = None
            try:
                batch = _prepare_batch(sched, runner, _store())
            except Exception:  # noqa: BLE001
                import traceback
                _debug("batch admit error:\n" + traceback.format_exc())
                batch = None
            if (batch is not None
                    and positions.shape[-1] < batch.T_total):
                _trace(
                    f"batch bypass shape {positions.shape[-1]} < "
                    f"{batch.T_total}")
                batch.release()
                batch = None
            _STATE["batch"] = batch
        batch = _STATE.get("batch")
        if (batch is not None and intermediate_tensors is None
                and (input_ids is not None or inputs_embeds is not None)
                and positions.shape[-1] >= batch.T_total):
            _trace(
                f"model batch pos={positions.shape[-1]} "
                f"T={batch.T_total} hits={len(batch.hits)} "
                f"path={batch.path}")
            if batch.path == PATH_REPRESENTATION:
                try:
                    return _run_batch_2a(self, positions, batch)
                except Exception:  # noqa: BLE001 - rerun the stock full path
                    import traceback
                    _debug("batch repr error:\n" + traceback.format_exc())
                    _STATE["batch"] = None
                    _STATE["batch_pool"] = None
                    batch.release()
                    return _stock_forward(
                        original, self, input_ids, positions,
                        intermediate_tensors, inputs_embeds, kwargs)
            if batch.hits:
                try:
                    return _run_batch_model(
                        self, input_ids, positions, inputs_embeds, batch,
                        original)
                except Exception:  # noqa: BLE001 - rerun the stock full path
                    import traceback
                    _debug("batch mixed error:\n" + traceback.format_exc())
                    _STATE["batch"] = None
                    _STATE["batch_pool"] = None
                    batch.release()
                    return _stock_forward(
                        original, self, input_ids, positions,
                        intermediate_tensors, inputs_embeds, kwargs)
            _STATE["batch_tap"] = batch
            plan = _plan_capture(
                batch.capture2a,
                [request.start for request in batch.capture2a], self)
            if plan is not None:
                _STATE["capture2a"] = plan
            try:
                hidden = original(self, input_ids, positions,
                                  intermediate_tensors, inputs_embeds,
                                  **kwargs)
            finally:
                _STATE["batch_tap"] = None
                _STATE["capture2a"] = None
            for request in batch.requests:
                request.output_start = request.start
                request.output_len = request.T
            _STATE["batch_pool"] = batch
            return hidden
        ctx = _STATE["ctx"]
        if isinstance(ctx, RepresentationContext):
            if (intermediate_tensors is not None
                    or (input_ids is None and inputs_embeds is None)
                    or positions.shape[-1] != ctx.T):
                return _stock_forward(
                    original, self, input_ids, positions,
                    intermediate_tensors, inputs_embeds, kwargs)
            started = time.perf_counter()
            try:
                _wait_repr(ctx)
                if (ctx.hidden.shape[0] != ctx.T
                        or ctx.residual.shape[0] != ctx.T):
                    raise RuntimeError(
                        f"tier2a: restored rows {ctx.hidden.shape[0]}/"
                        f"{ctx.residual.shape[0]} != {ctx.T}")
                core = self.language_model.model
                hidden = _run_layers_range(
                    core, ctx.hidden, ctx.residual, positions,
                    start=_repr_boundary())
                ctx.hidden = None
                ctx.residual = None
                ctx.restored = None
                ctx.model_done = True
                ctx.pool_done = True
                if ctx.lease is not None:
                    ctx.lease.release()
                    ctx.lease = None
                _STATE["ctx"] = None
                _trace(
                    f"repr resume_ms="
                    f"{(time.perf_counter() - started) * 1000.0:.2f} "
                    f"T={ctx.T} boundary={_repr_boundary()}")
                return hidden
            except Exception:  # noqa: BLE001 - rerun the stock full path
                import traceback
                _debug("repr resume error:\n" + traceback.format_exc())
                _retire_pool_ctx(ctx)
                return _stock_forward(
                    original, self, input_ids, positions,
                    intermediate_tensors, inputs_embeds, kwargs)
        if (ctx is None or intermediate_tensors is not None
                or (input_ids is None and inputs_embeds is None)
                or positions.shape[-1] != ctx.T):
            return _stock_forward(
                original, self, input_ids, positions,
                intermediate_tensors, inputs_embeds, kwargs)
        core = self.language_model.model
        try:
            if inputs_embeds is None:
                embeds = core.embed_input_ids(input_ids)
            else:
                embeds = inputs_embeds
        except Exception:  # noqa: BLE001
            return _stock_forward(
                original, self, input_ids, positions,
                intermediate_tensors, inputs_embeds, kwargs)
        if embeds.shape[0] != ctx.T:
            return _stock_forward(
                original, self, input_ids, positions,
                intermediate_tensors, inputs_embeds, kwargs)
        if ctx.mode == "populate":
            started = time.perf_counter()
            plan = _pending_capture_plan(_STATE.get("capture2a_pending"))
            if plan is not None:
                _STATE["capture2a"] = plan
            try:
                hidden = _run_layers(core, embeds, positions)
            except Exception:  # noqa: BLE001 - rerun the stock full path
                import traceback
                _debug("tap error:\n" + traceback.format_exc())
                _retire_pool_ctx(ctx)
                return _stock_forward(
                    original, self, input_ids, positions,
                    intermediate_tensors, inputs_embeds, kwargs)
            finally:
                _STATE["capture2a"] = None
            ctx.model_done = True
            _trace(
                f"populate tap_ms={(time.perf_counter() - started) * 1000.0:.2f}"
                f" T={ctx.T} L={ctx.L}")
            return hidden
        started = time.perf_counter()
        try:
            _wait_restored(ctx)
            hidden = _run_layers(core, embeds[ctx.L:], positions[..., ctx.L:])
        except Exception:  # noqa: BLE001 - rerun the stock full path
            import traceback
            _debug("resume error:\n" + traceback.format_exc())
            _retire_pool_ctx(ctx)
            return _stock_forward(
                original, self, input_ids, positions,
                intermediate_tensors, inputs_embeds, kwargs)
        ctx.model_done = True
        _trace(
            f"hit suffix_ms={(time.perf_counter() - started) * 1000.0:.2f}"
            f" S={ctx.S} T={ctx.T}")
        return hidden
    return forward


# ----------------------------------------------------------------------
# full attention: tap post-RoPE K/V; resume with continuous cat
def _attention_forward(original):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        batch = _STATE.get("batch_tap")
        layer_index = _layer_index(_layer_name(layer))
        if batch is not None:
            if (layer_index is not None and output is not None
                    and key is not None and value is not None):
                _tap_batch_kv(batch, layer_index, key, value)
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        sub = _STATE.get("sub")
        if sub is not None:
            if layer_index is None or output is None or key is None:
                raise RuntimeError("tier2: sub attention missing tensors")
            if sub.kind == "fresh":
                for request, begin in zip(sub.requests, sub.offsets):
                    if request.decision != "populate":
                        continue
                    end = begin + request.L
                    if end <= key.shape[0]:
                        request.captured_kv[layer_index] = (
                            key[begin:end].detach().clone(),
                            value[begin:end].detach().clone())
                attn = _sub_fia(
                    self, query, key, value, sub.cu[1:], sub.cu[1:],
                    attn_metadata)
            else:
                parts_k = []
                parts_v = []
                kv_cu = [0]
                for request, begin in zip(sub.requests, sub.offsets):
                    cached = request.restored_kv.get(layer_index)
                    if cached is None:
                        raise RuntimeError(
                            "tier2: missing KV state for layer "
                            f"{layer_index}")
                    end = begin + request.S
                    parts_k.append(cached[0])
                    parts_k.append(key[begin:end])
                    parts_v.append(cached[1])
                    parts_v.append(value[begin:end])
                    kv_cu.append(kv_cu[-1] + request.T)
                attn = _sub_fia(
                    self, query, torch.cat(parts_k, dim=0),
                    torch.cat(parts_v, dim=0), sub.cu[1:], kv_cu[1:],
                    attn_metadata)
            num_tokens = query.shape[0]
            output[:num_tokens] = (
                attn if output.dim() == 3
                else attn.reshape(num_tokens, -1))
            return output
        ctx = _STATE["ctx"]
        if (ctx is None or layer_index is None or output is None
                or key is None or value is None):
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        if ctx.mode == "populate" and ctx.phase == "tap":
            if 0 < ctx.L <= key.shape[0]:
                ctx.captured_kv[layer_index] = (
                    key[: ctx.L].detach().clone(),
                    value[: ctx.L].detach().clone())
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        if ctx.mode != "hit" or ctx.phase != "resume":
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        cached = ctx.restored_kv.get(layer_index)
        if cached is None:
            raise RuntimeError(
                f"tier2: missing KV state for layer {layer_index}")
        num_tokens = query.shape[0]
        k_full = torch.cat([cached[0], key], dim=0)
        v_full = torch.cat([cached[1], value], dim=0)
        from vllm_ascend.device.device_op import DeviceOperator

        attn = DeviceOperator.npu_fused_infer_attention_score(
            query=query,
            key=k_full.contiguous(),
            value=v_full.contiguous(),
            attn_metadata=attn_metadata,
            key_cache=None,
            value_cache=None,
            current_key=key,
            current_value=value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_size=self.head_size,
            scale=self.scale,
            is_prefill_no_cache=True,
            atten_mask=_mask(),
            input_layout="TND",
            actual_seq_lengths=[num_tokens],
            actual_seq_lengths_kv=[k_full.shape[0]],
            block_table=None,
            block_size=0,
            sparse_mode=3,
        )[0]
        output[:num_tokens] = (
            attn if output.dim() == 3 else attn.reshape(num_tokens, -1))
        return output
    return forward


# ----------------------------------------------------------------------
# GDN: tap conv/ssm at L; resume with reference conv + initial_state
def _gdn_forward_core(original):
    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        batch = _STATE.get("batch_tap")
        layer_index = _layer_index(str(getattr(self, "prefix", "")))
        if batch is not None:
            if layer_index is not None:
                _tap_batch_conv(batch, layer_index, mixed_qkv)
            _STATE["batch_layer"] = layer_index
            try:
                return original(self, mixed_qkv, b, a, core_attn_out)
            finally:
                _STATE["batch_layer"] = None
        sub = _STATE.get("sub")
        if sub is not None:
            if layer_index is None:
                raise RuntimeError("tier2: sub GDN missing layer")
            if sub.kind == "fresh":
                _fresh_gdn_sub(
                    self, sub, layer_index, mixed_qkv, b, a,
                    core_attn_out)
            else:
                _resume_gdn_sub(
                    self, sub, layer_index, mixed_qkv, b, a,
                    core_attn_out)
            return None
        ctx = _STATE["ctx"]
        if ctx is None or layer_index is None:
            return original(self, mixed_qkv, b, a, core_attn_out)
        if ctx.mode == "populate" and ctx.phase == "tap":
            tokens = mixed_qkv.shape[0]
            if tokens >= 3 and 0 < ctx.L <= tokens:
                ctx.captured_conv[layer_index] = mixed_qkv[
                    ctx.L - 3: ctx.L, :5120].detach().clone()
            _STATE["capture_layer"] = layer_index
            try:
                return original(self, mixed_qkv, b, a, core_attn_out)
            finally:
                _STATE["capture_layer"] = None
        if ctx.mode != "hit" or ctx.phase != "resume":
            return original(self, mixed_qkv, b, a, core_attn_out)
        conv_state = ctx.restored_conv.get(layer_index)
        ssm_state = ctx.restored_ssm.get(layer_index)
        if conv_state is None or ssm_state is None:
            raise RuntimeError(
                f"tier2: missing GDN state for layer {layer_index}")
        _resume_gdn(self, mixed_qkv, b, a, core_attn_out,
                    conv_state, ssm_state)
        return None
    return forward_core


_V4_RUNTIME = None
_V4_VENDOR = None


def _v4_qkv(x, weight):
    """Run the accepted single-sequence ClefGdnConvQkv kernel."""
    global _V4_RUNTIME, _V4_VENDOR
    import ctypes
    from fla_npu.ops.ascendc import _runtime as aclnn_runtime

    vendor = os.environ.get("CLEF_CONV_QKV_VENDOR")
    if not vendor:
        raise RuntimeError("CLEF_CONV_QKV_VENDOR is required for Tier 2 resume")
    if _V4_RUNTIME is None:
        library = ctypes.CDLL(
            f"{vendor}/op_api/lib/libcust_opapi.so",
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0))
        _V4_RUNTIME = aclnn_runtime.runtime()
        _V4_RUNTIME._libraries.insert(0, library)
        _V4_VENDOR = vendor
    elif vendor != _V4_VENDOR:
        raise RuntimeError("CLEF_CONV_QKV_VENDOR changed")
    tokens = x.shape[0]
    q = torch.empty((1, tokens, 8, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, 24, 128), dtype=x.dtype, device=x.device)
    tile = int(os.environ.get("CLEF_CONV_QKV_TILE", "64"))
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnConvQkv",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctypes.c_double(1e-6), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"),
                     ctx.tensor(v, "vOut")],
        (q, k, v),
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_double, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return q, k, v


def _resume_gdn(self, mixed_qkv, b, a, core_attn_out, conv_state, ssm_state):
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.gdn import (
        _get_fla_gdn_prefill_op,
        _get_packed_conv_weights,
    )

    tokens = mixed_qkv.shape[0]
    x = mixed_qkv[:, :5120].contiguous()
    weight = _get_packed_conv_weights(self)
    # Prepend the exact last three raw tokens and run the accepted v4 fused
    # kernel; its first three outputs are the partial-history rows, so drop
    # them.  The remaining rows are the suffix conv+QKV-norm outputs.
    x_aug = torch.cat([conv_state, x], dim=0).contiguous()
    q, k, v = _v4_qkv(x_aug, weight)
    q = q[:, 3:].contiguous()
    k = k[:, 3:].contiguous()
    v = v[:, 3:].contiguous()
    g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    meta = _capture_meta(tokens)
    runtime = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=lambda: runtime)
    output, _ = DeviceOperator.fla_gdn_prefill(
        q, k, v, g, beta, ssm_state,
        self.head_k_dim ** -0.5, meta, _get_fla_gdn_prefill_op(),
        decision_only=True, runtime=runtime, qkv_pre_normalized=True)
    core_attn_out[:tokens] = output.reshape(tokens, 24, 128).to(
        core_attn_out.dtype)


def _h_state_enabled() -> bool:
    """Opt-in per-chunk ``h`` capture (Ascend950 only).

    A3's stable launcher rejects ``return_intermediate_states`` with 161002
    ("prepare path is supported on Ascend950 only"), so the grouped varlen
    capture is the A3 path.  The flag stays for a future A950/ctypes build;
    when it fails, the wrapper falls back to the split tap.
    """
    return _flag("clef_tier2_h_state") or os.environ.get(
        "CLEF_TIER2_H_STATE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _tap_full_with_h(q, k, v, g, beta, scale, prebuilt_meta, fused_fwd,
                     runtime, qkv_pre_normalized):
    """Run the production decision prefill and materialise per-chunk states.

    ``return_intermediate_states`` exposes the chunk-boundary states the
    chunked algorithm already computes internally, so the state at
    ``L`` costs one extra output buffer instead of a second prefix pass.
    Returns ``(output, h)`` or ``None`` when this call shape is ineligible.
    """
    if (not qkv_pre_normalized or not getattr(runtime, "bsnd", False)
            or q.dim() != 4
            or q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16
            or getattr(prebuilt_meta, "keep_meta", None) is not None):
        return None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.to(torch.float32).contiguous()
    beta = beta.to(v.dtype).contiguous()
    skip_recompute = bool(getattr(runtime, "no_recompute", True))
    result = fused_fwd(
        q, k, v, g, beta,
        initial_state=None,
        output_final_state=False,
        chunk_size=128,
        cu_seqlens=prebuilt_meta.cu_seqlens_host,
        chunk_indices=prebuilt_meta.chunk_indices_chunk64_host,
        scale=scale,
        layout="BSND",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=not skip_recompute,
        return_intermediate_states=True,
        state_v_first=False,
    )
    if not isinstance(result, tuple) or not result:
        return None
    h = None
    for value in result[1:]:
        if value is not None and getattr(value, "dim", lambda: 0)() == 5:
            h = value
            break
    return result[0], h


def _ssm_group_limit() -> int:
    if _flag("clef_tier2_ssm_off"):
        return 1
    value = os.environ.get("CLEF_TIER2_SSM_GROUP", "")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            return 8
    return 8


def _ssm_packed_bytes(length: int) -> int:
    # q/k/v BF16 (8 + 8 + 24 heads x 128) plus float32 g and BF16 beta.
    return int(length) * ((40 * 128 * 2) + (24 * 4) + (24 * 2))


def _ssm_budget_bytes() -> int:
    return int(os.environ.get(
        "CLEF_TIER2_SSM_GROUP_MIB", "512")) << 20


def _stage_ssm(ctx: Context, layer: int, original, q, k, v, g, beta, scale,
               fused_fwd, runtime, qkv_pre_normalized) -> bool:
    """Copy one layer's prefix GDN inputs into the packed group buffers."""
    length = ctx.L
    if ctx.ssm_group == 1:
        return False
    if ctx.ssm_packed is None:
        group = min(_ssm_group_limit(),
                    max(1, _ssm_budget_bytes() // _ssm_packed_bytes(length)))
        if group <= 1:
            ctx.ssm_group = 1
            return False
        try:
            device = q.device
            ctx.ssm_packed = [
                torch.empty((1, group * length, 8, 128), dtype=q.dtype,
                            device=device),
                torch.empty((1, group * length, 8, 128), dtype=k.dtype,
                            device=device),
                torch.empty((1, group * length, 24, 128), dtype=v.dtype,
                            device=device),
                torch.empty((1, group * length, 24), dtype=torch.float32,
                            device=device),
                torch.empty((1, group * length, 24), dtype=beta.dtype,
                            device=device),
            ]
        except Exception:  # noqa: BLE001 - no HBM headroom: immediate path
            import traceback
            _debug("ssm group alloc error:\n" + traceback.format_exc())
            ctx.ssm_packed = None
            ctx.ssm_group = 1
            return False
        ctx.ssm_group = group
        ctx.ssm_scale = scale
        ctx.ssm_fused = fused_fwd
        ctx.ssm_original = original
    if ctx.ssm_group <= 1:
        return False
    if ctx.ssm_scale != scale or ctx.ssm_fused is not fused_fwd:
        return False
    slot = len(ctx.ssm_pending)
    packed_q, packed_k, packed_v, packed_g, packed_beta = ctx.ssm_packed
    begin, end = slot * length, (slot + 1) * length
    packed_q[0, begin:end].copy_(q[0, :length])
    packed_k[0, begin:end].copy_(k[0, :length])
    packed_v[0, begin:end].copy_(v[0, :length])
    packed_g[0, begin:end].copy_(g[0, :length])
    packed_beta[0, begin:end].copy_(beta[0, :length])
    ctx.ssm_pending.append((layer, slot))
    if len(ctx.ssm_pending) >= ctx.ssm_group:
        _flush_ssm(ctx)
    return True


def _flush_ssm(ctx: Context) -> None:
    pending = ctx.ssm_pending
    if not pending or ctx.ssm_packed is None:
        return
    ctx.ssm_pending = []
    length = ctx.L
    group = len(pending)
    total = group * length
    chunk = _gdn_chunk_size()
    chunks = (length + chunk - 1) // chunk
    meta = SimpleNamespace(
        cu_seqlens_host=tuple(index * length for index in range(group + 1)),
        chunk_indices_chunk64_host=tuple(
            value for seq in range(group) for index in range(chunks)
            for value in (seq, index)),
        keep_meta=None,
    )
    packed_q, packed_k, packed_v, packed_g, packed_beta = ctx.ssm_packed
    # A partial trailing group must pass exactly ``group * length`` tokens:
    # the op requires cu_seqlens[-1] == the token count.
    packed_q = packed_q[:, :total]
    packed_k = packed_k[:, :total]
    packed_v = packed_v[:, :total]
    packed_g = packed_g[:, :total]
    packed_beta = packed_beta[:, :total]
    zeros = torch.zeros((group, 24, 128, 128), dtype=torch.bfloat16,
                        device=packed_q.device)
    runtime_stub = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=lambda: runtime_stub)
    try:
        _, state = ctx.ssm_original(
            packed_q, packed_k, packed_v, packed_g, packed_beta,
            zeros, ctx.ssm_scale, meta, ctx.ssm_fused,
            decision_only=False, runtime=runtime_stub,
            qkv_pre_normalized=True)
        for layer, slot in pending:
            ctx.captured_ssm[layer] = state[slot].unsqueeze(0).clone()
    except Exception:  # noqa: BLE001 - skip publication, keep the response
        import traceback
        _debug("ssm group error:\n" + traceback.format_exc())
        for layer, _slot in pending:
            ctx.captured_ssm.pop(layer, None)


def _abort_ssm(ctx: Context) -> None:
    ctx.ssm_pending = []
    ctx.ssm_packed = None
    ctx.ssm_group = 0


def _fla_gdn_prefill(original):
    def wrapper(q, k, v, g, beta, initial_state, scale, prebuilt_meta,
                fused_fwd, *, decision_only=None, runtime=None,
                qkv_pre_normalized=False):
        batch = _STATE.get("batch_tap")
        batch_layer = _STATE.get("batch_layer")
        if (batch is not None and batch_layer is not None
                and initial_state is None):
            result = original(q, k, v, g, beta, initial_state, scale,
                              prebuilt_meta, fused_fwd,
                              decision_only=decision_only, runtime=runtime,
                              qkv_pre_normalized=qkv_pre_normalized)
            try:
                _tap_batch_ssm(
                    batch, batch_layer, q, k, v, g, beta, scale, original,
                    fused_fwd, runtime, qkv_pre_normalized)
            except Exception:  # noqa: BLE001 - keep the response
                import traceback
                _debug("batch ssm tap error:\n" + traceback.format_exc())
                for request in batch.populates:
                    request.captured_ssm.pop(batch_layer, None)
            return result
        ctx = _STATE["ctx"]
        layer = _STATE["capture_layer"]
        if not (ctx is not None and ctx.mode == "populate"
                and ctx.phase == "tap" and layer is not None
                and initial_state is None):
            return original(q, k, v, g, beta, initial_state, scale,
                            prebuilt_meta, fused_fwd,
                            decision_only=decision_only, runtime=runtime,
                            qkv_pre_normalized=qkv_pre_normalized)
        if (_h_state_enabled() and decision_only is not False
                and 0 < ctx.L <= int(q.shape[1])):
            try:
                tapped = _tap_full_with_h(
                    q, k, v, g, beta, scale, prebuilt_meta, fused_fwd,
                    runtime, qkv_pre_normalized)
            except Exception:  # noqa: BLE001 - fall back to the split tap
                import traceback
                _debug("h-state tap error:\n" + traceback.format_exc())
                tapped = None
            if tapped is not None:
                output, h = tapped
                chunk = ctx.L // 128
                if h is not None and 0 <= chunk < h.shape[1]:
                    ctx.captured_ssm[layer] = (
                        h[:, chunk].transpose(-1, -2).contiguous())
                return output, None
        # The stock full-length call is the response path; the tap only adds
        # a prefix re-run, because the fused kernel cannot return a mid-call
        # state.  Both reads share the stream, so the response is unchanged.
        result = original(q, k, v, g, beta, initial_state, scale,
                          prebuilt_meta, fused_fwd,
                          decision_only=decision_only, runtime=runtime,
                          qkv_pre_normalized=qkv_pre_normalized)
        try:
            tokens = int(q.shape[1])
            if 0 < ctx.L <= tokens:
                grouped = False
                if (_ssm_group_limit() > 1 and qkv_pre_normalized
                        and getattr(runtime, "bsnd", False)
                        and q.dim() == 4
                        and q.dtype == torch.bfloat16
                        and v.dtype == torch.bfloat16
                        and getattr(prebuilt_meta, "keep_meta", None) is None):
                    grouped = _stage_ssm(
                        ctx, layer, original, q, k, v, g, beta, scale,
                        fused_fwd, runtime, qkv_pre_normalized)
                if not grouped:
                    zeros = torch.zeros(
                        (1, 24, 128, 128), dtype=torch.bfloat16,
                        device=q.device)
                    runtime_stub = SimpleNamespace(
                        bsnd=True, no_recompute=True, decision_prefill=True,
                        state_free=lambda: runtime_stub)
                    _, state = original(
                        q[:, : ctx.L], k[:, : ctx.L], v[:, : ctx.L],
                        g[:, : ctx.L], beta[:, : ctx.L], zeros, scale,
                        _capture_meta(ctx.L), fused_fwd,
                        decision_only=False, runtime=runtime_stub,
                        qkv_pre_normalized=qkv_pre_normalized)
                    ctx.captured_ssm[layer] = state.detach()
        except Exception:  # noqa: BLE001 - skip publication, keep the response
            import traceback
            _debug("ssm tap error:\n" + traceback.format_exc())
            ctx.captured_ssm.pop(layer, None)
        return result
    return wrapper


# ----------------------------------------------------------------------
# pooler/head: tap prefix_memory on populate; consume it on hits
def _pooler_forward(original):
    def forward(self, hidden_states, pooling_metadata):
        batch = _STATE.get("batch_pool")
        if batch is not None:
            result = _batch_pooler(
                self, hidden_states, pooling_metadata, batch)
            if result is not None:
                return result
            _STATE["batch_pool"] = None
            _STATE["batch"] = None
            batch.release()
            return original(self, hidden_states, pooling_metadata)
        ctx = _STATE["ctx"]
        if ctx is None:
            return original(self, hidden_states, pooling_metadata)
        if ctx.mode == "populate" and ctx.phase == "tap":
            if hidden_states.shape[0] != ctx.T:
                return original(self, hidden_states, pooling_metadata)
            head = self.head
            projection = head.memory_projection
            captured: dict = {}

            def tap(hidden):
                memory = projection(hidden)
                captured["memory"] = memory[: ctx.L].clone()
                return memory

            object.__setattr__(head, "memory_projection", tap)
            try:
                result = original(self, hidden_states, pooling_metadata)
            finally:
                object.__setattr__(head, "memory_projection", projection)
            ctx.prefix_memory = captured.get("memory")
            _flush_ssm(ctx)
            _trace(
                f"pooler tap hidden={tuple(hidden_states.shape)} T={ctx.T} "
                f"L={ctx.L} memory="
                f"{'yes' if ctx.prefix_memory is not None else 'no'} "
                f"ssm={len(ctx.captured_ssm)}")
            _publish_populate(ctx)
            _retire_pool_ctx(ctx)
            return result
        if ctx.mode != "hit" or ctx.phase != "resume":
            return original(self, hidden_states, pooling_metadata)
        try:
            from vllm.model_executor.models import clef_joint_schema as schema_mod
            from vllm.utils.torch_utils import async_tensor_h2d

            prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
            if len(prompt_token_ids) != 1:
                return original(self, hidden_states, pooling_metadata)
            params = pooling_metadata.pooling_params[0]
            parsed = schema_mod.parse_joint_schema(
                params.extra_kwargs, ctx.T)
            if parsed is None:
                return original(self, hidden_states, pooling_metadata)
            schema, lexical_ids, lexical_span_ids = self._indices(
                *parsed, hidden_states.device)
            schema = schema_mod.SchemaIndices(
                positions=schema.positions - ctx.L,
                span_ids=schema.span_ids,
                span_lengths=schema.span_lengths,
                question_types=schema.question_types,
                option_question=schema.option_question,
                option_slots=schema.option_slots,
                option_mask=schema.option_mask,
            )
            token_ids = prompt_token_ids[0]
            lexical_tokens = [token_ids[index] for index in lexical_ids.tolist()]
            lexical_embeddings = self.embed_lexical(
                async_tensor_h2d(lexical_tokens, hidden_states.device))
            if _joint_rank0_only() and _tp_rank() != 0:
                # embed_lexical carries a collective lm_head all-reduce on
                # every rank; the head itself is discarded on rank != 0.
                options = sum(int(count) for count in parsed[2])
                _retire_pool_ctx(ctx)
                return [hidden_states.new_zeros(
                    options, dtype=torch.float32)]
            lexical = schema_mod.segment_mean(
                lexical_embeddings, lexical_span_ids,
                schema.span_lengths[len(parsed[0]):])
            logits = self.head(
                hidden_states, lexical, schema,
                prefix_memory=ctx.prefix_memory).float()
            if params.use_activation is not False:
                mask = schema.option_mask
                grid = logits.new_full((mask.numel(),), float("-inf"))
                grid[schema.option_slots] = logits
                probs = grid.view(mask.shape).softmax(dim=-1)
                logits = probs.view(-1)[schema.option_slots]
            _retire_pool_ctx(ctx)
            return [logits]
        except Exception:  # noqa: BLE001
            import traceback
            _debug("pooler error:\n" + traceback.format_exc())
            _retire_pool_ctx(ctx)
            return original(self, hidden_states, pooling_metadata)
    return forward


def _head_forward(original):
    def forward(self, hidden_states, lexical, schema, prefix_memory=None):
        if prefix_memory is None:
            return original(self, hidden_states, lexical, schema)
        projection = self.memory_projection

        def projected(hidden):
            suffix = projection(hidden)
            return torch.cat([prefix_memory, suffix], dim=0)

        object.__setattr__(self, "memory_projection", projected)
        try:
            return original(self, hidden_states, lexical, schema)
        finally:
            object.__setattr__(self, "memory_projection", projection)
    return forward


def _decoder_layer_forward(original):
    """Capture the boundary representation once per full-length pass."""

    def forward(self, hidden_states, residual, positions, **kwargs):
        output = original(self, hidden_states, residual, positions, **kwargs)
        plan = _STATE.get("capture2a")
        if (plan is None
                or getattr(self, "layer_idx", None) != plan["after_layer"]):
            return output
        _STATE["capture2a"] = None
        if not isinstance(output, tuple) or len(output) < 2:
            return output
        hidden, resid = output[0], output[1]
        if hidden is None or resid is None:
            return output
        for item in plan["items"]:
            begin, end = int(item["begin"]), int(item["end"])
            if not (0 <= begin < end <= hidden.shape[0]):
                continue
            if resid.shape[0] < end:
                continue
            _submit_publish_repr(
                item["key"], item["rank"], item["length"],
                hidden[begin:end].detach(), resid[begin:end].detach())
        return output

    return forward


def install() -> None:
    import torch_npu  # noqa: F401
    import vllm_ascend.worker.v2.model_runner as runner_mod
    import vllm_ascend.attention.attention_v1 as attn_mod
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm.model_executor.models.clef_flash import ClefFlashForJointSchema
    from vllm.model_executor.models import clef_joint_schema as schema_mod
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer

    original_execute = runner_mod.NPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            return original_execute(self, scheduler_output, *args, **kwargs)
        prepare_execute(scheduler_output, self)
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            finish_execute()

    runner_mod.NPUModelRunner.execute_model = execute_model

    original_prepare_inputs = runner_mod.NPUModelRunner.prepare_inputs

    def prepare_inputs(self, scheduler_output, batch_req_state, *args,
                       **kwargs):
        if _STATE.get("sched") is not None:
            try:
                _STATE["packed_req_ids"] = list(batch_req_state.req_ids)
            except Exception:  # noqa: BLE001
                _STATE["packed_req_ids"] = None
        return original_prepare_inputs(
            self, scheduler_output, batch_req_state, *args, **kwargs)

    runner_mod.NPUModelRunner.prepare_inputs = prepare_inputs
    ClefFlashForJointSchema.forward = _model_forward(
        ClefFlashForJointSchema.forward)
    attn_mod.AscendAttentionBackendImpl.forward = _attention_forward(
        attn_mod.AscendAttentionBackendImpl.forward)
    QwenGatedDeltaNetAttention._forward_core = _gdn_forward_core(
        QwenGatedDeltaNetAttention._forward_core)
    Qwen3NextDecoderLayer.forward = _decoder_layer_forward(
        Qwen3NextDecoderLayer.forward)
    descriptor = DeviceOperator.__dict__["fla_gdn_prefill"]
    original_prefill = (
        descriptor.__func__ if isinstance(descriptor, staticmethod)
        else descriptor)
    global _ORIGINAL_FLA
    _ORIGINAL_FLA = original_prefill
    setattr(DeviceOperator, "fla_gdn_prefill",
            staticmethod(_fla_gdn_prefill(original_prefill)))
    schema_mod.JointSchemaPooler.forward = _pooler_forward(
        schema_mod.JointSchemaPooler.forward)
    schema_mod.JointSchemaHead.forward = _head_forward(
        schema_mod.JointSchemaHead.forward)
    drop_pages = os.environ.get(
        "CLEF_DROP_WEIGHT_PAGES", "1").strip().lower() not in (
        "0", "false", "no", "off")
    if drop_pages:
        report = drop_safetensors_page_cache(
            os.environ.get("CLEF_MODEL_DIR"))
        print(
            f"[tier2-engine] weight page cache "
            f"files={report.files} bytes={report.bytes} "
            f"skipped={report.skipped} errors={report.errors}",
            flush=True)
    if _repr_enabled():
        # Allocate the Tier 2A ring and publisher in the main thread so the
        # first capture never pays a lazy allocation on the model path.
        _store2a()
        _capture2a()
        _repr_queue()
    print(
        f"[tier2-engine] hooks installed policy=aged_keep_longest "
        f"window={_store().active_window} rank0_only={_joint_rank0_only()} "
        f"tier2a={_repr_enabled()} boundary={_repr_boundary()} "
        f"repr_bytes={os.environ.get('CLEF_TIER2A_RANK_BYTES', str(TIER2A_RANK_BYTES_DEFAULT))} "
        f"state_bytes={os.environ.get('CLEF_PREFIX_CACHE_RANK_BYTES', str(TIER2B_RANK_BYTES_DEFAULT))}",
        flush=True)
