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
    compute_prefix_key,
    locate_state_prefix,
)

DELIMITER_TEXT = "\n\nSCHEMA FIELDS:\n"
ALIGN = 128
TEXT_MIN_PREFIX = 512
MM_MIN_PREFIX = 128
ACTIVE_WINDOW = 3000
#: TP2 state package per rank: 16 full-attention layers x K/V x 2 heads x
#: 256 elements x 2 bytes = 32 KiB/token, plus the rank-0 2 KiB/token memory
#: projection.  Both ranks use the same estimate so the admission decision
#: is byte-identical across TP workers.
PER_TOKEN_BYTES = 34 * 1024
STATE_TAIL_BYTES = 40 << 20
_STATE: dict = {
    "store": None,
    "tokenizer": None,
    "delimiter_ids": None,
    "mask": None,
    "ctx": None,
    "pool_ctx": None,
    "pool_only": False,
    "lock": threading.RLock(),
    "capture_layer": None,
    "capture": None,
    "prefetcher": None,
}


def _flag(name: str) -> bool:
    return Path(f"/tmp/{name}").is_file()


def _env_on() -> bool:
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
            "CLEF_PREFIX_CACHE_RANK_BYTES", str(32 << 30)))
        _STATE["store"] = HostStore(
            budget_bytes=budget, admission="always", use_shm=False,
            eviction_policy="aged_keep_longest",
            active_window=int(os.environ.get(
                "CLEF_PREFIX_ACTIVE_WINDOW", str(ACTIVE_WINDOW))),
            eviction_watermark=0.95)
    return _STATE["store"]


def _capture() -> AsyncCapture:
    if _STATE["capture"] is None:
        slot_bytes = int(os.environ.get(
            "CLEF_TIER2_RING_BYTES", str(640 << 20)))
        _STATE["capture"] = AsyncCapture(
            PinnedRing(slots=2, slot_bytes=slot_bytes))
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


def prepare_execute(scheduler_output, runner) -> None:
    """Set the B1 Tier 2 context before the model runs; fail closed to bypass."""
    _STATE["ctx"] = None
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
    lease = None
    try:
        fresh = _fresh_b1(scheduler_output)
        if fresh is None:
            _trace(
                f"bypass fresh new={len(scheduler_output.scheduled_new_reqs)} "
                f"total={scheduler_output.total_num_scheduled_tokens} "
                f"cached={len(scheduler_output.scheduled_cached_reqs.req_ids)}")
            return
        req, tokens = fresh
        delimiter = _delimiter_ids(runner.model_config)
        mm_items = tuple(
            MmItem(
                mm_hash=str(f.identifier or f.mm_hash or ""),
                placeholder_offset=int(f.mm_position.offset),
                placeholder_length=int(f.mm_position.length),
                expanded_offset=int(f.mm_position.offset),
                expanded_length=int(f.mm_position.length),
                modality=str(f.modality),
            )
            for f in req.mm_features)
        if any(not item.mm_hash for item in mm_items):
            return
        min_prefix = MM_MIN_PREFIX if mm_items else TEXT_MIN_PREFIX
        split = locate_state_prefix(
            tokens, delimiter, align=ALIGN, min_prefix=min_prefix,
            mm_items=mm_items)
        if split is None:
            _trace(
                f"bypass split T={len(tokens)} "
                f"mm={[(item.placeholder_offset, item.placeholder_length) for item in mm_items]}")
            return
        revision = str(getattr(runner.model_config, "revision", "") or "")
        key = compute_prefix_key(
            "clef-tier2", split, model_revision=revision, align=ALIGN, tp_size=2)
        rank = _tp_rank()
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


def finish_execute() -> None:
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


def _publish_populate(ctx: Context) -> None:
    """Hand the captured package to the async D2H ring; publish on completion."""
    if not (ctx.captured_kv and ctx.captured_conv and ctx.captured_ssm):
        _trace("populate skipped: empty capture")
        return
    if set(ctx.captured_conv) != set(ctx.captured_ssm):
        _trace(
            f"populate skipped: incomplete GDN capture "
            f"conv={len(ctx.captured_conv)} ssm={len(ctx.captured_ssm)}")
        return
    counts = (len(ctx.captured_kv), len(ctx.captured_conv),
              len(ctx.captured_ssm))
    expected = _STATE.get("layer_counts")
    if expected is None:
        _STATE["layer_counts"] = counts
    elif expected != counts:
        _trace(
            f"populate skipped: layer counts {counts} != {expected}")
        return
    try:
        payload = _pack_capture_device(ctx)
    except Exception:  # noqa: BLE001
        import traceback
        _debug("pack error:\n" + traceback.format_exc())
        return
    if ctx.prefix_memory is not None:
        payload["prefix_memory"] = ctx.prefix_memory.detach()
    store = _store()
    key = ctx.key
    rank = ctx.rank
    length = ctx.L
    total = int(ctx.T)

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
        _capture().submit(payload, on_ready)
    except Exception:  # noqa: BLE001
        import traceback
        _debug("capture submit error:\n" + traceback.format_exc())
        return
    _trace(
        f"populate submitted L={length} T={total} rank={rank} "
        f"submit_ms={(time.perf_counter() - started) * 1000.0:.2f}")
    ctx.captured_kv.clear()
    ctx.captured_conv.clear()
    ctx.captured_ssm.clear()


# ----------------------------------------------------------------------
# model forward: one full tap pass, or the suffix-only resume pass
def _run_layers(core, embeds, positions):
    hidden = embeds
    residual = None
    for layer in core.layers:
        hidden, residual = layer(
            positions=positions, hidden_states=hidden, residual=residual)
    return core.norm(hidden, residual)[0]


def _model_forward(original):
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        ctx = _STATE["ctx"]
        if (ctx is None or intermediate_tensors is not None
                or (input_ids is None and inputs_embeds is None)
                or positions.shape[-1] != ctx.T):
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
        core = self.language_model.model
        try:
            if inputs_embeds is None:
                embeds = core.embed_input_ids(input_ids)
            else:
                embeds = inputs_embeds
        except Exception:  # noqa: BLE001
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
        if embeds.shape[0] != ctx.T:
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
        if ctx.mode == "populate":
            started = time.perf_counter()
            try:
                hidden = _run_layers(core, embeds, positions)
            except Exception:  # noqa: BLE001 - rerun the stock full path
                import traceback
                _debug("tap error:\n" + traceback.format_exc())
                _retire_pool_ctx(ctx)
                return original(self, input_ids, positions,
                                intermediate_tensors, inputs_embeds, **kwargs)
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
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
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
        ctx = _STATE["ctx"]
        layer_index = _layer_index(_layer_name(layer))
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
        ctx = _STATE["ctx"]
        layer_index = _layer_index(str(getattr(self, "prefix", "")))
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
                captured["memory"] = memory[: ctx.L]
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
    ClefFlashForJointSchema.forward = _model_forward(
        ClefFlashForJointSchema.forward)
    attn_mod.AscendAttentionBackendImpl.forward = _attention_forward(
        attn_mod.AscendAttentionBackendImpl.forward)
    QwenGatedDeltaNetAttention._forward_core = _gdn_forward_core(
        QwenGatedDeltaNetAttention._forward_core)
    descriptor = DeviceOperator.__dict__["fla_gdn_prefill"]
    original_prefill = (
        descriptor.__func__ if isinstance(descriptor, staticmethod)
        else descriptor)
    setattr(DeviceOperator, "fla_gdn_prefill",
            staticmethod(_fla_gdn_prefill(original_prefill)))
    schema_mod.JointSchemaPooler.forward = _pooler_forward(
        schema_mod.JointSchemaPooler.forward)
    schema_mod.JointSchemaHead.forward = _head_forward(
        schema_mod.JointSchemaHead.forward)
    print(
        f"[tier2-engine] hooks installed policy=aged_keep_longest "
        f"window={ACTIVE_WINDOW} rank0_only={_joint_rank0_only()}", flush=True)
