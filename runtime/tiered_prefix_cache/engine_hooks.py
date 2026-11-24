"""M1 Tier 2 engine hooks: B=1 continuous-KV / GDN / joint-memory reuse.

This module is a probe-grade implementation installed by
``probe_engine/sitecustomize.py``.  It patches four points:

- ``NPUModelRunner.execute_model``: build the prefix split/key, vote-free B1
  admission, and load/publish HostStore state around the model call;
- ``ClefFlashForJointSchema.forward``: run the prefix capture pass, then the
  suffix pass (or suffix only on a hit);
- ``AscendAttentionBackendImpl.forward``: capture post-RoPE K/V and resume
  with ``cat([K_prefix, K_suffix])`` + FIA ``TND``/``sparse_mode=3``;
- ``QwenGatedDeltaNetAttention._forward_core``: capture conv/ssm and resume
  through the reference conv + ``fla_gdn_prefill(initial_state=...)``;
- ``JointSchemaPooler``/``JointSchemaHead``: consume suffix hidden with the
  cached ``prefix_memory`` and bypass ``AllPool``.

Only one fresh request per step is admitted in this milestone; anything else
falls back to the stock path.  It never touches vLLM's prefix-caching or KV
cache manager.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 - workers already have the backend loaded

from runtime.tiered_prefix_cache.host_store import HostStore, read_memory_budget
from runtime.tiered_prefix_cache.prefix_key import (
    MmItem,
    compute_prefix_key,
    locate_state_prefix,
)

DELIMITER_TEXT = "\n\nSCHEMA FIELDS:\n"
ALIGN = 128
MIN_PREFIX = 512
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
}


def _flag(name: str) -> bool:
    return Path(f"/tmp/{name}").is_file()


def _env_on() -> bool:
    return _flag("clef_tier2_engine") or os.environ.get(
        "CLEF_TIER2_ENGINE", "0").strip().lower() in ("1", "true", "yes", "on")


def _debug(msg: str) -> None:
    try:
        with open("/tmp/clef_tier2_engine.log", "a") as handle:
            handle.write(f"{os.getpid()} {time.time():.3f} {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _trace(msg: str) -> None:
    if _flag("clef_tier2_trace") or os.environ.get(
            "CLEF_TIER2_TRACE", "0").strip().lower() in (
            "1", "true", "yes", "on"):
        print(f"[tier2-timing] {msg}", flush=True)


def _device() -> str:
    return f"npu:{torch.npu.current_device()}"


def _mask() -> torch.Tensor:
    if _STATE["mask"] is None:
        tril = torch.ones((2048, 2048), dtype=torch.bool, device=_device()).tril_()
        _STATE["mask"] = ~tril
    return _STATE["mask"]


def _store() -> HostStore:
    if _STATE["store"] is None:
        # The 64 GiB total plan is split evenly across the two TP workers.
        budget = read_memory_budget()["budget_bytes"] // 2
        _STATE["store"] = HostStore(
            budget_bytes=budget, admission="always", use_shm=False,
            eviction_policy="keep_longest", eviction_watermark=0.95)
    return _STATE["store"]


def _delimiter_ids(model_config) -> list[int]:
    if _STATE["delimiter_ids"] is None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        path = model_config.tokenizer or model_config.model
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        _STATE["tokenizer"] = tokenizer
        _STATE["delimiter_ids"] = tokenizer.encode(
            DELIMITER_TEXT, add_special_tokens=False)
    return _STATE["delimiter_ids"]


class Context:
    def __init__(self, key, split, mode, rank):
        self.key = key
        self.split = split
        self.mode = mode
        self.rank = rank
        self.T = split.total_tokens
        self.L = split.l_aligned
        self.S = split.l_suffix
        self.phase = "capture"
        self.prefix_memory = None
        self.prefix_hidden = None
        self.captured_kv = {}
        self.captured_conv = {}
        self.captured_ssm = {}
        self.restored_kv = {}
        self.restored_conv = {}
        self.restored_ssm = {}
        self.lease = None


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


def prepare_execute(scheduler_output, runner) -> None:
    """Set the B1 Tier 2 context before the model runs; fail closed to bypass."""
    _STATE["ctx"] = None
    _STATE["pool_only"] = False
    if not _env_on():
        return
    if scheduler_output.total_num_scheduled_tokens == 0:
        pool_ctx = _STATE.get("pool_ctx")
        _debug(f"total0 pool_ctx={pool_ctx is not None}")
        if pool_ctx is not None:
            _STATE["ctx"] = pool_ctx
            _STATE["pool_only"] = True
            _debug(f"pool-only reuse L={pool_ctx.L} phase={pool_ctx.phase}")
        return
    old = _STATE.get("pool_ctx")
    if old is not None:
        if old.lease is not None:
            old.lease.release()
        _STATE["pool_ctx"] = None
    try:
        fresh = _fresh_b1(scheduler_output)
        if fresh is None:
            _debug(
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
        split = locate_state_prefix(
            tokens, delimiter, align=ALIGN, min_prefix=MIN_PREFIX,
            mm_items=mm_items)
        if split is None:
            _debug(f"bypass split T={len(tokens)}")
            return
        revision = str(getattr(runner.model_config, "revision", "") or "")
        key = compute_prefix_key(
            "clef-tier2", split, model_revision=revision, align=ALIGN, tp_size=2)
        rank = _tp_rank()
        decision = _store().observe(key)
        _debug(f"decision={decision} T={split.total_tokens} L={split.l_aligned} "
               f"rank={rank}")
        if decision == "miss":
            return
        ctx = Context(key, split, decision, rank)
        if decision == "hit":
            lease = _store().acquire(key)
            entry = lease.entry if lease else None
            if entry is None:
                return
            ctx.lease = lease
            started = time.perf_counter()
            shard = entry.shards.get(rank, {})
            if "kv_k" in shard:
                ctx.restored_kv, ctx.restored_conv, ctx.restored_ssm = (
                    _unpack_hit(shard))
            else:
                ctx.restored_kv = _load_group(shard, "kv_")
                ctx.restored_conv = _load_group(shard, "conv_")
                ctx.restored_ssm = _load_group(shard, "ssm_")
            if entry.prefix_memory is not None:
                ctx.prefix_memory = entry.prefix_memory.to(
                    _device(), non_blocking=False)
            hidden = shard.get("prefix_hidden")
            if hidden is not None:
                ctx.prefix_hidden = hidden.to(_device(), non_blocking=False)
            ctx.load_ms = (time.perf_counter() - started) * 1000.0
            _trace(f"hit load L={ctx.L} T={ctx.T} load_ms={ctx.load_ms:.2f}")
            if len(ctx.restored_kv) == 0 or len(ctx.restored_conv) == 0:
                return
        _STATE["ctx"] = ctx
    except Exception:  # noqa: BLE001 - never poison the stock fallback
        _STATE["ctx"] = None


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
            return
        _debug(f"finish mode={ctx.mode} model_done={getattr(ctx,'model_done',False)} "
               f"phase={ctx.phase} pool_only={_STATE.get('pool_only')}")
        if getattr(ctx, "model_done", False):
            _STATE["pool_ctx"] = ctx
            _STATE["ctx"] = ctx
        if ctx.mode == "populate":
            shards = _pack_capture(ctx)
            prefix_memory = (
                ctx.prefix_memory.detach().cpu()
                if ctx.prefix_memory is not None else None)
            _store().publish(
                ctx.key, {ctx.rank: shards},
                prefix_memory=prefix_memory, length=ctx.L)
        if ctx.lease is not None:
            ctx.lease.release()
    except Exception:  # noqa: BLE001
        pass


def _pack_capture(ctx) -> dict:
    """One contiguous host tensor per state family: three H2D copies, not 64."""
    shards = {}
    if ctx.captured_kv:
        layers = sorted(ctx.captured_kv)
        shards["kv_k"] = torch.stack(
            [ctx.captured_kv[layer][0] for layer in layers]).cpu()
        shards["kv_v"] = torch.stack(
            [ctx.captured_kv[layer][1] for layer in layers]).cpu()
        shards["kv_layers"] = torch.tensor(layers, dtype=torch.int64)
    if ctx.captured_conv:
        layers = sorted(ctx.captured_conv)
        shards["conv_all"] = torch.stack(
            [ctx.captured_conv[layer] for layer in layers]).cpu()
        shards["conv_layers"] = torch.tensor(layers, dtype=torch.int64)
    if ctx.prefix_hidden is not None:
        shards["prefix_hidden"] = ctx.prefix_hidden.cpu()
    if ctx.captured_ssm:
        layers = sorted(ctx.captured_ssm)
        shards["ssm_all"] = torch.stack(
            [ctx.captured_ssm[layer].squeeze(0) for layer in layers]).cpu()
        shards["ssm_layers"] = torch.tensor(layers, dtype=torch.int64)
    return shards


def _unpack_hit(shard) -> tuple[dict, dict, dict]:
    device = _device()
    kv_layers = shard["kv_layers"].tolist()
    kv_k = shard["kv_k"].to(device, non_blocking=False)
    kv_v = shard["kv_v"].to(device, non_blocking=False)
    kv = {layer: (kv_k[index], kv_v[index])
          for index, layer in enumerate(kv_layers)}
    conv_layers = shard["conv_layers"].tolist()
    conv_all = shard["conv_all"].to(device, non_blocking=False)
    conv = {layer: conv_all[index]
            for index, layer in enumerate(conv_layers)}
    ssm_layers = shard["ssm_layers"].tolist()
    ssm_all = shard["ssm_all"].to(device, non_blocking=False)
    ssm = {layer: ssm_all[index].unsqueeze(0)
           for index, layer in enumerate(ssm_layers)}
    return kv, conv, ssm


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def _load_group(shard, prefix):
    import torch as _torch

    out = {}
    for name, value in shard.items():
        if not name.startswith(prefix):
            continue
        layer = int(name[len(prefix):].split("_")[0])
        if prefix == "kv_":
            suffix = name.rsplit("_", 1)[-1]
            out.setdefault(layer, {})[suffix] = value.to(
                _device(), non_blocking=False)
        else:
            out[layer] = value.to(_device(), non_blocking=False)
    if prefix == "kv_":
        return {layer: (pair["k"], pair["v"]) for layer, pair in out.items()}
    return out


# ----------------------------------------------------------------------
# model forward: capture prefix then suffix, or suffix only
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
        _debug(
            f"model_forward ctx={None if ctx is None else ctx.mode} "
            f"ids={None if input_ids is None else tuple(input_ids.shape)} "
            f"emb={None if inputs_embeds is None else tuple(inputs_embeds.shape)} "
            f"pos={tuple(positions.shape)} inter={intermediate_tensors is not None} "
            f"T={None if ctx is None else ctx.T}")
        if (ctx is None or intermediate_tensors is not None
                or (input_ids is None and inputs_embeds is None)
                or positions.shape[-1] != ctx.T):
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
        try:
            core = self.language_model.model
            if inputs_embeds is None:
                embeds = core.embed_input_ids(input_ids)
            else:
                embeds = inputs_embeds
            if embeds.shape[0] != ctx.T:
                return original(self, input_ids, positions,
                                intermediate_tensors, inputs_embeds, **kwargs)
            head = self.pooler.head
            if ctx.mode == "populate":
                ctx.phase = "capture"
                prefix_hidden = _run_layers(
                    core, embeds[: ctx.L], positions[..., : ctx.L])
                ctx.prefix_memory = head.memory_projection(
                    head.hidden_norm(prefix_hidden)).contiguous()
                ctx.prefix_hidden = prefix_hidden.contiguous()
                ctx.phase = "resume"
            else:
                ctx.phase = "resume"
            started = time.perf_counter()
            hidden = _run_layers(core, embeds[ctx.L:], positions[..., ctx.L:])
            if _flag("clef_tier2_trace") or os.environ.get(
                    "CLEF_TIER2_TRACE", "0").strip().lower() in (
                    "1", "true", "yes", "on"):
                torch.npu.synchronize()
            _trace(f"{ctx.mode} suffix_ms={(time.perf_counter()-started)*1000:.2f} "
                   f"S={ctx.S} phase={ctx.phase}")
            ctx.model_done = True
            return hidden
        except Exception:  # noqa: BLE001
            import traceback
            _debug("model_forward error:\n" + traceback.format_exc())
            _STATE["ctx"] = None
            return original(self, input_ids, positions,
                            intermediate_tensors, inputs_embeds, **kwargs)
    return forward


# ----------------------------------------------------------------------
# full attention: capture post-RoPE K/V; resume with continuous cat
def _attention_forward(original):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        ctx = _STATE["ctx"]
        layer_index = _layer_index(_layer_name(layer))
        if (ctx is None or layer_index is None or output is None
                or ctx.phase not in ("capture", "resume")
                or key is None or value is None):
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        try:
            num_tokens = query.shape[0]
            if ctx.phase == "capture":
                ctx.captured_kv[layer_index] = (
                    key.detach().clone(), value.detach().clone())
                k_full, v_full = key, value
                q_cu, kv_cu = [num_tokens], [num_tokens]
            else:
                cached = (ctx.captured_kv.get(layer_index)
                          or ctx.restored_kv.get(layer_index))
                if cached is None:
                    return original(self, layer, query, key, value, kv_cache,
                                    attn_metadata, output, output_scale,
                                    output_block_scale)
                k_full = torch.cat([cached[0], key], dim=0)
                v_full = torch.cat([cached[1], value], dim=0)
                q_cu, kv_cu = [num_tokens], [k_full.shape[0]]
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
                actual_seq_lengths=q_cu,
                actual_seq_lengths_kv=kv_cu,
                block_table=None,
                block_size=0,
                sparse_mode=3,
            )[0]
            output[:num_tokens] = (
                attn if output.dim() == 3 else attn.reshape(num_tokens, -1))
            return output
        except Exception:  # noqa: BLE001
            import traceback
            _debug("attention hook error:\n" + traceback.format_exc())
            _STATE["ctx"] = None
            return original(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
    return forward


# ----------------------------------------------------------------------
# GDN: capture conv/ssm; resume with reference conv + initial_state
def _gdn_forward_core(original):
    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        ctx = _STATE["ctx"]
        layer_index = _layer_index(str(getattr(self, "prefix", "")))
        if ctx is None or layer_index is None:
            return original(self, mixed_qkv, b, a, core_attn_out)
        try:
            if ctx.phase == "capture":
                tokens = mixed_qkv.shape[0]
                if tokens >= 3:
                    ctx.captured_conv[layer_index] = mixed_qkv[
                        tokens - 3: tokens, :5120].detach().clone()
                _STATE["capture_layer"] = layer_index
                try:
                    return original(self, mixed_qkv, b, a, core_attn_out)
                finally:
                    _STATE["capture_layer"] = None
            if ctx.phase != "resume":
                return original(self, mixed_qkv, b, a, core_attn_out)
            conv_state = ctx.captured_conv.get(layer_index)
            if conv_state is None:
                conv_state = ctx.restored_conv.get(layer_index)
            ssm_state = ctx.captured_ssm.get(layer_index)
            if ssm_state is None:
                ssm_state = ctx.restored_ssm.get(layer_index)
            if conv_state is None or ssm_state is None:
                return original(self, mixed_qkv, b, a, core_attn_out)
            _resume_gdn(self, mixed_qkv, b, a, core_attn_out,
                        conv_state, ssm_state)
            return None
        except Exception:  # noqa: BLE001
            import traceback
            _debug("gdn hook error:\n" + traceback.format_exc())
            _STATE["ctx"] = None
            return original(self, mixed_qkv, b, a, core_attn_out)
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
    meta = SimpleNamespace(
        cu_seqlens_host=(0, tokens),
        chunk_indices_chunk64_host=tuple(
            value for index in range((tokens + 127) // 128)
            for value in (0, index)),
        keep_meta=None,
    )
    runtime = SimpleNamespace(
        bsnd=True, no_recompute=True, decision_prefill=True,
        state_free=lambda: runtime)
    output, _ = DeviceOperator.fla_gdn_prefill(
        q, k, v, g, beta, ssm_state,
        self.head_k_dim ** -0.5, meta, _get_fla_gdn_prefill_op(),
        decision_only=True, runtime=runtime, qkv_pre_normalized=True)
    core_attn_out[:tokens] = output.reshape(tokens, 24, 128).to(
        core_attn_out.dtype)


def _fla_gdn_prefill(original):
    def wrapper(q, k, v, g, beta, initial_state, scale, prebuilt_meta,
                fused_fwd, *, decision_only=None, runtime=None,
                qkv_pre_normalized=False):
        ctx = _STATE["ctx"]
        layer = _STATE["capture_layer"]
        if (ctx is not None and ctx.phase == "capture"
                and layer is not None and initial_state is None):
            zeros = torch.zeros((1, 24, 128, 128), dtype=torch.bfloat16,
                                device=q.device)
            runtime_stub = SimpleNamespace(
                bsnd=True, no_recompute=True, decision_prefill=True,
                state_free=lambda: runtime_stub)
            tokens = int(q.shape[1])
            capture_meta = SimpleNamespace(
                cu_seqlens_host=(0, tokens),
                chunk_indices_chunk64_host=tuple(
                    value for index in range((tokens + 127) // 128)
                    for value in (0, index)),
                keep_meta=None)
            output, state = original(
                q, k, v, g, beta, zeros, scale, capture_meta, fused_fwd,
                decision_only=False, runtime=runtime_stub,
                qkv_pre_normalized=qkv_pre_normalized)
            ctx.captured_ssm[layer] = state.detach().clone()
            return output, None
        return original(q, k, v, g, beta, initial_state, scale, prebuilt_meta,
                        fused_fwd, decision_only=decision_only,
                        runtime=runtime, qkv_pre_normalized=qkv_pre_normalized)
    return wrapper


# ----------------------------------------------------------------------
# pooler/head: bypass AllPool and consume prefix_memory
def _pooler_forward(original):
    def forward(self, hidden_states, pooling_metadata):
        ctx = _STATE["ctx"]
        _debug(f"pooler ctx={None if ctx is None else ctx.mode}/{None if ctx is None else ctx.phase} hidden={tuple(hidden_states.shape)}")
        if ctx is None or ctx.phase != "resume":
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
            full_hidden = None
            if ctx.prefix_hidden is not None:
                full_hidden = torch.cat([ctx.prefix_hidden, hidden_states], dim=0)
            else:
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
            lexical = schema_mod.segment_mean(
                lexical_embeddings, lexical_span_ids,
                schema.span_lengths[len(parsed[0]):])
            logits = self.head(
                full_hidden if full_hidden is not None else hidden_states,
                lexical, schema,
                prefix_memory=None if full_hidden is not None else ctx.prefix_memory
            ).float()
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
    import vllm_ascend.ops.gdn as gdn_mod
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm.model_executor.models.clef_flash import ClefFlashForJointSchema
    from vllm.model_executor.models import clef_joint_schema as schema_mod
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    original_execute = runner_mod.NPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
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
    print("[tier2-engine] hooks installed", flush=True)
