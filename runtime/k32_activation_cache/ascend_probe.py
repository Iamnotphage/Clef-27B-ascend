"""Opt-in Clef K32 adapter. Load only in the isolated deployment."""

from collections import Counter
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from importlib import import_module
from inspect import getsourcefile, isclass, isfunction, ismodule
import json
import os
import signal
from pathlib import Path
from time import perf_counter_ns
from types import MethodType

from activation_cache import ActivationCache, BOUNDARY, POST_NORM, text_key
from fixed_pair import pack_pair, run_fixed_pair
from step_protocol import fresh_text_items, fresh_text_step, run_candidate, run_final_candidate, vote_integers, vote_result


ACTIVE = ContextVar("clef_private_k32_step", default=None)
GDN_FLAGS = (
    "CLEF_GDN_DECISION_PREFILL", "CLEF_GDN_BSND",
    "CLEF_GDN_SKIP_INITIAL_STATE", "CLEF_GDN_SKIP_CONV_STATE",
    "CLEF_GDN_PREPROCESS_FUSION",
)
SOURCE_LOCK = {
    "ascend_runner": "e866aeb27d04afe2141850122c03343c38d2a9b14cf1f9f136866003b35968a6",
    "base_runner": "6ebb6b8f2eed517dfe28af7d7aa1c5dd324b47f2eb724b1a4861cd5a8703f308",
    "forward": "6439d8ed22f5c256baddc8074a20634c7cecf7a68b20c958201ec8a0346b8a60",
    "qwen": "f9f370f7b4988bd917c1dad4a4d6939e74c5fe4f5b7cea9cedfcf4c2a5637afe",
    "clef": "e51277a80e5e6790cd3d4ecb27e1f2a85c4f328425a8262f799a0fc43b1103d9",
    "ascend_clef": "e7d196facbf605bacba418602f14e41fd48be42b29b3e7bc76e2d294765f60a4",
    "ascend_registry": "c68e1b8295b6feb4371511b325ef6afd1fb2e42cdb7dbf285e19e72c61f5ae30",
    "ascend_preprocess": "50706a86aa2a4f3ac83766dfbb3eb4c392872b746f6c138de6d30c34bb203c25",
    "head": "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503",
    "attention": "a221dd76ada427354d856990033a9ec9fd925a806f272b385e5f49edec7bcd19",
    "attn_state": "3b9def8922419f73710348b79f6fa54b7c8adb676d9e0617b42e55f4973ec195",
    "gdn_metadata": "ea723bc5bd8258caf736295162183f2bc5ff46b5a87b1cd5abb92a1126864fd0",
    "gdn_builder": "f81eeefa957790398adbbde35fd3e7b2c5aaf3eeac9273400c7412f753ba4a8a",
    "gdn_device": "bf027ec99a3d069f913600ddfcb76283f73a4114febe46b151c113c62450c226",
    "block_table": "3433729cad5aa5fceb312e0a8d9ffabb3f7327dbdd18966d576e64d090313020",
    "ascend_block_table": "ec82c11f6b1935f17ac2581ad5e8a9601ec7a9889dbd9cee559d8e52f24ca0a8",
    "block_table_patch": "d6ed9dac65aa8fd1eff882931df1b1ab171e976ae9961f3ce5fa3cbed4867460",
    "slot_kernel": "ecd831ed6f16451e5ee41c47d845b4d5bdc675203353ff6bfd6cb35b117887a5",
    "runner_patch": "79da5e641e3518bfa443b80c25b4647705a961fba6416f73cb79af685cc733f0",
    "ascend_hybrid_model_state": "c00f1205aff7f3a0f1854d2310319a7891a161c86929c45df5dbd236ed77337d",
    "ascend_model_state": "015f644876a40118b132929a54b55c47e62bee143d307f6845da1f227e30c227",
    "storage_block_size": "e9e146d84b804520974ddc8ad57299135b29cec7a6458c52c8fc5a3a6ce2caa5",
    "attention_layer": "5ae48902abd82367138555b7393942cf4b5341942cec3a2a3817ebab00a36198",
    "gdn": "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78",
}
if POST_NORM:
    SOURCE_LOCK.update({
        "all_pool": "d6970def31d87bd8ad2052487e91d2d6234bc79243dcc8e9a032f8bd1f830d3d",
        "pooling_runner": "3a36931da3d807a8f7926ee5382a22de704247c74ccc28f3a3a40610a26c67e9",
        "base_model_state": "453eb32c5efef9f4064c333c3e5e714aef3adc16645db1f4a311f8a397860f02",
        "encoder_runner": "894ffc53e702e9bf9ca06bf21dcd4c7a169a36d7bd3de2387bbe43c972427006",
    })


@dataclass
class Step:
    runner: object
    request: object
    batch: object = None
    times: dict = field(default_factory=dict)
    boundary: list | None = None
    contract: dict | None = None
    ordered_items: tuple | None = None
    kv_proof: list | None = None
    guard_facts: dict = field(default_factory=dict)


def check_sources(targets):
    if set(targets) != set(SOURCE_LOCK):
        raise RuntimeError(f"K32 source coverage mismatch: missing={sorted(SOURCE_LOCK.keys() - targets.keys())}"
                           f" extra={sorted(targets.keys() - SOURCE_LOCK.keys())}")
    checks = {"module": ismodule, "class": isclass, "function": isfunction,
              "path": lambda value: isinstance(value, Path)}
    rows = []
    for name, (kind, value) in targets.items():
        row = {"name": name, "expected_type": kind, "actual_type": type(value).__name__,
               "module": value.__name__ if ismodule(value) else getattr(value, "__module__", None),
               "qualname": getattr(value, "__qualname__", None),
               "expected_sha256": SOURCE_LOCK[name], "actual_sha256": None, "realpath": None}
        rows.append(row)
        if not checks[kind](value):
            row["failure"] = "type"
            continue
        try:
            source = value if kind == "path" else Path(getsourcefile(value))
            path = source.resolve(strict=True)
            row["realpath"] = str(path)
            row["actual_sha256"] = sha256(path.read_bytes()).hexdigest()
            if path.suffix != ".py" or row["actual_sha256"] != row["expected_sha256"]:
                row["failure"] = "origin or SHA"
        except (OSError, TypeError, ValueError) as error:
            row["failure"] = type(error).__name__
    return rows


def symbol_info(symbol):
    if symbol is None:
        return None
    row = {"module": getattr(symbol, "__module__", None),
           "qualname": getattr(symbol, "__qualname__", None)}
    try:
        source = Path(getsourcefile(symbol)).resolve()
        row.update(origin=str(source), sha256=sha256(source.read_bytes()).hexdigest())
    except (OSError, TypeError):
        pass
    code = getattr(symbol, "__code__", None)
    if code is not None:
        row.update(code_file=code.co_filename, code_line=code.co_firstlineno)
    return row


def wrapper_chain(symbol):
    info = []
    seen = set()
    while symbol is not None and id(symbol) not in seen and len(info) < 8:
        seen.add(id(symbol))
        info.append(symbol_info(symbol))
        symbol = getattr(symbol, "__wrapped__", None)
    return info


def layout(tensor):
    return str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride())


def identity_failures(outer, core, tp, outer_class, pooler_class, core_class, forward):
    """Return the exact failed predicates without changing admission."""
    kind = lambda item: f"{type(item).__module__}.{type(item).__qualname__}"
    method = getattr(core.forward, "__func__", None)
    boundary_types = list(core.config.layer_types)[BOUNDARY - 1:BOUNDARY + 1]
    expected_types = ["full_attention"] if POST_NORM else ["full_attention", "linear_attention"]
    checks = {
        "outer_type": (type(outer) is outer_class, kind(outer)),
        "pooler_type": (type(outer.pooler) is pooler_class, kind(outer.pooler)),
        "core_type": (type(core) is core_class, kind(core)),
        "forward_identity": (method is forward, getattr(method, "__qualname__", None)),
        "layer_count": (len(core.layers) == 64, len(core.layers)),
        "boundary_types": (boundary_types == expected_types, boundary_types),
        "tp_size": (tp.world_size == 2, tp.world_size),
    }
    return {field: value for field, (valid, value) in checks.items() if not valid}


def timing(step, name, call):
    t0 = perf_counter_ns()
    try:
        return call()
    finally:
        step.times[name] = step.times.get(name, 0) + perf_counter_ns() - t0


def metadata_checks(context, batch, n, full_type, gdn_type, lengths=None):
    if batch is None:
        return {"missing_batch": False}
    lengths = list(lengths) if lengths is not None else [n]
    count = len(lengths)
    offsets = [0]
    for width in lengths:
        offsets.append(offsets[-1] + width)
    checks = {
        "eager": getattr(context.cudagraph_runtime_mode, "name", None) == "NONE",
        "no_ubatch": context.ubatch_slices is None,
        "batch_descriptor": (context.batch_descriptor is not None and context.batch_descriptor.num_tokens == n),
        "per_layer_metadata": isinstance(context.attn_metadata, dict),
        "num_reqs": batch.num_reqs == batch.num_reqs_after_padding == count,
        "tokens": batch.num_tokens == batch.num_tokens_after_padding == n,
        "num_drafts": batch.num_draft_tokens == 0,
        "query_start": batch.query_start_loc_np.tolist()[:count+1] == offsets,
        "initial_computed": batch.num_computed_tokens_np.tolist() == [0] * count,
        "prefill_computed": batch.num_computed_prefill_tokens_np.tolist() == [0] * count,
        "prefill_len": batch.prefill_len_np.tolist() == lengths,
        "seq_len": batch.seq_lens_np[:count].tolist() == lengths,
        "batch_attn_enum": getattr(batch.attn_state, "name", None) == "PrefillCacheHit",
    }
    if not checks["per_layer_metadata"]:
        return checks
    unique = {id(meta): meta for meta in context.attn_metadata.values()}
    full = [m for m in unique.values() if type(m) is full_type]
    gdn = [m for m in unique.values() if type(m) is gdn_type]
    checks["metadata_types"] = bool(full and gdn and len(full) + len(gdn) == len(unique))
    checks["metadata_prefill_counts"] = all(
        m.num_prefills == count and m.num_decodes == m.num_decode_tokens == 0 and m.num_actual_tokens == n
        for m in (*full, *gdn)
    )
    checks["full_metadata"] = all(
        m.attn_state is batch.attn_state and m.actual_seq_lengths_q == offsets[1:] and m.seq_lens_list == lengths
        and m.causal is True and m.model_runner_type == "pooling"
        and not m.uses_pcp_sharded_layout
        for m in full
    )
    checks["gdn_metadata"] = all(
        m.num_prefill_tokens == n and m.num_spec_decodes == m.num_spec_decode_tokens == 0
        and m.spec_sequence_masks is None for m in gdn
    )
    return checks


def slots_cover_full_prompt(blocks, slots, block_size, total_blocks, n):
    """Use the actual physical block geometry of the attention backend."""
    needed = (n + block_size - 1) // block_size
    return (len(blocks) == len(set(blocks)) == needed
            and all(0 < block < total_blocks for block in blocks)
            and len(slots) == n
            and all(slot == blocks[i // block_size] * block_size + i % block_size
                    for i, slot in enumerate(slots)))


def valid_manager_ids(ids, kernel_blocks, n):
    needed = (n + 511) // 512
    return (n > 0 and kernel_blocks % 4 == 0 and len(ids) == len(set(ids)) == needed
            and all(0 < block < kernel_blocks // 4 for block in ids))


def verify_mixed_pages(manager_ids, kernel_ids, slots, total_kernel_blocks, n):
    """Compare the scheduler's manager IDs with the already expanded metadata."""
    manager_tokens, kernel_tokens = 512, 128
    ratio = manager_tokens // kernel_tokens
    if not valid_manager_ids(manager_ids, total_kernel_blocks, n):
        return False
    expected_kernel = [m * ratio + subpage for m in manager_ids for subpage in range(ratio)]
    used_kernel = (n + kernel_tokens - 1) // kernel_tokens
    return (list(kernel_ids) == expected_kernel[:used_kernel]
            and slots_cover_full_prompt(kernel_ids, slots, kernel_tokens, total_kernel_blocks, n))


def check_fact(facts, field, actual, expected, *, ok=None):
    if ok is None:
        ok = actual == expected
    if facts is not None:
        facts[field] = {"expected": expected, "actual": actual, "ok": ok}
    return ok


def allocation_geometry(address, shape, stride, item_bytes):
    if (len(shape) != 4 or tuple(shape[1:]) != (128, 2, 256) or shape[0] < 1
            or tuple(stride) != (131072, 512, 256, 1) or item_bytes != 2):
        return None
    return address, stride[0] * item_bytes, shape[0], 128 * 2 * 256 * item_bytes


def verify_interleaved_pair(proofs, regions):
    first, second = proofs
    for proof, (address, pitch, count, size) in zip(proofs, regions, strict=True):
        start = proof["offset_elements"] * 2
        end = start + (count - 1) * pitch + size
        if start < 0 or address != proof["storage_base"] + start or end > proof["storage_bytes"]:
            return False
    return (first["storage_base"] == second["storage_base"]
            and first["storage_bytes"] == second["storage_bytes"]
            and second["offset_elements"] - first["offset_elements"] == 65536)


def no_layer_overlap(layer_ranges):
    ordered = sorted(layer_ranges)
    return all(stop <= start for (_, stop), (start, _) in zip(ordered, ordered[1:]))


def verify_runtime_sources(runner_class):
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.models.clef_flash import ClefFlashForJointSchema
    from vllm.model_executor.models.clef_joint_schema import AllPool, JointSchemaPooler
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model
    from vllm.model_executor.models.qwen3_next import Qwen3NextModel
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend, AscendAttentionBackendImpl, AscendMetadata, AttentionType,
    )
    import vllm_ascend.device.device_op as device_op
    import vllm_ascend.models as ascend_models
    from vllm_ascend.models.clef_flash import AscendClefFlashForJointSchema
    from vllm_ascend.models.qwen3_5_vl import _AscendVLPreprocessMixin
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    from vllm_ascend.patch.worker import patch_block_table, patch_model_runner
    from vllm_ascend.worker.v2.attn_utils import build_attn_state
    from vllm_ascend.worker.v2.block_table import AscendBlockTables
    from vllm_ascend.worker.v2.model_states.default import AscendModelState
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    block_table_module = import_module("vllm_ascend.worker.v2.block_table")
    slot_kernel_module = import_module("vllm_ascend.ops.triton.block_table.compute_slot_mappings")
    exported_kernel = slot_kernel_module._compute_slot_mappings_kernel
    bound_kernel = block_table_module._compute_slot_mappings_kernel
    targets = {
        "ascend_runner": ("class", runner_class),
        "base_runner": ("class", GPUModelRunner),
        "forward": ("function", Qwen3NextModel.forward),
        "qwen": ("class", Qwen3_5Model),
        "clef": ("class", ClefFlashForJointSchema),
        "head": ("class", JointSchemaPooler),
        "attention": ("class", AscendMetadata),
        "gdn": ("class", AscendGatedDeltaNetAttention),
        "ascend_clef": ("class", AscendClefFlashForJointSchema),
        "ascend_registry": ("module", ascend_models),
        "ascend_preprocess": ("class", _AscendVLPreprocessMixin),
        "attn_state": ("function", build_attn_state),
        "block_table": ("class", BlockTables),
        "gdn_metadata": ("class", GDNAttentionMetadata),
        "gdn_builder": ("class", AscendGDNAttentionMetadataBuilder),
        "gdn_device": ("module", device_op),
        "ascend_block_table": ("class", AscendBlockTables),
        "block_table_patch": ("module", patch_block_table),
        "slot_kernel": ("module", slot_kernel_module),
        "runner_patch": ("module", patch_model_runner),
        "ascend_model_state": ("class", AscendModelState),
        "ascend_hybrid_model_state": ("class", AscendMambaHybridModelState),
        "storage_block_size": ("function", get_storage_block_size),
        "attention_layer": ("class", Attention),
    }
    if POST_NORM:
        from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState
        from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
        targets.update(all_pool=("class", AllPool), pooling_runner=("class", PoolingRunner),
                       base_model_state=("class", DefaultModelState),
                       encoder_runner=("class", EncoderRunner))
    rows = check_sources(targets)
    slot = next(row for row in rows if row["name"] == "slot_kernel")
    slot["exported_type"] = type(exported_kernel).__name__
    slot["table_uses_exported_kernel"] = bound_kernel is exported_kernel
    if bound_kernel is not exported_kernel:
        slot["failure"] = "bound kernel differs from module export"
    failed = [row for row in rows if "failure" in row]
    if failed:
        raise RuntimeError(f"K32 source contract rejected: {json.dumps(failed, sort_keys=True)}")
    return rows


def install(runner_class):
    import torch
    import torch_npu
    from memory_budget import MemoryBudget
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.clef_flash import ClefFlashForJointSchema
    from vllm.model_executor.models.clef_joint_schema import AllPool, JointSchemaPooler
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model
    from vllm.model_executor.models.qwen3_next import Qwen3NextModel
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend, AscendAttentionBackendImpl, AscendMetadata, AttentionType,
    )
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    from vllm_ascend.models.clef_flash import AscendClefFlashForJointSchema
    from vllm_ascend.models.qwen3_5_vl import _AscendVLPreprocessMixin
    from vllm_ascend.ops.gdn import _get_fla_gdn_prefill_op
    from vllm_ascend.ops.gdn_decision_flags import snapshot_for
    from vllm_ascend.patch.worker import patch_model_runner
    from vllm_ascend.worker.v2.block_table import AscendBlockTables
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    verify_runtime_sources(runner_class)
    if POST_NORM:
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
        from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner

    mode = os.environ["CLEF_PRIVATE_K32"]
    fixed_pair = os.environ.get("CLEF_PRIVATE_K64_FIXED_PAIR") == "1"
    if fixed_pair and not POST_NORM:
        raise RuntimeError("fixed-pair cache requires the K64 post-norm boundary")
    namespace = bytes.fromhex(os.environ["CLEF_PRIVATE_K32_NAMESPACE"])
    # The supervisor supplies one namespace through an environment inherited
    # by both workers. The mode carries its own domain in the TP key.
    assert len(namespace) == 32 and mode in {"split", "cache"}
    cap = int(os.environ.get("CLEF_PRIVATE_K32_RANK_BYTES", 256 << 20))
    formal = os.environ.get("CLEF_PRIVATE_K32_FORMAL") == "1"
    approved_cap = (41 << 29) if formal else (256 << 20)
    assert 0 < cap <= approved_cap
    audit = os.environ.get("CLEF_PRIVATE_K32_AUDIT") == "1"
    control = os.environ.get("CLEF_PRIVATE_K32_CONTROL")
    diagnostics = Path(os.environ["CLEF_PRIVATE_K32_DIAG_DIR"])
    if not diagnostics.is_dir():
        raise RuntimeError("K32 diagnostic directory unavailable")
    if formal and (audit or control is not None or mode != "cache"):
        raise RuntimeError("formal K32 cannot use audit or per-request control")

    load, execute, prepare = (runner_class.load_model, runner_class.execute_model,
                              runner_class.prepare_inputs)

    @wraps(load)
    def load_model(runner, *args, **kwargs):
        previous = getattr(runner, "_clef_private_k32_cache", None)
        if previous is not None:
            previous.invalidate()
        output = load(runner, *args, **kwargs)
        outer = runner.model
        core = outer.language_model.model
        tp = get_tp_group()
        pp = get_pp_group()
        failures = identity_failures(
            outer, core, tp, AscendClefFlashForJointSchema, JointSchemaPooler,
            Qwen3_5Model, Qwen3NextModel.forward,
        )
        actual_types = [layer.layer_type for layer in core.layers]
        expected_types = (["linear_attention"] * 3 + ["full_attention"]) * 16
        config = runner.parallel_config
        expected_forward = Qwen3_5ForConditionalGeneration.forward
        outer_forward = getattr(outer.forward, "__func__", None)
        predicates = {
            "registered_bases": (AscendClefFlashForJointSchema.__bases__ ==
                                 (_AscendVLPreprocessMixin, ClefFlashForJointSchema)),
            "outer_forward": outer_forward is expected_forward,
            "all_pool": type(outer.pooler.pooling) is AllPool,
            "model_partition": (core.start_layer, core.end_layer) == (0, 64),
            "pp": (pp.world_size == 1 and pp.is_first_rank and pp.is_last_rank
                   and config.pipeline_parallel_size == 1),
            "tp": (tp.rank_in_group in (0, 1) and config.tensor_parallel_size == 2),
            "layer_inventory": (list(core.config.layer_types) == actual_types == expected_types),
            "runner": (runner.is_pooling_model and runner.model_config.dtype == torch.bfloat16
                       and runner.model_config.enforce_eager),
        }
        if fixed_pair and runner.scheduler_config.max_num_seqs != 2:
            failures["fixed_pair_profile"] = False
        if not all(predicates.values()):
            failures.update({key: False for key, ok in predicates.items() if not ok})
        facts = {
            "rank": tp.rank_in_group, "pid": os.getpid(), "failures": failures,
            "activation_boundary": BOUNDARY, "activation_generation_sha256": namespace.hex(),
            "activation_representation": "post_norm" if POST_NORM else "hidden_residual",
            "fixed_pair_only": fixed_pair,
            "old_base_exact": type(outer) is ClefFlashForJointSchema,
            "expected_outer": symbol_info(AscendClefFlashForJointSchema),
            "actual_outer": symbol_info(type(outer)),
            "actual_mro": [symbol_info(cls) for cls in type(outer).__mro__[:6]],
            "expected_outer_forward": symbol_info(expected_forward),
            "outer_forward_chain": wrapper_chain(outer_forward),
            "expected_core": symbol_info(Qwen3_5Model), "actual_core": symbol_info(type(core)),
            "expected_core_forward": symbol_info(Qwen3NextModel.forward),
            "core_forward_chain": wrapper_chain(getattr(core.forward, "__func__", None)),
            "expected_pooler": symbol_info(JointSchemaPooler),
            "actual_pooler": symbol_info(type(outer.pooler)),
            "actual_pooling": symbol_info(type(outer.pooler.pooling)),
            "layer_count": len(core.layers), "layer_start": core.start_layer,
            "layer_end": core.end_layer, "expected_layer_types": expected_types,
            "actual_layer_types": actual_types,
            "pp": {"size": pp.world_size, "first": pp.is_first_rank, "last": pp.is_last_rank},
            "tp": {"size": tp.world_size, "rank": tp.rank_in_group},
            "runner": {"type": symbol_info(type(runner)), "eager": runner.model_config.enforce_eager,
                       "pooling": runner.is_pooling_model, "dtype": str(runner.model_config.dtype),
                       "batch": runner.scheduler_config.max_num_seqs},
        }
        with (diagnostics / f"identity-{tp.rank_in_group}.json").open("x") as out:
            json.dump(facts, out)
        if failures:
            raise RuntimeError(f"K32 model identity or partition rejected: {failures!r}")
        stock = core.forward
        budget = (MemoryBudget(cap, shared_reserve=10 << 30, partners=2)
                  if formal or fixed_pair else MemoryBudget(cap))
        cache = ActivationCache(budget)
        cache.scheduler_bypasses = 0
        if fixed_pair:
            cache.batch_steps = Counter()
        runner._clef_private_k32_cache = cache
        gdn_layers = [x.linear_attn for x in core.layers if x.layer_type == "linear_attention"]
        attention = [x.self_attn.attn for x in core.layers if x.layer_type == "full_attention"]
        if len(gdn_layers) != 48 or len(attention) != 16:
            raise RuntimeError("K32 attention inventory rejected")
        verified_bindings = None
        audit_sequence = 0
        head_audit = None
        if POST_NORM and (audit or control is not None):
            def observe_head(_module, args, _output):
                nonlocal head_audit
                if head_audit is None:
                    return
                sequence, request_id, n = head_audit
                head_audit = None
                hidden, metadata = args
                row = {"rank":tp.rank_in_group, "pid":os.getpid(), "request_id":request_id,
                       "shape":tuple(hidden.shape), "expected_tokens":n,
                       "scheduled":metadata.get_pooling_cursor().num_scheduled_tokens_cpu.tolist()}
                with (diagnostics / f"head-{sequence:04d}-{tp.rank_in_group}.json").open("x") as out:
                    json.dump(row,out)
            outer.pooler.register_forward_hook(observe_head)

        counters = Counter()
        snapshots = 0
        if formal:
            def write_counters(_number, _frame):
                nonlocal snapshots
                snapshots += 1
                row = {"pid": os.getpid(), "rank": tp.rank_in_group, "sequence": snapshots,
                       "counters": dict(counters), "published": cache.published,
                       "evicted": cache.evictions, "cache": cache.snapshot(),
                       "scheduler_bypasses": cache.scheduler_bypasses, "activation_boundary": BOUNDARY}
                if fixed_pair:
                    row["batch_steps"] = dict(cache.batch_steps)
                with (diagnostics / f"endpoint-{snapshots:02d}-{tp.rank_in_group}.json").open("x") as out:
                    json.dump(row, out)
            if signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
                raise RuntimeError("private endpoint receipt signal already owned")
            signal.signal(signal.SIGUSR2, write_counters)

        def cache_binding(context, n, record, step):
            nonlocal verified_bindings
            full = context.attn_metadata
            tables = runner.block_tables
            items = step.ordered_items if fixed_pair else (step.request,)
            lengths = tuple(len(item.token_ids) for item in items)

            def manager_scope(group_counts):
                for group, count in group_counts:
                    used = set()
                    for item, width in zip(items, lengths, strict=True):
                        ids = item.manager_block_ids
                        if group >= len(ids) or not valid_manager_ids(ids[group], count, width):
                            return False
                        if fixed_pair:
                            if used.intersection(ids[group]):
                                return False
                            used.update(ids[group])
                return True

            if not fixed_pair and not record and verified_bindings is not None:
                known_tables, identities, group_counts = verified_bindings
                return (tables is known_tables and manager_scope(group_counts) and all(
                    context.no_compile_layers.get(layer.layer_name) is layer
                    and type(full.get(layer.layer_name)) is AscendMetadata
                    and layer.kv_cache is kv and layer.impl is impl
                    and (impl.key_cache is key or key is None and impl.key_cache is kv[0])
                    and (impl.value_cache is value or value is None and impl.value_cache is kv[1])
                    for layer, kv, impl, key, value in identities
                ))
            if not fixed_pair and not record and not formal:
                return False

            facts = {}
            step.guard_facts["kv"] = facts
            group_checks = {}
            facts["groups"] = group_checks
            layers_facts = []
            facts["layers"] = layers_facts
            valid = check_fact(facts, "table_type", symbol_info(type(tables)),
                               symbol_info(AscendBlockTables), ok=type(tables) is AscendBlockTables)
            valid &= check_fact(facts, "initialize_kv_cache",
                                symbol_info(getattr(runner.initialize_kv_cache, "__func__", None)),
                                symbol_info(runner_class.initialize_kv_cache),
                                ok=getattr(runner.initialize_kv_cache, "__func__", None)
                                is runner_class.initialize_kv_cache)
            valid &= check_fact(facts, "base_initialize_kv_cache",
                                symbol_info(GPUModelRunner.initialize_kv_cache),
                                symbol_info(patch_model_runner.initialize_kv_cache),
                                ok=GPUModelRunner.initialize_kv_cache is patch_model_runner.initialize_kv_cache)
            valid &= check_fact(facts, "model_state_type", symbol_info(type(runner.model_state)),
                                symbol_info(AscendMambaHybridModelState),
                                ok=type(runner.model_state) is AscendMambaHybridModelState)
            valid &= check_fact(facts, "context_parallel", getattr(tables, "cp_size", None), 1)
            groups = runner.kv_cache_config.kv_cache_groups
            layer_groups = {name: group for group, spec in enumerate(groups) for name in spec.layer_names}
            allocations = []
            bindings = []
            proofs = {}
            kind = lambda cls: f"{cls.__module__}.{cls.__qualname__}"
            for layer in attention:
                name = layer.layer_name
                m = full.get(name)
                impl, kv = layer.impl, layer.kv_cache
                group = layer_groups.get(name)
                checks = {}
                layers_facts.append({"name": name, "group": group, "checks": checks})
                layer_ok = check_fact(checks, "context_layer_identity",
                                      context.no_compile_layers.get(name) is layer, True)
                layer_ok &= check_fact(checks, "meta_type", kind(type(m)), kind(AscendMetadata),
                                       ok=type(m) is AscendMetadata)
                layer_ok &= check_fact(checks, "impl_type", kind(type(impl)),
                                       kind(AscendAttentionBackendImpl),
                                       ok=type(impl) is AscendAttentionBackendImpl)
                backend = layer.get_attn_backend()
                layer_ok &= check_fact(checks, "backend", kind(backend), kind(AscendAttentionBackend),
                                       ok=backend is AscendAttentionBackend)
                layer_ok &= check_fact(checks, "attn_type", str(impl.attn_type), str(AttentionType.DECODER),
                                       ok=impl.attn_type == AttentionType.DECODER)
                layer_ok &= check_fact(checks, "use_bnsd_kv_cache", impl.use_bnsd_kv_cache, False,
                                       ok=impl.use_bnsd_kv_cache is False)
                layer_ok &= check_fact(checks, "kv_heads_and_width", [impl.num_kv_heads, impl.head_size], [2, 256])
                layer_ok &= check_fact(checks, "layer_has_group", group is not None, True)
                layer_ok &= check_fact(checks, "layer_sharing_target", layer.kv_sharing_target_layer_name, None)
                layer_ok &= check_fact(checks, "impl_sharing_target", impl.kv_sharing_target_layer_name, None)
                layer_ok &= check_fact(checks, "kv_producer", bool(impl.is_kv_producer), False)
                layer_ok &= check_fact(checks, "kv_count", len(kv), 2)
                if group is not None and group not in group_checks:
                    spec = groups[group].kv_cache_spec
                    group_detail = {}
                    group_checks[group] = group_detail
                    for field, actual, expected in (
                        ("manager_spec", spec.block_size, 512),
                        ("manager_storage", get_storage_block_size(spec), 512),
                        ("manager_table", tables.block_sizes[group], 512),
                        ("kernel_runner", runner.kernel_block_sizes[group], 128),
                        ("kernel_table", getattr(tables, "kernel_block_sizes", [None] * len(groups))[group], 128),
                        ("kernel_pages_per_manager",
                         getattr(tables, "blocks_per_kv_block", [None] * len(groups))[group], 4),
                    ):
                        valid &= check_fact(group_detail, field, actual, expected)
                if len(kv) != 2:
                    valid = False
                    continue
                a, b = kv
                layer_ok &= check_fact(
                    checks, "kv_logical_layout", [layout(a), layout(b)],
                    ["identical BF16 NBHD (N,128,2,256)", "identical BF16 NBHD (N,128,2,256)"],
                    ok=(a.ndim == 4 and a.dtype == b.dtype == torch.bfloat16
                        and a.shape == b.shape and a.device == b.device == step.batch.positions.device
                        and a.numel() > 0))
                layer_ok &= check_fact(checks, "kv_device", [str(a.device), str(b.device)],
                                       [str(step.batch.positions.device)] * 2)
                for bound, tensor, label in ((impl.key_cache, a, "key"), (impl.value_cache, b, "value")):
                    expected = [tensor.data_ptr(), layout(tensor), str(tensor.device)]
                    actual = None if bound is None else [bound.data_ptr(), layout(bound), str(bound.device)]
                    layer_ok &= check_fact(checks, f"bound_impl_{label}", actual, expected,
                                           ok=(actual is None or actual == expected))
                regions = [allocation_geometry(t.data_ptr(), t.shape, t.stride(), t.element_size()) for t in (a, b)]
                geometry_ok = check_fact(checks, "kernel_page_geometry", regions,
                                         "BF16 NBHD page pitch 262144 bytes and length 131072 bytes",
                                         ok=all(region is not None for region in regions))
                layer_ok &= geometry_ok
                if not geometry_ok:
                    valid = False
                    continue
                storage_facts = []
                for t in (a, b):
                    storage = t.untyped_storage()
                    storage_facts.append({"data_ptr": t.data_ptr(), "storage_base": storage.data_ptr(),
                                          "storage_bytes": storage.nbytes(), "offset_elements": t.storage_offset(),
                                          "layout": layout(t)})
                layer_ok &= check_fact(checks, "interleaved_storage", storage_facts,
                                       "same storage; V offset minus K offset 65536 elements; both in bounds",
                                       ok=verify_interleaved_pair(storage_facts, regions))
                start, pitch, block_count, _ = regions[0]
                layer_ok &= check_fact(checks, "kernel_block_count_mod4", block_count % 4, 0)
                allocations.append((start, start + pitch * block_count))
                if group is not None and type(m) is AscendMetadata:
                    proofs[(group, id(m), block_count)] = m
                bindings.append((layer, kv, impl, impl.key_cache, impl.value_cache))
                valid &= layer_ok
            valid &= check_fact(facts, "nonoverlapping_layer_ranges", sorted(allocations),
                                "pairwise disjoint physical K/V spans", ok=no_layer_overlap(allocations))
            if not valid:
                verified_bindings = None
                return False
            group_counts = tuple(sorted({(group, count) for group, _, count in proofs}))
            if not group_counts or not manager_scope(group_counts):
                verified_bindings = None
                return False
            if not record and not fixed_pair:
                # The fixed planner derives slots from the fresh scheduler row
                # and expanded kernel pages. No per-request device audit here.
                verified_bindings = (tables, bindings, group_counts)
                return True
            kv_positions = step.batch.positions.to("cpu", non_blocking=False).tolist()
            valid &= check_fact(facts, "kv_positions", kv_positions,
                                [p for width in lengths for p in range(width)])
            observations = []
            step.kv_proof = observations
            for proof_id, ((group, _, count), meta) in enumerate(proofs.items()):
                shape = (meta.block_tables.ndim == 2 and meta.block_tables.shape[0] == len(items)
                         and meta.slot_mapping.ndim == 1 and meta.slot_mapping.numel() == n)
                start = 0
                occupied = set()
                for index, (item, width) in enumerate(zip(items, lengths, strict=True)):
                    needed = (width + 127) // 128
                    manager = item.manager_block_ids[group]
                    good = shape and meta.block_tables.shape[1] >= needed
                    blocks = meta.block_tables[index, :needed].to("cpu", non_blocking=False).tolist() if good else []
                    slots = meta.slot_mapping[start:start + width].to("cpu", non_blocking=False).tolist() if good else []
                    ok = good and verify_mixed_pages(manager, blocks, slots, count, width)
                    receipt = {"group": group, "manager_ids": manager, "kernel_pages": count,
                               "block_size": 128, "blocks": blocks, "slots": slots,
                               "table_shape": list(meta.block_tables.shape),
                               "slot_shape": list(meta.slot_mapping.shape)}
                    if fixed_pair:
                        ok = ok and not occupied.intersection(slots)
                        occupied.update(slots)
                        receipt.update(proof_id=proof_id, request_id=item.request_id)
                    receipt["verified"] = ok
                    observations.append(receipt)
                    valid &= ok
                    start += width
            gdn_metadata = [m for m in {id(m): m for m in full.values()
                                         if type(m) is GDNAttentionMetadata}.values()]
            initial_state = [None if m.prefill_has_initial_state is None else
                             m.prefill_has_initial_state.to("cpu", non_blocking=False).tolist()
                             for m in gdn_metadata]
            valid &= check_fact(facts, "gdn_initial_state", initial_state, [[False] * len(items)] * len(initial_state))
            if fixed_pair:
                offsets = [0, lengths[0], n]
                checked = set()
                for layer in gdn_layers:
                    group = layer_groups.get(layer.prefix)
                    meta = full.get(layer.prefix)
                    states = layer.kv_cache
                    if group is None or type(meta) is not GDNAttentionMetadata or len(states) != 2:
                        return False
                    capacity = min(state.shape[0] for state in states)
                    identity = (group, id(meta), capacity)
                    if identity in checked:
                        continue
                    checked.add(identity)
                    if any(group >= len(item.manager_block_ids) or len(item.manager_block_ids[group]) != 1
                           for item in items):
                        return False
                    expected = [item.manager_block_ids[group][0] for item in items]
                    slots = (meta.non_spec_state_indices_tensor, meta.prefill_state_indices)
                    starts = (meta.non_spec_query_start_loc, meta.prefill_query_start_loc)
                    actual = [x.to("cpu", non_blocking=False).tolist() if x is not None else None for x in slots]
                    queries = [x.to("cpu", non_blocking=False).tolist() if x is not None else None for x in starts]
                    valid &= check_fact(facts, f"gdn_state_slot_{len(checked)}",
                                        {"blocks": actual, "query_starts": queries, "capacity": capacity},
                                        {"blocks": [expected, expected], "query_starts": [offsets, offsets]},
                                        ok=(actual == [expected, expected] and queries == [offsets, offsets]
                                            and len(set(expected)) == 2 and all(0 < x < capacity for x in expected)))
            verified_bindings = (tables, bindings, group_counts) if valid else None
            return valid

        def contract(step, context, inputs_embeds, input_ids, positions, domain, record):
            pc = runner.parallel_config
            config = runner.vllm_config
            if fixed_pair:
                if step.batch is None:
                    return None
                by_id = {item.request_id:item for item in step.request}
                ids = step.batch.req_ids
                if len(ids)!=2 or set(ids)!=by_id.keys():
                    return None
                step.ordered_items = tuple(by_id[req_id] for req_id in ids)
                lengths = tuple(len(item.token_ids) for item in step.ordered_items)
            else:
                lengths = (len(step.request.token_ids),)
            n = sum(lengths)
            checks = {
                "parallel": (pc.pipeline_parallel_size == pc.data_parallel_size ==
                             pc.prefill_context_parallel_size == pc.decode_context_parallel_size == 1),
                "scheduler_batch" if fixed_pair else "single_sequence": runner.scheduler_config.max_num_seqs == len(lengths),
                "no_chunk": not runner.scheduler_config.enable_chunked_prefill,
                "no_prefix": not runner.cache_config.enable_prefix_caching,
                "no_mamba_prefix": runner.cache_config.mamba_cache_mode == "none",
                "no_transfer": config.kv_transfer_config is None and runner.kv_connector is NO_OP_KV_CONNECTOR,
                "no_spec_or_lora": runner.speculative_config is None and runner.lora_config is None,
                "no_aux_sp": not (runner.use_aux_hidden_state_outputs or core.use_sequence_parallel
                                  or core.aux_hidden_state_layers),
                "range": (core.start_layer, core.end_layer) == (0, 64),
                "gdn_flags": all(os.environ.get(k) == "1" for k in GDN_FLAGS)
                             and not os.environ.get("CLEF_GDN_FLAG_FILE"),
                "gdn_layers": all(getattr(g, "_decision_safe_config", False) for g in gdn_layers),
                "no_cache_sharing": all(getattr(a.impl, "kv_sharing_target_layer_name", True) is None for a in attention),
                "no_fast_prefill_sharing": not runner.cache_config.kv_sharing_fast_prefill,
                "gdn_fla": _get_fla_gdn_prefill_op() is not None,
            }
            checks.update(metadata_checks(context,step.batch,n,AscendMetadata,GDNAttentionMetadata,lengths))
            if fixed_pair:
                checks["scheduled_lengths"] = tuple(step.batch.num_scheduled_tokens.tolist()) == lengths
                checks["bound_total"] = n <= 4096
            if POST_NORM:
                state = runner.model_state
                checks.update(
                    final_pooler=(type(runner.pooling_runner) is PoolingRunner
                                  and runner.pooling_runner.model is outer),
                    no_postnorm_state_consumers=(not has_kv_transfer_group()
                        and runner.speculator is None and runner.kvpp.scheduler is None
                        and state._offload_req_ids is None and state._pending_bulk_copy is None
                        and getattr(outer, "prepare_engram_inputs", None) is None
                        and state.prompt_embeds_state is None),
                )
                if inputs_embeds is not None:
                    encoder = state.encoder_runner
                    checks["token_embedding_buffer"] = (
                        type(encoder) is EncoderRunner and encoder.model is outer
                        and inputs_embeds.data_ptr() == encoder.inputs_embeds.data_ptr()
                        and inputs_embeds.shape == (n, 5120) and inputs_embeds.stride() == (5120, 1)
                    )
            step.contract = checks
            if record and not fixed_pair:
                batch = step.batch
                expected = {
                    "parallel": [1, 1, 1, 1], "single_sequence": 1, "no_chunk": False,
                    "no_prefix": False, "no_mamba_prefix": "none", "no_transfer": [True, True],
                    "no_spec_or_lora": [True, True], "no_aux_sp": [False, False, False],
                    "range": [0, 64], "gdn_flags": ["1"] * len(GDN_FLAGS) + [False],
                    "gdn_layers": [True] * 48, "no_cache_sharing": [None] * 16,
                    "no_fast_prefill_sharing": False, "gdn_fla": True,
                }
                actual = {
                    "parallel": [pc.pipeline_parallel_size, pc.data_parallel_size,
                                 pc.prefill_context_parallel_size, pc.decode_context_parallel_size],
                    "single_sequence": runner.scheduler_config.max_num_seqs,
                    "no_chunk": runner.scheduler_config.enable_chunked_prefill,
                    "no_prefix": runner.cache_config.enable_prefix_caching,
                    "no_mamba_prefix": runner.cache_config.mamba_cache_mode,
                    "no_transfer": [config.kv_transfer_config is None,
                                    runner.kv_connector is NO_OP_KV_CONNECTOR],
                    "no_spec_or_lora": [runner.speculative_config is None, runner.lora_config is None],
                    "no_aux_sp": [bool(runner.use_aux_hidden_state_outputs), bool(core.use_sequence_parallel),
                                  bool(core.aux_hidden_state_layers)],
                    "range": [core.start_layer, core.end_layer],
                    "gdn_flags": [os.environ.get(k) for k in GDN_FLAGS]
                                 + [bool(os.environ.get("CLEF_GDN_FLAG_FILE"))],
                    "gdn_layers": [getattr(g, "_decision_safe_config", False) for g in gdn_layers],
                    "no_cache_sharing": [getattr(a.impl, "kv_sharing_target_layer_name", True)
                                         for a in attention],
                    "no_fast_prefill_sharing": runner.cache_config.kv_sharing_fast_prefill,
                    "gdn_fla": checks["gdn_fla"],
                }
                if batch is not None:
                    expected.update(eager="NONE", no_ubatch=True, batch_descriptor=n, per_layer_metadata=True,
                                    num_reqs=[1, 1], tokens=[n, n], num_drafts=0, query_start=[0, n],
                                    initial_computed=[0], prefill_computed=[0], prefill_len=[n], seq_len=[n],
                                    batch_attn_enum="PrefillCacheHit")
                    actual.update(
                        eager=getattr(context.cudagraph_runtime_mode, "name", None),
                        no_ubatch=context.ubatch_slices is None,
                        batch_descriptor=getattr(context.batch_descriptor, "num_tokens", None),
                        per_layer_metadata=isinstance(context.attn_metadata, dict),
                        num_reqs=[batch.num_reqs, batch.num_reqs_after_padding],
                        tokens=[batch.num_tokens, batch.num_tokens_after_padding],
                        num_drafts=batch.num_draft_tokens, query_start=batch.query_start_loc_np.tolist()[:2],
                        initial_computed=batch.num_computed_tokens_np.tolist(),
                        prefill_computed=batch.num_computed_prefill_tokens_np.tolist(),
                        prefill_len=batch.prefill_len_np.tolist(), seq_len=batch.seq_lens_np[:1].tolist(),
                        batch_attn_enum=getattr(batch.attn_state, "name", None),
                    )
                if isinstance(context.attn_metadata, dict):
                    names = {}
                    unique = {}
                    for layer_name, meta in context.attn_metadata.items():
                        names.setdefault(id(meta), []).append(layer_name)
                        unique[id(meta)] = meta
                    metadata = []
                    for meta_id, m in unique.items():
                        row = {"layers": names[meta_id], "type": type(m).__qualname__, "checks": {}}
                        metadata.append(row)
                        fields = row["checks"]
                        for attr, wanted in (("num_prefills", 1), ("num_decodes", 0),
                                             ("num_decode_tokens", 0), ("num_actual_tokens", n)):
                            check_fact(fields, attr, getattr(m, attr, None), wanted)
                        if type(m) is AscendMetadata:
                            for attr, wanted in (("actual_seq_lengths_q", [n]), ("seq_lens_list", [n]),
                                                 ("causal", True), ("model_runner_type", "pooling"),
                                                 ("uses_pcp_sharded_layout", False)):
                                check_fact(fields, attr, getattr(m, attr, None), wanted)
                            check_fact(fields, "attn_state", getattr(m.attn_state, "name", None),
                                       "PrefillCacheHit", ok=batch is not None and m.attn_state is batch.attn_state)
                        if type(m) is GDNAttentionMetadata:
                            for attr, wanted in (("num_prefill_tokens", n), ("num_spec_decodes", 0),
                                                 ("num_spec_decode_tokens", 0)):
                                check_fact(fields, attr, getattr(m, attr, None), wanted)
                            check_fact(fields, "spec_sequence_masks_absent", m.spec_sequence_masks is None, True)
                    step.guard_facts["metadata"] = metadata
                step.guard_facts["step"] = {
                    field: {"expected": expected.get(field, True), "actual": actual.get(field, ok), "ok": ok}
                    for field, ok in checks.items()
                }
            if record and fixed_pair:
                step.guard_facts["step"] = {
                    "checks": checks, "lengths": lengths, "packed_request_ids": list(step.batch.req_ids),
                }
            if not all(checks.values()):
                return None
            checks["verified_cache_binding"] = cache_binding(context, n, record, step)
            if not checks["verified_cache_binding"]:
                return None
            flags = snapshot_for(context)
            facts = step.guard_facts.setdefault("forward", {}) if record else None
            if not check_fact(facts, "gdn_snapshot_flags",
                              [flags.decision_prefill, flags.bsnd, flags.skip_conv_state, flags.skip_initial_state],
                              [True] * 4):
                return None
            if not check_fact(facts, "language_positions", [str(positions.dtype), tuple(positions.shape)],
                              ["torch.int64", [(n,), (3, n)]],
                              ok=positions.dtype == torch.int64 and tuple(positions.shape) in ((n,), (3, n))):
                return None
            if not check_fact(facts, "input_embeds", None if inputs_embeds is None else layout(inputs_embeds),
                              "absent or BF16 (T,5120)",
                              ok=inputs_embeds is None or (inputs_embeds.dtype == torch.bfloat16
                                                          and tuple(inputs_embeds.shape) == (n, 5120))):
                return None
            actual_tokens = step.batch.input_ids if input_ids is None else input_ids
            tokens = tuple(actual_tokens.to("cpu",non_blocking=False).tolist())
            if fixed_pair:
                packed_positions = positions.to("cpu",non_blocking=False).tolist()
                axes = [packed_positions] if positions.ndim == 1 else packed_positions
                return pack_pair(step.ordered_items,step.batch.req_ids,step.batch.query_start_loc_np.tolist()[:3],
                                 tokens,axes,positions.stride(),domain)
            if not check_fact(facts, "same_input_tokens", tokens == step.request.token_ids, True):
                return None
            pos = positions.to("cpu", non_blocking=False).flatten().tolist()
            if not check_fact(facts, "mrope_position_range",
                              pos == list(range(n)) * (positions.numel() // n), True):
                return None
            return text_key(domain, tokens, pos, positions.shape, positions.stride())

        def write_receipt(receipt):
            nonlocal audit_sequence
            audit_sequence += 1
            with (diagnostics / f"step-{audit_sequence:04d}-{tp.rank_in_group}.json").open("x") as out:
                json.dump(receipt, out)

        def tensor_digest(tensor):
            return sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()

        def tap_stock(step, n, original):
            layers = []
            detail = {"captured": False}

            def hook(index):
                def observe(_module, _args, result):
                    layers.append(index)
                    if index != BOUNDARY - 1:
                        return None
                    if (type(result) is not tuple or len(result) != 2
                            or any(not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != n
                                   for x in result)):
                        detail["reason"] = "unknown layer output or incomplete token axis"
                        return None
                    tensors = []
                    for tensor in result:
                        dtype, shape, stride = layout(tensor)
                        row = {"dtype": dtype, "shape": shape, "stride": stride,
                               "device": str(tensor.device),
                               "logical_bytes": tensor.numel() * tensor.element_size()}
                        try:
                            row["format"] = torch_npu.get_npu_format(tensor)
                        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                            row["format_unavailable"] = type(error).__name__
                        tensors.append(row)
                    detail["tensors"] = tensors
                    logical_bytes = sum(row["logical_bytes"] for row in tensors)
                    detail["logical_bytes"] = logical_bytes
                    if logical_bytes > cap:
                        detail["reason"] = "rank byte cap"
                        return None
                    try:
                        if logical_bytes > budget(0):
                            detail["reason"] = "memory budget"
                            return None
                        for tensor, row in zip(result, tensors, strict=True):
                            start = perf_counter_ns()
                            cpu = tensor.detach().to(device="cpu", non_blocking=False, copy=True,
                                                     memory_format=torch.contiguous_format)
                            row["d2h_ns"] = perf_counter_ns() - start
                            row["host_layout"] = layout(cpu)
                            row["host_storage_bytes"] = cpu.untyped_storage().nbytes()
                            if row["host_storage_bytes"] != row["logical_bytes"]:
                                detail["reason"] = "unexpected host allocation geometry"
                                return None
                            row["sha256"] = tensor_digest(cpu)
                            del cpu
                        detail["captured"] = True
                    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as error:
                        detail["reason"] = f"tap copy unavailable: {type(error).__name__}"
                    return None
                return observe

            with ExitStack() as handles:
                for index, decoder in enumerate(core.layers):
                    handle = decoder.register_forward_hook(hook(index))
                    handles.callback(handle.remove)
                output = original()
            receipt = {"rank": tp.rank_in_group, "pid": os.getpid(), "mode": "tap",
                       "request_id": step.request.request_id, "tokens": n,
                       "admitted_local": False, "hit": False, "layers": layers,
                       "boundary": [[x["dtype"], x["shape"], x["stride"]]
                                    for x in detail.get("tensors", [])],
                       "tap": detail, "contract": None, "kv_proof": None, "cache": cache.snapshot()}
            try:
                cpu = output.detach().to(device="cpu", non_blocking=False, copy=True,
                                         memory_format=torch.contiguous_format)
                receipt["final_hidden_sha256"] = tensor_digest(cpu)
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                receipt["final_hidden_unavailable"] = type(error).__name__
            write_receipt(receipt)
            return output

        def final_hidden_layout(tensor, n, device):
            if not (type(tensor) is torch.Tensor and tensor.device == device
                    and tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == (n, 5120)
                    and tuple(tensor.stride()) == (5120, 1) and tensor.storage_offset() == 0
                    and tensor.untyped_storage().nbytes() == n * 10240):
                return False
            # ATen's Ascend ND format id; reject unknown or transformed layouts.
            try:
                return device.type == "cpu" or torch_npu.get_npu_format(tensor) == 2
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            nonlocal head_audit
            step = ACTIVE.get()
            if step is None or step.runner is not runner:
                return stock(input_ids, positions, intermediate_tensors, inputs_embeds)
            chosen, record = mode, audit
            veto_restore = False
            if control is not None:
                try:
                    row = json.loads(Path(control).read_bytes())
                    chosen, record = row["mode"], row["audit"]
                    if chosen not in ("tap", "stock", "split", "cache") or type(record) is not bool:
                        chosen, record = "stock", False
                    veto_restore = POST_NORM and record and row.get("veto_restore_rank") == tp.rank_in_group
                except (OSError, ValueError, KeyError, TypeError):
                    chosen, record = "stock", False
            def original():
                return stock(input_ids, positions, intermediate_tensors, inputs_embeds)

            if fixed_pair:
                n = sum(len(item.token_ids) for item in step.request)
                request_id = "|".join(step.batch.req_ids) if step.batch is not None else ""
            else:
                n = len(step.request.token_ids)
                request_id = step.request.request_id
            if chosen == "tap":
                if record and not fixed_pair and intermediate_tensors is None:
                    return tap_stock(step, n, original)
                return original()
            ctx = get_forward_context()
            key = None
            if intermediate_tensors is None and chosen != "stock":
                domain = sha256(namespace + chosen.encode()).digest()
                try:
                    key = timing(step, "key_ns", lambda: contract(
                        step, ctx, inputs_embeds, input_ids, positions, domain, record
                    ))
                except (AttributeError, ValueError, TypeError, RuntimeError) as error:
                    # Both ranks still vote. An unavailable local contract
                    # must not permit one rank to enter the layer loop first.
                    step.contract = {"failure_type": type(error).__name__, "detail": str(error)[:240]}
            def exchange(digest, ready):
                local = vote_integers(digest, ready)
                vector = torch.tensor(local, dtype=torch.int32, device="cpu")
                timing(step, "vote_ns", lambda: tp.cpu_all_reduce_(vector))
                return vote_result(local, vector.tolist())

            def restore(pair):
                if veto_restore:
                    step.times["injected_restore_veto"] = True
                    return None

                def copy():
                    copied = tuple(x.to(positions.device, non_blocking=False, copy=True) for x in pair)
                    torch.npu.current_stream().synchronize()
                    return copied
                try:
                    copied = timing(step, "h2d_ns", copy)
                    if POST_NORM and (len(copied) != 1 or
                                      not final_hidden_layout(copied[0], n, positions.device)):
                        return None
                    if record:
                        step.boundary = [layout(x) for x in copied]
                        step.times["restored_boundary_sha256"] = [
                            sha256(memoryview(x.to("cpu", non_blocking=False).view(torch.uint8).numpy())).hexdigest()
                            for x in copied
                        ]
                    return copied
                except (RuntimeError, MemoryError):
                    return None

            def capture_tensors(digest, pair):
                step.boundary = [layout(x) for x in pair]
                if any(x.dtype != torch.bfloat16 or tuple(x.shape) != (n, 5120)
                       or tuple(x.stride()) != (5120, 1) for x in pair):
                    return None
                bytes_ = sum(x.numel() * x.element_size() for x in pair)
                ticket = cache.reserve(digest, bytes_)
                if ticket is None:
                    return None
                try:
                    host = timing(step, "d2h_ns", lambda: tuple(
                        x.to("cpu", non_blocking=False, copy=True) for x in pair
                    ))
                    if POST_NORM and not final_hidden_layout(host[0], n, torch.device("cpu")):
                        cache.cancel(ticket)
                        return None
                    if record:
                        step.times["boundary_sha256"] = [
                            sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest() for x in host
                        ]
                    return ticket, host, bytes_
                except (MemoryError, RuntimeError):
                    cache.cancel(ticket)
                    return None
                except BaseException:
                    cache.cancel(ticket)
                    raise

            def capture(digest, hidden, residual):
                return None if residual is None else capture_tensors(digest, (hidden, residual))

            def capture_final(digest, output):
                if not final_hidden_layout(output, n, positions.device):
                    return None
                return capture_tensors(digest, (output,))

            def restore_pair(pair, values):
                if veto_restore:
                    step.times["injected_restore_veto"] = True
                    return None

                def copy():
                    spans = tuple(zip(pair.offsets[:-1],pair.offsets[1:],values,strict=True))
                    if any(not final_hidden_layout(host,end-begin,torch.device("cpu"))
                           for begin,end,host in spans):
                        return None
                    packed = torch.empty((n,5120),dtype=torch.bfloat16,device=positions.device)
                    try:
                        for begin,end,host in spans:
                            packed[begin:end].copy_(host,non_blocking=False)
                    finally:
                        try:
                            torch.npu.current_stream().synchronize()
                        except RuntimeError as exc:
                            raise OSError("could not retire the paired H2D stream") from exc
                    return packed if final_hidden_layout(packed,n,positions.device) else None

                return timing(step,"h2d_ns",copy)

            def capture_pair(digest,begin,end,output):
                if not final_hidden_layout(output,n,positions.device):
                    return None
                size = (end-begin)*10240
                ticket = cache.reserve(digest,size)
                if ticket is None:
                    return None
                try:
                    host = timing(step,"d2h_ns",lambda: output[begin:end].to("cpu",non_blocking=False,copy=True))
                    if not final_hidden_layout(host,end-begin,torch.device("cpu")):
                        cache.cancel(ticket)
                        return None
                    return ticket,host,size
                except BaseException:
                    cache.cancel(ticket)
                    raise

            def input_pair():
                hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
                return hidden, None

            def observe_boundary(hidden, residual):
                step.boundary = [layout(x) for x in (hidden, residual) if x is not None]

            layers = [] if record else None
            if POST_NORM:
                with ExitStack() as handles:
                    if record:
                        step.times["norm_calls"] = 0
                        for index, decoder in enumerate(core.layers):
                            def observe(_module, _args, _output, index=index):
                                layers.append(index)
                            handles.callback(decoder.register_forward_hook(observe).remove)

                        def observe_norm(_module, _args, _output):
                            step.times["norm_calls"] += 1
                        handles.callback(core.norm.register_forward_hook(observe_norm).remove)

                    if fixed_pair:
                        result, hit = run_fixed_pair(
                            key,cache=cache,restore=restore_pair,capture=capture_pair,
                            exchange=exchange,stock=original,enabled=chosen=="cache",
                        )
                    else:
                        result, hit = run_final_candidate(
                            key=key,cache=cache,restore=restore,capture=capture_final,
                            exchange=exchange,stock=original,enabled=chosen=="cache",
                        )
                if record and type(result) is torch.Tensor:
                    step.times["final_return"] = {
                        "layout": layout(result), "format": torch_npu.get_npu_format(result),
                        "storage_bytes": result.untyped_storage().nbytes(),
                    }
            else:
                result, hit = run_candidate(
                    self.layers, positions, input_pair, self.norm, key=key,
                    cache=cache, restore=restore, capture=capture, exchange=exchange,
                    stock=original,
                    enabled=chosen == "cache", observe_boundary=observe_boundary, observed_layers=layers,
                )
            if formal:
                counters["calls"] += 1
                if fixed_pair:
                    counters["requests"] += 2
                    counters["hit_requests"] += 2 * int(hit)
                counters["hit" if hit else "miss" if key is not None else "guard_fallback"] += 1
                for name in ("d2h_ns", "h2d_ns", "vote_ns", "key_ns"):
                    counters[name] += step.times.get(name, 0)
            if record:
                host = result.to("cpu",non_blocking=False).contiguous()
                digest = tensor_digest(host)
                if fixed_pair:
                    step.times["packed_req_ids"] = list(step.batch.req_ids)
                    if key is not None:
                        step.times["packed_offsets"] = key.offsets
                        step.times["item_final_hidden_sha256"] = [
                            tensor_digest(host[a:b]) for a,b in zip(key.offsets[:-1],key.offsets[1:],strict=True)
                        ]
                receipt = {
                    "rank": tp.rank_in_group, "pid": os.getpid(), "tokens": n, "mode": chosen,
                    "request_id": request_id, "guard_facts": step.guard_facts,
                    "admitted_local": key is not None, "hit": hit, "boundary": step.boundary,
                    "layers": layers, "contract": step.contract,
                    "kv_proof": step.kv_proof,
                    "final_hidden_sha256": digest, "cache": cache.snapshot(), **step.times,
                }
                write_receipt(receipt)
                if POST_NORM:
                    head_audit = (audit_sequence,request_id,n)
            return result

        core.forward = MethodType(forward, core)
        print("K32_READY", tp.rank_in_group, os.getpid(), mode, flush=True)
        return output

    @wraps(prepare)
    def prepare_inputs(runner, *args, **kwargs):
        batch = prepare(runner, *args, **kwargs)
        step = ACTIVE.get()
        if step is not None and step.runner is runner:
            step.batch = batch
        return batch

    @wraps(execute)
    def execute_model(runner, scheduler_output, intermediate_tensors=None, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[0] if args else False)
        profiling = kwargs.get("is_profile", args[2] if len(args) > 2 else False)
        if fixed_pair:
            candidate = fresh_text_items(scheduler_output,count=2,dummy_run=dummy,is_profile=profiling)
        else:
            candidate = fresh_text_step(scheduler_output,dummy_run=dummy,is_profile=profiling)
        if candidate is None and not dummy and not profiling and control:
            try:
                diagnose = json.loads(Path(control).read_bytes()).get("audit") is True
            except (OSError, ValueError, TypeError, AttributeError):
                diagnose = False
            if diagnose:
                print("K32_SCHEDULE_BYPASS", json.dumps({
                    "pid": os.getpid(), "new": len(scheduler_output.scheduled_new_reqs),
                    "cached": len(scheduler_output.scheduled_cached_reqs.req_ids),
                    "scheduled": scheduler_output.total_num_scheduled_tokens,
                    "new_facts": [{"num_computed_tokens": x.num_computed_tokens,
                                   "prompt_length": len(x.prompt_token_ids or ()),
                                   "prefill_length": len(x.prefill_token_ids or ()),
                                   "task": str(getattr(x.pooling_params, "task", None)),
                                   "mm_features": len(x.mm_features)}
                                  for x in scheduler_output.scheduled_new_reqs],
                }), flush=True)
        if (formal and not dummy and not profiling and scheduler_output.total_num_scheduled_tokens > 0
                and hasattr(runner, "_clef_private_k32_cache")):
            state = runner._clef_private_k32_cache
            if candidate is None:
                state.scheduler_bypasses += 1
            if fixed_pair:
                state.batch_steps[len(scheduler_output.num_scheduled_tokens)] += 1
        stamp = ACTIVE.set(Step(runner, candidate) if candidate is not None else None)
        try:
            return execute(runner, scheduler_output, intermediate_tensors, *args, **kwargs)
        finally:
            ACTIVE.reset(stamp)

    runner_class.load_model = load_model
    runner_class.prepare_inputs = prepare_inputs
    runner_class.execute_model = execute_model


def preflight():
    # NPUWorker.__init__ installs these patches before the worker imports its
    # V2 runner. No model, process group, or device is initialized here.
    from vllm_ascend.worker import worker  # load worker-side environment and imports
    from vllm_ascend.utils import adapt_patch
    adapt_patch()
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    rows = verify_runtime_sources(NPUModelRunner)
    print("K32_SOURCE_RECEIPT " + json.dumps(rows, sort_keys=True), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["verify-sources"]:
        raise SystemExit("expected verify-sources")
    preflight()
