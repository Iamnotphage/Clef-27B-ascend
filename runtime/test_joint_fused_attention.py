#!/usr/bin/env python3
"""CPU equivalence test for the fused joint-head cross-attention candidate.

No real vllm or torch_npu package is imported.  The vLLM symbols used by the
postimage are stubbed and ``torch_npu.npu_fusion_attention`` is a CPU reference
implementation of the BSH attention it replaces.  The test only checks that the
fused path computes the same function as the original MHA path.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
POSTIMAGE = HERE / "clef_joint_schema.py"


def stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class Pooler(torch.nn.Module):
    pass


class PoolingParamsUpdate:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class AllPool(torch.nn.Module):
    pass


class PoolerOutput:
    pass


class PoolingMetadata:
    pass


stub("vllm.model_executor.layers.pooler", Pooler=Pooler,
     PoolingParamsUpdate=PoolingParamsUpdate)
stub("vllm.model_executor.layers.pooler.tokwise.methods", AllPool=AllPool)
stub("vllm.tasks", PoolingTask=object)
stub("vllm.utils.torch_utils", async_tensor_h2d=lambda value, device, dtype=None:
     torch.as_tensor(value, device=device, dtype=dtype))
stub("vllm.v1.outputs", PoolerOutput=PoolerOutput)
stub("vllm.v1.pool.metadata", PoolingMetadata=PoolingMetadata)


def reference_attention(query, key, value, head_num, input_layout, **kwargs):
    assert input_layout == "BSH"
    batch, q_len, width = query.shape
    kv_len = key.shape[1]
    head_dim = width // head_num
    q = query.view(batch, q_len, head_num, head_dim).transpose(1, 2)
    k = key.view(batch, kv_len, head_num, head_dim).transpose(1, 2)
    v = value.view(batch, kv_len, head_num, head_dim).transpose(1, 2)
    scale = kwargs.get("scale", 1.0 / math.sqrt(head_dim))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, q_len, width)
    return (out, None, None, None, 0, 0, 0)


fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.npu_fusion_attention = reference_attention
sys.modules["torch_npu"] = fake_torch_npu


def load_postimage():
    spec = importlib.util.spec_from_file_location("clef_joint_schema_fused", POSTIMAGE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_postimage()
    torch.manual_seed(20261119)
    hidden_size, width, heads, ff = 32, 16, 4, 32
    head = mod.JointSchemaHead(hidden_size, width, 1, 1, heads, ff).eval()
    q_len, options = 1, 3
    hidden = torch.randn(6, hidden_size)
    lexical = torch.randn(options, hidden_size)
    schema = mod.SchemaIndices(
        positions=torch.tensor([0, 1, 2, 3]),
        span_ids=torch.tensor([0, 0, 1, 1]),
        span_lengths=torch.tensor([2, 2]),
        question_types=torch.tensor([1]),
        option_question=torch.tensor([0, 0, 0]),
        option_slots=torch.tensor([0, 1, 2]),
        option_mask=torch.tensor([[True, True, True]]),
    )
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "0"
    with torch.no_grad():
        baseline = head(hidden, lexical, schema)
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "1"
    with torch.no_grad():
        fused = head(hidden, lexical, schema)
    delta = (baseline - fused).abs()
    same_shape = baseline.shape == fused.shape
    close = torch.allclose(baseline, fused, atol=1e-5, rtol=1e-4)
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "0"
    helper_off = mod._fused_cross_attention(
        head.evidence_layers[0].attention,
        torch.randn(1, 1, width), torch.randn(1, 5, width)) is None
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "1"
    q2 = mod._fused_cross_attention(
        head.evidence_layers[0].attention,
        torch.randn(1, 2, width), torch.randn(1, 5, width)) is None
    checks = {
        "shape": same_shape,
        "close": close,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "off_returns_none": helper_off,
        "q2_falls_back": q2,
    }
    print(checks)
    ok = same_shape and close and helper_off and q2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
