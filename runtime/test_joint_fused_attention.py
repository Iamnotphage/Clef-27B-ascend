#!/usr/bin/env python3
"""CPU equivalence test for the fused joint-head cross-attention candidate.

No real vllm or torch_npu package is imported.  The vLLM symbols used by the
postimage are stubbed and ``torch_npu.npu_fusion_attention`` is a CPU reference
implementation of the BSH attention it replaces.  The test checks the fused
path mathematically and counts the six single-field cross-attention calls
(two evidence-routing layers plus four decoder layers).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
POSTIMAGE = HERE / "joint_fused_attention" / "clef_joint_schema.py"
CALLS = {"n": 0}


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
    CALLS["n"] += 1
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
    routing_layers, layers = 2, 4
    head = mod.JointSchemaHead(hidden_size, width, routing_layers, layers,
                               heads, ff).eval()
    options = 3
    hidden = torch.randn(8, hidden_size)
    lexical = torch.randn(options, hidden_size)
    schema = mod.SchemaIndices(
        positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        span_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        span_lengths=torch.tensor([2, 2, 2, 2]),
        question_types=torch.tensor([1]),
        option_question=torch.tensor([0, 0, 0]),
        option_slots=torch.tensor([0, 1, 2]),
        option_mask=torch.tensor([[True, True, True]]),
    )
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
    with torch.inference_mode():
        os.environ["CLEF_JOINT_FUSED_ATTN"] = "0"
        CALLS["n"] = 0
        baseline = head(hidden, lexical, schema)
        off_calls = CALLS["n"]
        os.environ["CLEF_JOINT_FUSED_ATTN"] = "1"
        CALLS["n"] = 0
        fused = head(hidden, lexical, schema)
        on_calls = CALLS["n"]
        disabled = mod._fused_cross_attention(
            head.evidence_layers[0].attention,
            torch.randn(1, 3, width), torch.randn(1, 5, width), enabled=False)
    delta = (baseline - fused).abs()
    checks = {
        "shape": baseline.shape == fused.shape,
        "close": torch.allclose(baseline, fused, atol=1e-5, rtol=1e-4),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "off_calls": off_calls,
        "on_calls": on_calls,
        "six_fused_calls": on_calls == routing_layers + layers,
        "disabled_none": disabled is None,
    }
    print(checks)
    ok = (checks["shape"] and checks["close"] and checks["six_fused_calls"]
          and checks["disabled_none"] and off_calls == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
