"""Clef small-Q cross-attention, Triton Ascend; imported only when enabled.

BF16 inputs/output, FP32 products, online softmax and accumulation. No masks,
dropout, causality, padding or attention-weight output. One program per
(query, head) and one device launch; no QK/probability tensor in HBM.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["L"])
def clef_smallq_online(
    Q, K, V, Out,
    L, QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
    WIDTH: tl.constexpr = 1024, D: tl.constexpr = 64,
    BLOCK: tl.constexpr = 64,
):
    head = tl.program_id(0)
    query = tl.program_id(1)
    dims = tl.arange(0, D)
    rows = tl.arange(0, BLOCK)
    q = tl.load(Q + query * QS + head * D + dims).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(L, BLOCK)):
        tokens = start * BLOCK + rows
        k = tl.load(K + tokens[:, None] * KS + head * D + dims[None, :],
                    tokens[:, None] < L, 0).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * 0.125
        score = tl.where(tokens < L, score, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(score, 0))
        correction = tl.exp(maximum - next_maximum)
        probability = tl.exp(score - next_maximum)
        v = tl.load(V + tokens[:, None] * VS + head * D + dims[None, :],
                    tokens[:, None] < L, 0).to(tl.float32)
        acc = acc * correction + tl.sum(probability[:, None] * v, 0)
        denominator = denominator * correction + tl.sum(probability, 0)
        maximum = next_maximum
    tl.store(Out + query * WIDTH + head * D + dims, acc / denominator)


def smallq_attention(q, k, v):
    """Projected BSH inputs; the head adapter validates the supported domain."""
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    clef_smallq_online[(16, q.shape[1])](
        q, k, v, output, k.shape[1], q.stride(1), k.stride(1), v.stride(1),
    )
    return output
