# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared joint schema head for Clef decision models.

The head reads the final hidden states of the whole prompt, so it runs as a
``token_classify`` pooler.

Each request carries its schema in ``PoolingParams.extra_kwargs`` under
``"joint_schema"``:

- ``question_types``: 0 (noul), 1 (choice) or 2 (score) per question.
- ``question_spans``: the ``[start, end)`` of each question's instructions.
- ``option_spans``: the ``[start, end)`` of each option, question by question.
- ``option_counts``: the number of options of each question.

Spans count from the end of the prompt (``-len(prompt) <= start < end <= 0``),
so they hold after multimodal placeholders are expanded. The output has one
value per option, in ``option_spans`` order: its probability within its
question, or its logit with ``use_activation=False``.
"""

import math
from collections.abc import Callable, Set
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

JOINT_HEAD_WEIGHTS = "joint_head.safetensors"
NUM_QUESTION_TYPES = 3
MAX_LOGIT_SCALE = math.log(100.0)


def segment_mean(
    values: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    sums = values.new_zeros(lengths.shape[0], values.shape[-1], dtype=torch.float32)
    sums.index_add_(0, segment_ids, values.float())
    return (sums / lengths[:, None]).to(values.dtype)


@dataclass
class SchemaIndices:
    """Where one prompt's questions and options are, on the model's device."""

    positions: torch.Tensor
    """Every token of every span, questions first."""
    span_ids: torch.Tensor
    """The span of each of ``positions``."""
    span_lengths: torch.Tensor
    """``[Q + O]``."""
    question_types: torch.Tensor
    """``[Q]``."""
    option_question: torch.Tensor
    """The question of each option, ``[O]``."""
    option_slots: torch.Tensor
    """Each option's flat index in a ``[Q, max_options]`` grid, ``[O]``."""
    option_mask: torch.Tensor
    """``[Q, max_options]``, true at the slots of options."""



def _smallq_cross_attention(module, query, key, value, *, enabled):
    """Strict inference-only domain; all unsupported calls retain original MHA."""
    import os
    if not enabled or os.environ.get("CLEF_JOINT_SMALLQ", "0") != "1":
        return None
    if (module.training or torch.is_grad_enabled() or module.dropout != 0
            or not module.batch_first or not module._qkv_same_embed_dim
            or module.in_proj_weight is None or module.bias_k is not None
            or module.bias_v is not None or module.add_zero_attn
            or module.embed_dim != 1024 or module.num_heads != 16):
        return None
    # Snapshot metadata once per call; all dynamic guards remain live.
    same_kv = key is value
    q_shape, k_shape = query.shape, key.shape
    v_shape = k_shape if same_kv else value.shape
    if any(len(shape) != 3 or shape[0] != 1 or shape[2] != 1024
           for shape in ((q_shape, k_shape) if same_kv else (q_shape, k_shape, v_shape))):
        return None
    if not (1 <= q_shape[1] <= 8 and 1 <= k_shape[1] <= 2048
            and k_shape == v_shape):
        return None
    q_dtype, q_device = query.dtype, query.device
    if (q_dtype != torch.bfloat16 or q_device.type != "npu"
            or query.stride(-1) != 1 or query.stride(1) < 1024):
        return None
    # Shared decoder K/V has identical metadata and needs one validation.
    for tensor in ((key,) if same_kv else (key, value)):
        if (tensor.dtype != q_dtype or tensor.device != q_device
                or tensor.stride(-1) != 1 or tensor.stride(1) < 1024):
            return None
    out_proj = module.out_proj
    parameters = (module.in_proj_weight, module.in_proj_bias,
                  out_proj.weight, out_proj.bias)
    for parameter in parameters:
        if parameter is not None and (parameter.dtype != q_dtype
                                      or parameter.device != q_device):
            return None
    # Use the same packed/unpacked projection dispatch and tensor ranks as MHA.
    # Routing has separate norm outputs for K/V; decoder shares K/V identity.
    if key is value:
        q_in, k_in = query.transpose(1, 0), key.transpose(1, 0)
        v_in = k_in
    else:
        q_in, k_in, v_in = (t.transpose(1, 0) for t in (query, key, value))
    q, k, v = F._in_projection_packed(
        q_in, k_in, v_in, module.in_proj_weight, module.in_proj_bias)
    from .clef_smallq_kernel import smallq_attention
    output = smallq_attention(q.transpose(1, 0), k.transpose(1, 0),
                              v.transpose(1, 0))
    output = F.linear(output.view(query.shape[1], 1024),
                      module.out_proj.weight, module.out_proj.bias)
    return output.view(1, query.shape[1], 1024)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # The Identity takes the place of the checkpoint's dropout.
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, width),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor,
                single_field: bool = False) -> torch.Tensor:
        # Preserve the reference's separate K/V norms and unpacked projections.
        q_in = self.query_norm(queries)
        k_in = self.memory_norm(memory)
        v_in = self.memory_norm(memory)
        routed = _smallq_cross_attention(self.attention, q_in, k_in, v_in,
                                         enabled=single_field)
        if routed is None:
            routed, _ = self.attention(q_in, k_in, v_in, need_weights=False)
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(NUM_QUESTION_TYPES, width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(width, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        lexical: torch.Tensor,
        schema: SchemaIndices,
    ) -> torch.Tensor:
        """The ``[O]`` option logits of one prompt, from its ``[L, hidden]``
        final hidden states and each option's ``[O, hidden]`` mean output
        embedding."""
        import os

        num_questions = schema.question_types.shape[0]
        option_question = schema.option_question
        hidden_states = self.hidden_norm(hidden_states)
        memory = self.memory_projection(hidden_states).unsqueeze(0)
        global_vector = hidden_states[-1]
        means = segment_mean(
            hidden_states[schema.positions], schema.span_ids, schema.span_lengths
        )
        question_vectors, contexts = means[:num_questions], means[num_questions:]

        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors)[option_question]
        ).unsqueeze(0)
        for layer in self.evidence_layers:
            options = layer(options, memory, single_field=(num_questions == 1))
        options = options[0]

        # Each field summarizes its options, weighted by their match to it.
        base_fields = self.question_projection(question_vectors)
        width = options.shape[-1]
        grid = options.new_zeros(schema.option_mask.numel(), width)
        grid[schema.option_slots] = options
        grid = grid.view(*schema.option_mask.shape, width)
        if num_questions == 1 and os.environ.get("CLEF_JOINT_SINGLE_FIELD", "0") == "1":
            options_grid = grid[0]
            scores = (options_grid * base_fields[0]).sum(dim=-1) / math.sqrt(width)
            scores = scores.masked_fill(~schema.option_mask[0], -1e4)
            summaries = (
                scores.softmax(dim=-1).unsqueeze(-1) * options_grid
            ).sum(dim=0, keepdim=True)
        else:
            routing = torch.einsum("qow,qw->qo", grid, base_fields) / math.sqrt(width)
            routing = routing.masked_fill(~schema.option_mask, float("-inf"))
            summaries = torch.einsum("qo,qow->qw", routing.softmax(dim=-1), grid)
        fields = (
            base_fields
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(schema.question_types)
        ).unsqueeze(0)
        import os

        if num_questions == 1 and os.environ.get("CLEF_JOINT_SINGLE_FIELD", "0") == "1":
            single = self._single_field_decoder(fields, memory)
            if single is not None:
                fields = single
            else:
                for layer in self.layers:
                    fields = layer(fields, memory)
        else:
            for layer in self.layers:
                fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[option_question]

        anchors = F.normalize(question_vectors + global_vector, dim=-1)
        prior_scale = self.prior_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        prior = prior_scale * (
            F.normalize(lexical, dim=-1) * anchors[option_question]
        ).sum(dim=-1)
        options = self.option_norm(options)
        cosine = F.cosine_similarity(fields, options, dim=-1)
        features = torch.cat(
            [fields, options, fields * options, torch.abs(fields - options)], dim=-1
        )
        residual = self.residual_scorer(features).squeeze(-1)
        joint_scale = self.joint_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        joint = joint_scale * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


    def _single_field_decoder(self, fields: torch.Tensor, memory: torch.Tensor):
        """Q=1 decoder stack with the self-attention softmax folded to V.

        ``fields`` is [1, 1, W].  For one target token the self-attention
        softmax is 1, so only the V projection and output projection remain.
        Cross-attention, FFN, norms and residuals are unchanged.
        """
        for layer in self.layers:
            x = fields
            y = layer.norm1(x)
            weight = layer.self_attn.in_proj_weight
            if weight is None:
                return None
            bias = layer.self_attn.in_proj_bias
            width = layer.self_attn.embed_dim
            v = F.linear(
                y,
                weight[2 * width:3 * width],
                None if bias is None else bias[2 * width:3 * width],
            )
            self_attn = F.linear(
                v, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias
            )
            x = x + layer.dropout1(self_attn)
            q_in = layer.norm2(x)
            cross = _smallq_cross_attention(layer.multihead_attn, q_in, memory,
                                            memory, enabled=True)
            if cross is None:
                cross, _ = layer.multihead_attn(
                    q_in, memory, memory, need_weights=False
                )
            x = x + layer.dropout2(cross)
            x = x + layer.dropout3(
                layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))
                )
            )
            fields = x
        return fields


def parse_joint_schema(
    extra_kwargs: dict[str, Any] | None, prompt_len: int
) -> tuple[list[int], list[tuple[int, int]], list[int]] | None:
    """The question types, the absolute spans of the questions then of the
    options, and the option counts. None if the schema is missing or invalid."""
    schema = (extra_kwargs or {}).get("joint_schema")
    if not isinstance(schema, dict):
        return None
    try:
        types = [int(t) for t in schema["question_types"]]
        counts = [int(c) for c in schema["option_counts"]]
        question_spans = [
            (int(start) + prompt_len, int(end) + prompt_len)
            for start, end in schema["question_spans"]
        ]
        option_spans = [
            (int(start) + prompt_len, int(end) + prompt_len)
            for start, end in schema["option_spans"]
        ]
    except (KeyError, TypeError, ValueError):
        return None
    spans = question_spans + option_spans
    if (
        not types
        or len(counts) != len(types)
        or len(question_spans) != len(types)
        or len(option_spans) != sum(counts)
        or any(not 0 <= t < NUM_QUESTION_TYPES for t in types)
        or any(c < 1 for c in counts)
        or any(not 0 <= start < end <= prompt_len for start, end in spans)
    ):
        return None
    return types, spans, counts


class JointSchemaPooler(Pooler):
    """Runs the joint schema head on each finished prompt.

    It never raises, because it runs in the engine core: a prompt without a
    valid schema, such as a profiling dummy, gets an empty output.
    """

    def __init__(
        self,
        head: JointSchemaHead,
        embed_lexical: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.head = head
        self.embed_lexical = embed_lexical

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = []
        for hidden, token_ids, params in zip(
            pooled, prompt_token_ids, pooling_metadata.pooling_params
        ):
            if hidden is None:
                outputs.append(None)
                continue
            parsed = parse_joint_schema(params.extra_kwargs, hidden.shape[0])
            if parsed is None:
                outputs.append(hidden.new_empty(0, dtype=torch.float32))
                continue
            schema, lexical_ids, lexical_span_ids = self._indices(
                *parsed, hidden.device
            )
            lexical = segment_mean(
                self.embed_lexical(
                    async_tensor_h2d(token_ids[lexical_ids], hidden.device)
                ),
                lexical_span_ids,
                schema.span_lengths[len(parsed[0]) :],
            )
            logits = self.head(hidden, lexical, schema).float()
            if params.use_activation is not False:
                mask = schema.option_mask
                grid = logits.new_full((mask.numel(),), float("-inf"))
                grid[schema.option_slots] = logits
                probs = grid.view(mask.shape).softmax(dim=-1)
                logits = probs.view(-1)[schema.option_slots]
            outputs.append(logits)
        return outputs

    @staticmethod
    def _indices(
        types: list[int],
        spans: list[tuple[int, int]],
        counts: list[int],
        device: torch.device,
    ) -> tuple[SchemaIndices, torch.Tensor, torch.Tensor]:
        """The schema's indices, plus the option tokens' positions (on the CPU)
        and their option spans, for the lexical embeddings."""
        num_questions = len(types)
        lengths = np.array([end - start for start, end in spans], dtype=np.int64)
        span_ids = np.repeat(np.arange(len(spans)), lengths)
        positions = np.concatenate([np.arange(start, end) for start, end in spans])
        num_question_tokens = int(lengths[:num_questions].sum())
        counts_cpu = np.array(counts)
        max_options = max(counts)
        option_question = np.repeat(np.arange(num_questions), counts_cpu)
        option_slots = np.concatenate(
            [np.arange(count) + q * max_options for q, count in enumerate(counts)]
        )
        vectors = [
            positions,
            span_ids,
            lengths,
            np.array(types),
            option_question,
            option_slots,
            span_ids[num_question_tokens:] - num_questions,
        ]
        packed = async_tensor_h2d(np.concatenate(vectors), device, dtype=torch.long)
        (
            positions_d,
            span_ids_d,
            lengths_d,
            types_d,
            option_question_d,
            option_slots_d,
            lexical_span_ids,
        ) = packed.split([len(vector) for vector in vectors])
        schema = SchemaIndices(
            positions=positions_d,
            span_ids=span_ids_d,
            span_lengths=lengths_d,
            question_types=types_d,
            option_question=option_question_d,
            option_slots=option_slots_d,
            option_mask=async_tensor_h2d(
                np.arange(max_options) < counts_cpu[:, None], device
            ),
        )
        lexical_ids = torch.from_numpy(positions[num_question_tokens:])
        return schema, lexical_ids, lexical_span_ids
