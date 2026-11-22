"""Private same-memory KV experiment. No deployment installer or global patch.

Load the frozen aa3e head nodes without importing/initializing vLLM. Each
reference/candidate uses the same module globals, weights and BF16 boundaries.
"""
import ast
import dataclasses
import hashlib
import math
from pathlib import Path
import sys
import types

AA3E = 'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'


def load_reference(path):
    import torch
    from torch import nn
    import torch.nn.functional as F
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == AA3E
    names = {'SchemaIndices', 'segment_mean', '_fused_cross_attention',
             'EvidenceRoutingLayer', 'JointSchemaHead'}
    tree = ast.parse(source)
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    assert {n.name for n in nodes} == names
    module = types.ModuleType('_clef_aa3e_head_reference')
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=nn, F=F, math=math,
                           dataclass=dataclasses.dataclass,
                           NUM_QUESTION_TYPES=3, MAX_LOGIT_SCALE=math.log(100.0))
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def supported(module, query, memory, enabled):
    import torch
    w = getattr(module, 'in_proj_weight', None)
    return (enabled and w is not None and module.embed_dim == 1024
            and module.num_heads == 16 and tuple(w.shape) == (3072, 1024)
            and w.is_contiguous() and w.dtype == torch.bfloat16
            and query.dtype == memory.dtype == torch.bfloat16
            and query.ndim == memory.ndim == 3
            and query.shape[0] == memory.shape[0] == 1
            and 1 <= query.shape[1] <= 7 and 1 <= memory.shape[1] <= 1595
            and query.shape[2] == memory.shape[2] == 1024
            and (module.in_proj_bias is None or (
                tuple(module.in_proj_bias.shape) == (3072,)
                and module.in_proj_bias.is_contiguous()
                and module.in_proj_bias.dtype == torch.bfloat16)))


def projections(module, query, memory, mode):
    import torch.nn.functional as F
    width = module.embed_dim
    w, b = module.in_proj_weight, module.in_proj_bias
    q = F.linear(query, w[:width], None if b is None else b[:width])
    if mode == 'stock':
        k = F.linear(memory, w[width:2*width], None if b is None else b[width:2*width])
        v = F.linear(memory, w[2*width:], None if b is None else b[2*width:])
    else:
        kv = F.linear(memory, w[width:], None if b is None else b[width:])
        k, v = kv.split(width, dim=-1)
        if mode == 'contiguous':
            k, v = k.contiguous(), v.contiguous()
        else:
            assert mode == 'view'
    return q, k, v


def cross(module, query, memory, mode):
    import torch_npu
    import torch.nn.functional as F
    q, k, v = projections(module, query, memory, mode)
    out = torch_npu.npu_fusion_attention(
        q, k, v, module.num_heads, 'BSH',
        scale=1.0/math.sqrt(module.embed_dim // module.num_heads),
        keep_prob=1.0, sparse_mode=0)[0]
    return F.linear(out, module.out_proj.weight, module.out_proj.bias)


class Selection:
    """Switch only this private reference's attention function, outside timing."""
    def __init__(self, reference):
        self.reference = reference
        self.original = reference._fused_cross_attention

    def set(self, mode='stock', selected=None):
        import os
        if mode == 'stock':
            self.reference._fused_cross_attention = self.original
            return
        original = self.original

        def candidate(module, query, memory, enabled=True):
            if (os.environ.get('CLEF_JOINT_FUSED_ATTN', '0') != '1'
                    or (selected is not None and id(module) not in selected)
                    or not supported(module, query, memory, enabled)):
                return original(module, query, memory, enabled)
            return cross(module, query, memory, mode)
        self.reference._fused_cross_attention = candidate


def probabilities(logits, schema):
    logits = logits.float()
    mask = schema.option_mask
    grid = logits.new_full((mask.numel(),), float('-inf'))
    grid[schema.option_slots] = logits
    return grid.view(mask.shape).softmax(dim=-1).view(-1)[schema.option_slots]


def metadata(tensor):
    import torch_npu
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), storage_offset=tensor.storage_offset(),
                format=int(torch_npu.get_npu_format(tensor)), device=str(tensor.device))
