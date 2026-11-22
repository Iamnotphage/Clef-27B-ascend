"""Armed one-shot head capture in a private stock service; D2H outside timing."""
import builtins
import functools
import os
from pathlib import Path
import sys


def install():
    root = os.environ.get('CLEF_HEAD_KV_CAPTURE')
    if not root:
        return
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    original_import = builtins.__import__

    def patch():
        module = sys.modules.get('vllm.model_executor.models.clef_joint_schema')
        cls = getattr(module, 'JointSchemaHead', None)
        if cls is None or getattr(cls, '_clef_kv_capture', False):
            return
        import dataclasses
        import hashlib
        import json
        import torch
        from head_packed_kv import AA3E, metadata
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == AA3E
        original = cls.forward

        @functools.wraps(original)
        def forward(self, hidden_states, lexical, schema):
            arm = directory/'ARMED.json'
            if not arm.exists():
                return original(self, hidden_states, lexical, schema)
            spec = json.loads(arm.read_text())
            if hidden_states.shape[0] != spec['expected_tokens']:
                return original(self, hidden_states, lexical, schema)
            rank = hidden_states.device.index
            target = directory/f"{spec['label']}.rank{rank}.pt"
            try:
                target.with_suffix('.issued').open('x').close()
            except FileExistsError:
                return original(self, hidden_states, lexical, schema)
            payload = dict(source='actual aa3e head forward arguments', request=spec,
                source_sha256=AA3E, source_rank=rank,
                hidden=hidden_states.detach().cpu(), lexical=lexical.detach().cpu(),
                schema={f.name: getattr(schema, f.name).detach().cpu()
                        for f in dataclasses.fields(schema)},
                metadata={'hidden': metadata(hidden_states), 'lexical': metadata(lexical)})
            payload['weights'] = {}
            for name, value in self.state_dict().items():
                cpu = value.detach().cpu().contiguous()
                payload['weights'][name] = dict(metadata(value), sha256=hashlib.sha256(
                    cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
            captured = []
            old_cross = module._fused_cross_attention
            handles = []
            names = {id(m): name for name, m in self.named_modules()}
            if spec.get('capture_attention'):
                def capture_cross(m, q, mem, enabled=True):
                    out = old_cross(m, q, mem, enabled)
                    captured.append(dict(name=names[id(m)], query=q.detach().cpu(),
                        memory=mem.detach().cpu(), output=out.detach().cpu(),
                        metadata={'query': metadata(q), 'memory': metadata(mem)}))
                    return out
                module._fused_cross_attention = capture_cross
                for idx in (0, 1):
                    def prehook(m, args, idx=idx):
                        payload[f'fields{idx}'] = args[0].detach().cpu()
                    handles.append(self.layers[idx].norm1.register_forward_pre_hook(prehook))
            try:
                out = original(self, hidden_states, lexical, schema)
                payload.update(logits=out.detach().cpu(), attentions=captured)
                temp = target.with_suffix('.part')
                torch.save(payload, temp)
                temp.replace(target)
            finally:
                module._fused_cross_attention = old_cross
                for handle in handles:
                    handle.remove()
            return out

        cls.forward = forward
        cls._clef_kv_capture = True

    def hooked_import(*args, **kwargs):
        result = original_import(*args, **kwargs)
        if args and 'clef_joint_schema' in args[0]:
            patch()
        return result
    builtins.__import__ = hooked_import
    patch()
