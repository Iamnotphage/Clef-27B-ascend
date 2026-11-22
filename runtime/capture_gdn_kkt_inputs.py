"""One-shot private capture hook; enabled only by an explicit temporary env.

Use from a private sitecustomize.py in a temporary capture service. The deployed
source/wheel stay untouched. It saves the first layer's actual post-normalization
inputs for the designated 1595-token request, then leaves all calls unchanged.
"""
import builtins
import functools
import os
from pathlib import Path
import sys


def install():
    root = os.environ.get('CLEF_GDN_KKT_CAPTURE')
    if not root:
        return
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    original_import = builtins.__import__

    def patch():
        module = sys.modules.get('fla_npu.ops.ascendc')
        fn = getattr(module, 'chunk_gated_delta_rule_fwd', None)
        if fn is None or getattr(fn, '_clef_kkt_capture', False):
            return

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            q = args[0] if args else kwargs['q']
            target = directory/f'rank{q.device.index}.pt'
            if ((directory/'ARMED').exists() and tuple(q.shape) == (1, 1595, 8, 128)
                    and not target.exists()):
                # Exclusive marker prevents duplicate capture across layers/processes.
                marker = target.with_suffix('.issued')
                try:
                    handle = marker.open('x'); handle.close()
                except FileExistsError:
                    return fn(*args, **kwargs)
                import torch
                payload = {name: (args[i] if len(args) > i else kwargs[name]).detach().cpu()
                           for i, name in enumerate(('q', 'k', 'v', 'g', 'beta'))}
                payload['contract'] = {k: v for k, v in kwargs.items()
                    if isinstance(v, (bool, int, float, str, tuple)) or v is None}
                payload['source'] = 'first production GDN call, actual post-normalization operands'
                torch.save(payload, target)
            return fn(*args, **kwargs)

        wrapped._clef_kkt_capture = True
        module.chunk_gated_delta_rule_fwd = wrapped

    def hooked_import(*args, **kwargs):
        result = original_import(*args, **kwargs)
        if args and args[0].startswith('fla_npu'):
            patch()
        return result

    builtins.__import__ = hooked_import
    patch()
