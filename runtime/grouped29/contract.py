"""CPU-only identity and reproducible-input helpers for the owner27 probe."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

OWNER = 'clef-27b-ascend-27'
GATE_SHA = '041a66aa81c52d2a53bd825cf406323d7185759242669dff64b3551f86eba849'
SDK = Path('/usr/local/Ascend/cann-9.2.1/aarch64-linux')
PINNED = {
    str(SDK / 'lib64/libopapi.so'): 'eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456',
    str(SDK / 'lib64/libnnopbase.so'): '72cd43b4af1c9c451b566ccb7b32071793402949578f838d8345e369b056477c',
    str(SDK / 'include/aclnnop/aclnn_grouped_mat_mul_all_reduce.h'): 'c015a1eb0e28e8137240f34ad2695e77ef720a00509ba495020dbbe6ced91339',
}
SOURCE_NAMES = ('binding.cpp', 'compile.sh', 'contract.py', 'run_probe.py',
                'timing_contract.py', 'aggregate.py', 'window_common.py',
                'window_hook.py', 'window_probe.py', 'supervised_window.py')
N1_CELL = Path('/workspace/yangchen/logs/n1-v2-aron-c24-timing-owner27-20261119T0310/probe/cell')
N1_GROUP = 'group_name_0'
N1_META_SHAS = ('5b2261c6fe4fa2e0b5e948ee341186137ec07c312da493b7cb1e2ebeee0aeec6',
                'a14f9649e63f6d4f2d4b45dd650c905b77c6591443823cb1100ef590f2a8e499')


def n1_inputs(rank, cell=N1_CELL):
    if rank not in (0, 1) or Path(cell).resolve() != N1_CELL:
        raise ValueError('First cell requires frozen owner27 N1 c24 path')
    path = N1_CELL / f'rank{rank}/meta.json'
    check_sha(path, N1_META_SHAS[rank])
    meta = json.loads(path.read_text())
    if meta.get('rows') != 967 or meta.get('seed') != 20261118:
        raise ValueError('N1 source cell parameters mismatch')
    return {name: meta['inputs'][name] for name in ('x', 'w', 'r', 'gamma')}

WEIGHT_KEY = 'model.language_model.layers.0.mlp.down_proj.weight'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_sha(path, expected):
    got = sha256(path)
    if got != expected:
        raise ValueError(f'SHA mismatch: {path}: {got} != {expected}')


def verify(identity, adapter, gate):
    if (identity.get('schema') != 'grouped29-aot/2' or identity.get('pinned') != PINNED
            or identity.get('gate_sha256') != GATE_SHA):
        raise ValueError('Unknown identity or system-library pins')
    for path, digest in PINNED.items():
        check_sha(path, digest)
    if set(identity['sources']) != set(SOURCE_NAMES):
        raise ValueError('Source identity incomplete')
    for name, digest in identity['sources'].items():
        check_sha(Path(__file__).with_name(name), digest)
    if not Path(adapter).is_absolute() or Path(adapter).name != 'clef_grouped29.so':
        raise ValueError('Absolute independently named adapter required')
    if identity.get('n1_cell') != {'path': str(N1_CELL), 'meta_shas': list(N1_META_SHAS)}:
        raise ValueError('N1 cell identity mismatch')
    for rank in (0, 1):
        n1_inputs(rank)
    check_sha(adapter, identity['adapter']['sha256'])
    check_sha(gate, GATE_SHA)


def check_owner(owner, environ=None):
    env = os.environ if environ is None else environ
    if owner != OWNER or env.get('CLEF_GROUPED29_EXECUTION_OWNER') != OWNER:
        raise ValueError('Exclusive owner27 window required')
    rank, local, world = (int(env[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 2 or rank not in (0, 1) or local != rank or env.get('ASCEND_RT_VISIBLE_DEVICES') != '0,1':
        raise ValueError('Requires TP2 rank/local 0,1 on logical devices 0,1')
    return rank, local


def load_gate(path):
    check_sha(path, GATE_SHA)
    spec = importlib.util.spec_from_file_location('grouped29_frozen_gate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def cpu_inputs(torch, full_weight, rows, seed, rank):
    if rank not in (0, 1) or full_weight.ndim != 2 or full_weight.shape[1] % 2:
        raise ValueError('Invalid TP2 weight/rank')
    width = full_weight.shape[1] // 2
    weight = full_weight[:, rank * width:(rank + 1) * width].t().contiguous()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    # Keep exact N1 ordering: x, residual, gamma; rank only selects the weight.
    x = torch.randn(rows, width, generator=gen).to(torch.bfloat16)
    residual = torch.randn(rows, full_weight.shape[0], generator=gen).to(torch.bfloat16)
    gamma = torch.rand(full_weight.shape[0], generator=gen).to(torch.bfloat16)
    return {'x': x, 'w': weight, 'r': residual, 'gamma': gamma}


def tensor_bytes(torch, tensor):
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def record_tensor(torch, tensor):
    raw = tensor_bytes(torch, tensor)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for path, digest in PINNED.items():
        check_sha(path, digest)
    check_sha(a.gate, GATE_SHA)
    identity = {'schema': 'grouped29-aot/2', 'pinned': PINNED,
                'n1_cell': {'path': str(N1_CELL), 'meta_shas': list(N1_META_SHAS)},
                'adapter': {'build_path': str(a.adapter.resolve()), 'sha256': sha256(a.adapter)},
                'gate_sha256': GATE_SHA,
                'sources': {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_NAMES}}
    with a.output.open('x') as f:
        f.write(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
