"""Frozen 5c supervision and first-cell paths; CPU-only integrity checks."""
import json
import os
from pathlib import Path
from contract import OWNER, N1_CELL, SOURCE_NAMES, check_sha, sha256, verify

WRAPPER = Path('/workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py')
WRAPPER_SHA = '9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5'
REPO = WRAPPER.parent.parent
STOP = Path('/tmp/d0cc/d0_stop.py')
STOP_SHA = 'd6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886'
TREE = Path('/workspace/yangchen/work/src/vllm')
PREIMAGE_SHA = 'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863'
PYTHON = '/workspace/yangchen/venvs/clef-py312/bin/python'
REL = 'runtime/grouped29/'
CELL = {'rows': 967, 'seed': 20261118, 'eps': 1e-6}
PARAMETERS = {'splitItem': 0, 'groupList': None, 'reduceOp': 'sum', 'commTurn': 0,
              'streamMode': 1, 'groups': 1, 'bias': 'FP32 zero [5120]'}


def check_head(tree):
    if Path(tree).resolve() != TREE:
        raise ValueError('Fixed vLLM tree required')
    check_sha(TREE / 'vllm/model_executor/models/clef_joint_schema.py', PREIMAGE_SHA)


def staged_paths(script, run_dir):
    stage, run = Path(script).resolve().parent, Path(run_dir).resolve()
    if not run.is_relative_to(Path('/workspace/yangchen/logs')) or stage != run / 'stage/runtime/grouped29':
        raise ValueError('Requires staged 5c supervised Grouped29 payload')
    return stage, run


def frozen(stage, payload_sha):
    stage = Path(stage)
    check_sha(stage / 'payload.json', payload_sha)
    manifest = json.loads((stage / 'payload.json').read_text())
    if manifest.get('schema') != 'grouped29-window-payload/1':
        raise ValueError('Unknown payload schema')
    if set(manifest.get('files', {})) != set(SOURCE_NAMES) | {'identity.json', 'fixed_gate.py', 'clef_grouped29.so'}:
        raise ValueError('Incomplete frozen payload')
    for name, digest in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe payload member')
        check_sha(stage / name, digest)
    identity = json.loads((stage / 'identity.json').read_text())
    verify(identity, (stage / 'clef_grouped29.so').resolve(), stage / 'fixed_gate.py')
    return identity


def check_window_owner():
    if os.environ.get('CLEF_GROUPED29_EXECUTION_OWNER') != OWNER or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '0,1':
        raise ValueError('Requires owner27, physical card7/logical0,1 supervised window')
