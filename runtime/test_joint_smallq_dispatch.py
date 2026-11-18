#!/usr/bin/env python3
"""CPU control-flow equivalence, lifecycle and binding checks; no NPU claims."""
import ast
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# This test is CPU-only even on the fixed environment with torch_npu installed.
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

from joint_smallq_bd_probe import ArmController, ORDERS, cpu_cases, tensor_hash, verify_single_variable
from test_joint_smallq import load_head

HERE = Path(__file__).parent
B = HERE / 'joint_smallq/clef_joint_schema.py'
D = HERE / 'joint_smallq_dispatch/clef_joint_schema.py'


@dataclass(frozen=True)
class Device:
    type: str = 'npu'
    index: int = 0


class Metadata:
    """Descriptor stand-in for guard decisions only; never runs tensor math."""
    def __init__(self, shape=(1, 4, 1024), dtype='bf16', device=Device(), strides=(4096, 1024, 1)):
        self.values = dict(shape=shape, dtype=dtype, device=device, strides=strides)
        self.reads = Counter()

    def read(self, name):
        self.reads[name] += 1
        return self.values[name]

    shape = property(lambda self: self.read('shape'))
    ndim = property(lambda self: len(self.read('shape')))
    dtype = property(lambda self: self.read('dtype'))
    device = property(lambda self: self.read('device'))

    def stride(self, dim):
        self.reads['stride'] += 1
        return self.values['strides'][dim]


def guards(path):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_smallq_cross_attention')
    cut = next(i for i, n in enumerate(node.body)
               if isinstance(n, ast.If) and ast.unparse(n.test) == 'key is value')
    node.body = node.body[:cut] + [ast.Return(value=ast.Constant(value=True))]
    namespace = {'torch': SimpleNamespace(bfloat16='bf16', is_grad_enabled=lambda: False)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['_smallq_cross_attention'], namespace['torch']


def fixture(shared=False):
    q, k = Metadata(), Metadata((1, 307, 1024))
    v = k if shared else Metadata((1, 307, 1024))
    m = SimpleNamespace(training=False, dropout=0, batch_first=True, _qkv_same_embed_dim=True,
                        in_proj_weight=Metadata(), in_proj_bias=Metadata(), bias_k=None,
                        bias_v=None, add_zero_attn=False, embed_dim=1024, num_heads=16,
                        out_proj=SimpleNamespace(weight=Metadata(), bias=Metadata()))
    return m, q, k, v


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.b, self.bt = guards(B)
        self.d, self.dt = guards(D)
        self.env = patch.dict(os.environ, {'CLEF_JOINT_SMALLQ': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def same(self, args, accepted, enabled=True):
        self.assertEqual(self.b(*args, enabled=enabled), True if accepted else None)
        self.assertEqual(self.d(*args, enabled=enabled), True if accepted else None)

    def test_boundaries_and_same_object(self):
        for shared in (False, True):
            for qlen in (0, 1, 4, 8, 9):
                for length in (0, 1, 63, 64, 65, 307, 967, 1595, 2048, 2049):
                    args = fixture(shared)
                    args[1].values['shape'] = (1, qlen, 1024)
                    args[2].values['shape'] = args[3].values['shape'] = (1, length, 1024)
                    with self.subTest(shared=shared, Q=qlen, L=length):
                        self.same(args, 1 <= qlen <= 8 and 1 <= length <= 2048)

    def test_each_tensor_metadata_rejection_and_live_recovery(self):
        mutations = [('shape', ()), ('shape', (4, 1024)), ('shape', (1, 4, 1024, 1)),
                     ('shape', (2, 4, 1024)), ('shape', (1, 4, 1023)),
                     ('dtype', 'fp16'), ('device', Device('cpu')), ('device', Device('npu', 1)),
                     ('strides', (4096, 1024, 2)), ('strides', (4096, 1023, 1))]
        for tensor_index in (1, 2, 3):
            for name, value in mutations:
                args = fixture()
                self.same(args, True)
                tensor = args[tensor_index]
                before = tensor.values[name]
                tensor.values[name] = value
                with self.subTest(tensor=tensor_index, name=name, value=value):
                    self.same(args, False)
                    tensor.values[name] = before
                    self.same(args, True)
        args = fixture()
        args[3].values['shape'] = (1, 306, 1024)
        self.same(args, False)

    def test_each_module_and_parameter_guard_stays_dynamic(self):
        mutations = [('training', True), ('dropout', .1), ('batch_first', False),
                     ('_qkv_same_embed_dim', False), ('in_proj_weight', None),
                     ('bias_k', object()), ('bias_v', object()), ('add_zero_attn', True),
                     ('embed_dim', 512), ('num_heads', 8)]
        args = fixture()
        for name, value in mutations:
            before = getattr(args[0], name)
            setattr(args[0], name, value)
            self.same(args, False)
            setattr(args[0], name, before)
            self.same(args, True)
        for parameter in (args[0].in_proj_weight, args[0].in_proj_bias,
                          args[0].out_proj.weight, args[0].out_proj.bias):
            for key, value in (('dtype', 'fp32'), ('device', Device('npu', 1))):
                before = parameter.values[key]
                parameter.values[key] = value
                self.same(args, False)
                parameter.values[key] = before
                self.same(args, True)
        args[0].in_proj_bias = args[0].out_proj.bias = None
        self.same(args, True)
        self.bt.is_grad_enabled = self.dt.is_grad_enabled = lambda: True
        self.same(args, False)
        self.bt.is_grad_enabled = self.dt.is_grad_enabled = lambda: False
        self.same(args, True)

    def test_default_off_and_disabled_never_read_objects(self):
        self.same((None,) * 4, False, enabled=False)
        for value in ('0', '', 'true'):
            os.environ['CLEF_JOINT_SMALLQ'] = value
            self.same((None,) * 4, False)
        os.environ.pop('CLEF_JOINT_SMALLQ')
        self.same((None,) * 4, False)
        os.environ['CLEF_JOINT_SMALLQ'] = '1'
        self.same(fixture(), True)

    def test_metadata_reads_deduplicated_without_cache(self):
        b_args, d_args = fixture(True), fixture(True)
        self.assertTrue(self.b(*b_args, enabled=True))
        self.assertTrue(self.d(*d_args, enabled=True))
        self.assertEqual(d_args[2].reads['stride'], 2)
        self.assertEqual(b_args[2].reads['stride'], 4)
        self.assertEqual(d_args[1].reads['device'], 1)
        self.assertGreater(b_args[1].reads['device'], 1)
        previous = d_args[2].reads.copy()
        self.assertTrue(self.d(*d_args, enabled=True))
        self.assertEqual(d_args[2].reads, Counter({k: v * 2 for k, v in previous.items()}))

    def test_only_guard_changes_and_orders_balanced(self):
        self.assertTrue(verify_single_variable(B, D))
        self.assertEqual(sum(order[0] == 'B' for order in ORDERS), 2)
        self.assertEqual((HERE / 'joint_smallq/clef_smallq_kernel.py').read_bytes(),
                         (HERE / 'joint_smallq_dispatch/clef_smallq_kernel.py').read_bytes())


class InstallerTest(unittest.TestCase):
    def test_roundtrip_and_mixed_B_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'vllm/model_executor/models'
            target.mkdir(parents=True)
            head = target / 'clef_joint_schema.py'
            preimage = (HERE / 'joint_single_field/clef_joint_schema.py').read_bytes()
            head.write_bytes(preimage)
            command = [sys.executable, str(HERE / 'apply_joint_smallq_dispatch.py'),
                       '--tree', str(root), '--backup-dir', str(root / 'backup')]
            def run(action, rc=0):
                p = subprocess.run(command + [action], capture_output=True, text=True)
                self.assertEqual(p.returncode, rc, p.stdout + p.stderr)
                return json.loads(p.stdout)
            self.assertFalse(run('--check')['ready'])
            self.assertTrue(run('--apply')['ready'])
            self.assertEqual(head.read_bytes(), D.read_bytes())
            self.assertTrue(run('--check')['ready'])
            run('--revert')
            self.assertEqual(head.read_bytes(), preimage)
            self.assertFalse((target / 'clef_smallq_kernel.py').exists())
            head.write_bytes(B.read_bytes())
            run('--apply', 1)
            self.assertEqual(head.read_bytes(), B.read_bytes())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch unavailable')
class BindingTest(unittest.TestCase):
    def test_same_bf16_input_hashes(self):
        expected = json.loads((HERE / 'joint_smallq_acb/inputs.json').read_text())['inputs']
        actual = {}
        for label, tensors in cpu_cases(torch):
            if label['scope'] == 'fallback':
                continue
            key = f"{label['scope']}/{label['length']}/{label.get('queries', 4)}"
            actual[key] = {name + '_sha256': tensor_hash(torch, tensor) for name, tensor in tensors.items()}
        self.assertEqual(actual, expected)

    def test_full_head_fallback_storage_and_exception_restore(self):
        original = load_head(HERE / 'joint_single_field/clef_joint_schema.py', 'bd_cpu_a')
        b, d = load_head(B, 'bd_cpu_b'), load_head(D, 'bd_cpu_d')
        head = d.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        ptrs = [p.data_ptr() for p in head.parameters()]
        hidden, lexical = torch.randn(40, 32), torch.randn(4, 32)
        backend = object()
        kernel = SimpleNamespace(smallq_attention=backend)
        with patch.dict(os.environ, {'CLEF_JOINT_SINGLE_FIELD': '1', 'CLEF_JOINT_SMALLQ': 'sentinel'}):
            with torch.inference_mode(), ArmController(head, original, b, d, kernel) as controller:
                for questions in (1, 2):
                    spans = questions + 4
                    schema = original.SchemaIndices(torch.arange(spans * 2), torch.arange(spans).repeat_interleave(2),
                        torch.full((spans,), 2), torch.ones(questions, dtype=torch.long),
                        torch.arange(questions).repeat_interleave(4 // questions), torch.arange(4),
                        torch.ones(questions, 4 // questions, dtype=torch.bool))
                    outputs = []
                    for arm, mod in (('A', original), ('B', b), ('D', d), ('B', b)):
                        controller.select(arm)
                        self.assertIs(head.forward.__func__, mod.JointSchemaHead.forward)
                        self.assertIs(head._single_field_decoder.__func__, mod.JointSchemaHead._single_field_decoder)
                        self.assertTrue(all(x.forward.__func__ is mod.EvidenceRoutingLayer.forward for x in head.evidence_layers))
                        outputs.append(head(hidden, lexical, schema))
                        self.assertEqual(ptrs, [p.data_ptr() for p in head.parameters()])
                        self.assertIs(kernel.smallq_attention, backend)
                    self.assertTrue(all(torch.equal(outputs[0], x) for x in outputs))
            self.assertNotIn('forward', head.__dict__)
            self.assertEqual(os.environ['CLEF_JOINT_SMALLQ'], 'sentinel')
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                with ArmController(head, original, b, d, kernel) as controller:
                    controller.select('D')
                    raise RuntimeError('deliberate')
            self.assertNotIn('forward', head.__dict__)
            self.assertIs(kernel.smallq_attention, backend)
            self.assertEqual(os.environ['CLEF_JOINT_SMALLQ'], 'sentinel')
        self.assertNotIn('triton', sys.modules)
        self.assertNotIn('torch_npu', sys.modules)


if __name__ == '__main__':
    unittest.main()
