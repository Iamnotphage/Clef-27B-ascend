#!/usr/bin/env python3
"""CPU-only controller/source/input checks, with no fake device timings."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from joint_smallq_acb_probe import ArmController, ORDERS, cpu_cases, tensor_hash
from test_joint_smallq import load_head

HERE = Path(__file__).resolve().parent
try:
    import torch
except ImportError:
    torch = None


class OrderingTest(unittest.TestCase):
    def test_each_pair_has_both_orders_twice(self):
        for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
            self.assertEqual(sum(order.index(a) < order.index(b) for order in ORDERS), 2)
        for arm in "ABC":
            self.assertEqual(sum(order.index(arm) for order in ORDERS), 4)


@unittest.skipIf(torch is None, "CPU torch unavailable")
class ControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_head(HERE / "joint_single_field/clef_joint_schema.py", "acb_cpu_f810")
        cls.candidate = load_head(HERE / "joint_smallq/clef_joint_schema.py", "acb_cpu_2e8")

    def test_shared_storage_exact_method_binding_fallback_and_restore(self):
        torch.manual_seed(31)
        head = self.candidate.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        reference = self.original.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        reference.load_state_dict(head.state_dict())
        ptrs = [p.data_ptr() for p in head.parameters()]
        hidden, lexical = torch.randn(40, 32), torch.randn(4, 32)

        def frozen_backend(*_):
            raise AssertionError("CPU domain must not reach device backend")

        def sdpa_backend(*_):
            raise AssertionError("CPU domain must not reach replacement backend")

        kernel = SimpleNamespace(smallq_attention=frozen_backend)
        with patch.dict(os.environ, {"CLEF_JOINT_SINGLE_FIELD": "1", "CLEF_JOINT_SMALLQ": "sentinel"}):
            with torch.inference_mode(), ArmController(head, self.original, self.candidate, kernel, sdpa_backend) as controller:
                for questions in (1, 2):
                    spans = questions + 4
                    schema = self.original.SchemaIndices(
                        torch.arange(spans * 2), torch.arange(spans).repeat_interleave(2),
                        torch.full((spans,), 2), torch.ones(questions, dtype=torch.long),
                        torch.arange(questions).repeat_interleave(4 // questions), torch.arange(4),
                        torch.ones(questions, 4 // questions, dtype=torch.bool))
                    expected = reference(hidden, lexical, schema)
                    for arm in ("A", "C", "B", "C", "A"):
                        controller.select(arm)
                        module = self.original if arm == "A" else self.candidate
                        self.assertIs(head.forward.__func__, module.JointSchemaHead.forward)
                        self.assertIs(head._single_field_decoder.__func__, module.JointSchemaHead._single_field_decoder)
                        self.assertTrue(all(layer.forward.__func__ is module.EvidenceRoutingLayer.forward for layer in head.evidence_layers))
                        self.assertIs(kernel.smallq_attention, sdpa_backend if arm == "C" else frozen_backend)
                        self.assertTrue(torch.equal(expected, head(hidden, lexical, schema)))
                        self.assertEqual(ptrs, [p.data_ptr() for p in head.parameters()])
            self.assertNotIn("forward", head.__dict__)
            self.assertNotIn("_single_field_decoder", head.__dict__)
            self.assertTrue(all("forward" not in layer.__dict__ for layer in head.evidence_layers))
            self.assertIs(kernel.smallq_attention, frozen_backend)
            self.assertEqual(os.environ["CLEF_JOINT_SMALLQ"], "sentinel")

    def test_exception_restores_backend_methods_and_environment(self):
        head = self.candidate.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        saved = object()
        kernel = SimpleNamespace(smallq_attention=saved)
        with patch.dict(os.environ, {"CLEF_JOINT_SMALLQ": "0"}):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with ArmController(head, self.original, self.candidate, kernel, lambda *_: None) as controller:
                    controller.select("C")
                    raise RuntimeError("deliberate")
            self.assertIs(kernel.smallq_attention, saved)
            self.assertNotIn("forward", head.__dict__)
            self.assertEqual(os.environ["CLEF_JOINT_SMALLQ"], "0")

    def test_cpu_bf16_inputs_match_previous_real_probe_hashes(self):
        fixture = json.loads((HERE / "joint_smallq_acb/inputs.json").read_text())
        seen = set()
        for label, tensors in cpu_cases(torch):
            if label["scope"] == "fallback":
                continue
            key = f"{label['scope']}/{label['length']}/{label.get('queries', 4)}"
            hashes = {name + "_sha256": tensor_hash(torch, tensor) for name, tensor in tensors.items()}
            self.assertEqual(hashes, fixture["inputs"][key])
            seen.add(key)
        self.assertEqual(seen, set(fixture["inputs"]))
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("triton", sys.modules)


if __name__ == "__main__":
    unittest.main()
