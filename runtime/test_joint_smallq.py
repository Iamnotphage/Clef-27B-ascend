#!/usr/bin/env python3
"""CPU math, default-off/domain fallback and installer tests; no device claims."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent


class InstallerTest(unittest.TestCase):
    def test_apply_check_revert_and_unknown_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "vllm/model_executor/models"
            target.mkdir(parents=True)
            head = target / "clef_joint_schema.py"
            preimage = (HERE / "joint_single_field/clef_joint_schema.py").read_bytes()
            head.write_bytes(preimage)
            command = [sys.executable, str(HERE / "apply_joint_smallq.py"),
                       "--tree", str(root), "--backup-dir", str(root / "backup")]

            def invoke(action, expected=0):
                result = subprocess.run(command + [action], capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return json.loads(result.stdout)

            self.assertFalse(invoke("--check")["ready"])
            self.assertTrue(invoke("--apply")["ready"])
            self.assertTrue(invoke("--check")["ready"])
            invoke("--apply")
            self.assertTrue((target / "clef_smallq_kernel.py").exists())
            invoke("--revert")
            self.assertEqual(head.read_bytes(), preimage)
            self.assertFalse((target / "clef_smallq_kernel.py").exists())
            invoke("--revert")
            head.write_bytes(preimage + b"# unknown\n")
            invoke("--apply", 1)
            self.assertEqual(head.read_bytes(), preimage + b"# unknown\n")


try:
    import torch
except ImportError:
    torch = None


def load_head(path, name):
    # Execute the actual head definitions. The unused pooler integration imports
    # are omitted so this CPU test never imports vLLM, torch_npu or Triton.
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith("vllm"):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "parse_joint_schema":
            break
        nodes.append(node)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@unittest.skipIf(torch is None, "CPU torch unavailable")
class HeadTest(unittest.TestCase):
    def test_default_off_and_cpu_domain_fallback_match_actual_preimage(self):
        original = load_head(HERE / "joint_single_field/clef_joint_schema.py", "smallq_cpu_original")
        candidate = load_head(HERE / "joint_smallq/clef_joint_schema.py", "smallq_cpu_candidate")
        torch.manual_seed(17)
        a = original.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        b = candidate.JointSchemaHead(32, 16, 2, 4, 4, 32).eval()
        b.load_state_dict(a.state_dict())
        hidden, lexical = torch.randn(40, 32), torch.randn(4, 32)
        os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
        with torch.inference_mode():
            for questions in (1, 2):
                spans = questions + 4
                schema = original.SchemaIndices(
                    torch.arange(spans * 2), torch.arange(spans).repeat_interleave(2),
                    torch.full((spans,), 2), torch.ones(questions, dtype=torch.long),
                    torch.arange(questions).repeat_interleave(4 // questions),
                    torch.arange(4), torch.ones(questions, 4 // questions, dtype=torch.bool))
                reference = a(hidden, lexical, schema)
                for flag in ("0", "1"):
                    os.environ["CLEF_JOINT_SMALLQ"] = flag
                    self.assertTrue(torch.equal(reference, b(hidden, lexical, schema)))
        self.assertNotIn("triton", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)
        os.environ["CLEF_JOINT_SMALLQ"] = "0"
        self.assertIsNone(candidate._smallq_cross_attention(None, None, None, None, enabled=True))
        os.environ["CLEF_JOINT_SMALLQ"] = "1"
        self.assertIsNone(candidate._smallq_cross_attention(None, None, None, None, enabled=False))

    def test_online_reduction_math_and_tail_boundaries(self):
        torch.manual_seed(20261119)
        for length in (1, 63, 64, 65, 307, 967, 1595, 2048):
            for queries in (1, 4):
                q = torch.randn(queries, 64).to(torch.bfloat16).float()
                k = torch.randn(length, 64).to(torch.bfloat16).float()
                v = torch.randn(length, 64).to(torch.bfloat16).float()
                for amplitude in (1.0, 20.0):
                    scores = (q * amplitude) @ k.T * 0.125
                    expected = scores.softmax(-1) @ v
                    maximum = torch.full((queries,), -float("inf"))
                    denominator = torch.zeros(queries)
                    accumulator = torch.zeros(queries, 64)
                    for offset in range(0, length, 64):
                        s = scores[:, offset:offset + 64]
                        updated = torch.maximum(maximum, s.max(-1).values)
                        correction = (maximum - updated).exp()
                        p = (s - updated[:, None]).exp()
                        accumulator = accumulator * correction[:, None] + p @ v[offset:offset + 64]
                        denominator = denominator * correction + p.sum(-1)
                        maximum = updated
                    torch.testing.assert_close(accumulator / denominator[:, None], expected,
                                               rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
