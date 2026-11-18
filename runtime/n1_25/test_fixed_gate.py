"""CPU regression of the unchanged admission thresholds (requires CPU torch)."""
import unittest
import torch
from fixed_gate import evaluate, gate_output


class GateTests(unittest.TestCase):
    def test_identical(self):
        d = {"y": torch.tensor([0., 1.], dtype=torch.bfloat16),
             "residual": torch.tensor([0., 1.], dtype=torch.bfloat16),
             "rstd": torch.tensor([1.])}
        self.assertTrue(evaluate(d, d)["pass"])

    def test_residual_mixed_all_elements(self):
        ref = torch.tensor([0., 1., -1.])
        self.assertTrue(gate_output("res", ref + 0.000001, ref, "mixed")["pass"])
        bad = ref.clone(); bad[0] = 0.00002
        self.assertFalse(gate_output("res", bad, ref, "mixed")["pass"])

    def test_y_nearzero_still_abs_and_finite(self):
        self.assertFalse(gate_output("y", torch.tensor([0.021]), torch.tensor([0.]), "y")["pass"])
        self.assertFalse(gate_output("y", torch.tensor([float("nan")]), torch.tensor([0.]), "y")["pass"])
        self.assertFalse(gate_output("y", torch.tensor([0.]), torch.tensor([float("nan")]), "y")["pass"])

    def test_y_relative_threshold_domain(self):
        self.assertFalse(gate_output("y", torch.tensor([1.011]), torch.tensor([1.]), "y")["pass"])
        self.assertFalse(gate_output("y", torch.tensor([0.00002]), torch.tensor([0.00001]), "y")["pass"])

    def test_shape_dtype_and_rstd(self):
        self.assertFalse(gate_output("y", torch.zeros(1,1), torch.zeros(1), "y")["pass"])
        self.assertFalse(gate_output("y", torch.zeros(1).bfloat16(), torch.zeros(1), "y")["pass"])
        self.assertFalse(gate_output("rstd", torch.tensor([1.002]), torch.tensor([1.]), "abs")["pass"])


if __name__ == "__main__":
    unittest.main()
