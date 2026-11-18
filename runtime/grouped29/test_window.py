"""CPU-only tests: supervision wiring and corrupt raw evidence rejection."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys
from pathlib import Path
import tempfile
import unittest
import torch
from aggregate import load_raw
from contract import record_tensor, tensor_bytes
from window_common import staged_paths
from window_probe import command


class WindowTests(unittest.TestCase):
    def test_exact_two_rank_supervised_command(self):
        run = Path('/workspace/yangchen/logs/grouped29-test')
        stage = run / 'stage/runtime/grouped29'
        cmd = command(stage, run, '/fixed/python')
        for flag in ('--run-path', '--start-method=spawn', '--max-restarts=0', '--nproc-per-node=2', '--timing', '--profile'):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--rows') + 1], '967')
        self.assertEqual(cmd[cmd.index('--seed') + 1], '20261118')
        self.assertEqual(cmd[cmd.index('--adapter') + 1], str(stage / 'clef_grouped29.so'))
        self.assertEqual(cmd[cmd.index('--identity') + 1], str(stage / 'identity.json'))
        self.assertEqual(staged_paths(stage / 'window_probe.py', run), (stage, run))
        with self.assertRaises(ValueError):
            staged_paths('/tmp/window_probe.py', run)

    def test_raw_evidence_tampering_and_nonfinite_rejected(self):
        self.assertNotIn('torch_npu', sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
            spec = record_tensor(torch, value)
            path = root / 'y.bin'
            path.write_bytes(tensor_bytes(torch, value))
            self.assertTrue(torch.equal(load_raw(root, 'y', spec, [2, 3], 'torch.bfloat16'), value))
            with self.assertRaises(ValueError):
                load_raw(root, 'y', spec, [3, 2], 'torch.bfloat16')
            raw = path.read_bytes(); path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.assertRaises(ValueError):
                load_raw(root, 'y', spec, [2, 3], 'torch.bfloat16')
            bad = value.clone(); bad[0, 0] = float('nan')
            path.write_bytes(tensor_bytes(torch, bad))
            with self.assertRaises(ValueError):
                load_raw(root, 'y', record_tensor(torch, bad), [2, 3], 'torch.bfloat16')


if __name__ == '__main__':
    unittest.main()
