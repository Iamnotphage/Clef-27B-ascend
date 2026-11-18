"""Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0; CPU only, no extension loading."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import tempfile
import sys
from pathlib import Path
import unittest
import torch
from contract import OWNER, check_owner, check_sha, cpu_inputs, record_tensor, sha256


class ContractTests(unittest.TestCase):
    def test_two_ranks_share_inputs_but_select_distinct_real_slices(self):
        full = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
        a = cpu_inputs(torch, full, 3, 20261118, 0)
        b = cpu_inputs(torch, full, 3, 20261118, 1)
        for name in ('x', 'r', 'gamma'):
            self.assertEqual(record_tensor(torch, a[name]), record_tensor(torch, b[name]))
        self.assertTrue(torch.equal(a['w'], full[:, :4].t()))
        self.assertTrue(torch.equal(b['w'], full[:, 4:].t()))
        self.assertNotEqual(record_tensor(torch, a['w'])['sha256'], record_tensor(torch, b['w'])['sha256'])
        # Exact N1 draw order, including conversion only after each random draw.
        gen = torch.Generator().manual_seed(20261118)
        for name, values in (('x', torch.randn(3, 4, generator=gen)),
                             ('r', torch.randn(3, 4, generator=gen)),
                             ('gamma', torch.rand(4, generator=gen))):
            self.assertTrue(torch.equal(a[name], values.to(torch.bfloat16)))

    def test_requires_exclusive_owner_and_exact_rank_mapping(self):
        env = {'CLEF_GROUPED29_EXECUTION_OWNER': OWNER, 'ASCEND_RT_VISIBLE_DEVICES': '0,1',
               'RANK': '1', 'LOCAL_RANK': '1', 'WORLD_SIZE': '2'}
        self.assertEqual(check_owner(OWNER, env), (1, 1))
        for key, value in (('CLEF_GROUPED29_EXECUTION_OWNER', 'other'), ('WORLD_SIZE', '4'),
                           ('LOCAL_RANK', '0'), ('ASCEND_RT_VISIBLE_DEVICES', '2,3')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_owner(OWNER, {**env, key: value})

    def test_identity_tampering_is_rejected(self):
        self.assertNotIn('torch_npu', sys.modules)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'source'
            p.write_bytes(b'frozen source')
            digest = sha256(p)
            check_sha(p, digest)
            p.write_bytes(b'mutated source')
            with self.assertRaises(ValueError):
                check_sha(p, digest)


if __name__ == '__main__':
    unittest.main()
