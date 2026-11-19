"""CPU-only tests for the candidate manifest whitelist transform."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

import candidate_manifest as cm

BASE = {
    'source_files': [
        {'path': '/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py',
         'sha256': cm.F810_HEAD},
        {'path': '/workspace/yangchen/clef/start.sh', 'sha256': 'a' * 64},
        {'path': '/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py',
         'sha256': cm.GUARD_PRE},
    ],
    'source_selection': {'production_head_sha': cm.F810_HEAD, 'guard_sha': cm.GUARD_PRE,
                         'candidate_head_sha': cm.AA3E_HEAD, 'winner': 'pending'},
    'forbidden_environment': [cm.FLAG_ENV, 'CLEF_PROFILE_MARKERS'],
    'phases': {p: {'service_environment': {'CLEF_MODEL_NAME': 'clef', 'CLEF_CONFIG': 's1'}}
               for p in cm.PHASES},
    'baseline_metrics': {'capability': 74.42547887630508},
    'quality_gate': {'od_absolute_points': 2.0},
}


class DeriveTest(unittest.TestCase):
    def test_derive_binds_candidate_exactly(self):
        candidate, expected, indexes = cm.derive(BASE)
        self.assertEqual(indexes, {'head_index': 0, 'guard_index': 2})
        self.assertEqual(sorted(cm.leaf_diff(BASE, candidate)), expected)
        self.assertEqual(len(expected), 13)
        self.assertEqual(candidate['source_files'][0]['sha256'], cm.AA3E_HEAD)
        self.assertEqual(candidate['source_files'][2]['sha256'], cm.GUARD_POST)
        self.assertEqual(candidate['source_selection']['production_head_sha'], cm.AA3E_HEAD)
        self.assertEqual(candidate['source_selection']['guard_sha'], cm.GUARD_POST)
        self.assertNotIn(cm.FLAG_ENV, candidate['forbidden_environment'])
        self.assertIn('CLEF_PROFILE_MARKERS', candidate['forbidden_environment'])
        for phase in cm.PHASES:
            env = candidate['phases'][phase]['service_environment']
            self.assertEqual(env[cm.FLAG_ENV], '1')
            self.assertEqual(env[cm.ALLOW_ENV], cm.AA3E_HEAD)
        cm.verify(BASE, candidate)

    def test_derive_is_deterministic(self):
        first, _, _ = cm.derive(copy.deepcopy(BASE))
        second, _, _ = cm.derive(copy.deepcopy(BASE))
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        digest = hashlib.sha256((json.dumps(first, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
        digest2 = hashlib.sha256((json.dumps(second, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
        self.assertEqual(digest, digest2)

    def test_drift_outside_whitelist_rejected(self):
        candidate, _, _ = cm.derive(BASE)
        candidate['baseline_metrics']['capability'] = 0.0
        with self.assertRaises(RuntimeError):
            cm.verify(BASE, candidate)

    def test_extra_phase_env_rejected(self):
        candidate, _, _ = cm.derive(BASE)
        candidate['phases']['s1']['service_environment']['CLEF_JOINT_SMALLQ'] = '1'
        with self.assertRaises(RuntimeError):
            cm.verify(BASE, candidate)

    def test_base_must_be_f810(self):
        base = copy.deepcopy(BASE)
        base['source_files'][0]['sha256'] = cm.AA3E_HEAD
        with self.assertRaises(RuntimeError):
            cm.derive(base)

    def test_base_must_be_preimage_guard(self):
        base = copy.deepcopy(BASE)
        base['source_files'][2]['sha256'] = cm.GUARD_POST
        with self.assertRaises(RuntimeError):
            cm.derive(base)

    def test_base_forbidden_requires_flag(self):
        base = copy.deepcopy(BASE)
        base['forbidden_environment'] = ['CLEF_PROFILE_MARKERS']
        with self.assertRaises(RuntimeError):
            cm.derive(base)


@unittest.skipUnless(os.environ.get('CLEF_BASE_MANIFEST'), 'real frozen manifest not provided')
class FrozenManifestTest(unittest.TestCase):
    def test_frozen_base_sha_and_candidate_sha(self):
        path = Path(os.environ['CLEF_BASE_MANIFEST'])
        self.assertEqual(cm.sha(path), cm.BASE_SHA)
        base = json.loads(path.read_text())
        candidate, expected, _ = cm.derive(base)
        cm.verify(base, candidate)
        digest = hashlib.sha256((json.dumps(candidate, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
        self.assertEqual(digest, '31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4')
        self.assertEqual(len(expected), 13)


if __name__ == '__main__':
    unittest.main()
