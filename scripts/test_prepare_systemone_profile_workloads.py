#!/usr/bin/env python3
"""Check workload provenance and single-variable transforms without ML imports."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import prepare_systemone_profile_workloads as prepare
from generate_multimodal_fixtures import prepare_fixtures


def rows_fixture():
    return [{
        "name": f"real-{index}", "source_row_index": index,
        "source": {"revision": "fixed", "record_sha256": f"source-{index}"},
        "source_request_sha256": f"request-{index}",
        "request": {"model": "clef", "state": "A natural source paragraph. " * (index + 1), "questions": {"answer": {"type": "choice", "instructions": "Choose the correct label.", "criteria": {f"key{option}": f"Real candidate {index} {option}" for option in range(4)}}}},
    } for index in range(3)]


def fake_encode(request, full=False):
    # Deliberately differ from character counts so target selection must use
    # the supplied encoder result rather than a text-length estimate.
    count = 20 + len(request["state"]) // 2
    return {"input_tokens": count, "input_ids": list(range(count)), "questions": [{"candidate_span_input_ids": [[10, 11], [12, 13]]}]}


class WorkloadTests(unittest.TestCase):
    def test_dataset_file_hash_and_source_requests_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            document = {"cases": rows_fixture()}
            path.write_text(json.dumps(document))
            rows, source = prepare.read_cases(path, "clef")
            self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(rows[0]["request"], document["cases"][0]["request"])
            self.assertEqual(rows[0]["source_request_sha256"], prepare.sha256(prepare.json_bytes(document["cases"][0]["request"])))
            path.write_text('{"cases": []}')
            with self.assertRaises(ValueError):
                prepare.read_cases(path, "clef")

    def test_natural_selection_uses_encoded_lengths_and_names_confounders(self):
        rows = rows_fixture()
        samples = prepare.select_natural_lengths(rows, [256, 2000, 4032], [256, 2048, 4032], 4096)
        self.assertEqual([sample["metadata"]["selected_full_input_tokens"] for sample in samples], [256, 2000, 4032])
        self.assertEqual(samples[1]["request"], rows[1]["request"])
        self.assertIn("schema", samples[1]["metadata"]["confounders"])
        self.assertFalse(samples[1]["metadata"]["derived_probe"])

    def test_prefix_search_returns_actual_encoding_and_retains_schema(self):
        request = rows_fixture()[2]["request"]
        prefix, metadata = prepare.prefix_for_target(request, 31, fake_encode)
        self.assertEqual(fake_encode(prefix)["input_tokens"], 31)
        self.assertEqual(prefix["questions"], request["questions"])
        self.assertEqual(prefix["state"], request["state"][:metadata["state_prefix_characters"]])
        self.assertTrue(metadata["state_prefix_truncated"])
        self.assertEqual(metadata["actual_full_input_tokens"], 31)
        self.assertEqual(metadata["target_delta_tokens"], 0)

    def test_field_candidate_scans_keep_state_and_candidate_prefix(self):
        rows = rows_fixture()
        original = copy.deepcopy(rows)
        samples = prepare.make_controlled_scans(rows, [40, 60, 80], fake_encode, [30, 40], [1, 2, 4], [2, 4, 8])
        self.assertEqual(rows, original)
        fields = [sample for sample in samples if sample["metadata"]["group"] == "field_count"]
        self.assertEqual([len(sample["request"]["questions"]) for sample in fields], [1, 2, 4])
        self.assertEqual({sample["request"]["state"] for sample in fields}, {rows[0]["request"]["state"]})
        candidates = [sample for sample in samples if sample["metadata"]["group"] == "candidate_count"]
        previous = []
        for sample in candidates:
            request = sample["request"]
            entries = list(request["questions"]["field_00"]["criteria"].items())
            self.assertEqual(entries[:len(previous)], previous)
            self.assertEqual(request["state"], rows[0]["request"]["state"])
            self.assertEqual(len(sample["metadata"]["candidate_sources"]), len(entries))
            self.assertTrue(sample["metadata"]["derived_probe"])
            previous = entries

    def test_existing_mm_hash_validation_and_no_estimated_token_count(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prepare_fixtures(directory)
            samples = prepare.existing_mm_samples(directory / "manifest.json", "clef")
            self.assertEqual(len(samples), 4)
            for sample in samples:
                self.assertIsNone(sample["metadata"]["actual_input_tokens"])
                self.assertFalse(sample["metadata"]["encoding_complete"])
                self.assertIn("source_fixture", sample["metadata"])
            (directory / "red.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepare.existing_mm_samples(directory / "manifest.json", "clef")


if __name__ == "__main__":
    unittest.main()
