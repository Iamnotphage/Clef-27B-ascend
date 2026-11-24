#!/usr/bin/env python3
"""Offline tests for the official ERQA state-list wire format.

Run with:
  python3 scripts/official_eval/test_erqa_state_list.py
  # or
  python3 -m unittest scripts.official_eval.test_erqa_state_list

The full 400-sample audit runs automatically when the pinned extracted
snapshot is present and is skipped (with a recorded reason) otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import erqa_baseline  # noqa: E402
import erqa_protocol  # noqa: E402


class ERQAProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)

    def sample(self, question, indices, count):
        images = []
        for index in range(count):
            blob = f"image-{index}".encode()
            path = f"images/{index}.jpg"
            (self.data / path).parent.mkdir(parents=True, exist_ok=True)
            (self.data / path).write_bytes(blob)
            images.append({
                "path": path,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "format": "jpeg",
            })
        return {
            "id": "erqa-test", "record_index": 0, "question": question,
            "answer": "A", "question_type": "Other", "image_count": count,
            "images": images, "visual_indices": indices,
            "visual_index_pattern": (
                "empty" if not indices
                else "all-zero" if all(value == 0 for value in indices)
                else "interleaved"),
        }

    def ordered_urls(self, sample):
        request = erqa_protocol.build_request(sample, self.data)
        return request["images"]

    def raw_segments(self, sample):
        order = erqa_protocol.image_order(sample)
        urls = [
            erqa_protocol.data_url(
                (self.data / sample["images"][index]["path"]).read_bytes(),
                sample["images"][index]["format"])
            for index in order
        ]
        return urls, erqa_protocol.content_segments(sample, urls)

    def assert_raw_segments(self, sample):
        urls, segments = self.raw_segments(sample)
        self.assertEqual("".join(
            value for kind, value in segments if kind == "text"),
            sample["question"])
        self.assertEqual(
            [value for kind, value in segments if kind == "image"], urls)
        return segments

    def assert_segments(self, sample):
        request = erqa_protocol.build_request(
            sample, self.data, wire_format="state-list")
        state = request["state"]
        self.assertNotIn("images", request)
        self.assertEqual("".join(
            part["text"] for part in state if part["type"] == "text"),
            sample["question"])
        self.assertEqual(
            [part["image_url"]["url"] for part in state
             if part["type"] == "image_url"],
            self.ordered_urls(sample))
        return state

    def test_empty_indices_put_images_first(self):
        sample = self.sample("Which option? A. one B. two", [], 3)
        state = self.assert_segments(sample)
        self.assertEqual([part["type"] for part in state],
                         ["image_url", "image_url", "image_url", "text"])
        self.assertEqual(state[-1]["text"], sample["question"])

    def test_all_zero_indices_put_images_first_in_original_order(self):
        sample = self.sample("Which option? A. one B. two", [0, 0, 0], 3)
        state = self.assert_segments(sample)
        self.assertEqual([part["type"] for part in state],
                         ["image_url", "image_url", "image_url", "text"])

    def test_interleaved_slices_match_the_official_harness(self):
        question = "0123456789abcdefghij0123456789"
        sample = self.sample(question, [20, 0, 10], 3)
        segments = self.assert_raw_segments(sample)
        self.assertEqual([kind for kind, _ in segments], [
            "image", "text", "image", "text", "image", "text"])
        texts = [value for kind, value in segments if kind == "text"]
        self.assertEqual(texts, [question[:10], question[10:20], question[20:]])

    def test_index_beyond_question_length_appends_the_image(self):
        question = "short question A. x B. y"
        sample = self.sample(question, [0, len(question) + 5], 2)
        state = self.assert_segments(sample)
        self.assertEqual(state[0]["type"], "image_url")
        self.assertEqual(state[-1], {"type": "text", "text": question})

    def test_repeated_question_fuzz_always_joins_back(self):
        rng = random.Random(20261124)
        for _ in range(200):
            length = rng.randint(5, 300)
            question = "".join(rng.choice("abcdef ") for _ in range(length))
            count = rng.randint(1, 6)
            indices = [rng.choice([0, 0, 1, length // 2, length, length + 3])
                       for _ in range(count)]
            if rng.random() < 0.3:
                indices = [0] * count
            if rng.random() < 0.2:
                indices = []
                count = rng.randint(1, 4)
            sample = self.sample(question, indices, count)
            with self.subTest(question=question, indices=indices):
                self.assert_raw_segments(sample)

    def test_top_level_images_keeps_the_historical_body(self):
        sample = self.sample("Which option? A. one B. two", [0], 1)
        request = erqa_protocol.build_request(sample, self.data)
        self.assertEqual(set(request), {"model", "state", "questions", "images"})
        self.assertEqual(request["state"], {"question": sample["question"]})
        self.assertEqual(len(request["images"]), 1)
        historical = {
            "model": request["model"],
            "state": {"question": sample["question"]},
            "questions": request["questions"],
            "images": request["images"],
        }
        self.assertEqual(
            erqa_protocol.wire_body(request),
            erqa_protocol.wire_body(historical))
        state_list = erqa_protocol.build_request(
            sample, self.data, wire_format="state-list")
        self.assertNotEqual(
            erqa_protocol.request_summary(request, sample),
            erqa_protocol.request_summary(state_list, sample))

    def test_structural_hash_round_trips_for_both_wire_formats(self):
        sample = self.sample("0123456789abcdef A. x B. y", [10, 0], 2)
        for wire_format in erqa_protocol.WIRE_FORMATS:
            with self.subTest(wire_format=wire_format):
                first = erqa_protocol.build_request(
                    sample, self.data, wire_format=wire_format)
                second = erqa_protocol.build_request(
                    sample, self.data, wire_format=wire_format)
                self.assertEqual(
                    erqa_protocol.structural_sha256(first, sample),
                    erqa_protocol.structural_sha256(second, sample))

    def test_full_erqa_audit_when_available(self):
        candidates = [
            Path("/workspace/yangchen/datasets/erqa-official-20261114/"
                 "samples.jsonl"),
            Path(__file__).resolve().parents[2] / "datasets" /
            "erqa-official-20261114" / "samples.jsonl",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self.skipTest("ERQA samples.jsonl is not present on this host")
        data_dir = path.parent
        samples = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(samples), 400)
        for sample in samples:
            with self.subTest(sample=sample["id"]):
                self.assertGreaterEqual(sample["image_count"], 1)
                request = erqa_protocol.build_request(
                    sample, data_dir, wire_format="state-list")
                self.assertEqual(erqa_protocol.state_text(request["state"]),
                                 sample["question"])


class ERQARunnerWireTests(unittest.TestCase):
    def test_endpoint_path_normalizes_and_routes(self):
        self.assertEqual(erqa_baseline.endpoint_path("systemone"), "/systemone")
        self.assertEqual(erqa_baseline.endpoint_path("/decide"), "/decide")
        for endpoint, expected in (
            ("/systemone", "http://127.0.0.1:18143/v1/systemone"),
            ("/decide", "http://127.0.0.1:18143/v1/decide"),
        ):
            with self.subTest(endpoint=endpoint), \
                    mock.patch.object(erqa_baseline.urllib.request, "urlopen") as urlopen:
                response = urlopen.return_value.__enter__.return_value
                response.status = 200
                response.read.return_value = b"{}"
                erqa_baseline.post(
                    "http://127.0.0.1:18143/v1", b"{}", 5, endpoint)
                self.assertEqual(urlopen.call_args.args[0].full_url, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
