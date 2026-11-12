#!/usr/bin/env python3
"""Check real decoded-pixel uniqueness without HTTP, processors or NPU."""

import base64
import json
from pathlib import Path
import tempfile
import unittest

import prepare_systemone_uncached_media as media


class UncachedMediaTests(unittest.TestCase):
    def test_png_has_one_real_pixel_tag_and_deterministic_bytes(self):
        data, evidence = media.tagged_png(31)
        width, height, raw, pixels, chunks = media.decode_png(data)
        self.assertEqual((width, height), (256, 256))
        self.assertEqual(pixels[:3], bytes((255, 31, 0)))
        self.assertEqual(pixels[3:], bytes((255, 0, 0)) * (256 * 256 - 1))
        self.assertEqual(evidence["decoded_channel_differences_vs_base"], 1)
        self.assertEqual(evidence["decoded_rgb_sha256"], media.digest(pixels))
        self.assertEqual(data, media.tagged_png(31)[0])
        self.assertEqual(len(data), 196947)
        self.assertEqual({kind for kind, payload in chunks}, {b"IHDR", b"IDAT", b"IEND"})
        self.assertTrue(all(raw[row * (1 + width * 3)] == 0 for row in range(height)))

    def test_avi_changes_one_payload_byte_and_preserves_header_index(self):
        data, evidence = media.tagged_avi(35)
        base = media.avi(["red", "blue"], width=96, height=96, fps=2)
        changed = [index for index, pair in enumerate(zip(data, base)) if pair[0] != pair[1]]
        self.assertEqual(changed, [evidence["modified_payload_byte_offset"]])
        self.assertTrue(evidence["header_and_idx_unchanged"])
        self.assertEqual(len(data), len(base))
        self.assertEqual(len(data), 55576)
        frames = media.decode_avi_pixels(data)
        self.assertEqual(frames[0][:3], bytes((255, 35, 0)))
        self.assertEqual(frames[0][3:], bytes((255, 0, 0)) * (96 * 96 - 1))
        self.assertEqual(frames[1], bytes((0, 0, 255)) * (96 * 96))
        self.assertEqual(evidence["decoded_frame_rgb_sha256"], [media.digest(frame) for frame in frames])

    def test_warmup_and_profile_contents_are_unique_and_shape_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report = media.prepare(directory)
            self.assertTrue(report["passed"])
            self.assertFalse(report["http_requests_sent"])
            self.assertFalse(report["npu_operations_executed"])
            for modality, expected_tokens in (("image", 203), ("video", 173)):
                definition = report["modalities"][modality]
                self.assertTrue(definition["decoded_pixel_hashes_unique"])
                self.assertTrue(definition["warmup_profile_decoded_hashes_disjoint"])
                self.assertEqual(len(definition["fixtures"]), 15)
                self.assertEqual(len({row["decoded_rgb_sha256"] for row in definition["fixtures"]}), 15)
                for phase, expected_tags in (("warmup", list(range(1, 11))), ("profile", list(range(31, 36)))):
                    path = directory / modality / (phase + "-samples.json")
                    samples = json.loads(path.read_text())["samples"]
                    self.assertEqual([row["metadata"]["tag"] for row in samples], expected_tags)
                    self.assertEqual(len(samples), 10 if phase == "warmup" else 5)
                    for sample in samples:
                        metadata = sample["metadata"]
                        self.assertEqual(metadata["expected_input_tokens"], expected_tokens)
                        key = "images" if modality == "image" else "videos"
                        data = base64.b64decode(sample["request"][key][0].split(",", 1)[1])
                        self.assertEqual(media.digest(data), metadata["fixture"]["sha256"])
                        self.assertEqual(data, Path(metadata["fixture"]["path"]).read_bytes())
                        self.assertEqual(metadata["expected_choice"], "red" if modality == "image" else "red_then_blue")

    def test_existing_changed_fixture_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            media.prepare(directory)
            path = directory / "image/warmup-tag-01-red.png"
            path.write_bytes(b"user change")
            with self.assertRaises(ValueError):
                media.prepare(directory)
            self.assertEqual(path.read_bytes(), b"user change")


if __name__ == "__main__":
    unittest.main()
