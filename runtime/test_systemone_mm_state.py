#!/usr/bin/env python3
"""Offline tests for the /v1/systemone multimodal-state contract and installer.

Run with:
  python3 -m unittest runtime/test_systemone_mm_state.py

The OpenDecisions 2,352-sample scan runs automatically when the pinned snapshot
is present on this host and is skipped (with a recorded reason) otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "testdata" / "systemone_mm_state" / "serving.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


state_module = load("systemone_mm_state_test", HERE / "systemone_mm_state.py")
installer = load("apply_systemone_mm_state_patch_test",
                 HERE / "apply_systemone_mm_state_patch.py")

IMAGE = "data:image/jpeg;base64,AAAA"
PNG = "data:image/png;base64,BBBB"
WEBP = "Data:Image/WEBP;base64,CCCC"
HTTP = "https://example.test/frame.jpg"
FILE = "file:///tmp/frame.png"


def image_part(url=IMAGE, **extra):
    part = {"type": "image_url", "image_url": {"url": url}}
    part.update(extra)
    return part


def text_part(value):
    return {"type": "text", "text": value}


class NormalizeTests(unittest.TestCase):
    def test_standard_single_image_and_text(self):
        result = state_module.normalize_systemone_state(
            [image_part(), text_part("Which option? A. one B. two")])
        self.assertTrue(result.was_multimodal)
        self.assertEqual(result.mode, "extract_plain")
        self.assertEqual(result.state, "Which option? A. one B. two")
        self.assertEqual(result.images, [IMAGE])
        self.assertEqual(result.segments, [
            ("image", IMAGE), ("text", "Which option? A. one B. two")])

    def test_sixteen_images_preserve_wire_order(self):
        parts = [image_part(f"{IMAGE}{index}") for index in range(16)]
        parts.append(text_part("Q"))
        result = state_module.normalize_systemone_state(parts)
        self.assertEqual(result.images, [f"{IMAGE}{index}" for index in range(16)])
        self.assertEqual(result.state, "Q")

    def test_interleaved_text_joins_back_to_the_question(self):
        question = "The cat sat. Which is right? A. red B. blue"
        parts = [
            text_part(question[:10]),
            image_part(PNG),
            text_part(question[10:25]),
            image_part(WEBP),
            text_part(question[25:]),
        ]
        result = state_module.normalize_systemone_state(parts)
        self.assertEqual(result.images, [PNG, WEBP])
        self.assertEqual(result.state, question)
        self.assertEqual("".join(
            value for kind, value in result.segments if kind == "text"), question)
        self.assertEqual([kind for kind, _ in result.segments],
                         ["text", "image", "text", "image", "text"])

    def test_pure_images_yield_empty_state(self):
        result = state_module.normalize_systemone_state(
            [image_part(PNG), image_part(WEBP)])
        self.assertEqual(result.state, "")
        self.assertEqual(result.images, [PNG, WEBP])
        self.assertTrue(result.was_multimodal)

    def test_mime_formats_detail_and_string_shorthand(self):
        parts = [
            image_part("data:image/jpg;base64,AAAA"),
            image_part(WEBP),
            {"type": "image_url", "image_url": {"url": HTTP, "detail": "high"}},
            {"type": "image_url", "image_url": FILE},
            text_part("Q"),
        ]
        result = state_module.normalize_systemone_state(parts)
        self.assertEqual(result.images[-1], FILE)
        self.assertEqual(result.images[0], "data:image/jpg;base64,AAAA")
        self.assertEqual(result.images[2], HTTP)

    def test_extract_question_dict_matches_the_old_baseline_state(self):
        question = "The cat sat. A. yes B. no"
        result = state_module.normalize_systemone_state(
            [image_part(), text_part(question)], mode="extract_question_dict")
        self.assertEqual(result.state, {"question": question})
        self.assertEqual(result.images, [IMAGE])

    def test_interleave_mode_keeps_segments(self):
        result = state_module.normalize_systemone_state(
            [text_part("a"), image_part(PNG), text_part("b")], mode="interleave")
        self.assertEqual(result.mode, "interleave")
        self.assertEqual(result.segments, [("text", "a"), ("image", PNG), ("text", "b")])
        self.assertEqual(result.state, "ab")

    def test_mode_env_is_read_and_validated(self):
        with mock.patch.dict(os.environ, {"CLEF_STATE_MM_MODE": "extract_question_dict"}):
            self.assertEqual(state_module.resolve_mode(), "extract_question_dict")
        with mock.patch.dict(os.environ, {"CLEF_STATE_MM_MODE": "nope"}):
            with self.assertRaisesRegex(ValueError, "unknown CLEF_STATE_MM_MODE"):
                state_module.normalize_systemone_state("text")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(state_module.resolve_mode(), "extract_plain")


class BackwardCompatibilityTests(unittest.TestCase):
    def test_old_dict_state_with_top_level_images_is_unchanged(self):
        state = {"question": "Which option? A. one B. two"}
        result = state_module.normalize_systemone_state(state, [IMAGE, PNG])
        self.assertFalse(result.was_multimodal)
        self.assertIs(result.state, state)
        self.assertEqual(result.images, [IMAGE, PNG])
        self.assertIsNone(result.segments)

    def test_top_level_images_merge_before_state_images(self):
        result = state_module.normalize_systemone_state(
            [image_part(PNG), text_part("Q")], [IMAGE])
        self.assertEqual(result.images, [IMAGE, PNG])
        self.assertEqual(result.state, "Q")

    def test_no_images_returns_none(self):
        result = state_module.normalize_systemone_state("plain text")
        self.assertIsNone(result.images)
        self.assertEqual(result.state, "plain text")
        self.assertFalse(result.was_multimodal)


class ZeroMisjudgmentTests(unittest.TestCase):
    def test_plain_values_pass_through(self):
        values = [
            "plain text",
            {"question": "hi"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
            None, 0, True, [], ["a", "b"],
            [{"type": "text", "text": "hi"}],
            [{"type": "event", "value": 1}],
            [{"type": "IMAGE_URL", "image_url": {"url": IMAGE}}],
        ]
        for value in values:
            with self.subTest(value=value):
                result = state_module.normalize_systemone_state(value)
                self.assertFalse(result.was_multimodal)
                self.assertEqual(result.state, value)
                self.assertIsNone(result.images)

    def test_malformed_image_lists_fail_closed(self):
        for value in (
            [image_part(), "not-a-part"],
            [{"type": "image_url", "image_url": {"url": IMAGE}}, {"type": 3}],
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    state_module.normalize_systemone_state(value)

    def test_opendecisions_full_scan_when_available(self):
        candidates = [
            Path("/workspace/yangchen/datasets/clef-full-20261112/full-data/"
                 "open-decisions2352.jsonl"),
            Path("/workspace/yangchen/datasets/clef-fixed-20261123/"
                 "opendecisions-1.0.1/open-decisions2352.jsonl"),
            Path(__file__).resolve().parent.parent / "datasets" /
            "open-decisions2352.jsonl",
            Path("/private/tmp/clef-systemone-mm-20261124/datasets/"
                 "open-decisions2352.jsonl"),
        ]
        candidates.extend(sorted(Path("/workspace/yangchen/datasets").glob(
            "clef-fixed-*/opendecisions-1.0.1/open-decisions2352.jsonl")))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self.skipTest("OpenDecisions 2,352 snapshot is not present on this host")
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 2352)
        recognized = [
            row.get("id") for row in rows
            if state_module.is_multimodal_state_list(row.get("state"))
        ]
        self.assertEqual(recognized, [])
        malformed = [
            row.get("id") for row in rows
            if state_module._contains_image_url_part(row.get("state"))
        ]
        self.assertEqual(malformed, [])


class FailClosedTests(unittest.TestCase):
    def assert_invalid(self, parts):
        with self.assertRaises(ValueError):
            state_module.normalize_systemone_state(parts)

    def test_bad_image_parts(self):
        self.assert_invalid([{"type": "image_url"}])
        self.assert_invalid([{"type": "image_url", "image_url": {}}])
        self.assert_invalid([{"type": "image_url", "image_url": {"url": ""}}])
        self.assert_invalid([{"type": "image_url", "image_url": {"url": 3}}])
        self.assert_invalid([{"type": "image_url", "image_url": None}])
        self.assert_invalid([{"type": "image_url", "image_url": "   "}])
        self.assert_invalid([{"type": "image_url",
                              "image_url": {"url": "ftp://example.test/x.jpg"}}])

    def test_bad_text_parts(self):
        self.assert_invalid([image_part(), {"type": "text"}])
        self.assert_invalid([image_part(), {"type": "text", "text": 3}])
        self.assert_invalid([image_part(), {"type": "text", "text": None}])

    def test_unsupported_content_type(self):
        self.assert_invalid([image_part(), {"type": "audio_url",
                                            "audio_url": {"url": "x"}}])


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.serving = (
            self.root / "vllm/entrypoints/generate/structured_decisions/serving.py")
        self.serving.parent.mkdir(parents=True)
        self.serving.write_bytes(FIXTURE.read_bytes())
        self.module = self.serving.parent / "_clef_mm_state.py"

    def run_installer(self, flag):
        return subprocess.run(
            [sys.executable, str(HERE / "apply_systemone_mm_state_patch.py"),
             "--source", str(self.serving), flag],
            capture_output=True, text=True, check=False)

    def report(self, flag, rc=0):
        result = self.run_installer(flag)
        self.assertEqual(result.returncode, rc, result.stderr)
        return json.loads(result.stdout)

    def test_lifecycle_check_apply_idempotent_revert(self):
        before = self.report("--check")
        self.assertEqual(before["state"], "preimage")
        self.assertEqual(before["sha256"], installer.UPSTREAM_SHA256)
        self.assertFalse(before["ready"])

        applied = self.report("--apply")
        self.assertEqual(applied["result"], "applied")
        self.assertEqual(applied["installed_state"], "postimage")
        self.assertEqual(applied["installed_sha256"], installer.POSTIMAGE_SHA256)
        self.assertEqual(self.module.read_bytes(),
                         (HERE / "systemone_mm_state.py").read_bytes())

        ready = self.report("--check")
        self.assertTrue(ready["ready"])
        serving_stat = self.serving.stat()
        module_stat = self.module.stat()
        again = self.report("--apply")
        self.assertEqual(again["result"], "already-applied")
        self.assertEqual(self.serving.stat().st_ino, serving_stat.st_ino)
        self.assertEqual(self.serving.stat().st_mtime_ns, serving_stat.st_mtime_ns)
        self.assertEqual(self.module.stat().st_ino, module_stat.st_ino)
        self.assertEqual(self.module.stat().st_mtime_ns, module_stat.st_mtime_ns)

        reverted = self.report("--revert")
        self.assertEqual(reverted["result"], "reverted")
        self.assertEqual(reverted["sha256"], installer.UPSTREAM_SHA256)
        self.assertFalse(self.module.exists())
        self.assertEqual(self.report("--check")["state"], "preimage")

    def test_partial_postimage_installs_the_module(self):
        self.report("--apply")
        self.module.unlink()
        partial = self.report("--check")
        self.assertEqual(partial["state"], "postimage")
        self.assertEqual(partial["module_state"], "missing")
        self.assertFalse(partial["ready"])
        self.report("--apply")
        self.assertTrue(self.report("--check")["ready"])

    def test_unknown_serving_source_is_refused(self):
        self.serving.write_bytes(b"unknown source\n")
        self.assertEqual(self.run_installer("--check").returncode, 1)
        result = self.run_installer("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.serving.read_bytes(), b"unknown source\n")
        self.assertFalse(self.module.exists())

    def test_unknown_module_is_refused(self):
        self.module.write_bytes(b"foreign module\n")
        self.assertEqual(self.run_installer("--check").returncode, 1)
        result = self.run_installer("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.module.read_bytes(), b"foreign module\n")
        self.assertTrue(self.serving.read_bytes() == FIXTURE.read_bytes())

    def test_postimage_contains_the_normalizer_call(self):
        self.report("--apply")
        text = self.serving.read_text()
        self.assertIn("from ._clef_mm_state import normalize_systemone_state", text)
        self.assertIn("normalize_systemone_state(\n", text)
        self.assertIn("self.strategy.render_state(normalized.state)", text)
        self.assertNotIn("self.strategy.render_state(request.state)", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
