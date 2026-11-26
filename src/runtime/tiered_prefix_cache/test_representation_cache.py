"""Offline tests for the Tier 2A representation cache contract.

No NPU and no torch: keys are byte strings, packages are plain bytes, and the
weight-page helper is exercised with a mocked ``posix_fadvise``.  Run with::

    PYTHONPATH=src python -m unittest \
        runtime.tiered_prefix_cache.test_representation_cache
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime.tiered_prefix_cache.host_store import HostStore
from runtime.tiered_prefix_cache.prefix_key import (
    MmItem,
    REPRESENTATION_BOUNDARY,
    compute_representation_key,
)
from runtime.tiered_prefix_cache.representation_plan import (
    MODE_HIT,
    MODE_MISS,
    MODE_POPULATE,
    PATH_FULL,
    PATH_REPRESENTATION,
    PATH_STATE_RESUME,
    choose_single_path,
    plan_batch,
    should_capture,
)
from runtime.tiered_prefix_cache.weight_pages import (
    drop_safetensors_page_cache,
)


def image_item(*, content: str = "a" * 64, offset: int = 10,
               length: int = 256, modality: str = "image") -> MmItem:
    return MmItem(
        mm_hash=content,
        placeholder_offset=offset,
        placeholder_length=4,
        expanded_offset=offset,
        expanded_length=length,
        modality=modality,
    )


def key(tokens=(1, 2, 3, 4), items=(), **kwargs) -> bytes:
    return compute_representation_key(
        "clef-tier2a", tokens, items,
        model_revision=kwargs.pop("model_revision", "rev-1"),
        tp_size=kwargs.pop("tp_size", 2),
        boundary=kwargs.pop("boundary", REPRESENTATION_BOUNDARY),
        **kwargs)


class RepresentationKeyTests(unittest.TestCase):
    def test_key_is_stable_and_32_bytes(self):
        first = key()
        second = key()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_full_prompt_is_in_the_key(self):
        # Unlike the Tier 2B prefix key, the schema/option suffix matters.
        self.assertNotEqual(key((1, 2, 3)), key((1, 2, 4)))
        self.assertNotEqual(key((1, 2, 3)), key((1, 2, 3, 4)))

    def test_namespace_revision_tp_and_boundary_are_isolated(self):
        self.assertNotEqual(key(), key(model_revision="rev-2"))
        self.assertNotEqual(key(), key(tp_size=1))
        self.assertNotEqual(key(), key(boundary=57))
        with self.assertRaises(ValueError):
            compute_representation_key(
                "", (1, 2), model_revision="r", boundary=56)
        with self.assertRaises(ValueError):
            compute_representation_key(
                "ns", (1, 2), model_revision="r", boundary=0)

    def test_multimodal_content_and_spans_are_isolated(self):
        item = image_item()
        self.assertNotEqual(key(), key(items=(item,)))
        self.assertNotEqual(
            key(items=(item,)),
            key(items=(image_item(content="b" * 64),)))
        self.assertNotEqual(
            key(items=(item,)),
            key(items=(image_item(offset=11),)))
        self.assertNotEqual(
            key(items=(item,)),
            key(items=(image_item(modality="video"),)))
        self.assertNotEqual(
            key(items=(item,)),
            key(items=(image_item(), image_item(offset=300))))

    def test_text_and_multimodal_prompts_do_not_collide(self):
        self.assertNotEqual(key((1, 2, 3)), key((1, 2, 3), (image_item(),)))


class RepresentationPlanTests(unittest.TestCase):
    def test_single_cascade_priority(self):
        self.assertEqual(
            choose_single_path(MODE_HIT, MODE_HIT), PATH_REPRESENTATION)
        self.assertEqual(
            choose_single_path(MODE_MISS, MODE_HIT), PATH_STATE_RESUME)
        self.assertEqual(
            choose_single_path(MODE_POPULATE, MODE_MISS), PATH_FULL)
        self.assertEqual(choose_single_path(None, None), PATH_FULL)

    def test_whole_batch_hit_takes_representation_path(self):
        plan = plan_batch([MODE_HIT] * 16)
        self.assertEqual(plan.path, PATH_REPRESENTATION)
        self.assertEqual(plan.capture_indices, ())
        self.assertEqual(plan.demoted_hit_indices, ())

    def test_one_miss_falls_back_and_captures_only_full_passes(self):
        plan = plan_batch([MODE_HIT, MODE_MISS, MODE_POPULATE, None])
        self.assertEqual(plan.path, "cascade")
        self.assertEqual(plan.capture_indices, (1, 2))
        self.assertEqual(plan.demoted_hit_indices, (0,))

    def test_empty_and_ineligible_batches_never_resume_representations(self):
        self.assertEqual(plan_batch([]).path, "cascade")
        self.assertEqual(plan_batch([None, None]).capture_indices, ())

    def test_capture_skips_state_resume_and_confirmed_hits(self):
        self.assertTrue(should_capture(MODE_MISS, PATH_FULL))
        self.assertTrue(should_capture(MODE_POPULATE, PATH_FULL))
        self.assertFalse(should_capture(MODE_POPULATE, PATH_STATE_RESUME))
        self.assertFalse(should_capture(MODE_HIT, PATH_FULL))


class RepresentationStoreTests(unittest.TestCase):
    """The Tier 2A pool is pageable and evicts only under real pressure."""

    def make_store(self, budget=1 << 20):
        return HostStore(
            budget_bytes=budget, admission="always", use_shm=False,
            eviction_policy="lru", eviction_watermark=0.98)

    def test_whole_dataset_fits_with_zero_evictions(self):
        store = self.make_store(budget=5 << 20)
        keys = [bytes([index]) * 32 for index in range(8)]
        for index, digest in enumerate(keys):
            self.assertTrue(store.publish(
                digest, {0: {"hidden": bytes(256 << 10),
                             "residual": bytes(256 << 10)}},
                length=1024))
        snapshot = store.snapshot()
        self.assertEqual(snapshot["entries"], 8)
        self.assertEqual(snapshot["counters"]["evictions"], 0)
        for digest in keys:
            lease = store.acquire(digest)
            self.assertIsNotNone(lease)
            assert lease is not None
            self.assertEqual(lease.entry.shards[0]["hidden"], bytes(256 << 10))
            lease.release()

    def test_lru_evicts_under_pressure_and_keeps_a_lease(self):
        store = self.make_store(budget=2 << 20)
        first, second, third = (bytes([value]) * 32 for value in (1, 2, 3))
        package = {0: {"hidden": bytes(350 << 10), "residual": bytes(350 << 10)}}
        self.assertTrue(store.publish(first, package, length=512))
        lease = store.acquire(first)
        self.assertIsNotNone(lease)
        self.assertTrue(store.publish(second, package, length=512))
        # The third package only fits by evicting the idle second entry; the
        # leased first entry must survive.
        self.assertTrue(store.publish(third, package, length=512))
        self.assertIsNotNone(store.get(first))
        self.assertIsNone(store.get(second))
        self.assertIsNotNone(store.get(third))
        self.assertGreaterEqual(store.snapshot()["counters"]["evictions"], 1)
        assert lease is not None
        lease.release()

    def test_repr_package_round_trip(self):
        store = self.make_store()
        digest = key()
        self.assertTrue(store.publish(
            digest, {0: {"hidden": b"h" * 64, "residual": b"r" * 64}},
            length=32))
        lease = store.acquire(digest)
        assert lease is not None
        self.assertEqual(lease.entry.length, 32)
        self.assertEqual(lease.entry.shards[0]["residual"], b"r" * 64)
        lease.release()


class WeightPageTests(unittest.TestCase):
    def test_every_safetensors_file_gets_dontneed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model-00001.safetensors").write_bytes(b"a" * 128)
            (root / "joint_head.safetensors").write_bytes(b"b" * 64)
            (root / "config.json").write_bytes(b"{}")
            (root / "nested").mkdir()
            (root / "nested" / "ignored.safetensors").write_bytes(b"c")
            calls = []
            with mock.patch.object(os, "posix_fadvise",
                                   side_effect=lambda *args: calls.append(args),
                                   create=True), \
                    mock.patch.object(os, "POSIX_FADV_DONTNEED", 4,
                                      create=True):
                report = drop_safetensors_page_cache(root)
        self.assertEqual(report.files, 2)
        self.assertEqual(report.bytes, 192)
        self.assertEqual(report.errors, [])
        self.assertEqual(len(calls), 2)
        for fd, offset, length, advice in calls:
            self.assertEqual((offset, length, advice), (0, 0, 4))
            self.assertGreaterEqual(fd, 0)

    def test_missing_dir_and_unavailable_advice_are_best_effort(self):
        report = drop_safetensors_page_cache(None)
        self.assertEqual(report.files, 0)
        self.assertTrue(report.skipped)
        report = drop_safetensors_page_cache("/definitely/missing")
        self.assertEqual(report.files, 0)
        report = drop_safetensors_page_cache("/tmp", apply=False)
        self.assertEqual(report.files, 0)
        self.assertIn("posix_fadvise unavailable", report.skipped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
