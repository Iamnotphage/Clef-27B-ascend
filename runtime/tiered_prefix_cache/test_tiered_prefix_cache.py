"""Offline unit tests for the Tier 2 pure-CPU core.

Run with::

    python -m unittest runtime.tiered_prefix_cache.test_tiered_prefix_cache

The suite needs no NPU and no torch; concurrency and shared-memory tests use
plain bytes so every invariant is exercised on a laptop or the A3 CPU side.
"""

from __future__ import annotations

import importlib.util
import random
import threading
import time
import unittest

from runtime.tiered_prefix_cache.host_store import (
    HostStore,
    read_memory_budget,
    shm_available,
    tp_vote_result,
    tp_vote_vector,
)
from runtime.tiered_prefix_cache.prefix_key import (
    ALIGN,
    MmItem,
    SchemaSpan,
    compute_prefix_key,
    find_last_subsequence,
    locate_state_prefix,
)


DELIMITER = [101, 102, 103]


def make_prompt(l_prefix: int, total: int, *, start: int = 7) -> list[int]:
    tokens = [start + (index % 50) for index in range(l_prefix)]
    tokens += list(DELIMITER)
    tokens += [200 + (index % 80) for index in range(total - l_prefix - len(DELIMITER))]
    return tokens


def synthetic_mm(offset: int = 10, length: int = 3, expanded: int = 256,
                 content: str = "a" * 64) -> MmItem:
    return MmItem(
        mm_hash=content,
        placeholder_offset=offset,
        placeholder_length=length,
        expanded_offset=offset,
        expanded_length=expanded,
    )


class PrefixSplitTests(unittest.TestCase):
    def test_find_last_subsequence(self):
        self.assertEqual(find_last_subsequence([1, 2, 3, 1, 2, 3], [2, 3]), 4)
        self.assertEqual(find_last_subsequence([1, 2], [3]), -1)
        self.assertEqual(find_last_subsequence([], [1]), -1)

    def test_alignment_boundaries(self):
        cases = [
            # (l_prefix, total, expected l_aligned)
            (187, 307, 128),
            (813, 967, 768),
            (1314, 1595, 1280),
            (10161, 10294, 10112),
        ]
        for l_prefix, total, expected in cases:
            split = locate_state_prefix(make_prompt(l_prefix, total), DELIMITER)
            self.assertIsNotNone(split, (l_prefix, total))
            assert split is not None
            self.assertEqual(split.l_prefix, l_prefix)
            self.assertEqual(split.l_aligned, expected)
            self.assertEqual(split.l_aligned % ALIGN, 0)
            self.assertEqual(split.l_suffix, total - expected)
            self.assertEqual(len(split.prefix_token_ids), expected)
            self.assertEqual(
                split.prefix_token_ids + split.suffix_token_ids,
                tuple(make_prompt(l_prefix, total)))

    def test_min_prefix_and_missing_delimiter(self):
        self.assertIsNone(locate_state_prefix(make_prompt(100, 200), DELIMITER))
        self.assertIsNone(locate_state_prefix([1, 2, 3, 4], DELIMITER))
        self.assertIsNone(locate_state_prefix(make_prompt(40, 120), DELIMITER))

    def test_schema_spans_must_stay_in_suffix(self):
        prompt = make_prompt(813, 967)
        good = SchemaSpan(820, 900, "question")
        self.assertIsNotNone(locate_state_prefix(
            prompt, DELIMITER, schema_spans=[good]))
        crossing = SchemaSpan(700, 900, "option")
        self.assertIsNone(locate_state_prefix(
            prompt, DELIMITER, schema_spans=[crossing]))
        outside = SchemaSpan(900, 1200, "option")
        self.assertIsNone(locate_state_prefix(
            prompt, DELIMITER, schema_spans=[outside]))

    def test_delimiter_is_last_occurrence(self):
        tokens = make_prompt(300, 400)
        tokens += list(DELIMITER)
        tokens += [9] * 50
        split = locate_state_prefix(tokens, DELIMITER)
        assert split is not None
        self.assertEqual(split.l_prefix, 400)

    def test_multimodal_item_must_be_in_prefix(self):
        prompt = make_prompt(813, 967)
        inside = synthetic_mm(offset=100, expanded=256)
        self.assertIsNotNone(locate_state_prefix(
            prompt, DELIMITER, mm_items=[inside]))
        after = synthetic_mm(offset=900, expanded=10)
        self.assertIsNone(locate_state_prefix(
            prompt, DELIMITER, mm_items=[after]))
        too_long = synthetic_mm(offset=100, expanded=2000)
        self.assertIsNone(locate_state_prefix(
            prompt, DELIMITER, mm_items=[too_long]))


class PrefixKeyTests(unittest.TestCase):
    def _split(self):
        split = locate_state_prefix(make_prompt(813, 967), DELIMITER)
        assert split is not None
        return split

    def test_key_is_stable_and_namespaced(self):
        split = self._split()
        one = compute_prefix_key("clef", split)
        two = compute_prefix_key("clef", split)
        other = compute_prefix_key("other", split)
        revision = compute_prefix_key("clef", split, model_revision="rev-1")
        self.assertEqual(one, two)
        self.assertNotEqual(one, other)
        self.assertNotEqual(one, revision)
        self.assertEqual(len(one), 32)

    def test_same_placeholder_different_image_never_collides(self):
        prompt = make_prompt(813, 967)
        left = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm(content="a" * 64)])
        right = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm(content="b" * 64)])
        assert left is not None and right is not None
        self.assertNotEqual(
            compute_prefix_key("clef", left), compute_prefix_key("clef", right))

    def test_placeholder_offset_and_length_are_in_key(self):
        prompt = make_prompt(813, 967)
        base = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm(offset=10, length=3)])
        moved = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm(offset=20, length=3)])
        longer = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm(offset=10, length=4)])
        assert base is not None and moved is not None and longer is not None
        keys = {compute_prefix_key("clef", split)
                for split in (base, moved, longer)}
        self.assertEqual(len(keys), 3)

    def test_text_and_multimodal_keys_differ(self):
        prompt = make_prompt(813, 967)
        text = locate_state_prefix(prompt, DELIMITER)
        image = locate_state_prefix(
            prompt, DELIMITER, mm_items=[synthetic_mm()])
        assert text is not None and image is not None
        self.assertNotEqual(
            compute_prefix_key("clef", text), compute_prefix_key("clef", image))


class AdmissionTests(unittest.TestCase):
    def store(self, admission="second_hit", budget=10_000):
        return HostStore(budget_bytes=budget, admission=admission, use_shm=False)

    def test_second_hit_transitions(self):
        store = self.store()
        key = b"k" * 32
        self.assertEqual(store.observe(key), "miss")
        self.assertEqual(store.observe(key), "populate")
        self.assertTrue(store.publish(key, {0: {"kv": b"x" * 100}}))
        self.assertEqual(store.observe(key), "hit")
        store.close()

    def test_always_populates_on_first_sight(self):
        store = self.store("always")
        key = b"k" * 32
        self.assertEqual(store.observe(key), "populate")
        self.assertTrue(store.publish(key, {0: {"kv": b"x" * 100}}))
        self.assertEqual(store.observe(key), "hit")
        store.close()

    def test_ghost_capacity_is_bounded(self):
        store = HostStore(budget_bytes=10_000, ghost_capacity=4, use_shm=False)
        for index in range(10):
            store.observe(bytes([index]) * 32)
        self.assertLessEqual(store.snapshot()["ghost"], 4)
        self.assertEqual(store.snapshot()["counters"]["ghost_evictions"], 6)
        store.close()

    def test_validation(self):
        store = self.store()
        with self.assertRaises(ValueError):
            store.observe(b"short")
        with self.assertRaises(ValueError):
            store.publish(b"short", {})
        with self.assertRaises(ValueError):
            HostStore(budget_bytes=0, use_shm=False)
        with self.assertRaises(ValueError):
            HostStore(budget_bytes=10, admission="sometimes", use_shm=False)
        store.close()


class BudgetTests(unittest.TestCase):
    def test_lru_evicts_idle_and_keeps_leased(self):
        store = HostStore(budget_bytes=1_000, use_shm=False,
                          eviction_watermark=0.99)
        a, b, c = b"a" * 32, b"b" * 32, b"c" * 32
        self.assertTrue(store.publish(a, {0: {"kv": b"x" * 400}}))
        self.assertTrue(store.publish(b, {0: {"kv": b"y" * 400}}))
        lease = store.acquire(b)
        self.assertIsNotNone(lease)
        # c needs 400 bytes; a is idle and must go, b stays leased.
        self.assertTrue(store.publish(c, {0: {"kv": b"z" * 400}}))
        self.assertFalse(store.contains(a))
        self.assertTrue(store.contains(b))
        self.assertLessEqual(store.snapshot()["bytes"], 1_000)
        lease.release()
        store.close()

    def test_rejects_when_only_leased_entries_exist(self):
        store = HostStore(budget_bytes=500, use_shm=False,
                          eviction_watermark=0.99)
        a = b"a" * 32
        self.assertTrue(store.publish(a, {0: {"kv": b"x" * 400}}))
        lease = store.acquire(a)
        self.assertIsNotNone(lease)
        self.assertFalse(store.publish(b"b" * 32, {0: {"kv": b"y" * 400}}))
        self.assertEqual(store.snapshot()["counters"]["rejections"], 1)
        lease.release()
        self.assertTrue(store.publish(b"b" * 32, {0: {"kv": b"y" * 400}}))
        store.close()

    def test_drop_refuses_leased_entries(self):
        store = HostStore(budget_bytes=1_000, use_shm=False)
        key = b"a" * 32
        self.assertTrue(store.publish(key, {0: {"kv": b"x" * 100}}))
        lease = store.acquire(key)
        self.assertFalse(store.drop(key))
        lease.release()
        self.assertTrue(store.drop(key))
        self.assertFalse(store.contains(key))
        store.close()

    def test_concurrent_pressure_keeps_invariants(self):
        store = HostStore(budget_bytes=8_000, use_shm=False,
                          eviction_watermark=0.95)
        stop = threading.Event()
        errors: list[BaseException] = []

        def worker(seed: int) -> None:
            rng = random.Random(seed)
            try:
                for _ in range(400):
                    key = bytes([rng.randrange(64)]) * 32
                    store.observe(key)
                    lease = store.acquire(key)
                    if lease is not None:
                        time.sleep(0)
                        self.assertIsNotNone(lease.entry)
                        lease.release()
                    else:
                        store.publish(
                            key, {0: {"kv": bytes(rng.randrange(256)
                                                  for _ in range(120))}})
                    if rng.random() < 0.1:
                        store.drop(key)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(index,))
                   for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        snapshot = store.snapshot()
        self.assertLessEqual(snapshot["bytes"], store.budget_bytes)
        self.assertEqual(snapshot["leases"], 0)
        self.assertGreaterEqual(snapshot["entries"], 0)
        store.close()

    def test_budget_reader_is_positive(self):
        budget = read_memory_budget(default_gib=1, safety_gib=0)
        self.assertGreaterEqual(budget["budget_bytes"], 0)
        self.assertEqual(budget["safety_gib"], 0)


class SharedMemoryTests(unittest.TestCase):
    @unittest.skipUnless(shm_available(), "POSIX /dev/shm is unavailable")
    def test_prefix_memory_round_trip_and_cleanup(self):
        from multiprocessing import shared_memory

        store = HostStore(budget_bytes=1_000_000, use_shm=True)
        key = b"k" * 32
        payload = bytes(range(256)) * 16
        self.assertTrue(store.publish(key, {0: {"kv": b"x" * 64}},
                                      prefix_memory=payload))
        entry = store.get(key)
        assert entry is not None
        self.assertIsNotNone(entry.prefix_memory_shm)
        attached = store.attach_shm(entry.prefix_memory_shm,
                                    entry.prefix_memory_nbytes)
        self.assertEqual(bytes(attached[: len(payload)]), payload)
        name = entry.prefix_memory_shm
        store.close()
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class AsyncTransferTests(unittest.TestCase):
    def test_synchronous_capture_round_trip(self):
        import torch  # noqa: PLC0415

        from runtime.tiered_prefix_cache.async_transfer import (
            AsyncCapture,
            PinnedRing,
            Prefetcher,
        )

        ring = PinnedRing(slots=2, slot_bytes=1 << 20)
        capture = AsyncCapture(ring, synchronous=True)
        received = {}
        event = threading.Event()
        payload = {
            "kv": torch.arange(32, dtype=torch.float32),
            "state": torch.ones(4, 4, dtype=torch.float32),
        }
        capture.submit(payload, lambda host: (received.update(host), event.set()))
        self.assertTrue(event.wait(5))
        self.assertTrue(torch.equal(received["kv"], payload["kv"]))
        self.assertTrue(torch.equal(received["state"], payload["state"]))
        restored = Prefetcher(synchronous=True).load(received)
        self.assertTrue(torch.equal(restored.tensors["kv"], payload["kv"]))
        restored.close()
        capture.close()

    def test_ring_release_reuses_slots(self):
        import torch  # noqa: PLC0415

        from runtime.tiered_prefix_cache.async_transfer import PinnedRing

        ring = PinnedRing(slots=1, slot_bytes=1024)
        buffer = ring.acquire(0)
        self.assertEqual(buffer.numel(), 1024)
        ring.release(buffer)
        self.assertIs(ring.acquire(0), buffer)
        self.assertTrue(buffer.is_pinned() or buffer.device.type == "cpu")


class VoteTests(unittest.TestCase):
    def test_consensus(self):
        key = bytes(range(32))
        local = tp_vote_vector(key, True)
        summed = tuple(value * 2 for value in local)
        self.assertEqual(tp_vote_result(local, summed), (True, True))

    def test_presence_mismatch(self):
        local = tp_vote_vector(bytes(range(32)), True)
        other = tp_vote_vector(None, True)
        summed = tuple(a + b for a, b in zip(local, other))
        self.assertEqual(tp_vote_result(local, summed), (False, False))

    def test_ready_mismatch(self):
        key = bytes(range(32))
        local = tp_vote_vector(key, True)
        other = tp_vote_vector(key, False)
        summed = tuple(a + b for a, b in zip(local, other))
        eligible, resume = tp_vote_result(local, summed)
        self.assertTrue(eligible)
        self.assertFalse(resume)

    def test_key_byte_mismatch(self):
        left = tp_vote_vector(bytes([0] * 31 + [1]), True)
        right = tp_vote_vector(bytes([0] * 31 + [2]), True)
        summed = tuple(a + b for a, b in zip(left, right))
        self.assertEqual(tp_vote_result(left, summed), (False, False))

    def test_length_validation(self):
        with self.assertRaises(ValueError):
            tp_vote_vector(b"short", True)
        with self.assertRaises(ValueError):
            tp_vote_result((1, 1), (2, 2))


if __name__ == "__main__":
    unittest.main()
