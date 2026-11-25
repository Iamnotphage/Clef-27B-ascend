from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

from activation_cache import (
    ActivationCache, BOUNDARY, ENTRY_OVERHEAD, LAYERS,
    namespace_digest, run_layers, text_key,
)
from step_protocol import vote_integers, vote_result


class Limit:
    def __init__(self, bytes_):
        self.value = bytes_

    def __call__(self, charged):
        return self.value


class ActivationCacheTest(unittest.TestCase):
    def setUp(self):
        self.limit = Limit(2 * (ENTRY_OVERHEAD + 4))
        self.cache = ActivationCache(self.limit)

    def fill(self, key, value=b"test"):
        ticket = self.cache.reserve(key, len(value))
        self.assertIsNotNone(ticket)
        self.assertTrue(self.cache.publish(ticket, value, len(value)))

    def test_full_key_binds_all_content_and_layout(self):
        identity = {
            "weights": "one", "head": "aa3e", "processor": "proc", "source": "sha",
            "tokenizer": "tok", "template": "text", "rank_contract": "tp2",
            "dtype": "bfloat16", "positions_dtype": "int64", "mask": "causal",
            "graph": "eager", "reload": "g1", "media": [],
        }
        tokens = (1, 2, 3)
        pos = (0, 1, 2) * 3
        shape, stride = (3, 3), (8, 1)
        expected = text_key(namespace_digest(identity), tokens, pos, shape, stride)
        self.assertEqual(expected, text_key(namespace_digest(dict(reversed(
            tuple(identity.items())))), tokens, pos, shape, stride))
        for field in identity:
            replaced = dict(identity, **{field: "changed"})
            self.assertNotEqual(expected, text_key(
                namespace_digest(replaced), tokens, pos, shape, stride
            ), field)
        for args in (((1, 2, 4), pos, shape, stride), (tokens, pos[:-1] + (3,), shape, stride),
                     (tokens, pos[-3:], (3,), (1,)), (tokens, pos, shape, (3, 1))):
            self.assertNotEqual(expected, text_key(namespace_digest(identity), *args))
        with self.assertRaises(ValueError):
            text_key(namespace_digest(identity), tokens, pos, (1, 9), stride)

    def test_tp_consensus_requires_same_key_and_all_prepared(self):
        a, b = b"a" * 32, b"b" * 32
        for left, right, expected in (
            ((a, True), (a, True), (True, True)),
            ((a, True), (a, False), (True, False)),
            ((a, True), (b, True), (False, False)),
            ((None, True), (a, True), (False, False)),
        ):
            l, r = vote_integers(*left), vote_integers(*right)
            summed = [x + y for x, y in zip(l, r, strict=True)]
            self.assertEqual(vote_result(l, summed), expected)
            self.assertEqual(vote_result(r, summed), expected)

    def test_cap_and_lru_order(self):
        self.fill(b"a")
        self.fill(b"b")
        with self.cache.lease(b"a") as value:
            self.assertEqual(value, b"test")
        self.fill(b"c")
        with self.cache.lease(b"b") as value:
            self.assertIsNone(value)
        self.assertEqual(self.cache.snapshot(), (2, 0, self.limit.value))

    def test_pressure_keeps_detached_lease_charged(self):
        self.fill(b"a")
        with self.cache.lease(b"a") as original:
            self.limit.value = 0
            with self.cache.lease(b"a") as second:
                self.assertIsNone(second)
            self.assertEqual(self.cache.snapshot(), (0, 0, ENTRY_OVERHEAD + 4))
            self.assertEqual(original, b"test")
        self.assertEqual(self.cache.snapshot(), (0, 0, 0))

    def test_reservation_counts_before_copy_and_after_invalidation(self):
        ticket = self.cache.reserve(b"key", 4)
        self.assertEqual(self.cache.snapshot(), (0, 1, ENTRY_OVERHEAD + 4))
        self.assertIsNone(self.cache.reserve(b"key", 4))
        self.cache.invalidate()
        self.assertEqual(self.cache.snapshot(), (0, 1, ENTRY_OVERHEAD + 4))
        self.assertFalse(self.cache.publish(ticket, b"data", 4))
        self.assertEqual(self.cache.snapshot(), (0, 0, 0))
        self.fill(b"key", b"next")

    def test_oversize_or_failed_copy_does_not_publish(self):
        self.assertIsNone(self.cache.reserve(b"huge", self.limit.value + 1))
        t = self.cache.reserve(b"a", 3)
        self.assertFalse(self.cache.publish(t, b"A", 1))
        t = self.cache.reserve(b"a", 3)
        self.cache.cancel(t)
        self.cache.cancel(t)
        self.assertEqual(self.cache.snapshot(), (0, 0, 0))

    def test_concurrent_lease_and_invalidation(self):
        self.fill(b"a")
        ready = Barrier(2)
        dropped = Barrier(2)

        def with_lease():
            with self.cache.lease(b"a") as value:
                ready.wait(2)
                dropped.wait(2)
                return value

        with ThreadPoolExecutor(1) as pool:
            read = pool.submit(with_lease)
            ready.wait(2)
            self.cache.invalidate()
            self.assertEqual(self.cache.snapshot(), (0, 0, ENTRY_OVERHEAD + 4))
            dropped.wait(2)
            self.assertEqual(read.result(), b"test")
        self.assertEqual(self.cache.snapshot(), (0, 0, 0))

    def test_budget_read_failure_fails_closed(self):
        outcome = OSError()
        cache = ActivationCache(lambda amount: (_ for _ in ()).throw(outcome))
        self.assertIsNone(cache.reserve(b"key", 4))
        with cache.lease(b"key") as value:
            self.assertIsNone(value)


class FullPromptBoundaryTest(unittest.TestCase):
    def test_all_rows_pass_original_tail_and_norm(self):
        calls = []
        tokens = (2, 11, 9, 13, 3)
        positions = tuple(range(len(tokens)))

        def layer_fn(index):
            def run(*, positions, hidden_states, residual):
                calls.append(index)
                assert len(hidden_states) == len(positions) == len(tokens)
                previous = (0,) * len(tokens) if residual is None else residual
                updated = tuple(h + r for h, r in zip(hidden_states, previous))
                return tuple(((h ^ p) + index + 3) % 16381
                             for h, p in zip(updated, positions)), updated
            return run

        layers = [layer_fn(i) for i in range(LAYERS)]
        off = run_layers(layers, positions, (tokens, None))
        self.assertEqual(calls, list(range(64)))
        calls.clear()
        state = []
        split = run_layers(layers, positions, (tokens, None),
                           lambda a,b: state.append((a,b)))
        self.assertEqual(calls, list(range(64)))
        self.assertEqual(split, off)
        self.assertEqual(len(state), 1)
        self.assertEqual(tuple(map(len, state[0])), (5, 5))
        calls.clear()
        hit = run_layers(layers, positions, state[0], start=BOUNDARY)
        self.assertEqual(calls, list(range(BOUNDARY, 64)))
        self.assertEqual(hit, off)
        final_hidden = lambda pair: tuple(h - r for h,r in zip(*pair))
        self.assertEqual(final_hidden(hit), final_hidden(off))
        self.assertEqual(sum(final_hidden(hit)[:-1]), sum(final_hidden(off)[:-1]))


if __name__ == "__main__":
    unittest.main()
