from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from activation_cache import ActivationCache, ENTRY_OVERHEAD
import ascend_probe
from step_protocol import run_final_candidate, vote_integers, vote_result


class FinalHiddenTest(unittest.TestCase):
    def setUp(self):
        self.key = bytes(range(32))
        self.budgets = [4096, 4096]
        self.caches = [ActivationCache(lambda held, i=i: self.budgets[i]) for i in range(2)]

    def execute_two_ranks(self, *, veto_restore=(), veto_fill=(), keys=None):
        barrier = Barrier(2)
        votes = [None, None]
        key_by_rank = keys or [self.key] * 2
        stock_calls = [0, 0]
        vote_calls = [0, 0]
        final = (1, 5, 7)

        def compute(rank):
            cache = self.caches[rank]

            def exchange(key, ready):
                votes[rank] = vote_integers(key, ready)
                vote_calls[rank] += 1
                barrier.wait(3)
                summed = [a + b for a, b in zip(*votes, strict=True)]
                result = vote_result(votes[rank], summed)
                barrier.wait(3)
                return result

            def stock():
                stock_calls[rank] += 1
                return final

            def capture(digest, hidden):
                if rank in veto_fill:
                    raise MemoryError("injected fill failure")
                ticket = cache.reserve(digest, 6)
                return (ticket, (hidden,), 6) if ticket else None

            def restore(value):
                return None if rank in veto_restore else (tuple(list(value[0])),)

            return run_final_candidate(key=key_by_rank[rank], cache=cache, restore=restore,
                                       capture=capture, exchange=exchange, stock=stock)

        with ThreadPoolExecutor(2) as pool:
            result = list(pool.map(compute, range(2)))
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[0][0], final)
        return result[0][1], stock_calls, vote_calls

    def test_post_norm_identity_ends_on_full_attention(self):
        class Core:
            def forward(self):
                return None

        core = Core()
        core.layers = [None] * 64
        core.config = SimpleNamespace(layer_types=["linear_attention"] * 3 + ["full_attention"])
        core.config.layer_types *= 16
        outer = SimpleNamespace(pooler=object())
        tp = SimpleNamespace(world_size=2)
        args = outer, core, tp, type(outer), type(outer.pooler), Core, Core.forward
        with patch.multiple(ascend_probe, BOUNDARY=64, POST_NORM=True):
            self.assertEqual(ascend_probe.identity_failures(*args), {})
            core.config.layer_types[-1] = "linear_attention"
            self.assertIn("boundary_types", ascend_probe.identity_failures(*args))
        with patch.multiple(ascend_probe, BOUNDARY=56, POST_NORM=False):
            self.assertEqual(ascend_probe.identity_failures(*args), {})
            core.config.layer_types[56] = "full_attention"
            self.assertIn("boundary_types", ascend_probe.identity_failures(*args))

    def test_cold_norm_output_then_hit_without_stock(self):
        self.assertEqual(self.execute_two_ranks(), (False, [1, 1], [2, 2]))
        for cache in self.caches:
            self.assertEqual(cache.snapshot(), (1, 0, 6 + ENTRY_OVERHEAD))
        self.assertEqual(self.execute_two_ranks(), (True, [0, 0], [1, 1]))

    def test_failed_local_fill_releases_peer_preparation(self):
        self.assertEqual(self.execute_two_ranks(veto_fill=(1,)), (False, [1, 1], [2, 2]))
        self.assertEqual([cache.snapshot() for cache in self.caches], [(0, 0, 0)] * 2)

    def test_restore_veto_epoch_and_rank_pressure_are_symmetric(self):
        self.execute_two_ranks()
        self.assertEqual(self.execute_two_ranks(veto_restore=(1,)), (False, [1, 1], [2, 2]))
        self.assertEqual(self.execute_two_ranks(), (True, [0, 0], [1, 1]))
        self.caches[1].invalidate()
        self.assertEqual(self.execute_two_ranks(), (False, [1, 1], [2, 2]))
        self.budgets[1] = 0
        self.assertEqual(self.execute_two_ranks(), (False, [1, 1], [2, 2]))
        self.assertEqual([cache.snapshot() for cache in self.caches], [(0, 0, 0)] * 2)

    def test_staged_hidden_is_invisible_and_lifetime_follows_lease(self):
        cache = self.caches[0]
        ticket = cache.reserve(self.key, 6)
        self.assertTrue(cache.prepare(ticket, ((1, 5, 7),), 6))
        with cache.lease(self.key) as held:
            self.assertIsNone(held)
        self.assertTrue(cache.reveal(ticket))
        with cache.lease(self.key) as held:
            self.assertIsNotNone(held)
            cache.invalidate()
            self.assertEqual(cache.snapshot(), (0, 0, 6 + ENTRY_OVERHEAD))
        self.assertEqual(cache.snapshot(), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
