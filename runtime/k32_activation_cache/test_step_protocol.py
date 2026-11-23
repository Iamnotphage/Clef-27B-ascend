from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace as NS
import unittest

from activation_cache import ActivationCache, run_layers
from step_protocol import fresh_text_step, run_candidate, vote_integers, vote_result


def schedule():
    req = NS(req_id="request", prompt_token_ids=[3, 10, 8], block_ids=([7], [13]),
             prefill_token_ids=[3, 10, 8], mm_features=[], num_computed_tokens=0,
             lora_request=None, prompt_embeds=None, prompt_is_token_ids=None,
             replay_start=0, sampling_params=None, pooling_params=NS(task="token_classify"))
    return NS(scheduled_new_reqs=[req],
              scheduled_cached_reqs=NS(req_ids=[], resumed_req_ids=set()),
              scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
              encoder_input_updates={}, preempted_req_ids=set(),
              kv_connector_metadata=None, has_sync_kv_loads=False,
              finished_req_ids=set(), num_scheduled_tokens={"request": 3},
              total_num_scheduled_tokens=3)


class AdmissionTest(unittest.TestCase):
    def test_only_complete_fresh_text_pooling(self):
        baseline = schedule()
        self.assertEqual(fresh_text_step(baseline).token_ids, (3, 10, 8))
        self.assertIsNone(fresh_text_step(baseline, dummy_run=True))
        self.assertIsNone(fresh_text_step(baseline, is_profile=True))
        for obj, field, value in (
            ("req", "mm_features", [object()]),
            ("req", "num_computed_tokens", 1),
            ("req", "prefill_token_ids", [3, 8, 10]),
            ("req", "prompt_embeds", object()),
            ("req", "lora_request", object()),
            ("req", "prompt_is_token_ids", [True, False, True]),
            ("req", "replay_start", 1),
            ("req", "sampling_params", object()),
            ("req", "pooling_params", NS(task="embed")),
            ("out", "total_num_scheduled_tokens", 2),
            ("out", "num_scheduled_tokens", {"request": 2}),
            ("out", "kv_connector_metadata", object()),
            ("out", "has_sync_kv_loads", True),
            ("out", "scheduled_spec_decode_tokens", {"request": [0]}),
            ("out", "scheduled_encoder_inputs", {"request": [0]}),
            ("out", "finished_req_ids", {"request"}),
            ("out", "preempted_req_ids", {"other"}),
            ("out", "scheduled_cached_reqs", NS(req_ids=["cache"], resumed_req_ids=set())),
            ("out", "scheduled_new_reqs", [baseline.scheduled_new_reqs[0]] * 2),
        ):
            s = deepcopy(baseline)
            setattr(s.scheduled_new_reqs[0] if obj == "req" else s, field, value)
            self.assertIsNone(fresh_text_step(s), field)


class ConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.key = b"b" * 32
        self.limits = [4096, 4096]
        self.caches = [ActivationCache(lambda held, i=i: self.limits[i])
                       for i in range(2)]

    def exchange_run(self, *, keys=None, copy_failed=(), fill_failed=(), split=False):
        keys = keys or [self.key, self.key]
        ranks = Barrier(2)
        votes = [None, None]
        ranges = [[], []]
        stocks = [False, False]
        restore_attempts = []
        positions = (0, 1, 2)
        tokens = (11, 6, 3)

        def call_rank(i):
            def exchange(key, ready):
                vote = vote_integers(key, ready)
                votes[i] = vote
                ranks.wait(3)
                result = vote_result(vote, [a + b for a, b in zip(*votes, strict=True)])
                ranks.wait(3)
                return result

            def layer(level):
                def go(*, positions, hidden_states, residual):
                    ranges[i].append(level)
                    if residual is None:
                        residual = (0, 0, 0)
                    values = tuple((h + r) % 10007 for h, r in zip(hidden_states, residual))
                    return tuple(v + p + level for v, p in zip(values, positions)), values
                return go

            layers = [layer(level) for level in range(64)]
            norm = lambda h, r: (tuple(i - j for i, j in zip(h, r)), None)

            def stock():
                stocks[i] = True
                return norm(*run_layers(layers, positions, (tokens, None)))[0]

            def restore(value):
                restore_attempts.append(i)
                return None if i in copy_failed else value

            def fill(key, h, r):
                if i in fill_failed:
                    return None
                ticket = self.caches[i].reserve(key, 48)
                return (ticket, (h, r), 48) if ticket else None

            return run_candidate(layers, positions, lambda: (tokens, None), norm,
                                 key=keys[i], cache=self.caches[i], restore=restore,
                                 capture=fill, exchange=exchange, stock=stock,
                                 enabled=not split)

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(call_rank, range(2)))
        self.assertEqual(results[0], results[1])
        return results[0], ranges, stocks, sorted(restore_attempts)

    def test_miss_fill_then_both_hit_with_matching_full_output(self):
        (reference, hit), ranges, _, _ = self.exchange_run(split=True)
        self.assertFalse(hit)
        self.assertEqual(ranges, [list(range(64))] * 2)
        (miss, hit), ranges, stocks, _ = self.exchange_run()
        self.assertEqual(miss, reference)
        self.assertFalse(hit)
        self.assertEqual(ranges, [list(range(64))] * 2)
        self.assertEqual(stocks, [False, False])
        (replay, hit), ranges, stocks, _ = self.exchange_run()
        self.assertEqual(replay, reference)
        self.assertTrue(hit)
        self.assertEqual(ranges, [list(range(32, 64))] * 2)

    def test_asymmetric_guard_or_key_mismatch_reroutes_both_to_stock(self):
        self.exchange_run()
        for keys in ((None, self.key), (b"n" * 32, self.key)):
            (_, hit), ranges, stocks, _ = self.exchange_run(keys=keys)
            self.assertFalse(hit)
            self.assertEqual(ranges, [list(range(64))] * 2)
            self.assertEqual(stocks, [True, True])

    def test_copy_or_eviction_on_one_rank_reroutes_both_to_full(self):
        self.exchange_run()
        (_, hit), ranges, _, restores = self.exchange_run(copy_failed=(1,))
        self.assertFalse(hit)
        self.assertEqual(restores, [0, 1])
        self.assertEqual(ranges, [list(range(64))] * 2)
        self.limits[1] = 0
        (_, hit), ranges, _, restores = self.exchange_run()
        self.assertFalse(hit)
        self.assertEqual(restores, [0])
        self.assertEqual(ranges, [list(range(64))] * 2)
        self.assertEqual(self.caches[1].snapshot(), (0, 0, 0))

    def test_failed_fill_releases_all_staged_budget(self):
        self.exchange_run(fill_failed=(0,))
        self.assertEqual([c.snapshot() for c in self.caches], [(0, 0, 0)] * 2)


if __name__ == "__main__":
    unittest.main()
