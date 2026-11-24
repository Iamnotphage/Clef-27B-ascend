from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
import unittest

from activation_cache import ActivationCache
from fixed_pair import pack_pair, run_fixed_pair
from step_protocol import TextStep, fresh_text_items, vote_integers, vote_result
from test_step_protocol import schedule


class FixedPairTest(unittest.TestCase):
    def setUp(self):
        self.items = (TextStep("a",(4,7,9),((1,),)),TextStep("b",(6,2),((2,),)))
        self.domain = bytes(range(32))

    def make_pair(self, items=None, order=("a","b"), strides=(8,1), positions=None):
        items = self.items if items is None else items
        by_id = {x.request_id:x for x in items}
        tokens = tuple(t for i in order for t in by_id[i].token_ids)
        start = len(by_id[order[0]].token_ids)
        axes = tuple(range(start)) + tuple(range(len(tokens)-start))
        return pack_pair(items,order,(0,start,len(tokens)),tokens,(axes,)*3 if positions is None else positions,
                         strides,self.domain)

    def test_actual_unequal_pack_changes_on_order_partner_and_stride(self):
        pair = self.make_pair()
        self.assertEqual(pair.offsets,(0,3,5))
        self.assertNotEqual(pair.step_key,self.make_pair(order=("b","a")).step_key)
        new = (self.items[0],TextStep("b",(6,3),((2,),)))
        self.assertNotEqual(pair.step_key,self.make_pair(items=new).step_key)
        self.assertNotEqual(pair.step_key,self.make_pair(strides=(9,1)).step_key)
        self.assertIsNone(self.make_pair(strides=(4,1)))
        self.assertIsNone(self.make_pair(positions=((0,1,2,3,4),)*3))
        self.assertIsNone(pack_pair(self.items,("a","a"),(0,3,5),(4,7,9,6,2),((0,1,2,0,1),),(1,),self.domain))

    def test_duplicate_tokens_keep_occurrence_specific_numeric_domain(self):
        twin = TextStep("b",self.items[0].token_ids,((2,),))
        pair = self.make_pair(items=(self.items[0],twin))
        self.assertEqual(pair.offsets,(0,3,6))
        self.assertNotEqual(pair.keys[0],pair.keys[1])

    def test_every_scheduler_row_must_be_fresh(self):
        output = schedule()
        first = output.scheduled_new_reqs[0]
        second = deepcopy(first)
        second.req_id = "second"
        second.prompt_token_ids = second.prefill_token_ids = [55,43]
        output.scheduled_new_reqs.append(second)
        output.num_scheduled_tokens = {first.req_id:3,second.req_id:2}
        output.total_num_scheduled_tokens = 5
        self.assertEqual([len(x.token_ids) for x in fresh_text_items(output,count=2)],[3,2])
        second.num_computed_tokens = 1
        self.assertIsNone(fresh_text_items(output,count=2))
        second.num_computed_tokens = 0
        second.req_id = first.req_id
        self.assertIsNone(fresh_text_items(output,count=2))

    def test_pair_protocol_two_rank_fallback_and_commit(self):
        limits = [5000,5000]
        caches = [ActivationCache(lambda held,i=i:limits[i]) for i in range(2)]
        proof_pair = self.make_pair()
        stock_out = (3,6,2,9,7)

        def run(veto=None,reversed_rank=None):
            barrier = Barrier(2)
            votes = [None,None]
            stock_calls = [0,0]

            def rank(i):
                def exchange(key,ready):
                    votes[i] = vote_integers(key,ready)
                    barrier.wait(2)
                    decision = vote_result(votes[i],[a+b for a,b in zip(*votes,strict=True)])
                    barrier.wait(2)
                    return decision

                def stock():
                    stock_calls[i] += 1
                    return stock_out

                def capture(key,begin,end,output):
                    data = output[begin:end]
                    ticket = caches[i].reserve(key,len(data))
                    return (ticket,data,len(data)) if ticket is not None else None

                def restore(_,values):
                    return None if i==veto else values[0]+values[1]

                pair = self.make_pair(order=("b","a")) if i==reversed_rank else proof_pair
                return run_fixed_pair(pair,cache=caches[i],restore=restore,capture=capture,
                                      exchange=exchange,stock=stock,enabled=True)

            with ThreadPoolExecutor(2) as pool:
                values = list(pool.map(rank,range(2)))
            return values,stock_calls

        self.assertEqual(run(), ([(stock_out,False)]*2,[1,1]))
        self.assertEqual([c.published for c in caches],[2,2])
        self.assertEqual(run(), ([(stock_out,True)]*2,[0,0]))
        self.assertEqual(run(veto=1), ([(stock_out,False)]*2,[1,1]))
        self.assertEqual(run(reversed_rank=0), ([(stock_out,False)]*2,[1,1]))
        limits[1]=0
        self.assertEqual(run(), ([(stock_out,False)]*2,[1,1]))
        self.assertEqual([c.snapshot() for c in caches],[(0,0,0)]*2)


    def test_partial_capture_cancels_all_reservations(self):
        pair = self.make_pair()
        cache = ActivationCache(lambda _: 5000)
        observations = []

        def capture(key, begin, end, output):
            if begin:
                raise MemoryError("second allocation")
            value = output[begin:end]
            return cache.reserve(key, len(value)), value, len(value)

        def exchange(key, ready):
            observations.append((key, ready))
            return True, False

        expected = (4, 7, 9, 6, 2)
        value, hit = run_fixed_pair(pair, cache=cache, restore=None, capture=capture,
                                    exchange=exchange, stock=lambda: expected, enabled=True)
        self.assertEqual(value, expected)
        self.assertFalse(hit)
        self.assertEqual(observations, [(pair.step_key, False)] * 2)
        self.assertEqual(cache.snapshot(), (0, 0, 0))
        self.assertEqual(cache.published, 0)

    def test_restore_holds_both_leases_when_pressure_detaches_entries(self):
        pair = self.make_pair()
        cache = ActivationCache(lambda _: 5000)
        expected = ((4, 7, 9), (6, 2))
        for key, item in zip(pair.keys, expected, strict=True):
            self.assertTrue(cache.publish(cache.reserve(key, len(item)), item, len(item)))

        def restore(_, values):
            self.assertEqual(values, expected)
            cache.invalidate()
            self.assertEqual(cache.snapshot(), (0, 0, 2053))
            return values[0] + values[1]

        value, hit = run_fixed_pair(pair, cache=cache, restore=restore, capture=None,
                                    exchange=lambda _, ready: (True, ready), stock=None, enabled=True)
        self.assertTrue(hit)
        self.assertEqual(value, expected[0] + expected[1])
        self.assertEqual(cache.snapshot(), (0, 0, 0))
