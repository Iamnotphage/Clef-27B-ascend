"""CPU synthetic raw-schema checks; no backend import or device execution."""
import copy
import json
import math
import unittest

from timing_contract import ROUND_ORDERS, REPS, WARMUP, aggregate_samples, schedule


def fixture():
    metas = [{"rank": rank, "hostname": "cpu-test-host", "boot_id": "cpu-test-boot",
              "clock": {"monotonic": True, "implementation": "clock_gettime(CLOCK_MONOTONIC)"}}
             for rank in (0, 1)]
    ranks = [[], []]
    for index, item in enumerate(schedule()):
        for rank in (0, 1):
            start = 1_000_000_000 + index * 10_000_000 + rank * 100_000
            duration = (1_000_000 if item["arm"] == "baseline" else 500_000) + rank * 100_000
            event = (0.8 if item["arm"] == "baseline" else 0.4) + rank * (
                0.1 if item["arm"] == "baseline" else 0.05)
            if item["phase"] != "measure":
                duration, event = 4_000_000, 3.0
            ranks[rank].append({**item, "rank": rank, "start_ns": start,
                                "end_ns": start + duration, "wall_ms": duration / 1e6,
                                "event_ms": event})
    # Exercise the actual JSON-compatible schema rather than private structures.
    return json.loads(json.dumps(ranks)), json.loads(json.dumps(metas))


class TimingContractTests(unittest.TestCase):
    def test_schedule_and_json_raw_aggregation(self):
        self.assertEqual(REPS, 5)
        self.assertEqual(WARMUP, 3)
        seq = schedule()
        self.assertEqual(len(seq), 48)
        self.assertEqual([s["arm"] for s in seq[:8]], ["baseline", "candidate"] * 4)
        for rnd, order in enumerate(ROUND_ORDERS):
            self.assertEqual([s["arm"] for s in seq if s["round"] == rnd], list(order) * 5)
        records, metas = fixture()
        before = copy.deepcopy((records, metas))
        result = aggregate_samples(records, metas)
        self.assertEqual((records, metas), before)
        self.assertEqual(len(result["calls"]), 48)
        self.assertEqual(len(result["first_call"]["calls"]), 2)
        self.assertEqual(len(result["warmup"]["calls"]), 6)
        call = result["calls"][8]
        self.assertEqual(call["samples"], [records[0][8], records[1][8]])
        self.assertAlmostEqual(call["critical_path_wall_ms"], 1.2)
        self.assertAlmostEqual(call["slow_rank_wall_ms"], 1.1)
        self.assertAlmostEqual(call["slow_rank_event_ms"], 0.9)
        overall = result["measure"]["overall"]
        self.assertEqual(overall["pair_count"], 20)
        self.assertEqual(overall["arms"]["baseline"]["count"], 20)
        self.assertEqual(overall["arms"]["candidate"]["count"], 20)
        self.assertEqual(overall["arms"]["baseline"]["critical_path_wall_ms"]["median"], 1.2)
        self.assertAlmostEqual(overall["paired_speedups"]["critical_path_wall_ms"]["median"], 12 / 7)
        self.assertAlmostEqual(overall["paired_speedups"]["slow_rank_event_ms"]["median"], 2)
        for rnd in result["measure"]["rounds"]:
            self.assertEqual(rnd["pair_count"], 5)
            self.assertEqual(rnd["arms"]["baseline"]["count"], 5)
            self.assertEqual(rnd["arms"]["candidate"]["count"], 5)
        json.dumps(result, allow_nan=False)

    def test_strict_missing_extra_duplicate_and_reordered_samples(self):
        for change in (lambda r: r[0].pop(),
                       lambda r: r[1].append(copy.deepcopy(r[1][-1])),
                       lambda r: r[0].__setitem__(9, copy.deepcopy(r[0][8])),
                       lambda r: r[1].__setitem__(slice(8, 10), r[1][8:10][::-1])):
            records, metas = fixture()
            change(records)
            with self.assertRaises(ValueError):
                aggregate_samples(records, metas)

    def test_strict_field_schema_and_types(self):
        changes = [lambda s: s.pop("event_ms"), lambda s: s.update(extra=1),
                   lambda s: s.update(rank=1), lambda s: s.update(position=True),
                   lambda s: s.update(start_ns=float(s["start_ns"])),
                   lambda s: s.update(event_ms=True)]
        for change in changes:
            records, metas = fixture()
            change(records[0][8])
            with self.assertRaises(ValueError):
                aggregate_samples(records, metas)

    def test_nonfinite_negative_and_incorrect_wall_rejected(self):
        for field in ("event_ms", "wall_ms"):
            for bad in (math.nan, math.inf, -math.inf, -0.1, "1", 10 ** 1000):
                records, metas = fixture()
                records[1][9][field] = bad
                with self.assertRaises(ValueError):
                    aggregate_samples(records, metas)
        records, metas = fixture()
        records[1][9]["wall_ms"] += 1e-10
        with self.assertRaises(ValueError):
            aggregate_samples(records, metas)

    def test_timestamp_reversal_and_rank_local_overlap_rejected(self):
        records, metas = fixture()
        records[0][8]["end_ns"] = records[0][8]["start_ns"] - 1
        with self.assertRaises(ValueError):
            aggregate_samples(records, metas)
        records, metas = fixture()
        records[0][8]["start_ns"] = records[0][7]["end_ns"] - 1
        records[0][8]["wall_ms"] = (records[0][8]["end_ns"] - records[0][8]["start_ns"]) / 1e6
        with self.assertRaises(ValueError):
            aggregate_samples(records, metas)

    def test_clock_host_boot_and_monotonic_contract(self):
        mutations = [lambda m: m.update(hostname="other"),
                     lambda m: m.update(boot_id="other"),
                     lambda m: m["clock"].update(implementation="other"),
                     lambda m: m["clock"].update(monotonic=False),
                     lambda m: m["clock"].update(monotonic=1),
                     lambda m: m.pop("boot_id"),
                     lambda m: m.update(rank=0)]
        for change in mutations:
            records, metas = fixture()
            change(metas[1])
            with self.assertRaises(ValueError):
                aggregate_samples(records, metas)

    def test_exactly_two_ranks_and_mapping_inputs(self):
        records, metas = fixture()
        a = aggregate_samples(records, metas)
        b = aggregate_samples({0: records[0], 1: records[1]}, {"0": metas[0], "1": metas[1]})
        self.assertEqual(a, b)
        for bad in (records[:1], records + [records[0]], {0: records[0], 2: records[1]},
                    {False: records[0], True: records[1]}):
            with self.assertRaises(ValueError):
                aggregate_samples(bad, metas)
        with self.assertRaises(ValueError):
            aggregate_samples(records, metas[:1])

    def test_zero_duration_is_explicit_undefined_ratio(self):
        records, metas = fixture()
        for sample_list in records:
            for sample in sample_list:
                if sample["phase"] == "measure" and sample["arm"] == "candidate":
                    sample["start_ns"] = 1_000_000_000 + schedule().index(
                        {k: sample[k] for k in ("phase", "round", "rep", "position", "arm")}) * 10_000_000
                    sample["end_ns"] = sample["start_ns"]
                    sample["wall_ms"] = sample["event_ms"] = 0
        result = aggregate_samples(records, metas)
        ratios = result["measure"]["overall"]["paired_speedups"]
        for metric in ("critical_path_wall_ms", "slow_rank_event_ms"):
            self.assertEqual(ratios[metric], {"count": 0, "median": None, "p95": None,
                                             "undefined_count": 20})
        json.dumps(result, allow_nan=False)

    def test_p95_linear_interpolation(self):
        records, metas = fixture()
        for samples in records:
            for sample in samples:
                if sample["phase"] == "measure" and sample["arm"] == "baseline":
                    sample["event_ms"] = sample["round"] * REPS + sample["rep"] + 1
        result = aggregate_samples(records, metas)
        dist = result["measure"]["overall"]["arms"]["baseline"]["slow_rank_event_ms"]
        self.assertEqual(dist["count"], 20)
        self.assertEqual(dist["median"], 10.5)
        self.assertAlmostEqual(dist["p95"], 19.05)
        round0 = result["measure"]["rounds"][0]["arms"]["baseline"]["slow_rank_event_ms"]
        self.assertAlmostEqual(round0["p95"], 4.8)

    def test_finite_extreme_events_do_not_emit_infinite_statistics(self):
        records, metas = fixture()
        for samples in records:
            for sample in samples:
                if sample["phase"] == "measure":
                    sample["event_ms"] = 1e308 if sample["arm"] == "baseline" else 1e-308
        result = aggregate_samples(records, metas)
        overall = result["measure"]["overall"]
        self.assertEqual(overall["arms"]["baseline"]["slow_rank_event_ms"]["median"], 1e308)
        self.assertEqual(overall["paired_speedups"]["slow_rank_event_ms"]["undefined_count"], 20)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
