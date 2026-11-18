"""CPU/stdlib tests for complete two-rank GroupedMMAR timing evidence."""
from copy import deepcopy
import json
import unittest

from timing_contract import aggregate_samples, schedule


def fixture():
    metas = [{"rank": rank, "hostname": "fixed-host", "boot_id": "same-boot",
              "clock": {"implementation": "clock_gettime(CLOCK_MONOTONIC)",
                        "monotonic": True}} for rank in (0, 1)]
    records = []
    for rank in (0, 1):
        rows = []
        for index, item in enumerate(schedule()):
            start = 100_000_000 + index * 10_000_000 + rank * 100_000
            duration = (2_000_000 if item["arm"] == "baseline" else 1_000_000) + rank * 200_000
            rows.append({**item, "rank": rank, "start_ns": start, "end_ns": start + duration,
                         "wall_ms": duration / 1e6, "event_ms": duration / 1e6 - 0.25})
        records.append(rows)
    return records, metas


class TimingContractTests(unittest.TestCase):
    def test_complete_statistics_use_slow_rank_and_whole_span(self):
        records, metas = fixture()
        snapshot = deepcopy((records, metas))
        out = aggregate_samples(records, metas)
        self.assertEqual((records, metas), snapshot)
        self.assertEqual(out["protocol"]["samples_per_rank"], 80)
        self.assertEqual(len(out["calls"]), 80)
        self.assertAlmostEqual(out["calls"][0]["slow_rank_event_ms"], 1.95)
        self.assertEqual(out["calls"][0]["slow_rank_wall_ms"], 2.2)
        self.assertEqual(out["calls"][0]["critical_path_wall_ms"], 2.3)
        overall = out["measure"]["overall"]
        self.assertEqual(overall["pair_count"], 40)
        for key in ("mean", "p50", "p95"):
            self.assertAlmostEqual(overall["arms"]["baseline"]["critical_path_wall_ms"][key], 2.3)
        self.assertAlmostEqual(overall["paired_speedups"]["critical_path_wall_ms"]["p50"], 2.3 / 1.3)
        self.assertEqual([r["order"] for r in out["measure"]["rounds"]],
                         [["baseline", "candidate"], ["candidate", "baseline"]] * 2)
        self.assertTrue(all(r["pair_count"] == 10 for r in out["measure"]["rounds"]))
        json.dumps(out, allow_nan=False)

    def test_requires_exactly_two_complete_ranks(self):
        for mutate in (lambda r: r.pop(), lambda r: r[1].pop(),
                       lambda r: r[0].append(r[0][-1])):
            rows, metas = fixture()
            mutate(rows)
            with self.assertRaises(ValueError):
                aggregate_samples(rows, metas)

    def test_schedule_and_rank_tampering_rejected(self):
        for field, value in (("rank", 1), ("round", 3), ("rep", 4), ("arm", "candidate"),
                             ("start_ns", True), ("round", False)):
            rows, metas = fixture()
            rows[0][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                aggregate_samples(rows, metas)
        rows, metas = fixture()
        rows[1][0], rows[1][1] = rows[1][1], rows[1][0]
        with self.assertRaises(ValueError):
            aggregate_samples(rows, metas)

    def test_nonfinite_negative_and_boolean_measurements_rejected(self):
        for field in ("wall_ms", "event_ms"):
            for value in (float("nan"), float("inf"), -1.0, True):
                rows, metas = fixture()
                rows[0][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    aggregate_samples(rows, metas)

    def test_exact_timestamp_duration_required(self):
        rows, metas = fixture()
        rows[0][0]["wall_ms"] += 1e-9
        with self.assertRaises(ValueError):
            aggregate_samples(rows, metas)
        rows, metas = fixture()
        rows[0][1]["start_ns"] = rows[0][0]["end_ns"] - 1
        rows[0][1]["wall_ms"] = (rows[0][1]["end_ns"] - rows[0][1]["start_ns"]) / 1e6
        with self.assertRaises(ValueError):
            aggregate_samples(rows, metas)

    def test_clock_identity_is_required_and_common(self):
        for field, value in (("hostname", "other"), ("boot_id", "other"), ("hostname", ""),
                             ("clock", {"implementation": "other", "monotonic": True}),
                             ("clock", {"implementation": "clock_gettime(CLOCK_MONOTONIC)", "monotonic": False}),
                             ("clock", None)):
            rows, metas = fixture()
            metas[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                aggregate_samples(rows, metas)

    def test_extra_sample_fields_rejected(self):
        rows, metas = fixture()
        rows[0][0]["phase"] = "measure"
        with self.assertRaises(ValueError):
            aggregate_samples(rows, metas)

    def test_zero_candidate_ratio_remains_null(self):
        rows, metas = fixture()
        for records in rows:
            for record in records:
                if record["arm"] == "candidate":
                    record["event_ms"] = 0
        out = aggregate_samples(rows, metas)
        stats = out["measure"]["overall"]["paired_speedups"]["slow_rank_event_ms"]
        self.assertEqual(stats, {"count": 0, "mean": None, "p50": None, "p95": None,
                                 "undefined_count": 40})
        self.assertIsNone(out["measure"]["overall"]["pairs"][0]
                          ["speedup_baseline_over_candidate"]["slow_rank_event_ms"])
        json.dumps(out, allow_nan=False)

    def test_percentile_interpolates_formal_samples(self):
        rows, metas = fixture()
        for records in rows:
            counts = {"baseline": 0, "candidate": 0}
            for record in records:
                counts[record["arm"]] += 1
                record["event_ms"] = float(counts[record["arm"]])
        stats = aggregate_samples(rows, metas)["measure"]["overall"]["arms"]["baseline"]["slow_rank_event_ms"]
        self.assertEqual(stats["count"], 40)
        self.assertEqual(stats["p50"], 20.5)
        self.assertEqual(stats["mean"], 20.5)
        self.assertAlmostEqual(stats["p95"], 38.05)


if __name__ == "__main__":
    unittest.main()
