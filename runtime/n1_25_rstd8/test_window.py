"""CPU tests of the N1 supervision adapters; never imports a device backend."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import struct
from unittest.mock import patch

import window_common as common
import window_probe as probe
import window_vendor as vendor
from contract import MAGIC, checksum, intervals


class WindowContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stage = self.root / "stage/runtime/n1_25_rstd8"
        self.stage.mkdir(parents=True)
        self.identity = {"adapter_sha256": "adapter", "vendor_files": {"op_api/lib/libcust_opapi.so": "api"}}

    def reports(self):
        cell = self.root / "cell"
        names = {"partial", "r", "gamma", "diag", "reduced.reference", "reduced.candidate",
                 "y.reference", "y.candidate", "residual.reference", "residual.candidate",
                 "rstd.reference", "rstd.candidate"}
        for rank in (0, 1):
            root = cell / f"rank{rank}"
            root.mkdir(parents=True)
            raw = b"test raw"
            item = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            meta = {**common.CELL, "rank": rank, "owner": "clef-27b-ascend-21", "stage": "complete",
                    "identity_sha256": common.IDENTITY_SHA, "adapter_sha256": "adapter", "api_sha256": "api",
                    "eps": 1e-6, "pass": True, "files": {n: item for n in names},
                    "inputs": {n: copy.deepcopy(item) for n in ("partial", "x", "w", "r", "gamma")}}
            result = {"pass": True, "fixed_gate": {"pass": True}, "reduced_bit_equal": True, "diag": {"pass": True}}
            diag = [[0] * 32 for _ in range(48)]
            for core, (start, end) in enumerate(intervals(common.CELL["rows"], common.CELL["cores"])):
                diag[core][:13] = [MAGIC, 1, core, common.CELL["cores"], start, end, 1,
                                  rank if core == 0 else 0xFFFFFFFF, 2 if core == 0 else 0xFFFFFFFF,
                                  0, 0, 967, 5120]
                diag[core][31] = checksum(diag[core][:31])
            diag_raw = struct.pack("<1536I", *(v for row in diag for v in row))
            meta["files"]["diag"] = {"bytes": len(diag_raw), "sha256": hashlib.sha256(diag_raw).hexdigest()}
            (root / "meta.json").write_text(json.dumps(meta))
            (root / "result.json").write_text(json.dumps(result))
            for name in names:
                (root / (name + ".bin")).write_bytes(diag_raw if name == "diag" else raw)
        return cell

    def change(self, path, fn):
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))

    def test_aggregate_complete_and_raw_hashes(self):
        self.assertEqual(len(probe.aggregate(self.reports(), self.identity)), 2)

    def test_reject_raw_corruption(self):
        cell = self.reports()
        (cell / "rank1/partial.bin").write_bytes(b"wrong!!!")
        with self.assertRaisesRegex(ValueError, "raw byte"):
            probe.aggregate(cell, self.identity)

    def test_reject_fixed_cell_change(self):
        cell = self.reports()
        self.change(cell / "rank1/meta.json", lambda m: m.update(cores=3))
        with self.assertRaisesRegex(ValueError, "identity"):
            probe.aggregate(cell, self.identity)

    def test_reject_truthy_string_gate(self):
        cell = self.reports()
        self.change(cell / "rank0/result.json", lambda r: r["fixed_gate"].update({"pass": "true"}))
        with self.assertRaisesRegex(ValueError, "gate"):
            probe.aggregate(cell, self.identity)

    def test_reject_cross_rank_inputs(self):
        cell = self.reports()
        self.change(cell / "rank1/meta.json", lambda m: m["inputs"]["x"].update(sha256="other"))
        with self.assertRaisesRegex(ValueError, "cross-rank"):
            probe.aggregate(cell, self.identity)

    def test_reject_missing_reduced_bytes(self):
        cell = self.reports()
        self.change(cell / "rank1/meta.json", lambda m: m["files"].pop("reduced.candidate"))
        with self.assertRaisesRegex(ValueError, "missing raw"):
            probe.aggregate(cell, self.identity)

    def test_reject_derived_or_unbound_partial(self):
        cell = self.reports()
        self.change(cell / "rank1/meta.json", lambda m: m["inputs"]["partial"].update(sha256="derived"))
        with self.assertRaisesRegex(ValueError, "input/raw"):
            probe.aggregate(cell, self.identity)

    def test_staged_path_is_required(self):
        self.assertEqual(common.staged_paths(self.stage / "window_probe.py", self.root), (self.stage, self.root))
        with self.assertRaises(ValueError):
            common.staged_paths(self.root / "window_probe.py", self.root)

    def test_command_pins_run_path_and_aron_cell(self):
        cmd = probe.command(self.stage, self.root, "/fixed/python")
        self.assertIn("--run-path", cmd)
        self.assertIn("--start-method=spawn", cmd)
        self.assertIn("--max-restarts=0", cmd)
        self.assertEqual(cmd[cmd.index("--cores") + 1], "24")
        self.assertEqual(cmd[cmd.index("--use-hccl") + 1], "1")

    def test_requested_partition_and_observed_fixture_are_separate(self):
        parts = intervals(967, 24)
        self.assertEqual(parts[0][0], 0)
        self.assertEqual(parts[-1][1], 967)
        self.assertTrue(all(parts[i][1] == parts[i + 1][0] for i in range(23)))
        self.assertEqual(sorted(end - start for start, end in parts), [40] * 17 + [41] * 7)
        observed = probe.observed_diag(self.reports())
        self.assertTrue(all(r["observed_nonzero_slot_count"] == 24 for r in observed))
        self.assertEqual([(s["row_start"], s["row_end"]) for s in observed[0]["observed_nonzero_slots"]], parts)

    def test_wrong_core_gate_never_starts_timing_child(self):
        target = self.root / "probe"
        target.mkdir()
        self.reports().rename(target / "cell")
        self.change(target / "cell/rank0/meta.json", lambda m: m.update(cores=2))
        env = {"ASCEND_RT_VISIBLE_DEVICES": "0,1",
               "ASCEND_CUSTOM_OPP_PATH": str(self.root / "isolated-opp/vendors" / common.VENDOR),
               "LD_LIBRARY_PATH": str(self.root / "isolated-opp/vendors" / common.VENDOR / "op_api/lib")}
        with patch.object(probe, "staged_paths", return_value=(self.stage, self.root)), \
             patch.object(probe, "check_head"), patch.object(probe, "frozen", return_value=self.identity), \
             patch("contract.verify_identity"), patch.dict("os.environ", env), \
             patch("sys.argv", ["probe", "--tree", str(self.root), "--device", "0", "--output", str(target / "head-probe.json")]), \
             patch.object(probe.subprocess, "call", return_value=0) as child:
            self.assertEqual(probe.main(), 1)
        self.assertEqual(child.call_count, 1)
        result = json.loads((target / "head-probe.json").read_text())
        self.assertFalse(result["gate_passed"])
        self.assertNotIn("timing_argv", result)

    def lifecycle(self, action):
        with patch.object(vendor, "check_head"), patch.object(vendor, "frozen", return_value={}), \
             patch.object(vendor, "archive_bytes", return_value={"op_api/lib/fake": b"synthetic CPU fixture"}):
            return vendor.lifecycle(action, self.stage, self.root, self.root / "tree")

    def test_install_revert_retains_bytes_and_is_idempotent(self):
        with patch("contract.verify_identity"):
            self.assertTrue(self.lifecycle("check")["pass"])
            self.assertFalse((self.root / "isolated-opp").exists())
            self.assertTrue(self.lifecycle("install")["pass"])
        self.assertTrue(self.lifecycle("revert")["active_path_absent"])
        self.assertEqual((self.root / "isolated-opp.retired/vendors" / common.VENDOR / "op_api/lib/fake").read_bytes(), b"synthetic CPU fixture")
        self.assertTrue(self.lifecycle("revert")["pass"])

    def test_partial_install_retired_after_verification_failure(self):
        with patch("contract.verify_identity", side_effect=ValueError("failure")):
            with self.assertRaises(ValueError):
                self.lifecycle("install")
        self.assertTrue(self.lifecycle("revert")["active_path_absent"])

    def test_unowned_or_symlink_root_is_rejected(self):
        root = self.root / "isolated-opp"
        root.mkdir()
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.lifecycle("revert")
        root.rmdir()
        root.symlink_to(self.stage, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ordinary"):
            self.lifecycle("revert")


if __name__ == "__main__":
    unittest.main()
