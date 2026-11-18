"""CPU tests of the N1 supervision adapters; never imports a device backend."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import window_common as common
import window_probe as probe
import window_vendor as vendor


class WindowContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stage = self.root / "stage/runtime/n1_25"
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
            (root / "meta.json").write_text(json.dumps(meta))
            (root / "result.json").write_text(json.dumps(result))
            for name in names:
                (root / (name + ".bin")).write_bytes(raw)
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
        self.change(cell / "rank1/meta.json", lambda m: m.update(cores=2))
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

    def test_command_pins_run_path_and_first_cell(self):
        cmd = probe.command(self.stage, self.root, "/fixed/python")
        self.assertIn("--run-path", cmd)
        self.assertIn("--start-method=spawn", cmd)
        self.assertIn("--max-restarts=0", cmd)
        self.assertEqual(cmd[cmd.index("--cores") + 1], "1")
        self.assertEqual(cmd[cmd.index("--use-hccl") + 1], "0")

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
