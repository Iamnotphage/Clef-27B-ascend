#!/usr/bin/env python3
"""Observation contract checks; standard library only, no NPU required."""

import asyncio
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import profile_instrumentation as instrumentation


class InstrumentationTests(unittest.TestCase):
    def observer(self, directory=None, scope_filter=None):
        environment = dict(os.environ)
        for key in ("CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR", "CLEF_PROFILE_SCOPE_FILTER", "RANK", "LOCAL_RANK"):
            environment.pop(key, None)
        if directory is not None:
            environment["CLEF_PROFILE_OBSERVE_DIR"] = str(directory)
        if scope_filter is not None:
            environment["CLEF_PROFILE_SCOPE_FILTER"] = scope_filter
        namespace = {}
        with patch.dict(os.environ, environment, clear=True):
            exec(compile(instrumentation.OBSERVER, "clef_observe.py", "exec"), namespace)
        return namespace

    @staticmethod
    def rows(directory):
        return [json.loads(line) for file in Path(directory).glob("cpu-*.jsonl")
                for line in file.read_text().splitlines()]

    def test_disabled_returns_original(self):
        observer = self.observer()

        def original(value):
            return value + 1

        self.assertIs(observer["observe"]("disabled")(original), original)

    def test_scope_filter_selects_only_exact_names_and_empty_disables(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.observer(directory, " clef::joint_head, clef::worker_execute_model, ")

            def original(value):
                return value + 1

            skipped = namespace["observe"]("clef::joint_head_extra")(original)
            self.assertIs(skipped, original)
            selected = namespace["observe"]("clef::joint_head")(original)
            self.assertIsNot(selected, original)
            self.assertEqual(skipped(1), 2)
            self.assertEqual(selected(2), 3)
            rows = self.rows(directory)
            self.assertEqual([row["name"] for row in rows], ["clef::joint_head"])
            disabled = self.observer(directory, "")
            self.assertIs(disabled["observe"]("clef::joint_head")(original), original)

    def test_header_bound_method_async_context_and_explicit_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            observe = self.observer(directory)["observe"]

            class Service:
                @observe("clef::tokenizer")
                def tokens(self, text):
                    return text.upper()

                @observe("clef::frontend_read_including_engine")
                async def read(self, *, request_id):
                    await asyncio.sleep(0)
                    return self.tokens(request_id)

                @observe("clef::frontend_request")
                async def create(self, request, raw_request):
                    await asyncio.sleep(0)
                    return await self.read(
                        request_id="decision-" + raw_request.headers["X-Request-Id"])

            service = Service()

            @observe("clef::api_handler_including_json")
            async def handler(request, raw_request):
                return await service.create(request, raw_request)

            async def requests():
                return await asyncio.gather(*[
                    handler(SimpleNamespace(request_id="body-ignored"),
                            SimpleNamespace(headers={"X-Request-Id": value}))
                    for value in ("first", "second")])

            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"torch": None}):
                self.assertEqual(asyncio.run(requests()), ["DECISION-FIRST", "DECISION-SECOND"])
            rows = self.rows(directory)
            self.assertEqual(len(rows), 8)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["request_id"]].append(row)
                self.assertIsNone(row["rank"])
                self.assertIsNone(row["rank_source"])
                self.assertGreater(row["duration_ns"], 0)
                self.assertEqual(row["timing_scope"], "host_wall_inclusive")
            self.assertEqual(set(grouped), {"decision-first", "decision-second"})
            for request_rows in grouped.values():
                self.assertEqual(len(request_rows), 4)
                by_name = {row["name"]: row for row in request_rows}
                self.assertEqual(by_name["clef::api_handler_including_json"]["request_id_source"], "http_header")
                self.assertEqual(by_name["clef::frontend_request"]["request_id_source"], "http_header")
                self.assertEqual(by_name["clef::frontend_read_including_engine"]["request_id_source"], "explicit_argument")
                self.assertEqual(by_name["clef::tokenizer"]["request_id_source"], "context")

    def test_rank_uses_initialized_group_without_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            observe = self.observer(directory)["observe"]
            distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                                          get_rank=Mock(return_value=1), init_process_group=Mock())
            observed = observe("clef::rank_test")(lambda: "preserved")
            with patch.dict(os.environ, {}, clear=True), patch.dict(
                    sys.modules, {"torch": SimpleNamespace(distributed=distributed)}):
                self.assertEqual(observed(), "preserved")
            row = self.rows(directory)[0]
            self.assertEqual(row["rank"], 1)
            self.assertEqual(row["rank_source"], "initialized_torch_distributed")
            distributed.get_rank.assert_called_once_with()
            distributed.init_process_group.assert_not_called()

    def test_exception_preserved_and_context_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.observer(directory)

            @namespace["observe"]("clef::error_test")
            async def failing(*, request_id):
                raise ValueError("original exception")

            with self.assertRaisesRegex(ValueError, "original exception"):
                asyncio.run(failing(request_id="decision-error"))
            self.assertIsNone(namespace["_REQUEST_ID"].get())
            self.assertEqual(self.rows(directory)[0]["request_id"], "decision-error")

    def test_native_queue_metrics_use_engine_event_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            observe = self.observer(directory)["observe"]

            class Stats:
                @observe("clef::request_finished_metrics")
                def update_from_finished_request(self, *, request_id, req_stats, num_prompt_tokens):
                    return "unchanged result"

            state = SimpleNamespace(queued_ts=10.0, scheduled_ts=10.02,
                                    first_token_ts=10.15, last_token_ts=10.15, num_preemptions=0)
            self.assertEqual(Stats().update_from_finished_request(
                request_id="decision-native", req_stats=state, num_prompt_tokens=307), "unchanged result")
            row = self.rows(directory)[0]
            self.assertEqual(row["request_id"], "decision-native")
            self.assertAlmostEqual(row["engine_metrics"]["queue_seconds"], 0.02)
            self.assertAlmostEqual(row["engine_metrics"]["scheduled_to_output_seconds"], 0.13)
            self.assertEqual(row["engine_metrics"]["num_prompt_tokens"], 307)

    def test_worker_batch_ids_single_multiple_and_async_none(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.observer(directory)
            observe = namespace["observe"]

            class Worker:
                @observe("clef::joint_head")
                def head(self):
                    return "head output"

                @observe("clef::worker_execute_model")
                def execute_model(self, scheduler_output):
                    return self.head()

            worker = Worker()
            self.assertEqual(worker.execute_model(
                SimpleNamespace(num_scheduled_tokens={"decision-one-0": 307})), "head output")
            worker.execute_model(SimpleNamespace(num_scheduled_tokens={"decision-a-0": 100, "decision-b-0": 200}))
            worker.execute_model(None)
            rows = self.rows(directory)
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["request_id"] for row in rows[:2]], ["decision-one-0"] * 2)
            self.assertEqual(rows[1]["request_id_source"], "scheduler_output_single_request")
            for row in rows[2:4]:
                self.assertIsNone(row["request_id"])
                self.assertEqual(row["batch_request_ids"], ["decision-a-0", "decision-b-0"])
            for row in rows[4:]:
                self.assertIsNone(row["request_id"])
                self.assertEqual(row["batch_request_ids"], [])
            self.assertIsNone(namespace["_REQUEST_ID"].get())
            self.assertEqual(namespace["_BATCH_REQUEST_IDS"].get(), ())

    def test_cache_metadata_deduplicates_shared_storage(self):
        namespace = self.observer()

        class Storage:
            def data_ptr(self):
                return 4096

            def nbytes(self):
                return 1024

        class Tensor:
            shape = (128,)
            dtype = "bfloat16"
            device = "npu:0"

            def untyped_storage(self):
                return Storage()

            def stride(self):
                return (1,)

            def storage_offset(self):
                return 0

            def numel(self):
                return 128

            def element_size(self):
                return 2

        FullAttentionSpec = type("FullAttentionSpec", (), {"page_size_bytes": 256})
        MambaSpec = type("MambaSpec", (), {"page_size_bytes": 512, "state_content_size_bytes": 384,
                                          "shapes": ((128,), (64,)), "dtypes": ("bfloat16", "bfloat16")})
        config = SimpleNamespace(num_blocks=2, kv_cache_layout="hybrid", kv_cache_tensors=[],
            kv_cache_groups=[SimpleNamespace(layer_names=["attention"], kv_cache_spec=FullAttentionSpec()),
                             SimpleNamespace(layer_names=["mamba"], kv_cache_spec=MambaSpec())])
        runner = SimpleNamespace(kv_cache_config=config, kv_caches=[Tensor(), Tensor()],
            compilation_config=SimpleNamespace(static_forward_context={
                "attention": SimpleNamespace(kv_cache=Tensor()),
                "mamba": SimpleNamespace(kv_cache=(Tensor(), Tensor()))}))
        result = namespace["cache_metadata"](SimpleNamespace(model_runner=runner), config)
        self.assertEqual(result["unique_storage_bytes"], 1024)
        self.assertEqual(len(result["unique_storages"]), 1)
        self.assertEqual(result["unique_storage_bytes_by_kind"],
                         {"shared:FullAttentionSpec+MambaSpec": 1024})
        self.assertEqual(set(result["unique_storages"][0]["layers"]), {"attention", "mamba"})
        self.assertEqual(result["spec_by_layer"]["mamba"]["state_content_size_bytes"], 384)
        json.dumps(result, allow_nan=False)

    @staticmethod
    def fixture_sources(vllm_root, ascend_root):
        for source_root, relative, targets in (
                [(vllm_root, path, targets) for path, targets in instrumentation.TARGETS.items()] +
                [(ascend_root, path, targets) for path, targets in instrumentation.ASCEND_TARGETS.items()]):
            groups = defaultdict(list)
            for cls, function in targets:
                groups[cls].append(function)
            lines = ["import json\n"]
            for cls, functions in groups.items():
                indent = "    " if cls else ""
                if cls:
                    lines.append(f"class {cls}:\n")
                for function in functions:
                    lines.extend([f"{indent}def {function}(*args, **kwargs):\n",
                                  f"{indent}    return ('original', args, kwargs)\n"])
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(lines))
        (vllm_root / "vllm/profiler").mkdir(parents=True, exist_ok=True)
        profiler = ascend_root / "vllm_ascend/profiler/torch_npu_profiler.py"
        profiler.parent.mkdir(parents=True, exist_ok=True)
        profiler.write_text(
            "def create(profiler_config):\n"
            "    return profile(\n"
            "            profile_memory=profiler_config.torch_profiler_with_memory,\n"
            "    )\n")

    def test_apply_restore_and_concurrent_edit_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vllm_root, ascend_root = root / "vllm", root / "ascend"
            self.fixture_sources(vllm_root, ascend_root)
            initial = {str(path): path.read_bytes() for tree in (vllm_root, ascend_root)
                       for path in tree.rglob("*.py")}
            output = root / "observation"

            def git_head(command, **kwargs):
                return ("24420b7a341af37a9bc3186e6fc6062512e0aa58\n"
                        if Path(command[2]).resolve() == vllm_root.resolve() else
                        "89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d\n")

            command = ["profile_instrumentation.py", "--vllm-root", str(vllm_root),
                       "--ascend-root", str(ascend_root), "--output-dir", str(output), "--apply"]
            with patch.object(sys, "argv", command), patch.object(
                    instrumentation.subprocess, "check_output", side_effect=git_head), redirect_stdout(io.StringIO()):
                instrumentation.main()
            manifest_path = output / "instrumentation-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["applied"])
            self.assertEqual(manifest["marker_coverage"]["status"],
                             "static_targets_prepared_pending_runtime_trace_acceptance")
            helper = vllm_root / "vllm/profiler/clef_observe.py"
            self.assertTrue(helper.exists())
            for path in (*vllm_root.rglob("*.py"), *ascend_root.rglob("*.py")):
                compile(path.read_text(), str(path), "exec")
            target = Path(manifest["files"][1]["path"])
            applied = target.read_bytes()
            target.write_bytes(applied + b"# concurrent user edit\n")
            with self.assertRaisesRegex(ValueError, "Concurrent source edit"):
                instrumentation.restore(manifest_path)
            self.assertTrue(helper.exists())
            target.write_bytes(applied)
            with redirect_stdout(io.StringIO()):
                instrumentation.restore(manifest_path)
            self.assertFalse(helper.exists())
            self.assertEqual(initial, {str(path): path.read_bytes() for tree in (vllm_root, ascend_root)
                                       for path in tree.rglob("*.py")})

    def test_different_source_head_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                instrumentation.subprocess, "check_output", return_value="different-head\n"):
            with self.assertRaisesRegex(ValueError, "Pinned source HEAD differs"):
                instrumentation.prepare(Path(directory), Path(directory) / "ascend")


if __name__ == "__main__":
    unittest.main()
