"""CPU-only constructor contract; no vLLM, torch, or device imports."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ENTRY = Path(__file__).resolve().parents[1] / "Clef-27B" / "clef_encoder_cache_scheduler.py"
PARENT_MODULE = "vllm_ascend.patch.platform.patch_balance_schedule"


class ParentScheduler:
    calls = []
    wrong_capacity = False

    def __init__(self, vllm_config, *args, **kwargs):
        cfg = vllm_config.scheduler_config
        self._balance_enabled = getattr(cfg, "balance_scheduling", False)
        self.calls.append((cfg.encoder_cache_size, args, kwargs))
        self.max_num_encoder_input_tokens = max(
            cfg.max_num_encoder_input_tokens, vllm_config.max_item)
        capacity = max(cfg.encoder_cache_size, vllm_config.max_item)
        if self.wrong_capacity:
            capacity = 4096
        self.encoder_cache_manager = SimpleNamespace(
            cache_size=capacity, num_free_slots=capacity, num_freeable_slots=capacity)

    def schedule(self):
        return "parent schedule unchanged"


class EncoderCacheSchedulerTests(unittest.TestCase):
    def setUp(self):
        ParentScheduler.calls = []
        ParentScheduler.wrong_capacity = False
        modules = {name: ModuleType(name) for name in (
            "vllm_ascend", "vllm_ascend.patch", "vllm_ascend.patch.platform", PARENT_MODULE)}
        modules[PARENT_MODULE].BalanceScheduler = ParentScheduler
        with mock.patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("encoder_cache_cpu_test", ENTRY)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.scheduler = module.EncoderCacheBudgetScheduler
        self.environment = mock.patch.dict(os.environ, {"CLEF_ENCODER_CACHE_TOKENS": "327680"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def config(self, budget=4096, max_item=8192):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                async_scheduling=False, max_num_encoder_input_tokens=budget,
                encoder_cache_size=budget, max_num_batched_tokens=budget),
            model_config=SimpleNamespace(runner_type="pooling", supports_multimodal_inputs=True),
            max_item=max_item)

    def test_capacity_reaches_parent_before_manager_construction(self):
        config = self.config()
        scheduler = self.scheduler(config, "kv-cache", log_stats=True)
        self.assertEqual(ParentScheduler.calls, [(327680, ("kv-cache",), {"log_stats": True})])
        self.assertEqual(scheduler.encoder_cache_manager.cache_size, 327680)
        self.assertEqual(scheduler.max_num_encoder_input_tokens, 8192)
        self.assertEqual(config.scheduler_config.max_num_encoder_input_tokens, 4096)
        self.assertEqual(config.scheduler_config.max_num_batched_tokens, 4096)
        self.assertIs(self.scheduler.schedule, ParentScheduler.schedule)
        record = scheduler.encoder_cache_budget_record
        self.assertFalse(record["balance_scheduling"])
        self.assertFalse(record["async_scheduling"])
        self.assertEqual(record["compute_effective"], 8192)
        self.assertEqual(record["cache_effective"], 327680)
        self.assertEqual(record["parent"], ParentScheduler.__module__ + ".ParentScheduler")

    def test_erqa_budget_is_preserved_and_kwargs_signature_works(self):
        config = self.config(budget=16384, max_item=1014)
        scheduler = self.scheduler(vllm_config=config, kv_cache_config="kv")
        self.assertEqual(scheduler.max_num_encoder_input_tokens, 16384)
        self.assertEqual(config.scheduler_config.max_num_batched_tokens, 16384)

    def test_budget_record_uses_configured_vllm_logger(self):
        with self.assertLogs("vllm.clef_encoder_cache", level="INFO") as logs:
            self.scheduler(self.config())
        record = json.loads(logs.records[0].getMessage().split(": ", 1)[1])
        self.assertEqual(record["cache_effective"], 327680)
        self.assertEqual(record["compute_effective"], 8192)
        self.assertIs(record["async_scheduling"], False)
        self.assertIs(record["balance_scheduling"], False)

    def test_model_max_item_floor_still_applies(self):
        os.environ["CLEF_ENCODER_CACHE_TOKENS"] = "8192"
        scheduler = self.scheduler(self.config(max_item=16384))
        self.assertEqual(scheduler.encoder_cache_manager.cache_size, 16384)
        self.assertEqual(scheduler.max_num_encoder_input_tokens, 16384)

    def test_async_and_unresolved_async_rejected_before_parent(self):
        for value in (True, None):
            config = self.config()
            config.scheduler_config.async_scheduling = value
            with self.assertRaisesRegex(ValueError, "async_scheduling"):
                self.scheduler(config)
        self.assertEqual(ParentScheduler.calls, [])

    def test_invalid_or_shrinking_capacity_rejected_before_parent(self):
        for value in ("0", "-1", "bad", "4095"):
            os.environ["CLEF_ENCODER_CACHE_TOKENS"] = value
            with self.assertRaises(ValueError):
                self.scheduler(self.config())
        self.assertEqual(ParentScheduler.calls, [])

    def test_existing_larger_cache_is_preserved_by_rejection(self):
        config = self.config()
        config.scheduler_config.encoder_cache_size = 400000
        with self.assertRaises(ValueError):
            self.scheduler(config)
        self.assertEqual(config.scheduler_config.encoder_cache_size, 400000)

    def test_incompatible_model_rejected(self):
        config = self.config()
        config.model_config.runner_type = "generate"
        with self.assertRaisesRegex(ValueError, "pooling"):
            self.scheduler(config)
        config.model_config.runner_type = "pooling"
        config.model_config.supports_multimodal_inputs = False
        with self.assertRaisesRegex(ValueError, "multimodal"):
            self.scheduler(config)
        self.assertEqual(ParentScheduler.calls, [])

    def test_stale_manager_capacity_is_detected(self):
        ParentScheduler.wrong_capacity = True
        with self.assertRaisesRegex(RuntimeError, "manager"):
            self.scheduler(self.config())

    def test_balance_enabled_and_unknown_parent_state_rejected(self):
        for value in (True, None):
            config = self.config()
            config.scheduler_config.balance_scheduling = value
            with self.assertRaisesRegex(RuntimeError, "balance_scheduling"):
                self.scheduler(config)


if __name__ == "__main__":
    unittest.main()
