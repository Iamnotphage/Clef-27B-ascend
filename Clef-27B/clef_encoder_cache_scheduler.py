"""Opt-in encoder-output retention for the pinned synchronous pooling service.

vLLM 24420b7a + vLLM-Ascend 89eb8a5: BalanceScheduler calls Scheduler,
which builds MultiModalBudget and EncoderCacheManager in its constructor.
Set only the capacity before that call; inherit schedule() without changes.
Capacity counts merged encoder embeddings, not input patches or bytes.
This does not reserve HBM: runner tensors grow on misses and remain until
eviction. TP ranks retain full output vectors, including shared split storage.
"""

import json
import logging
import os

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


class EncoderCacheBudgetScheduler(BalanceScheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        config = vllm_config.scheduler_config
        if config.async_scheduling is not False:
            raise ValueError("Encoder cache candidate requires resolved async_scheduling=False")
        if vllm_config.model_config.runner_type != "pooling":
            raise ValueError("Encoder cache candidate supports pooling only")
        if not vllm_config.model_config.supports_multimodal_inputs:
            raise ValueError("Encoder cache candidate requires a multimodal model")

        requested = int(os.environ.get("CLEF_ENCODER_CACHE_TOKENS", "0"))
        compute_before = config.max_num_encoder_input_tokens
        capacity_before = config.encoder_cache_size
        if requested <= 0 or requested < max(compute_before, capacity_before):
            raise ValueError("Encoder cache capacity must be positive and at least the original budgets")

        # MultiModalBudget applies the model's max-item floor independently to
        # compute and cache. Changing a config field after super() is too late.
        config.encoder_cache_size = requested
        super().__init__(vllm_config, *args, **kwargs)

        # This parent delegates to upstream schedule() only with balance off.
        # Reject unknown parent implementations instead of changing scheduling.
        if getattr(self, "_balance_enabled", None) is not False:
            raise RuntimeError("Encoder cache candidate requires verified balance_scheduling=False")
        if config.async_scheduling is not False:
            raise RuntimeError("Encoder cache candidate requires async_scheduling=False after construction")
        manager = self.encoder_cache_manager
        expected_capacity = max(requested, self.max_num_encoder_input_tokens)
        if config.max_num_encoder_input_tokens != compute_before:
            raise RuntimeError("Encoder compute configuration changed during construction")
        if config.encoder_cache_size != requested:
            raise RuntimeError("Encoder cache configuration changed during construction")
        if (manager.cache_size, manager.num_free_slots, manager.num_freeable_slots) != (
            expected_capacity, expected_capacity, expected_capacity
        ):
            raise RuntimeError("Encoder cache manager was not initialized with the requested capacity")
        self.encoder_cache_budget_record = {
            "parent": BalanceScheduler.__module__ + "." + BalanceScheduler.__name__,
            "mro": [cls.__module__ + "." + cls.__name__ for cls in type(self).__mro__],
            "async_scheduling": config.async_scheduling,
            "balance_scheduling": self._balance_enabled,
            "compute_config": compute_before,
            "compute_effective": self.max_num_encoder_input_tokens,
            "cache_config_before": capacity_before,
            "cache_requested": requested,
            "cache_effective": manager.cache_size,
            "unit": "encoder_embeddings",
        }
        logging.getLogger("vllm.clef_encoder_cache").info(
            "Clef encoder cache candidate initialized: %s",
            json.dumps(self.encoder_cache_budget_record, sort_keys=True))
