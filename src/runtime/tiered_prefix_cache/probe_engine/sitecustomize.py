"""Install the M1 Tier 2 engine hooks when ``CLEF_TIER2_ENGINE=1``.

The worker imports the vLLM modules lazily; this sitecustomize waits for
them, then patches the four mount points through
``runtime.tiered_prefix_cache.engine_hooks``.  It is inert otherwise.
"""

from __future__ import annotations

import os
import sys
import threading
import time


def _enabled() -> bool:
    from pathlib import Path
    return (Path("/tmp/clef_tier2_engine").is_file()
            or os.environ.get("CLEF_TIER2_ENGINE", "0").strip().lower() in (
                "1", "true", "yes", "on"))


def _waiter() -> None:
    deadline = time.monotonic() + 600
    needed = (
        "vllm.model_executor.models.clef_flash",
        "vllm.model_executor.models.clef_joint_schema",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "vllm_ascend.attention.attention_v1",
        "vllm_ascend.worker.v2.model_runner",
        "vllm_ascend.ops.gdn",
    )
    while time.monotonic() < deadline:
        if all(name in sys.modules for name in needed):
            try:
                from runtime.tiered_prefix_cache.engine_hooks import install

                install()
            except Exception:  # noqa: BLE001 - never break the service
                import traceback

                traceback.print_exc()
            return
        time.sleep(0.2)


if _enabled():
    threading.Thread(target=_waiter, name="clef-tier2-engine",
                     daemon=True).start()
