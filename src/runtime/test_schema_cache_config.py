#!/usr/bin/env python3
"""Check the schema-cache v2 switch and budgets from the serving constructor path."""

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/yangchen/work/src/vllm")
from vllm.entrypoints.openai.decisions import strategies  # noqa: E402

failures = []


def check(name, actual, expected):
    if actual != expected:
        failures.append(f"{name}: {actual!r} != {expected!r}")


class ModelConfig:
    def __init__(self, additional=None):
        self.additional_config = additional


def with_env(value, fn):
    old = os.environ.get("CLEF_SCHEMA_TOKEN_CACHE")
    if value is None:
        os.environ.pop("CLEF_SCHEMA_TOKEN_CACHE", None)
    else:
        os.environ["CLEF_SCHEMA_TOKEN_CACHE"] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("CLEF_SCHEMA_TOKEN_CACHE", None)
        else:
            os.environ["CLEF_SCHEMA_TOKEN_CACHE"] = old


check("version", strategies.SCHEMA_CACHE_VERSION, 2)

check(
    "defaults",
    with_env(None, lambda: strategies._schema_cache_config(ModelConfig())),
    (False, 4096, 4_000_000, 65_536),
)
check(
    "env-on",
    with_env("1", lambda: strategies._schema_cache_config(ModelConfig())),
    (True, 4096, 4_000_000, 65_536),
)
check(
    "env-off",
    with_env("0", lambda: strategies._schema_cache_config(ModelConfig())),
    (False, 4096, 4_000_000, 65_536),
)
check(
    "section-budgets",
    with_env("1", lambda: strategies._schema_cache_config(ModelConfig(
        {"schema_token_cache": {
            "enabled": True, "max_entries": 8,
            "max_tokens": 100, "max_key_bytes": 2048}}))),
    (True, 8, 100, 2048),
)
check(
    "section-disabled-overrides-env",
    with_env("1", lambda: strategies._schema_cache_config(ModelConfig(
        {"schema_token_cache": {"enabled": False}}))),
    (False, 4096, 4_000_000, 65_536),
)

cache = strategies._SchemaTokenCache(
    object(), True, max_entries=2, max_tokens=100, max_key_bytes=1024)
# Return-value isolation: mutating a hit must not change the stored tuple.
class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


tokenizer = Tokenizer()
first = cache.tokens(tokenizer, "abcd")
first[0] = -1
second = cache.tokens(tokenizer, "abcd")
check("return-isolation", second, [0, 1, 2, 3])
check("hit-count", cache.stats()["hits"], 1)

# Oversized key bypass never stores an entry.
long_text = "x" * 2000
ids = cache.tokens(tokenizer, long_text)
check("oversized-bypass-ids", ids, list(range(len(long_text))))
check("oversized-bypass-count", cache.stats()["oversized_skips"], 1)
check("oversized-bypass-entries", cache.stats()["entries"], 1)

# Entry eviction keeps the newest key and the token budget.
tiny = strategies._SchemaTokenCache(object(), True, max_entries=1, max_tokens=4)
tiny.tokens(tokenizer, "aa")
tiny.tokens(tokenizer, "bbb")
check("eviction-entries", tiny.stats()["entries"], 1)
check("eviction-token-total", tiny.stats()["token_total"], 3)

print("failures:", failures if failures else "none")
sys.exit(1 if failures else 0)
