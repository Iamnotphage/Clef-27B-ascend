#!/usr/bin/env python3
"""Apply the bounded schema-token cache patch to the pinned vLLM strategies.py.

Idempotent: refuses to run when the marker is already present. Revert with
git checkout -- vllm/entrypoints/openai/decisions/strategies.py (the file had
no pre-existing modifications; baseline SHA256 is recorded in the tuning run).
"""
import sys

PATH = "/workspace/yangchen/work/src/vllm/vllm/entrypoints/openai/decisions/strategies.py"
MARKER = "_SchemaTokenCache"

text = open(PATH).read()
if MARKER in text:
    print("already patched")
    sys.exit(0)

# 1. imports: add os and OrderedDict
old_import = """import asyncio
import hashlib
import json
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence"""
new_import = """import asyncio
import hashlib
import json
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence"""
assert old_import in text
text = text.replace(old_import, new_import, 1)

# 2. cache class before JointSchemaStrategy
anchor = "class JointSchemaStrategy(ReadStrategy):"
cache_class = '''class _SchemaTokenCache:
    """Bounded LRU over schema-fragment tokenization for the joint schema read.

    Keys carry the tokenizer identity and the encoding options together with
    the full rendered fragment; values are immutable token tuples; hits return
    an independent list so callers can mutate their copy freely. The request
    state is never routed through this cache. Disabled by default; enabled via
    additional_config {"schema_token_cache": {"enabled": true,
    "max_entries": int, "max_tokens": int}} or CLEF_SCHEMA_TOKEN_CACHE=1.
    """

    def __init__(
        self,
        tokenizer: Any,
        enabled: bool,
        max_entries: int,
        max_tokens: int,
    ) -> None:
        self._key_prefix = (
            str(getattr(tokenizer, "name_or_path", "") or ""),
            str(getattr(tokenizer, "revision", "") or ""),
            False,  # add_special_tokens=False at the schema call sites
        )
        self._enabled = enabled
        self._entries: OrderedDict[tuple, tuple[int, ...]] = OrderedDict()
        self._max_entries = max(1, max_entries)
        self._max_tokens = max(1, max_tokens)
        self._token_total = 0
        self.hits = 0
        self.misses = 0
        self.evicted_entries = 0
        self.evicted_tokens = 0

    def tokens(self, tokenizer: Any, text: str) -> list[int]:
        if not self._enabled:
            self.misses += 1
            return tokenizer.encode(text, add_special_tokens=False)
        key = (self._key_prefix, text)
        found = self._entries.get(key)
        if found is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return list(found)
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        self._entries[key] = ids
        self._token_total += len(ids)
        self.misses += 1
        while len(self._entries) > self._max_entries or (
            self._token_total > self._max_tokens and len(self._entries) > 1
        ):
            _, evicted = self._entries.popitem(last=False)
            self._token_total -= len(evicted)
            self.evicted_entries += 1
            self.evicted_tokens += len(evicted)
        return list(ids)

    def stats(self) -> dict[str, int]:
        return {
            "enabled": int(self._enabled),
            "entries": len(self._entries),
            "token_total": self._token_total,
            "hits": self.hits,
            "misses": self.misses,
            "evicted_entries": self.evicted_entries,
            "evicted_tokens": self.evicted_tokens,
        }


def _schema_cache_config(model_config: Any) -> tuple[bool, int, int]:
    """Cache switch and budgets from additional_config, with an env fallback."""
    additional = getattr(model_config, "additional_config", None) or {}
    section = additional.get("schema_token_cache")
    env = os.environ.get("CLEF_SCHEMA_TOKEN_CACHE")
    if isinstance(section, dict):
        enabled = bool(section.get("enabled", False))
    else:
        enabled = env not in (None, "", "0", "false", "False")
    max_entries = 4096
    max_tokens = 4_000_000
    if isinstance(section, dict):
        max_entries = int(section.get("max_entries", max_entries))
        max_tokens = int(section.get("max_tokens", max_tokens))
    return enabled, max_entries, max_tokens


class JointSchemaStrategy(ReadStrategy):'''
assert anchor in text
text = text.replace(anchor, cache_class, 1)

# 3. init: build the cache after prefix/suffix
old_init_tail = '''        self.suffix = self._tokens(
            "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
            "JOINT SCHEMA DECISIONS:"
        )'''
new_init_tail = old_init_tail + '''
        cache_enabled, cache_entries, cache_tokens = _schema_cache_config(
            context.engine_client.model_config
        )
        self._schema_cache = _SchemaTokenCache(
            self.tokenizer, cache_enabled, cache_entries, cache_tokens
        )
        self._encode_calls = 0'''
assert old_init_tail in text
text = text.replace(old_init_tail, new_init_tail, 1)

# 4. encode(): route schema fragments through the cache; state stays direct
replacements = [
    ('        schema_ids = self._tokens("\\n\\nSCHEMA FIELDS:\\n")',
     '        schema_ids = self._schema_cache.tokens(self.tokenizer, "\\n\\nSCHEMA FIELDS:\\n")'),
    ('''            schema_ids += self._tokens(
                f"\\nFIELD {n}\\nID: {q.id}\\nTYPE: {q.type.name}\\nINSTRUCTION: "
            )''',
     '''            schema_ids += self._schema_cache.tokens(
                self.tokenizer,
                f"\\nFIELD {n}\\nID: {q.id}\\nTYPE: {q.type.name}\\nINSTRUCTION: ",
            )'''),
    ('            schema_ids += self._tokens(q.instructions or q.id)',
     '            schema_ids += self._schema_cache.tokens(self.tokenizer, q.instructions or q.id)'),
    ('            schema_ids += self._tokens("\\nALLOWED OPTIONS:\\n")',
     '            schema_ids += self._schema_cache.tokens(self.tokenizer, "\\nALLOWED OPTIONS:\\n")'),
    ('                schema_ids += self._tokens(f"OPTION {m}: ")',
     '                schema_ids += self._schema_cache.tokens(self.tokenizer, f"OPTION {m}: ")'),
    ('                schema_ids += self._tokens(self._render(semantics))',
     '                schema_ids += self._schema_cache.tokens(self.tokenizer, self._render(semantics))'),
    ('                schema_ids += self._tokens("\\n")',
     '                schema_ids += self._schema_cache.tokens(self.tokenizer, "\\n")'),
    ('            schema_ids += self._tokens("END FIELD\\n")',
     '            schema_ids += self._schema_cache.tokens(self.tokenizer, "END FIELD\\n")'),
]
for old, new in replacements:
    assert old in text, old[:60]
    text = text.replace(old, new, 1)

# 5. state tokenization stays uncached; annotate the call site
old_state = '        state_ids = self._tokens(state)[: self.max_length - fixed]'
new_state = ('        # The request state is per-request content; it never uses the cache.\n'
             '        state_ids = self._tokens(state)[: self.max_length - fixed]')
assert old_state in text
text = text.replace(old_state, new_state, 1)

# 6. periodic stats log at the end of encode()
old_return = '''        return prompt, schema, orders

    async def read('''
new_return = '''        self._encode_calls += 1
        if self._encode_calls % 500 == 0:
            logger.info(
                "schema token cache stats after %d encodes: %s",
                self._encode_calls, self._schema_cache.stats(),
            )
        return prompt, schema, orders

    async def read('''
assert old_return in text
text = text.replace(old_return, new_return, 1)

open(PATH, "w").write(text)
print("patched", PATH)
