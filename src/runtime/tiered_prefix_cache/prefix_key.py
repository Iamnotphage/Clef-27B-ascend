"""Semantic prefix split and collision-resistant keys for Tier 2.

The Tier 2 hybrid prefix-state cache keys on the *token-level aligned prefix*
``[0, L_aligned)`` of a decision prompt.  The fixed ``ClefFlashStrategy``
places the schema after the literal delimiter ``"\\n\\nSCHEMA FIELDS:\\n"``;
``L_aligned`` is that delimiter's token offset rounded down to 128 so it
matches the A3 kernel's chunk boundary and vLLM-Ascend's 128-token KV page.

Nothing here touches a device: the module is imported by the host store, the
scheduler shim and the offline unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable, Sequence

ALIGN = 128
MIN_PREFIX_TOKENS = 128
KEY_SCHEMA = b"clef-prefix-state-key/1"
MM_SCHEMA = b"clef-prefix-mm/1"


@dataclass(frozen=True)
class SchemaSpan:
    """An absolute ``[start, end)`` token span that must stay in the suffix."""

    start: int
    end: int
    kind: str = "option"


@dataclass(frozen=True)
class MmItem:
    """One multimodal item that belongs to the prefix.

    ``mm_hash`` is the content hash the media connector produced (hex).
    ``placeholder_offset`` and ``placeholder_length`` locate the item in the
    *unexpanded* token stream; ``expanded_offset`` and ``expanded_length``
    locate it after the image/video tokens replace the placeholder.  The key
    covers both so two images of equal placeholder size can never collide.
    """

    mm_hash: str
    placeholder_offset: int
    placeholder_length: int
    expanded_offset: int
    expanded_length: int
    modality: str = "image"


@dataclass(frozen=True)
class PrefixSplit:
    """The aligned split of one prompt."""

    total_tokens: int
    l_prefix: int
    l_aligned: int
    l_suffix: int
    prefix_token_ids: tuple[int, ...]
    suffix_token_ids: tuple[int, ...]
    mm_items: tuple[MmItem, ...] = ()


def find_last_subsequence(
    token_ids: Sequence[int], pattern: Sequence[int]
) -> int:
    """The index of the last occurrence of ``pattern``, or ``-1``."""
    if not pattern or len(pattern) > len(token_ids):
        return -1
    pattern = tuple(int(value) for value in pattern)
    limit = len(token_ids) - len(pattern)
    for index in range(limit, -1, -1):
        if tuple(token_ids[index:index + len(pattern)]) == pattern:
            return index
    return -1


def locate_state_prefix(
    token_ids: Sequence[int],
    schema_delimiter_ids: Sequence[int],
    *,
    align: int = ALIGN,
    min_prefix: int = MIN_PREFIX_TOKENS,
    schema_spans: Iterable[SchemaSpan] = (),
    mm_items: Sequence[MmItem] = (),
) -> PrefixSplit | None:
    """Split a prompt at the last schema delimiter and 128-align it.

    Returns ``None`` when the prompt has no delimiter, the delimiter starts
    before ``min_prefix``, or any schema span would cross into the prefix.
    ``schema_spans`` are absolute token spans; the joint head's negative spans
    are converted by the caller before this call.
    """
    token_ids = tuple(int(token) for token in token_ids)
    delimiter_at = find_last_subsequence(token_ids, schema_delimiter_ids)
    if delimiter_at < 0:
        return None
    l_aligned = (delimiter_at // align) * align
    # A boundary may fall inside one image's expanded block.  The runner has
    # already scattered the image's full vision features into ``inputs_embeds``
    # and ``positions``, so the suffix pass simply consumes the remaining
    # visual tokens; no half-image re-encoding happens.  The key therefore
    # includes every item that *starts* before the boundary (whole content
    # hash and span), which keeps prefixes from different images distinct even
    # when the cut lands inside the image.
    if l_aligned < min_prefix:
        return None
    for span in schema_spans:
        if not (l_aligned <= span.start < span.end <= len(token_ids)):
            return None
    prefix_items = tuple(sorted(
        (item for item in mm_items if item.expanded_offset < l_aligned),
        key=lambda item: item.expanded_offset))
    split = PrefixSplit(
        total_tokens=len(token_ids),
        l_prefix=delimiter_at,
        l_aligned=l_aligned,
        l_suffix=len(token_ids) - l_aligned,
        prefix_token_ids=token_ids[:l_aligned],
        suffix_token_ids=token_ids[l_aligned:],
        mm_items=prefix_items,
    )
    if not prefix_mm_matches(split, prefix_items):
        return None
    return split


def prefix_mm_matches(split: PrefixSplit, mm_items: Sequence[MmItem]) -> bool:
    """Every described item must *start* before the aligned boundary.

    An item may extend past ``L_aligned``: the runner has already materialised
    its full vision features in ``inputs_embeds``/``positions``, so the
    boundary can cut inside the expansion.  Its whole content hash and span
    are part of the key, which keeps two prefixes with different images
    distinct.  Items starting after the boundary, empty placeholders, and
    spans outside the prompt stay invalid.
    """
    for item in mm_items:
        if item.expanded_offset >= split.l_aligned:
            return False
        if item.expanded_offset + item.expanded_length > split.total_tokens:
            return False
        if item.placeholder_length <= 0:
            return False
    return True


def compute_prefix_key(
    namespace: str,
    split: PrefixSplit,
    *,
    model_revision: str | None = None,
    align: int = ALIGN,
    tp_size: int = 2,
) -> bytes:
    """A 32-byte SHA-256 identity of the aligned prefix.

    The key covers the namespace, the optional model revision, the alignment
    and TP partition, every prefix token id, and every prefix multimodal
    item's content hash and both span descriptions.  Image bytes never enter
    the key directly; the connector's content hash does.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("prefix key requires a non-empty namespace")
    digest = hashlib.sha256()
    digest.update(KEY_SCHEMA)
    _update_text(digest, namespace)
    _update_text(digest, model_revision or "")
    digest.update(struct.pack("<II", int(align), int(tp_size)))
    token_count = len(split.prefix_token_ids)
    digest.update(token_count.to_bytes(8, "little"))
    if token_count:
        digest.update(struct.pack(
            f"<{token_count}i", *split.prefix_token_ids))
    digest.update(len(split.mm_items).to_bytes(4, "little"))
    for item in split.mm_items:
        digest.update(MM_SCHEMA)
        _update_text(digest, item.modality)
        _update_text(digest, item.mm_hash)
        digest.update(struct.pack("<QQQQ", int(item.placeholder_offset),
                                  int(item.placeholder_length),
                                  int(item.expanded_offset),
                                  int(item.expanded_length)))
    return digest.digest()


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "little"))
    digest.update(encoded)
