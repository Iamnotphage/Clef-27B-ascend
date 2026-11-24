"""Normalize the structured-decisions ``state`` field for multimodal requests.

The 2026-11-24 competition protocol sends ERQA images as ``state`` content
parts (``[{"type": "image_url", ...}, {"type": "text", ...}]``) instead of the
historical top-level ``images`` list.  This module is deliberately free of
third-party imports so the exact same bytes can be installed next to the
pinned vLLM frontend as ``_clef_mm_state.py`` and imported by its worker
process without any extra ``PYTHONPATH`` entry.

Contract:

* A ``state`` value is multimodal only when it is a non-empty list whose every
  element is an object with a string ``type`` and at least one element has
  ``type == "image_url"``.  Every other value (``str``, ``dict``, ``None``,
  numbers, ``[]``, lists of strings, plain JSON objects, and lists without an
  ``image_url`` part) is returned unchanged, so the OpenDecisions text path is
  byte-for-byte identical to the pre-patch service.
* Once a list is recognized as multimodal it is fail-closed: a missing,
  empty or non-string image URL, a non-string text part, or an unsupported
  content type raises ``ValueError`` (the caller maps it to HTTP 400) instead
  of falling back to ``json.dumps`` (which would tokenize the base64 payload
  as text).
* ``extract_plain`` (default) joins the text parts with ``""``; official
  ERQA interleaving slices the question string by character index, so the
  join restores the original question exactly.  ``extract_question_dict``
  wraps that text as ``{"question": ...}``, reproducing the 2026-11-14
  baseline state.  ``interleave`` keeps the original ``(kind, value)``
  sequence for the conditional native-rendering arm; the installed frontend
  rejects it until that arm exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Literal, Sequence

MODE_ENV = "CLEF_STATE_MM_MODE"
DEFAULT_MODE = "extract_plain"
MODES = ("extract_plain", "extract_question_dict", "interleave")

Segment = tuple[Literal["text", "image"], str]

#: Data URLs the competition harness emits, plus the URL schemes the pinned
#: MediaConnector understands.  Decoding and network policy stay in vLLM.
_DATA_IMAGE = re.compile(
    r"^data:image/(?:jpeg|jpg|png|webp);base64,", re.IGNORECASE)
_URL_SCHEMES = ("http://", "https://", "file://")


@dataclass(frozen=True)
class NormalizedSystemOneState:
    """Result of :func:`normalize_systemone_state`.

    ``state`` is what the strategy's ``render_state`` receives, ``images`` is
    the merged media list (top-level images first, then state images), and
    ``segments`` preserves the original content order for the conditional
    native-interleaving arm.
    """

    state: Any
    images: list[str] | None
    segments: list[Segment] | None
    was_multimodal: bool
    mode: str


def resolve_mode(mode: str | None = None) -> str:
    """Return a validated mode, reading ``CLEF_STATE_MM_MODE`` when omitted."""
    value = mode if mode is not None else os.environ.get(MODE_ENV)
    value = DEFAULT_MODE if value is None or not str(value).strip() else str(value).strip()
    if value not in MODES:
        raise ValueError(
            f"unknown {MODE_ENV} mode {value!r}; expected one of {', '.join(MODES)}")
    return value


def is_multimodal_state_list(state: Any) -> bool:
    """True only for the well-formed new-protocol content-part list."""
    if not isinstance(state, list) or not state:
        return False
    for part in state:
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            return False
    return any(part.get("type") == "image_url" for part in state)


def _contains_image_url_part(state: Any) -> bool:
    return isinstance(state, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in state
    )


def extract_image_url(item: dict[str, Any]) -> str:
    """Validate one ``image_url`` part and return its stripped URL.

    The standard shape is ``{"type": "image_url", "image_url": {"url": ...}}``;
    the string shorthand ``{"type": "image_url", "image_url": "<url>"}`` is
    accepted defensively.  Optional fields such as ``detail`` are ignored.
    """
    if "image_url" not in item:
        raise ValueError('image_url content part is missing its "image_url" field')
    value = item["image_url"]
    if isinstance(value, dict):
        if "url" not in value:
            raise ValueError('image_url object is missing its "url" field')
        url = value["url"]
    elif isinstance(value, str):
        url = value
    else:
        raise ValueError("image_url must be an object with a string url or a string")
    if not isinstance(url, str):
        raise ValueError("image_url url must be a string")
    url = url.strip()
    if not url:
        raise ValueError("image_url url must not be empty")
    if not (_DATA_IMAGE.match(url)
            or any(url.lower().startswith(scheme) for scheme in _URL_SCHEMES)):
        raise ValueError(
            "unsupported image_url; expected a data:image/(jpeg|jpg|png|webp);"
            "base64 URL or an http://, https:// or file:// URL")
    return url


def extract_text(item: dict[str, Any]) -> str:
    if "text" not in item:
        raise ValueError('text content part is missing its "text" field')
    value = item["text"]
    if not isinstance(value, str):
        raise ValueError("text content part must contain a string")
    return value


def normalize_systemone_state(
    state: Any,
    existing_images: Sequence[str] | None = None,
    mode: str | None = None,
) -> NormalizedSystemOneState:
    """Normalize a request ``state`` and merge its images into the media list."""
    mode = resolve_mode(mode)
    images = list(existing_images) if existing_images else []
    if not is_multimodal_state_list(state):
        if _contains_image_url_part(state):
            # A malformed list that still carries an image part must not fall
            # back to json.dumps: the base64 payload would become prompt text.
            raise ValueError(
                "multimodal state list is malformed: every element must be an "
                "object with a string type")
        return NormalizedSystemOneState(
            state=state,
            images=images or None,
            segments=None,
            was_multimodal=False,
            mode=mode,
        )

    state_images: list[str] = []
    text_parts: list[str] = []
    segments: list[Segment] = []
    for part in state:
        kind = part["type"]
        if kind == "image_url":
            url = extract_image_url(part)
            state_images.append(url)
            segments.append(("image", url))
        elif kind == "text":
            text = extract_text(part)
            text_parts.append(text)
            segments.append(("text", text))
        else:
            raise ValueError(
                f"unsupported state content type {kind!r}; expected "
                '"image_url" or "text"')
    joined = "".join(text_parts)
    normalized_state: Any = (
        {"question": joined} if mode == "extract_question_dict" else joined)
    merged = [*images, *state_images]
    return NormalizedSystemOneState(
        state=normalized_state,
        images=merged or None,
        segments=segments,
        was_multimodal=True,
        mode=mode,
    )
