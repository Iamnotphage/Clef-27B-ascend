"""Runtime switches for the Clef decision-only GDN prefill path.

Every request in this service is one fresh full prefill and the model answers
in that single forward, so no decode step ever runs.  The GDN layers still
gather and scatter a recurrent state on every layer and still write two
backward-only intermediates.  This module owns the switches that let the
prefill path skip that work.

``CLEF_GDN_DECISION_PREFILL`` is the process default, read like the other CLEF
switches.  ``/dev/shm/clef-gdn-flags.json`` overrides it at runtime, which lets
an A/B alternate OFF/ON inside one service process instead of comparing two
service starts (the ordering confound that invalidated the round-3 A/B).  The
file is re-read at most once a second.

Keys in the flag file:

``decision_prefill``
    Skip the final-state export, its transpose, and the state write-back.
``no_recompute``
    Pass ``disable_recompute=False`` to the fused prefill kernel so it stops
    allocating and writing ``g_cumsum``/``A``, which only the backward uses.
    Defaults to ``decision_prefill`` when neither the file nor
    ``CLEF_GDN_NO_RECOMPUTE`` sets it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FLAG_FILE = Path("/dev/shm/clef-gdn-flags.json")
_POLL_SECONDS = 1.0
_cache: dict[str, object] = {"at": 0.0, "mtime": None, "values": {}}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _file_values() -> dict:
    now = time.monotonic()
    if now - float(_cache["at"]) < _POLL_SECONDS:
        return _cache["values"]  # type: ignore[return-value]
    _cache["at"] = now
    try:
        mtime = FLAG_FILE.stat().st_mtime
    except OSError:
        _cache["mtime"] = None
        _cache["values"] = {}
        return {}
    if mtime != _cache["mtime"]:
        try:
            values = json.loads(FLAG_FILE.read_text(encoding="utf-8"))
            _cache["values"] = values if isinstance(values, dict) else {}
        except (OSError, ValueError):
            _cache["values"] = {}
        _cache["mtime"] = mtime
    return _cache["values"]  # type: ignore[return-value]


def decision_prefill_enabled() -> bool:
    """True when the prefill path may skip state export and write-back."""
    return bool(
        _file_values().get(
            "decision_prefill", _env_bool("CLEF_GDN_DECISION_PREFILL")
        )
    )


def no_recompute_enabled() -> bool:
    """True when the fused prefill kernel may skip backward-only outputs."""
    values = _file_values()
    if "no_recompute" in values:
        return bool(values["no_recompute"])
    if os.environ.get("CLEF_GDN_NO_RECOMPUTE") is not None:
        return _env_bool("CLEF_GDN_NO_RECOMPUTE")
    return decision_prefill_enabled()


def reset_cache() -> None:
    """Drop the cached view of the flag file (tests and the A/B driver)."""
    _cache.update(at=0.0, mtime=None, values={})
