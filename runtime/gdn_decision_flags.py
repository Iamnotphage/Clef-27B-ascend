"""Runtime switches for the Clef decision-only GDN prefill path.

Every request in this service is one fresh full prefill and the model answers
in that single forward, so no decode step ever runs.  The GDN layers still
export and write back a recurrent state nobody reads and still write two
backward-only intermediates.  This module owns the switches that let the
prefill path skip that work.

``CLEF_GDN_DECISION_PREFILL`` is the process default and the only switch a
normal launch uses.  A runtime file override exists for experiments, but it is
**opt-in**: the service must be started with ``CLEF_GDN_FLAG_FILE=<path>``.
Without that variable no file is read, so a stray file can never change a
submission run.  The override lets an A/B alternate OFF/ON inside one service
process instead of comparing two service starts (the ordering confound that
invalidated the round-3 A/B).  The file is re-read at most once a second.

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

_POLL_SECONDS = 1.0
_cache: dict[str, object] = {
    "path": None,
    "at": 0.0,
    "mtime": None,
    "values": {},
}


def _flag_path() -> Path | None:
    value = os.environ.get("CLEF_GDN_FLAG_FILE")
    return Path(value) if value else None


def runtime_override_enabled() -> bool:
    """True only when the launch explicitly opted into the file override."""
    return _flag_path() is not None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _file_values() -> dict:
    path = _flag_path()
    if path is None:
        return {}
    now = time.monotonic()
    if _cache["path"] == str(path) and now - float(_cache["at"]) < _POLL_SECONDS:
        return _cache["values"]  # type: ignore[return-value]
    _cache["path"] = str(path)
    _cache["at"] = now
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _cache["mtime"] = None
        _cache["values"] = {}
        return {}
    if mtime != _cache["mtime"]:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
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
    _cache.update(path=None, at=0.0, mtime=None, values={})
