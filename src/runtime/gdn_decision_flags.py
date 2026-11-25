"""Runtime switches for the Clef decision-only GDN prefill data path.

Every request in this service is one fresh full prefill and the model answers
in that single forward, so no decode step ever runs.  The GDN layers still
export and write back a recurrent state nobody reads, still write two
backward-only intermediates, still copy Q/K/V twice for the BNSD layout, still
gather a zero initial state, and still write a convolution state nobody reads.
This module owns the switches that let the prefill path skip that work.

``CLEF_GDN_DECISION_PREFILL`` is the process default and the only switch a
normal launch uses.  A runtime file override exists for experiments, but it is
**opt-in**: the service must be started with ``CLEF_GDN_FLAG_FILE=<path>``.
Without that variable no file is read, so a stray file can never change a
submission run.  The override lets an A/B alternate configurations inside one
service process instead of comparing two service starts (the ordering
confound that invalidated the round-3 A/B).  The file is re-read at most once
a second.

Keys in the flag file (all values must be JSON booleans; a non-boolean value is
ignored and reported once):

``epoch``
    Strictly increasing integer written by the experiment driver.  The service
    logs one host-side line per rank when it adopts a new epoch, which is how
    the driver proves both TP ranks loaded the same configuration.
``decision_prefill``
    Skip the final-state export, its transpose, and the state write-back.
``no_recompute``
    Pass ``disable_recompute=False`` to the fused prefill kernel so it stops
    allocating and writing ``g_cumsum``/``A``, which only the backward uses.
    Defaults to ``decision_prefill`` when neither the file nor
    ``CLEF_GDN_NO_RECOMPUTE`` sets it.
``bsnd``
    Pass sequence-major Q/K/V with ``layout="BSND"`` and normalize the
    head-first convolution storage before the layout swap.
``skip_initial_state``
    Pass ``initial_state=None`` to the fused kernel instead of gathering the
    (absent) initial recurrent state and transposing it.
``skip_conv_state``
    Call the prefill convolution with ``conv_states=None`` so it does not read
    or write the persistent convolution cache.
``preprocess_fusion``
    Reserved for the fused Ascend C preprocessing prototype.  It is recorded
    in the manifest and the epoch log; the prototype only honours it once its
    operator is installed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

try:  # keep the module importable by the plain unit tests
    from vllm.logger import init_logger

    _logger = init_logger(__name__)
except Exception:  # pragma: no cover - the vLLM import is optional in tests
    import logging

    _logger = logging.getLogger(__name__)

_POLL_SECONDS = 1.0
_BOOL_KEYS = (
    "decision_prefill",
    "no_recompute",
    "bsnd",
    "skip_initial_state",
    "skip_conv_state",
    "preprocess_fusion",
)
_ENV_KEYS = {
    "decision_prefill": "CLEF_GDN_DECISION_PREFILL",
    "no_recompute": "CLEF_GDN_NO_RECOMPUTE",
    "bsnd": "CLEF_GDN_BSND",
    "skip_initial_state": "CLEF_GDN_SKIP_INITIAL_STATE",
    "skip_conv_state": "CLEF_GDN_SKIP_CONV_STATE",
    "preprocess_fusion": "CLEF_GDN_PREPROCESS_FUSION",
}
_cache: dict[str, object] = {
    "path": None,
    "at": 0.0,
    "mtime": None,
    "values": {},
    "warned": set(),
}
_UNSET = object()
_snapshot: dict[str, object] = {
    "key": _UNSET,
    "flags": None,
    "logged_epoch": None,
}


@dataclass(frozen=True)
class GdnFlags:
    """One consistent view of every switch a forward may consult."""

    epoch: int = 0
    decision_prefill: bool = False
    no_recompute: bool = False
    bsnd: bool = False
    skip_initial_state: bool = False
    skip_conv_state: bool = False
    preprocess_fusion: bool = False

    def disabled(self) -> "GdnFlags":
        """Keep the epoch but turn every data-path specialization off."""
        return replace(
            self,
            decision_prefill=False,
            no_recompute=False,
            bsnd=False,
            skip_initial_state=False,
            skip_conv_state=False,
            preprocess_fusion=False,
        )

    def state_free(self) -> "GdnFlags":
        """Keep the decision/recompute switches, drop state-bearing skips."""
        return replace(
            self,
            bsnd=False,
            skip_initial_state=False,
            skip_conv_state=False,
            preprocess_fusion=False,
        )


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


def _warn_once(message: str, *args: object) -> None:
    warned = _cache["warned"]
    assert isinstance(warned, set)
    key = message % args if args else message
    if key in warned:
        return
    warned.add(key)
    _logger.warning(key)


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
        mtime = path.stat().st_mtime_ns
    except OSError:
        _cache["mtime"] = None
        _cache["values"] = {}
        return {}
    if mtime != _cache["mtime"]:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _warn_once("clef-gdn-runtime: unreadable flag file %s: %r", path, exc)
            values = {}
        if not isinstance(values, dict):
            _warn_once("clef-gdn-runtime: flag file %s is not a JSON object", path)
            values = {}
        _cache["values"] = values
        _cache["mtime"] = mtime
    return _cache["values"]  # type: ignore[return-value]


def _strict_bool(key: str, value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    _warn_once(
        "clef-gdn-runtime: %s must be a JSON boolean, got %r; ignoring it",
        key,
        value,
    )
    return None


def read_flags() -> GdnFlags:
    """Read the process defaults and the opt-in experiment file."""
    values = _file_values()
    resolved: dict[str, bool] = {}
    for key in _BOOL_KEYS:
        if key in values:
            parsed = _strict_bool(key, values[key])
            if parsed is not None:
                resolved[key] = parsed
                continue
        env_name = _ENV_KEYS[key]
        if os.environ.get(env_name) is not None:
            resolved[key] = _env_bool(env_name)
        else:
            resolved[key] = False
    if "no_recompute" not in values and os.environ.get(_ENV_KEYS["no_recompute"]) is None:
        resolved["no_recompute"] = resolved["decision_prefill"]
    epoch = values.get("epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        _warn_once("clef-gdn-runtime: epoch must be a non-negative integer, got %r", epoch)
        epoch = 0
    return GdnFlags(epoch=epoch, **resolved)


def decision_prefill_enabled() -> bool:
    """True when the prefill path may skip state export and write-back."""
    return read_flags().decision_prefill


def no_recompute_enabled() -> bool:
    """True when the fused prefill kernel may skip backward-only outputs."""
    return read_flags().no_recompute


def bsnd_enabled() -> bool:
    """True when Q/K/V may use the sequence-major BSND layout."""
    return read_flags().bsnd


def skip_initial_state_enabled() -> bool:
    """True when a fresh full prefill may pass no initial recurrent state."""
    return read_flags().skip_initial_state


def skip_conv_state_enabled() -> bool:
    """True when the prefill convolution may skip the persistent state cache."""
    return read_flags().skip_conv_state


def preprocess_fusion_enabled() -> bool:
    """True when the fused Ascend C preprocessing prototype is requested."""
    return read_flags().preprocess_fusion


def snapshot_for(key: object) -> GdnFlags:
    """Return one forward's flags, keyed by the vLLM forward context.

    vLLM constructs a fresh ``ForwardContext`` per forward pass, so every GDN
    layer of one request sees the same values even if the experiment file is
    rewritten between layers.  A new epoch is logged once per worker process;
    the driver uses those lines to prove both TP ranks adopted the switch.
    """
    if _snapshot["key"] is not key:
        flags = read_flags()
        _snapshot["key"] = key
        _snapshot["flags"] = flags
        if flags.epoch != _snapshot["logged_epoch"]:
            _snapshot["logged_epoch"] = flags.epoch
            _logger.info(
                "clef-gdn-runtime epoch=%d pid=%d decision_prefill=%d "
                "no_recompute=%d bsnd=%d skip_initial_state=%d "
                "skip_conv_state=%d preprocess_fusion=%d",
                flags.epoch,
                os.getpid(),
                int(flags.decision_prefill),
                int(flags.no_recompute),
                int(flags.bsnd),
                int(flags.skip_initial_state),
                int(flags.skip_conv_state),
                int(flags.preprocess_fusion),
            )
    return _snapshot["flags"]  # type: ignore[return-value]


def reset_cache() -> None:
    """Drop the cached view of the flag file (tests and the A/B driver)."""
    _cache.update(path=None, at=0.0, mtime=None, values={}, warned=set())
    _snapshot.update(key=_UNSET, flags=None, logged_epoch=None)
