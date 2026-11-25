"""Probe-only capture of production GDN prefill activations (sitecustomize).

Stage 0.2 probe B needs the real q/k/v/g/beta the A3 service feeds to its GDN
prefill kernel, plus the convolution boundary inputs, so the 1-pass vs 2-pass
split check runs on production tensors rather than random data.

The module is inert unless ``CLEF_TIERED_PROBE_DIR`` is set.  It never edits
the vLLM/vLLM-Ascend source tree: it waits until ``vllm_ascend.ops.gdn`` is
imported, then wraps

- ``AscendGatedDeltaNetAttention._forward_core`` to know the current layer;
- ``DeviceOperator.fla_gdn_prefill`` to capture the kernel operands;
- ``vllm_ascend.ops.gdn.causal_conv1d_fn`` to capture the last convolution
  inputs and weights.

An ``arm.json`` in the probe directory enables one request::

    {"arm_id": "od-0967-1", "request_name": "od-0967",
     "token_count": 967, "model_layers": [0, 8, 16, 24, 32, 40, 48, 56, 62],
     "split": 384}

Capture files land in ``<probe dir>/<arm_id>/`` as
``<request>-rank<r>-layer<l>-gdn.pt`` and ``...-conv.pt``.
"""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
import re
import threading
import time


_ARM_ID = contextvars.ContextVar("clef_tiered_probe_arm", default=None)
_LAYER = contextvars.ContextVar("clef_tiered_probe_layer", default=None)
_SAVED: set[tuple[str, int, int, str]] = set()
_LOCK = threading.Lock()
_PATCHED = False


def _debug(message: str) -> None:
    directory = _probe_dir()
    if directory is None:
        return
    try:
        with (directory / "capture-debug.log").open("a", encoding="utf-8") as log:
            log.write(f"{os.getpid()} {time.time():.3f} {message}\n")
    except Exception:  # noqa: BLE001
        pass


def _probe_dir() -> Path | None:
    value = os.environ.get("CLEF_TIERED_PROBE_DIR")
    return Path(value).expanduser().resolve() if value else None


def _read_arm() -> dict | None:
    directory = _probe_dir()
    if directory is None:
        return None
    path = directory / "arm.json"
    try:
        arm = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no arm, no capture
        return None
    if not isinstance(arm, dict) or not arm.get("arm_id"):
        return None
    return arm


def _current_layer() -> int | None:
    """The model layer of the enclosing GDN call, from the live stack.

    The production worker patches ``QwenGatedDeltaNetAttention._forward_core``
    with the Ascend implementation, so patching the Ascend class afterwards
    would not intercept the call.  Walking to the nearest ``self`` with a
    ``prefix`` attribute is implementation-agnostic and works for both the
    Base and the 310P adaptors.
    """
    import sys  # noqa: PLC0415

    frame = sys._getframe(1)
    for _ in range(8):
        if frame is None:
            break
        candidate = frame.f_locals.get("self")
        prefix = getattr(candidate, "prefix", None) if candidate is not None else None
        if isinstance(prefix, str):
            match = re.search(r"layers\.(\d+)", prefix)
            if match:
                return int(match.group(1))
        frame = frame.f_back
    return None


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - single rank fallback
        return 0


def _save_once(kind: str, arm: dict, layer: int, payload: dict) -> None:
    directory = _probe_dir()
    if directory is None:
        return
    rank = _rank()
    key = (str(arm["arm_id"]), int(layer), rank, kind)
    with _LOCK:
        if key in _SAVED:
            return
        _SAVED.add(key)
    import torch  # noqa: PLC0415 - sitecustomize stays import-light

    destination = directory / str(arm["arm_id"])
    destination.mkdir(parents=True, exist_ok=True)
    name = str(arm.get("request_name", "request"))
    path = destination / f"{name}-rank{rank}-layer{layer:02d}-{kind}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu(value):
    import torch  # noqa: PLC0415

    if value is None or not torch.is_tensor(value):
        return None
    return value.detach().to("cpu", copy=True)


def _install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import torch  # noqa: PLC0415
    import vllm_ascend.ops.gdn as gdn  # noqa: PLC0415
    from vllm_ascend.device.device_op import DeviceOperator  # noqa: PLC0415

    layer_pattern = re.compile(r"layers\.(\d+)")

    original_core = gdn.AscendGatedDeltaNetAttention._forward_core

    def forward_core(self, *args, **kwargs):
        match = layer_pattern.search(str(getattr(self, "prefix", "")))
        token = _LAYER.set(int(match.group(1)) if match else None)
        _debug(f"forward_core prefix={getattr(self, 'prefix', None)!r} layer={_LAYER.get()}")
        try:
            return original_core(self, *args, **kwargs)
        finally:
            _LAYER.reset(token)

    gdn.AscendGatedDeltaNetAttention._forward_core = forward_core

    descriptor = DeviceOperator.__dict__["fla_gdn_prefill"]
    original_prefill = (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor)

    def fla_gdn_prefill(q, k, v, g, beta, initial_state, scale, prebuilt_meta,
                        fused_fwd, *, decision_only=None, runtime=None,
                        qkv_pre_normalized=False):
        arm = _read_arm()
        layer = _LAYER.get() if _LAYER.get() is not None else _current_layer()
        capture = (
            arm is not None and layer is not None
            and int(layer) in {int(value) for value in arm.get("model_layers", [])}
            and int(q.shape[1]) == int(arm.get("token_count", -1))
        )
        if arm is not None:
            _debug(
                f"fla layer={layer} q_shape={tuple(q.shape)} v_shape={tuple(v.shape)} "
                f"arm={arm.get('arm_id')} want={arm.get('token_count')} "
                f"layers={arm.get('model_layers')} capture={capture}")
        if capture:
            try:
                _save_once("gdn", arm, int(layer), {
                    "request_name": arm.get("request_name"),
                    "layer": int(layer),
                    "rank": _rank(),
                    "tokens": int(q.shape[1]),
                    "q_shape": list(q.shape),
                    "v_shape": list(v.shape),
                    "q": _cpu(q),
                    "k": _cpu(k),
                    "v": _cpu(v),
                    "g": _cpu(g),
                    "beta": _cpu(beta),
                    "initial_state": _cpu(initial_state),
                    "scale": float(scale),
                    "decision_only": decision_only,
                    "qkv_pre_normalized": bool(qkv_pre_normalized),
                    "cu_seqlens_host": list(prebuilt_meta.cu_seqlens_host),
                    "chunk_indices_chunk64_host": list(
                        prebuilt_meta.chunk_indices_chunk64_host),
                    "keep_meta": _cpu(getattr(prebuilt_meta, "keep_meta", None)),
                    "chunk128": bool(getattr(prebuilt_meta, "chunk128", False)),
                })
            except Exception:  # noqa: BLE001 - capture must never break service
                pass
        return original_prefill(
            q, k, v, g, beta, initial_state, scale, prebuilt_meta, fused_fwd,
            decision_only=decision_only, runtime=runtime,
            qkv_pre_normalized=qkv_pre_normalized)

    setattr(DeviceOperator, "fla_gdn_prefill", staticmethod(fla_gdn_prefill))

    original_conv = gdn.causal_conv1d_fn

    def causal_conv1d_fn(x, weight, bias, **kwargs):
        output = original_conv(x, weight, bias, **kwargs)
        arm = _read_arm()
        layer = _LAYER.get() if _LAYER.get() is not None else _current_layer()
        if arm is not None and layer is not None:
            try:
                if int(layer) in {int(value) for value in arm.get("model_layers", [])}:
                    sequence = None
                    for axis, size in enumerate(x.shape):
                        if int(size) == int(arm.get("token_count", -1)):
                            sequence = axis
                            break
                    if sequence is not None:
                        boundary = int(arm.get("split", 0))
                        lower = max(0, boundary - 4)
                        upper = min(int(x.shape[sequence]), boundary + 8)
                        window = (
                            x if int(x.numel()) <= 64 * 1024 * 1024 else
                            x.index_select(
                                sequence,
                                torch.arange(lower, upper, device=x.device)))
                        _save_once("conv", arm, int(layer), {
                            "request_name": arm.get("request_name"),
                            "layer": int(layer),
                            "rank": _rank(),
                            "x_shape": list(x.shape),
                            "sequence_axis": sequence,
                            "window": [lower, upper],
                            "x": _cpu(window),
                            "weight": _cpu(weight),
                            "bias": _cpu(bias),
                            "kwargs": {
                                key: value for key, value in kwargs.items()
                                if isinstance(value, (str, int, float, bool, type(None)))
                            },
                        })
            except Exception:  # noqa: BLE001 - capture must never break service
                pass
        return output

    gdn.causal_conv1d_fn = causal_conv1d_fn
    _PATCHED = True
    _debug(f"hooks installed pid={os.getpid()}")
    print("[tiered-probe] capture hooks installed", flush=True)


def _waiter() -> None:
    deadline = time.monotonic() + 3600.0
    while time.monotonic() < deadline:
        if _PATCHED:
            return
        try:
            import sys

            if "vllm_ascend.ops.gdn" in sys.modules:
                _install()
                return
        except Exception:  # noqa: BLE001 - retry
            pass
        time.sleep(0.2)


if _probe_dir() is not None:
    threading.Thread(target=_waiter, name="clef-tiered-probe", daemon=True).start()
