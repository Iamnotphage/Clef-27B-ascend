"""Async D2H capture and pipelined H2D restore for Tier 2.

Cold capture must not add latency to the response path: the prefix state is
copied into a small pinned staging ring on a dedicated NPU stream, and a
background thread publishes it only after the copy's event completes.

Hit restore is grouped (all 16 attention layers, then the 48 GDN layers) so a
copy stream can fill group ``i + 1`` while the compute stream runs group
``i``.  The caller waits per group; the last group's event is also waited by
``close()`` so no staged buffer is freed while a copy is in flight.

The module imports ``torch`` only; NPU support is detected at construction, so
the offline unit tests run on CPU with the synchronous fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Iterable, Mapping


def _npu_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


@dataclass
class StagedCopy:
    name: str
    tensor: Any
    event: Any = None


class PinnedRing:
    """A small ring of pinned host buffers for zero-cold-block D2H copies."""

    def __init__(self, slots: int = 3, slot_bytes: int = 512 << 20):
        if slots < 1:
            raise ValueError("slots must be positive")
        self.slots = slots
        self.slot_bytes = int(slot_bytes)
        self._free: list[Any] = []
        self._lock = threading.Lock()
        import torch  # noqa: PLC0415

        self._torch = torch
        for _ in range(slots):
            self._free.append(self._allocate())

    def _allocate(self):
        try:
            return self._torch.empty(self.slot_bytes, dtype=self._torch.uint8,
                                     pin_memory=True)
        except Exception:  # noqa: BLE001 - pageable fallback is measured slower
            return self._torch.empty(self.slot_bytes, dtype=self._torch.uint8)

    def acquire(self, index: int):
        with self._lock:
            while not self._free:
                time.sleep(0.0005)
            return self._free.pop()

    def release(self, buffer) -> None:
        with self._lock:
            self._free.append(buffer)


class AsyncCapture:
    """Copies a payload to pinned host memory and publishes it off-stream."""

    def __init__(self, ring: PinnedRing | None = None, *, synchronous: bool | None = None):
        self.ring = ring or PinnedRing()
        self.synchronous = (not _npu_available()) if synchronous is None else synchronous
        self._torch = self.ring._torch
        self._stream = None if self.synchronous else self._torch.npu.Stream()
        self._errors: list[BaseException] = []

    def submit(self, payload: Mapping[str, Any],
               on_ready: Callable[[dict[str, Any]], None]) -> threading.Thread:
        host = {}
        offset = 0
        buffer = self.ring.acquire(0)
        try:
            for name, tensor in payload.items():
                if tensor is None:
                    host[name] = None
                    continue
                nbytes = tensor.numel() * tensor.element_size()
                if offset + nbytes > buffer.numel():
                    raise RuntimeError("pinned ring slot overflow")
                view = buffer[offset:offset + nbytes].view(
                    tensor.dtype).view(tensor.shape)
                view.copy_(tensor.detach(), non_blocking=not self.synchronous)
                host[name] = view
                offset += nbytes
        except BaseException:
            self.ring.release(buffer)
            raise
        if self.synchronous:
            on_ready(host)
            self.ring.release(buffer)
            return threading.Thread(target=lambda: None, daemon=True)

        event = self._torch.npu.Event()
        with self._torch.npu.stream(self._stream):
            event.record()

        def _wait() -> None:
            try:
                event.synchronize()
                on_ready({name: (value.clone() if value is not None else None)
                          for name, value in host.items()})
            except BaseException as error:  # noqa: BLE001
                self._errors.append(error)
            finally:
                self.ring.release(buffer)

        thread = threading.Thread(target=_wait, name="clef-tiered-d2h", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        if self._stream is not None:
            self._torch.npu.synchronize()


class RestoredState:
    """Device-side state buffers filled on a copy stream, waited per group."""

    def __init__(self, tensors: dict[str, Any], events: dict[str, Any], stream):
        self.tensors = tensors
        self._events = events
        self._stream = stream
        self._closed = False

    def wait(self, group: Iterable[str]) -> None:
        for name in group:
            event = self._events.get(name)
            if event is not None:
                event.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for event in self._events.values():
            event.synchronize()


class Prefetcher:
    """Pipelined host-to-device copies of a prefix package."""

    def __init__(self, *, synchronous: bool | None = None):
        import torch  # noqa: PLC0415

        self._torch = torch
        self.synchronous = (not _npu_available()) if synchronous is None else synchronous
        self._stream = None if self.synchronous else torch.npu.Stream()

    def load(self, shard: Mapping[str, Any], device: str = "npu:0") -> RestoredState:
        tensors = {}
        events = {}
        if self.synchronous:
            for name, value in shard.items():
                tensors[name] = value.clone()
            return RestoredState(tensors, events, None)
        with self._torch.npu.stream(self._stream):
            for name, value in shard.items():
                target = value.to(device, non_blocking=True)
                tensors[name] = target
                event = self._torch.npu.Event()
                event.record()
                events[name] = event
        return RestoredState(tensors, events, self._stream)

    def close(self) -> None:
        if self._stream is not None:
            self._torch.npu.synchronize()
