"""Async D2H capture and pipelined H2D restore for Tier 2.

Cold capture must not add latency to the response path: the prefix state is
copied into a small pinned staging ring on a dedicated NPU stream, and a
single background consumer publishes it only after the copy's event completes.

The copy protocol is:

1. record ``ready`` on the caller's compute stream;
2. make the copy stream wait for ``ready``;
3. ``record_stream`` every source tensor on the copy stream so the caching
   allocator cannot recycle it before the D2H completes;
4. enqueue all copies, then record ``done`` on the copy stream.

The consumer waits ``done``, deep-copies the staging views into independent
pinned host tensors, calls ``on_ready`` and only then returns the ring slot.
Callbacks and consumer errors are surfaced through ``errors``; ``drain()``
waits for every queued publish without closing the capture.

Hit restore uses a two-slot pinned staging pool (default 384 MiB each) when a
package is pageable or a SHM view, so the H2D always travels a pinned DMA
path.  ``RestoredState.wait()`` uses a pure stream-to-stream ``wait_event`` by
default; only ``sync_host=True`` or ``close()`` blocks the host.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping


def _npu_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


class PinnedRing:
    """A condition-guarded ring of pinned host buffers.

    ``acquire`` never sleeps while holding the lock: a full ring returns
    ``None`` after ``timeout`` (Fail-Closed), and the caller records the drop.
    """

    def __init__(self, slots: int = 3, slot_bytes: int = 512 << 20):
        if slots < 1:
            raise ValueError("slots must be positive")
        self.slots = slots
        self.slot_bytes = int(slot_bytes)
        self._cond = threading.Condition()
        self._free: list[Any] = []
        self.dropped = 0
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

    def acquire(self, timeout: float | None = 0.05):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._free:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._cond.wait(remaining)
            return self._free.pop()

    def release(self, buffer) -> None:
        with self._cond:
            self._free.append(buffer)
            self._cond.notify()


class AsyncCapture:
    """Copies a payload to pinned host memory and publishes it off-stream."""

    def __init__(self, ring: PinnedRing | None = None, *,
                 synchronous: bool | None = None):
        self.ring = ring or PinnedRing()
        self.synchronous = (not _npu_available()) if synchronous is None else synchronous
        self._torch = self.ring._torch
        self._stream = None if self.synchronous else self._torch.npu.Stream()
        self.errors: list[BaseException] = []
        self.dropped = 0
        self._queue: "queue.Queue" = queue.Queue()
        self._closed = False
        self._snapshot_lock = threading.Lock()
        self._snapshot_free: dict[tuple, list[Any]] = {}
        self._snapshot_pool_bytes = int(self._snapshot_bytes_cap())
        self._consumer = None
        if not self.synchronous:
            self._consumer = threading.Thread(
                target=self._consume, name="clef-tiered-d2h", daemon=True)
            self._consumer.start()

    def _snapshot_bytes_cap(self) -> int:
        return 1 << 30

    def _persistent_host(self, tensor):
        """Pinned host storage, or pageable if the pin budget is exhausted.

        The Prefetcher detects pageable input and routes it through its
        two-slot pinned staging pool, so a failed pin degrades bandwidth but
        never correctness.
        """
        try:
            return self._torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=True)
        except Exception:  # noqa: BLE001 - memlock/CANN pin budget
            return self._torch.empty(tensor.shape, dtype=tensor.dtype)

    def _take_snapshot(self, source):
        key = (tuple(source.shape), source.dtype, str(source.device))
        with self._snapshot_lock:
            free = self._snapshot_free.get(key)
            if free:
                return free.pop()
        return self._torch.empty_like(source)

    def _return_snapshots(self, snapshots: list[Any]) -> None:
        with self._snapshot_lock:
            for snapshot in snapshots:
                used = sum(
                    tensor.numel() * tensor.element_size()
                    for values in self._snapshot_free.values()
                    for tensor in values)
                nbytes = snapshot.numel() * snapshot.element_size()
                if used + nbytes > self._snapshot_pool_bytes:
                    continue
                key = (tuple(snapshot.shape), snapshot.dtype, str(snapshot.device))
                self._snapshot_free.setdefault(key, []).append(snapshot)

    # ------------------------------------------------------------------
    def submit(self, payload: Mapping[str, Any],
               on_ready: Callable[[dict[str, Any]], None]):
        """Queue one publish; returns ``None`` when the ring is full/dropped."""
        if self.synchronous:
            try:
                host = self._synchronous_host(payload)
                on_ready(host)
            except BaseException as error:  # noqa: BLE001
                self.errors.append(error)
            return None
        buffer = self.ring.acquire(timeout=0.05)
        if buffer is None:
            self.dropped += 1
            self.ring.dropped += 1
            return None
        compute_stream = self._torch.npu.current_stream()
        ready = self._torch.npu.Event()
        ready.record(compute_stream)
        held_sources: list[Any] = []
        host: dict[str, Any] = {}
        offset = 0
        snapshots: list[tuple[str, Any]] = []
        try:
            with self._torch.npu.stream(self._stream):
                self._stream.wait_event(ready)
                # Phase 1: every device-to-device snapshot.  The compute
                # stream is ordered after snapshot_done only, never after the
                # host transfer below.
                for name, tensor in payload.items():
                    if tensor is None:
                        host[name] = None
                        continue
                    source = tensor.detach()
                    if source.device.type in ("npu", "privateuseone"):
                        source.record_stream(self._stream)
                    if not source.is_contiguous():
                        source = source.contiguous()
                        if source.device.type in ("npu", "privateuseone"):
                            source.record_stream(self._stream)
                    snapshot = self._take_snapshot(source)
                    snapshot.copy_(source, non_blocking=True)
                    snapshot.record_stream(self._stream)
                    held_sources.append(snapshot)
                    snapshots.append((name, snapshot))
                snapshot_done = self._torch.npu.Event()
                snapshot_done.record(self._stream)

                # Phase 2: queue all D2H copies after every snapshot.
                for name, snapshot in snapshots:
                    nbytes = snapshot.numel() * snapshot.element_size()
                    if offset + nbytes > buffer.numel():
                        raise RuntimeError("pinned ring slot overflow")
                    view = buffer[offset:offset + nbytes].view(
                        snapshot.dtype).view(snapshot.shape)
                    view.copy_(snapshot, non_blocking=True)
                    host[name] = view
                    offset += nbytes
                done = self._torch.npu.Event()
                done.record(self._stream)
        except BaseException:
            self.ring.release(buffer)
            raise
        compute_stream.wait_event(snapshot_done)
        self._queue.put((done, host, held_sources, buffer, on_ready))
        return done

    def _synchronous_host(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        host: dict[str, Any] = {}
        for name, tensor in payload.items():
            if tensor is None:
                host[name] = None
                continue
            persistent = self._persistent_host(tensor)
            persistent.copy_(tensor.detach())
            host[name] = persistent
        return host

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            done, host, held_sources, buffer, on_ready = item
            try:
                done.synchronize()
                persistent = {}
                for name, view in host.items():
                    if view is None:
                        persistent[name] = None
                        continue
                    target = self._persistent_host(view)
                    target.copy_(view)
                    persistent[name] = target
                on_ready(persistent)
            except BaseException as error:  # noqa: BLE001
                self.errors.append(error)
            finally:
                self._return_snapshots(held_sources)
                self.ring.release(buffer)
                self._queue.task_done()

    def drain(self, timeout: float | None = 30.0) -> bool:
        """Wait for every queued publish; returns False on timeout."""
        if self.synchronous:
            return True
        end = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.001)
        return True

    def close(self) -> None:
        if self.synchronous or self._closed:
            return
        self.drain(timeout=None)
        self._closed = True
        self._queue.put(None)
        self._consumer.join()
        self._torch.npu.synchronize()

    @property
    def pending(self) -> int:
        return 0 if self.synchronous else self._queue.unfinished_tasks


class RestoredState:
    """Device-side state buffers filled on a copy stream, waited per group."""

    def __init__(self, tensors: dict[str, Any], events: dict[str, Any],
                 stream, torch_module):
        self.tensors = tensors
        self._events = events
        self._stream = stream
        self._torch = torch_module
        self._closed = False

    def wait(self, group: Iterable[str], *, sync_host: bool = False) -> None:
        compute_stream = (
            None if self._stream is None
            else self._torch.npu.current_stream())
        for name in group:
            event = self._events.get(name)
            target = self.tensors.get(name)
            if event is None:
                continue
            if sync_host:
                event.synchronize()
            else:
                compute_stream.wait_event(event)
            if target is not None and self._stream is not None:
                target.record_stream(compute_stream)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for event in self._events.values():
            event.synchronize()


class Prefetcher:
    """Pipelined host-to-device copies with a small pinned staging pool."""

    def __init__(self, *, synchronous: bool | None = None,
                 staging_slots: int = 2,
                 staging_bytes: int = 384 << 20):
        import torch  # noqa: PLC0415

        self._torch = torch
        self.synchronous = (not _npu_available()) if synchronous is None else synchronous
        self._stream = None if self.synchronous else torch.npu.Stream()
        self._staging_slots = max(1, int(staging_slots))
        self._staging_bytes = int(staging_bytes)
        self._staging_cond = threading.Condition()
        self._free_staging: list[Any] = []
        self._reaper: "queue.Queue" = queue.Queue()
        self._reaper_thread = None
        self.errors: list[BaseException] = []
        if not self.synchronous:
            self._reaper_thread = threading.Thread(
                target=self._reap, name="clef-tiered-h2d", daemon=True)
            self._reaper_thread.start()

    # ------------------------------------------------------------------
    def load(self, shard: Mapping[str, Any], device: str = "npu:0") -> RestoredState:
        tensors: dict[str, Any] = {}
        events: dict[str, Any] = {}
        if self.synchronous:
            for name, value in shard.items():
                tensors[name] = value.clone()
            return RestoredState(tensors, events, None, self._torch)
        with self._torch.npu.stream(self._stream):
            for name, value in shard.items():
                target = self._torch.empty(
                    value.shape, dtype=value.dtype, device=device)
                pinned = bool(getattr(value, "is_pinned", lambda: False)())
                staging = None
                source = value
                if not pinned:
                    staging, source = self._stage(value)
                target.copy_(source, non_blocking=True)
                event = self._torch.npu.Event()
                event.record()
                tensors[name] = target
                events[name] = event
                if staging is not None:
                    self._reaper.put((event, staging))
        return RestoredState(tensors, events, self._stream, self._torch)

    def _stage(self, value):
        nbytes = int(value.numel() * value.element_size())
        staging = self._acquire_staging(nbytes)
        view = staging[:nbytes].view(value.dtype).view(value.shape)
        view.copy_(value)
        return staging, view

    def _acquire_staging(self, nbytes: int):
        with self._staging_cond:
            for index, slot in enumerate(self._free_staging):
                if slot.numel() >= nbytes:
                    return self._free_staging.pop(index)
        if nbytes > self._staging_bytes:
            return self._torch.empty(nbytes, dtype=self._torch.uint8,
                                     pin_memory=True)
        return self._torch.empty(self._staging_bytes, dtype=self._torch.uint8,
                                 pin_memory=True)

    def _reap(self) -> None:
        while True:
            item = self._reaper.get()
            if item is None:
                self._reaper.task_done()
                return
            event, staging = item
            try:
                event.synchronize()
                with self._staging_cond:
                    if staging.numel() <= self._staging_bytes and \
                            len(self._free_staging) < self._staging_slots:
                        self._free_staging.append(staging)
                        self._staging_cond.notify()
            except BaseException as error:  # noqa: BLE001
                self.errors.append(error)
            finally:
                self._reaper.task_done()

    def drain(self, timeout: float | None = 30.0) -> bool:
        if self.synchronous:
            return True
        end = None if timeout is None else time.monotonic() + timeout
        while self._reaper.unfinished_tasks:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.001)
        return True

    def close(self) -> None:
        if self.synchronous:
            return
        self.drain(timeout=None)
        self._reaper.put(None)
        self._reaper_thread.join()
        self._torch.npu.synchronize()
