"""Single-slot K56 D2H fill. Only the request thread owns this state."""

from contextlib import nullcontext
from dataclasses import dataclass


WIDTH = 5120
MAX_ROWS = 2048
SNAPSHOT_BYTES = 2 * MAX_ROWS * WIDTH * 2
# Back the two views with one explicitly charged, power-of-two allocation.
PINNED_BYTES = 64 << 20
ND_FORMAT = 2


@dataclass
class _Flight:
    ticket: object
    source: tuple
    producer: object
    snapshot: tuple = ()
    destinations: tuple = ()
    ready: object = None
    done: object = None
    done_recorded: bool = False
    producer_submitted: bool = False
    copy_submitted: bool = False


class DeferredFill:
    def __init__(self, torch, get_npu_format, budget):
        self.torch = torch
        self.npu = torch.npu
        self.get_npu_format = get_npu_format
        self.budget = budget
        self.pinned = None
        self.pinned_ready = False
        self.pinned_views = ()
        self.copy_stream = None
        self.flight = None
        self.cache = None
        self.span = nullcontext
        self.inspect = None

    def entry_limit(self, charged):
        # Pinned memory is credited to the cgroup only after allocation, but
        # its full allocation and the device snapshot allowance always occupy
        # part of the rank's hard cap.
        pinned = PINNED_BYTES if self.pinned is not None else 0
        return max(0, self.budget(charged + pinned) - PINNED_BYTES - SNAPSHOT_BYTES)

    def _init_slot(self, device):
        if self.copy_stream is None:
            self.copy_stream = self.npu.Stream(device=device)
        if self.copy_stream.device != device:
            raise RuntimeError("K56 copy stream belongs to another device")
        if self.pinned is None:
            self.pinned = self.torch.empty(PINNED_BYTES // 2, device="cpu",
                                           dtype=self.torch.bfloat16, pin_memory=True)
            matrices = self.pinned[:2 * MAX_ROWS * WIDTH].view((2 * MAX_ROWS, WIDTH))
            self.pinned_views = matrices[:MAX_ROWS], matrices[MAX_ROWS:]
            self.pinned_ready = (self.pinned.untyped_storage().nbytes() == PINNED_BYTES
                                 and all(view.is_pinned() for view in self.pinned_views))
        if not self.pinned_ready:
            raise RuntimeError("K56 pinned allocation did not satisfy the slot contract")

    def _valid_pair(self, hidden, residual):
        return (
            residual is not None and hidden.device.type == "npu"
            and residual.device == hidden.device and 0 < hidden.shape[0] <= MAX_ROWS
            and all(x.dtype == self.torch.bfloat16
                    and tuple(x.shape) == (hidden.shape[0], WIDTH)
                    and tuple(x.stride()) == (WIDTH, 1)
                    and self.get_npu_format(x) == ND_FORMAT for x in (hidden, residual))
        )

    def begin(self, key, hidden, residual):
        if self.flight is not None:
            raise RuntimeError("K56 async fill still owns a previous flight")
        try:
            if not self._valid_pair(hidden, residual):
                return None
            producer = self.npu.current_stream(hidden.device)
        except (RuntimeError, MemoryError):
            return None
        payload = 2 * hidden.numel() * hidden.element_size()
        ticket = self.cache.reserve(key, payload)
        if ticket is None:
            return None
        flight = self.flight = _Flight(ticket, (hidden, residual), producer)
        try:
            self._init_slot(hidden.device)
            if self.copy_stream == producer:
                raise RuntimeError("K56 snapshot and copy must use distinct streams")
            flight.destinations = tuple(x[:hidden.shape[0]] for x in self.pinned_views)
            with self.span("k56_snapshot"):
                flight.snapshot = tuple(self.torch.empty_like(x) for x in flight.source)
                sources = {x.untyped_storage().data_ptr() for x in flight.source}
                snapshots = [x.untyped_storage() for x in flight.snapshot]
                pointers = {s.data_ptr() for s in snapshots}
                if (not self._valid_pair(*flight.snapshot) or len(pointers) != 2
                        or pointers & sources or any(s.nbytes() != payload // 2 for s in snapshots)):
                    raise RuntimeError("K56 snapshot geometry or storage alias mismatch")
                flight.producer_submitted = True
                for dst, src in zip(flight.snapshot, flight.source, strict=True):
                    dst.copy_(src, non_blocking=True)
                flight.ready = self.npu.Event()
                flight.ready.record(producer)
            with self.npu.stream(self.copy_stream), self.span("k56_enqueue_d2h"):
                flight.copy_submitted = True
                self.copy_stream.wait_event(flight.ready)
                for name, dst, src in zip(("k56_copy_hidden", "k56_copy_residual"),
                                          flight.destinations, flight.snapshot, strict=True):
                    with self.span(name):
                        dst.copy_(src, non_blocking=True)
                flight.done = self.npu.Event()
                flight.done.record(self.copy_stream)
                flight.done_recorded = True
            return ticket
        except (RuntimeError, MemoryError):
            self.discard(ticket)
            return None
        except BaseException:
            self.discard(ticket)
            raise

    def _retire(self):
        flight = self.flight
        if flight.done_recorded:
            try:
                with self.span("k56_wait_d2h"):
                    flight.done.synchronize()
                self.flight = None
                return
            except RuntimeError:
                # A broken event can still leave enqueued commands. Retire
                # both queues before reuse, and drop this fill regardless.
                flight.done_recorded = False
                self._drain_streams(flight)
                raise
        self._drain_streams(flight)

    def _drain_streams(self, flight):
        try:
            if flight.copy_submitted:
                self.copy_stream.synchronize()
            if flight.producer_submitted:
                flight.producer.synchronize()
        except RuntimeError as error:
            raise RuntimeError("K56 async flight could not retire; worker recovery required") from error
        self.flight = None

    def prepare(self, ticket):
        flight = self.flight
        if flight is None or flight.ticket is not ticket or not flight.done_recorded:
            raise RuntimeError("K56 preparation requires its completed enqueue")
        try:
            self._retire()
            with self.span("k56_pageable"):
                host = tuple(self.torch.empty_like(x, pin_memory=False).copy_(x)
                             for x in flight.destinations)
            if self.inspect is not None:
                self.inspect(flight, host)
            return self.cache.prepare(ticket, host, ticket.payload_bytes)
        except (RuntimeError, MemoryError):
            if self.flight is not None:
                raise
            return False

    def discard(self, ticket):
        if self.flight is not None and self.flight.ticket is ticket:
            self._retire()
        self.cache.cancel(ticket)
