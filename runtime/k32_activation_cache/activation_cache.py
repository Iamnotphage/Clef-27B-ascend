"""CPU primitives for the private, synchronous K32 activation probe."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
import os
from struct import pack
from threading import Lock


_boundary = os.environ.get("CLEF_PRIVATE_ACTIVATION_BOUNDARY", "32")
if _boundary not in {"32", "56"}:
    raise ValueError("Activation cache boundary must be 32 or 56")
BOUNDARY = int(_boundary)
LAYERS = 64
ENTRY_OVERHEAD = 1024


def namespace_digest(identity: dict) -> bytes:
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).digest()


def text_key(namespace, token_ids, positions, position_shape, position_stride):
    """Digest the full text input in token order, including its actual positions."""
    count = len(token_ids)
    shape = tuple(position_shape)
    stride = tuple(position_stride)
    if len(namespace) != 32 or count <= 0 or shape not in ((count,), (3, count)):
        raise ValueError("unexpected text activation key shape")
    axes = 1 if len(shape) == 1 else 3
    if len(stride) != len(shape) or len(positions) != axes * count:
        raise ValueError("unexpected position layout")
    digest = sha256(b"clef-text-bfloat16-boundary\0")
    digest.update(namespace)
    digest.update(pack("<IIQ", BOUNDARY, axes, count))
    digest.update(pack(f"<{len(stride)}q", *stride))
    digest.update(pack(f"<{count}I", *token_ids))
    digest.update(pack(f"<{len(positions)}q", *positions))
    return digest.digest()


@dataclass(frozen=True, eq=False)
class Reservation:
    key: bytes
    epoch: int
    payload_bytes: int

    @property
    def charge(self):
        return self.payload_bytes + ENTRY_OVERHEAD


@dataclass
class _Entry:
    value: object
    charge: int
    leases: int = 0
    resident: bool = True
    pending: Reservation | None = None
    next_published: int | None = None


class ActivationCache:
    """Per-rank state. A retained value is read only while its lease is held."""

    def __init__(self, budget):
        self._budget = budget
        self._lock = Lock()
        self._table = OrderedDict()
        self._inflight = set()
        self._live_bytes = 0
        self._epoch = 0
        self.evictions = 0
        self.published = 0

    def _evict(self):
        _, entry = self._table.popitem(last=False)
        self.evictions += 1
        self._detach(entry)

    def _detach(self, entry):
        entry.resident = False
        if not entry.leases:
            self._live_bytes -= entry.charge

    def _limit(self):
        try:
            return max(0, int(self._budget(self._live_bytes)))
        except (OSError, ValueError, TypeError):
            return 0

    def _trim(self, limit):
        while self._table and self._live_bytes > limit:
            self._evict()

    def reserve(self, key, payload_bytes):
        if not key or payload_bytes <= 0:
            raise ValueError("a fill requires a valid key and a positive size")
        with self._lock:
            ticket = Reservation(key, self._epoch, payload_bytes)
            limit = self._limit()
            if ticket.charge > limit:
                self._trim(limit)
                return None
            if key in self._table or any(t.key == key for t in self._inflight):
                self._trim(limit)
                return None
            self._trim(limit - ticket.charge)
            if self._live_bytes + ticket.charge > limit:
                return None
            self._inflight.add(ticket)
            self._live_bytes += ticket.charge
            return ticket

    def cancel(self, ticket):
        with self._lock:
            if ticket in self._inflight:
                self._inflight.remove(ticket)
                self._live_bytes -= ticket.charge
            entry = self._table.get(ticket.key)
            if entry is not None and entry.pending is ticket:
                self._detach(self._table.pop(ticket.key))

    def prepare(self, ticket, value, payload_bytes):
        """Store a hidden value and finalize its accounting before the vote."""
        with self._lock:
            if ticket not in self._inflight:
                return False
            limit = self._limit()
            self._trim(limit)
            if (ticket.epoch != self._epoch or ticket.key in self._table
                    or self._live_bytes > limit or value is None
                    or payload_bytes != ticket.payload_bytes):
                return False
            self._table[ticket.key] = _Entry(value, ticket.charge, pending=ticket,
                                             next_published=self.published + 1)
            self._inflight.remove(ticket)
            return True

    def reveal(self, ticket):
        """Commit by changing visibility, with no insertion or budget read."""
        with self._lock:
            entry = self._table.get(ticket.key)
            if entry is None or entry.pending is not ticket or ticket.epoch != self._epoch:
                return False
            # Asynchronous fills are serialized by the rank's request thread.
            self.published = entry.next_published
            entry.pending = None
            return True

    def publish(self, ticket, value, payload_bytes):
        """Publish only after a complete synchronous copy and positive TP vote."""
        with self._lock:
            if ticket not in self._inflight:
                return False
            self._inflight.remove(ticket)
            limit = self._limit()
            self._trim(limit)
            if (ticket.epoch != self._epoch or ticket.key in self._table
                    or self._live_bytes > limit or value is None
                    or payload_bytes != ticket.payload_bytes):
                self._live_bytes -= ticket.charge
                return False
            self._table[ticket.key] = _Entry(value, ticket.charge)
            self.published += 1
            return True

    @contextmanager
    def lease(self, key):
        with self._lock:
            limit = self._limit()
            self._trim(limit)
            entry = self._table.get(key)
            if entry is not None and entry.pending is not None:
                entry = None
            if entry is not None:
                entry.leases += 1
                self._table.move_to_end(key)
        try:
            yield entry.value if entry is not None else None
        finally:
            if entry is not None:
                with self._lock:
                    entry.leases -= 1
                    if not entry.resident and not entry.leases:
                        self._live_bytes -= entry.charge

    def invalidate(self):
        with self._lock:
            self._epoch += 1
            while self._table:
                self._evict()

    def discard(self, key):
        with self._lock:
            entry = self._table.pop(key, None)
            if entry is not None:
                self._detach(entry)

    def snapshot(self):
        with self._lock:
            pending = sum(entry.pending is not None for entry in self._table.values())
            return len(self._table) - pending, len(self._inflight) + pending, self._live_bytes


def run_layers(layers, positions, pair, capture=None, *, start=0, observed=None):
    """Preserve the original per-layer calls and full-row hidden/residual pair."""
    hidden_states, residual = pair
    for layer_idx, layer in enumerate(islice(layers, start, LAYERS), start=start):
        hidden_states, residual = layer(
            positions=positions, hidden_states=hidden_states, residual=residual
        )
        if observed is not None:
            observed.append(layer_idx)
        if capture is not None and layer_idx + 1 == BOUNDARY:
            capture(hidden_states, residual)
    return hidden_states, residual
