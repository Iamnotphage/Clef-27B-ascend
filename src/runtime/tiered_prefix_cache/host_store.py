"""Budgeted host store for Tier 2 prefix state, with SHM-shared head memory.

Layout follows the measured design:

- TP-sharded data (16 Full-Attention KV layers and 48 GDN state layers) stays
  in each rank's own pinned/anonymous pool; rank 0 and rank 1 hold different
  heads and never duplicate each other.
- ``prefix_memory`` is identical on both ranks, so rank 0 stores it in one
  POSIX shared-memory object and rank 1 only maps it read-only.

The store is deliberately device- and framework-agnostic: a buffer is any
object exposing ``nbytes`` (a CPU ``torch.Tensor``, a ``bytes``, a numpy
array).  Eviction is reference-counted LRU: an entry with a live lease is
never reclaimed.  Admission supports ``second_hit`` (the production default,
so a cold first pass pays nothing) and ``always`` (known one-state-many-ask
workloads).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import atexit
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, Iterable, Mapping
import uuid

DEFAULT_BUDGET_GIB = 64
DEFAULT_SAFETY_GIB = 16
GHOST_CAPACITY = 65_536
ADMISSIONS = ("second_hit", "always")
EVICTION_POLICIES = ("keep_longest", "aged_keep_longest", "lru")
DEFAULT_ACTIVE_WINDOW = 500


def buffer_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        return int(element_size()) * int(numel())
    raise TypeError(f"cannot size host buffer of type {type(value).__name__}")


def shm_available() -> bool:
    return Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK)


@dataclass
class HostEntry:
    key: bytes
    shards: dict[int, dict[str, Any]] = field(default_factory=dict)
    prefix_memory: Any = None
    prefix_memory_shm: str | None = None
    prefix_memory_nbytes: int = 0
    owner: bool = False
    length: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    last_obs_seq: int = 0
    leases: int = 0

    def nbytes(self) -> int:
        total = 0
        for shard in self.shards.values():
            for value in shard.values():
                total += buffer_nbytes(value)
        total += self.prefix_memory_nbytes
        return total


class Lease:
    """A live reference that prevents eviction until released."""

    __slots__ = ("_store", "key", "_released")

    def __init__(self, store: "HostStore", key: bytes):
        self._store = store
        self.key = key
        self._released = False

    @property
    def entry(self) -> HostEntry | None:
        return self._store._entries.get(self.key)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._store._release(self.key)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class AttachedShm:
    """A bounded read-only view of one shared-memory object.

    The view must be released before the owner closes the segment; the store
    releases any outstanding views itself on eviction/close, so a caller that
    forgets ``release()`` cannot trigger ``BufferError``.
    """

    def __init__(self, store: "HostStore", name: str, size: int):
        self._store = store
        self.name = name
        self.size = int(size)
        shm = store._shm_objects[name][0]
        if not 0 <= self.size <= shm.size:
            raise ValueError("SHM slice exceeds the mapped segment")
        self._view = shm.buf[: self.size]

    @property
    def view(self) -> memoryview:
        if self._view is None:
            raise RuntimeError("shared-memory view already released")
        return self._view

    def read(self) -> bytes:
        return bytes(self.view)

    def release(self) -> None:
        if self._view is not None:
            self._view.release()
            self._view = None
        self._store._detach(self)

    def __enter__(self) -> "AttachedShm":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class HostStore:
    """Thread-safe, budget-bounded LRU of prefix state packages."""

    def __init__(
        self,
        *,
        budget_bytes: int,
        admission: str = "always",
        ghost_capacity: int = GHOST_CAPACITY,
        use_shm: bool | None = None,
        session: str | None = None,
        eviction_watermark: float = 0.92,
        eviction_policy: str = "keep_longest",
        active_window: int = 0,
    ):
        if admission not in ADMISSIONS:
            raise ValueError(f"unknown admission policy: {admission}")
        if eviction_policy not in EVICTION_POLICIES:
            raise ValueError(f"unknown eviction policy: {eviction_policy}")
        if eviction_policy == "aged_keep_longest" and active_window <= 0:
            active_window = DEFAULT_ACTIVE_WINDOW
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = int(budget_bytes)
        self.admission = admission
        self.eviction_policy = eviction_policy
        self.ghost_capacity = int(ghost_capacity)
        self.eviction_watermark = float(eviction_watermark)
        self.active_window = int(active_window)
        self.use_shm = shm_available() if use_shm is None else bool(use_shm)
        self.session = session or (
            f"{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}")
        self._lock = threading.RLock()
        self._entries: "OrderedDict[bytes, HostEntry]" = OrderedDict()
        self._ghost: "OrderedDict[bytes, None]" = OrderedDict()
        self._bytes = 0
        self._sequence = 0
        self._obs_seq = 0
        self._shm_objects: dict[str, tuple[Any, bool]] = {}
        self._attachments: dict[str, list[AttachedShm]] = {}
        self._closed = False
        self._install_cleanup()
        self.counters = {
            "hits": 0,
            "misses": 0,
            "populates": 0,
            "publications": 0,
            "evictions": 0,
            "rejections": 0,
            "ghost_evictions": 0,
            "shm_created": 0,
            "shm_attached": 0,
            "shm_unlinked": 0,
        }

    # ------------------------------------------------------------------
    # admission
    def _ages(self) -> None:
        self._obs_seq += 1

    def _is_stale(self, entry: HostEntry) -> bool:
        return (self.active_window > 0 and not entry.leases
                and self._obs_seq - entry.last_obs_seq > self.active_window)

    def _stale_entries(self) -> list[HostEntry]:
        if self.active_window <= 0:
            return []
        return [entry for entry in self._entries.values()
                if self._is_stale(entry)]

    def observe(self, key: bytes, length: int = 0, est_bytes: int = 0) -> str:
        """Record one sighting and return ``hit`` / ``populate`` / ``miss``.

        A resident entry keeps its age current on every sighting.  With
        ``keep_longest`` a full pool rejects a newcomer that is no longer than
        every resident prefix, so a sequential scan cannot churn the pool and
        the caller pays no populate cost.  The check predicts the publish
        decision with ``est_bytes`` (data bytes plus the state tail) instead of
        the post-eviction watermark, which a full pool can fall below after an
        eviction and thereby re-admit every round.  With aging enabled an
        entry nobody has touched for ``active_window`` sightings is stale:
        stale entries are evicted first and never trigger an early rejection,
        so a new dataset can reclaim the pool even when its prefixes are
        shorter than the departed dataset's.
        """
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("prefix key must be 32 bytes")
        with self._lock:
            self._ages()
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_obs_seq = self._obs_seq
                self.counters["hits"] += 1
                return "hit"
            if (self.eviction_policy != "lru" and length > 0
                    and self._bytes + max(int(est_bytes), 0)
                    > self.budget_bytes * self.eviction_watermark
                    and not self._stale_entries()):
                resident = [item.length for item in self._entries.values()
                            if item.length > 0]
                if resident and length <= min(resident):
                    self.counters["misses"] += 1
                    return "miss"
            if self.admission == "always":
                self.counters["populates"] += 1
                return "populate"
            if key in self._ghost:
                self._ghost.move_to_end(key)
                self.counters["populates"] += 1
                return "populate"
            self._remember_ghost(key)
            self.counters["misses"] += 1
            return "miss"

    def _remember_ghost(self, key: bytes) -> None:
        self._ghost[key] = None
        while len(self._ghost) > self.ghost_capacity:
            self._ghost.popitem(last=False)
            self.counters["ghost_evictions"] += 1

    # ------------------------------------------------------------------
    # membership and leases
    def contains(self, key: bytes) -> bool:
        with self._lock:
            return key in self._entries

    def acquire(self, key: bytes) -> Lease | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.leases += 1
            entry.last_used = time.monotonic()
            entry.last_obs_seq = self._obs_seq
            self._entries.move_to_end(key)
            return Lease(self, key)

    def _release(self, key: bytes) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.leases <= 0:
                raise RuntimeError("lease counter underflow")
            entry.leases -= 1

    def get(self, key: bytes) -> HostEntry | None:
        with self._lock:
            return self._entries.get(key)

    # ------------------------------------------------------------------
    # publish / evict
    def publish(
        self,
        key: bytes,
        shards: Mapping[int, Mapping[str, Any]],
        *,
        prefix_memory: Any = None,
        length: int = 0,
    ) -> bool:
        """Insert a package, evicting idle LRU entries to make room.

        Returns ``False`` when the package cannot fit without touching a
        leased entry; the caller must then fall back to the stock path.
        """
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("prefix key must be 32 bytes")
        with self._lock:
            if self._closed or key in self._entries:
                self.counters["rejections"] += 1
                return False
            entry = HostEntry(key=key, length=int(length),
                              last_obs_seq=self._obs_seq)
            entry.shards = {
                int(rank): {str(name): value for name, value in shard.items()}
                for rank, shard in shards.items()
            }
            if prefix_memory is not None:
                if self.use_shm:
                    entry.prefix_memory_shm, entry.prefix_memory_nbytes = (
                        self._create_shm(prefix_memory))
                    entry.owner = True
                else:
                    entry.prefix_memory = prefix_memory
                    entry.prefix_memory_nbytes = buffer_nbytes(prefix_memory)
                    entry.owner = True
            size = entry.nbytes()
            if size > self.budget_bytes:
                self.counters["rejections"] += 1
                self._discard_entry(entry)
                return False
            self._evict_for(size, entry.length)
            if self._bytes + size > self.budget_bytes:
                self.counters["rejections"] += 1
                self._discard_entry(entry)
                return False
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._bytes += size
            self._ghost.pop(key, None)
            self.counters["publications"] += 1
            return True

    def _evict_for(self, incoming: int, incoming_length: int = 0) -> None:
        target = int(self.budget_bytes * self.eviction_watermark)
        if self._bytes + incoming <= target:
            return
        if self.eviction_policy == "lru":
            for key in list(self._entries.keys()):
                if self._bytes + incoming <= target:
                    break
                entry = self._entries[key]
                if entry.leases:
                    continue
                self._entries.pop(key)
                self._bytes -= entry.nbytes()
                self._discard_entry(entry)
                self.counters["evictions"] += 1
            return
        # Stale entries belong to a scan that has moved on; evict them first,
        # oldest sighting first, regardless of length.  This is what frees a
        # pool left behind by a previous dataset.
        for entry in sorted(self._stale_entries(),
                            key=lambda item: item.last_obs_seq):
            if self._bytes + incoming <= target:
                return
            if entry.leases or self._entries.get(entry.key) is not entry:
                continue
            self._entries.pop(entry.key)
            self._bytes -= entry.nbytes()
            self._discard_entry(entry)
            self.counters["evictions"] += 1
        if self._bytes + incoming <= target:
            return
        # A short newcomer must never flush a longer resident prefix: a
        # sequential three-round scan would otherwise cycle the whole pool.
        lengths = [entry.length for entry in self._entries.values()
                   if entry.length > 0]
        if (incoming_length > 0 and lengths
                and incoming_length <= min(lengths)):
            return
        candidates = sorted(
            ((entry.length, entry.last_used, key)
             for key, entry in self._entries.items() if not entry.leases),
            key=lambda item: (item[0], item[1]))
        for _length, _used, key in candidates:
            if self._bytes + incoming <= target:
                break
            entry = self._entries.pop(key)
            self._bytes -= entry.nbytes()
            self._discard_entry(entry)
            self.counters["evictions"] += 1

    def _discard_entry(self, entry: HostEntry) -> None:
        if entry.prefix_memory_shm is not None and entry.owner:
            self._close_shm(entry.prefix_memory_shm)

    def drop(self, key: bytes) -> bool:
        """Explicitly evict an idle entry; leased entries are never dropped."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases:
                return False
            self._entries.pop(key)
            self._bytes -= entry.nbytes()
            self._discard_entry(entry)
            self.counters["evictions"] += 1
            return True

    # ------------------------------------------------------------------
    # POSIX shared memory for prefix_memory (identical on both TP ranks)
    def _create_shm(self, value: Any) -> tuple[str, int]:
        from multiprocessing import shared_memory  # noqa: PLC0415

        payload = _as_bytes(value)
        name = f"clef_pfx_{self.session}_{self._sequence:08d}"
        self._sequence += 1
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(payload))
        shm.buf[: len(payload)] = payload
        self._shm_objects[name] = (shm, True)
        self.counters["shm_created"] += 1
        return name, len(payload)

    def attach_shm(self, name: str, size: int) -> AttachedShm:
        """Map an existing segment read-only; the caller must release it."""
        from multiprocessing import shared_memory  # noqa: PLC0415

        with self._lock:
            record = self._shm_objects.get(name)
            if record is None:
                shm = shared_memory.SharedMemory(name=name, create=False)
                self._shm_objects[name] = (shm, False)
                self.counters["shm_attached"] += 1
                record = (shm, False)
            shm, _owned = record
            if not 0 <= int(size) <= shm.size:
                raise ValueError("SHM slice exceeds the mapped segment")
            attached = AttachedShm(self, name, int(size))
            self._attachments.setdefault(name, []).append(attached)
            return attached

    def read_shm_bytes(self, name: str, size: int) -> bytes:
        attached = self.attach_shm(name, size)
        try:
            return attached.read()
        finally:
            attached.release()

    def _detach(self, attached: AttachedShm) -> None:
        with self._lock:
            current = self._attachments.get(attached.name)
            if current and attached in current:
                current.remove(attached)
                if not current:
                    self._attachments.pop(attached.name, None)

    def _close_shm(self, name: str) -> None:
        record = self._shm_objects.pop(name, None)
        if record is None:
            return
        for attached in list(self._attachments.pop(name, [])):
            if attached._view is not None:
                attached._view.release()
                attached._view = None
        shm, owned = record
        try:
            shm.close()
            if owned:
                shm.unlink()
                self.counters["shm_unlinked"] += 1
        except FileNotFoundError:
            pass

    def _install_cleanup(self) -> None:
        store = self

        def _cleanup() -> None:
            store.close()

        atexit.register(_cleanup)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for key in list(self._entries.keys()):
                entry = self._entries.pop(key)
                self._discard_entry(entry)
            self._bytes = 0
            for name in list(self._shm_objects.keys()):
                self._close_shm(name)

    # ------------------------------------------------------------------
    # observability
    def shm_names(self) -> set[str]:
        """The SHM objects this store currently owns (live or staged)."""
        with self._lock:
            return set(self._shm_objects.keys())

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "admission": self.admission,
                "budget_bytes": self.budget_bytes,
                "bytes": self._bytes,
                "entries": len(self._entries),
                "ghost": len(self._ghost),
                "leases": sum(entry.leases for entry in self._entries.values()),
                "counters": dict(self.counters),
                "use_shm": self.use_shm,
                "shm_objects": len(self._shm_objects),
                "obs_seq": self._obs_seq,
                "active_window": self.active_window,
                "stale_entries": len(self._stale_entries()),
                "eviction_policy": self.eviction_policy,
            }


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "detach") and hasattr(value, "element_size"):
        import torch  # noqa: PLC0415

        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint8)
        return tensor.numpy().tobytes()
    view = getattr(value, "view", None)
    if callable(view):
        try:
            import numpy as np  # noqa: PLC0415

            return np.asarray(view(-1)).tobytes()
        except Exception:  # noqa: BLE001
            pass
    raise TypeError(f"cannot serialize host buffer of type {type(value).__name__}")


# ----------------------------------------------------------------------
# TP2 consensus, byte-identical to the verified K32 step protocol.
def tp_vote_vector(key: bytes | None, ready: bool) -> tuple[int, ...]:
    if key is not None and len(key) != 32:
        raise ValueError("invalid TP prefix digest")
    return (int(key is not None), int(ready and key is not None), *(key or bytes(32)))


def tp_vote_result(local: tuple[int, ...], summed: tuple[int, ...]) -> tuple[bool, bool]:
    """Both ranks must agree on presence and on every key byte."""
    if len(local) != 34 or len(summed) != 34:
        raise ValueError("invalid TP vote vector")
    eligible = summed[0] == 2 and all(
        value * 2 == total for value, total in zip(local[2:], summed[2:])
    )
    return eligible, eligible and summed[1] == 2


def _cgroup_anon_bytes() -> int | None:
    """Anonymous RSS from cgroup v2 ``memory.stat``.

    ``memory.current`` also counts reclaimable file page cache, which the
    service creates by reading the 54 GiB of safetensors; subtracting it from
    the budget would falsely disable the cache.
    """
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] in ("anon", "total_anon"):
                return int(fields[1])
    except (OSError, ValueError):
        return None
    return None


def read_memory_budget(default_gib: int = DEFAULT_BUDGET_GIB,
                       safety_gib: int = DEFAULT_SAFETY_GIB) -> dict:
    """Budget from cgroup limit minus anonymous RSS and a safety reserve."""
    limit = None
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit():
            limit = int(raw)
            break
    anon = _cgroup_anon_bytes()
    current = None
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError):
        pass
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    budget = default_gib * (1 << 30)
    used = anon if anon is not None else current
    if limit is not None and used is not None:
        budget = min(budget, max(0, limit - used - safety_gib * (1 << 30)))
    elif limit is not None:
        budget = min(budget, max(0, limit - safety_gib * (1 << 30)))
    if available is not None:
        budget = min(budget, max(0, available - safety_gib * (1 << 30)))
    return {
        "cgroup_bytes": limit,
        "cgroup_anon_bytes": anon,
        "cgroup_current_bytes": current,
        "mem_available_bytes": available,
        "budget_bytes": budget,
        "safety_gib": safety_gib,
    }
