"""Shared cgroup and host headroom for the private K32 candidate."""

from pathlib import Path
from time import monotonic
import re

HOST_NS = Path("/proc")


def proc_unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def memory_hierarchy(proc):
    paths = {}
    for line in (proc / "self/cgroup").read_text().splitlines():
        index, controllers, path = line.split(":", 2)
        if "memory" in controllers.split(","):
            paths["v1"] = Path(path)
        elif index == "0" and not controllers:
            paths["v2"] = Path(path)
    for wanted in ("v1", "v2"):
        if wanted not in paths:
            continue
        for line in (proc / "self/mountinfo").read_text().splitlines():
            lhs, rhs = line.split(" - ", 1)
            fields = rhs.split()
            kind = "v2" if fields[0] == "cgroup2" else (
                "v1" if fields[0] == "cgroup" and "memory" in fields[2].split(",") else None)
            if kind != wanted:
                continue
            spec = lhs.split()
            root, mount = map(lambda path: Path(proc_unescape(path)), spec[3:5])
            path = paths[wanted]
            if path.is_relative_to(root):
                leaf = mount / path.relative_to(root)
            elif path == Path("/"):
                leaf = mount
            else:
                continue
            name = "memory.max" if wanted == "v2" else "memory.limit_in_bytes"
            if (leaf / name).is_file():
                return wanted, tuple(p for p in (leaf, *leaf.parents) if p.is_relative_to(mount))
    if paths:
        raise ValueError("memory controller is not readable")
    return None, ()


class MemoryBudget:
    def __init__(self, requested, proc=HOST_NS, clock=monotonic, *, shared_reserve=None, partners=1):
        self.requested = requested
        self.shared_reserve = shared_reserve
        self.partners = partners
        self.proc = proc
        self.clock = clock
        self.events = None
        self.frozen_until = 0.0
        try:
            self.version, self.parents = memory_hierarchy(proc)
        except (OSError, ValueError):
            self.version, self.parents = "unknown", ()

    def __call__(self, already_retained):
        if self.version == "unknown":
            return 0
        try:
            host = {}
            for line in (self.proc / "meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    host[key] = int(value.split()[0]) * 1024
            reserve = max(1 << 30, min(4 << 30, host["MemTotal"] // 10))
            limits = [self.requested, already_retained + (host["MemAvailable"] - reserve) // self.partners]
            events = []
            for parent in self.parents:
                if self.version == "v2":
                    max_names, usage, event = ("memory.max", "memory.high"), "memory.current", "memory.events"
                else:
                    max_names, usage, event = ("memory.limit_in_bytes",), "memory.usage_in_bytes", "memory.failcnt"
                if parent==self.parents[-1] and parent!=self.parents[0] and not (parent / max_names[0]).exists():
                    continue
                use = int((parent / usage).read_text())
                for name in max_names:
                    raw = (parent / name).read_text().strip()
                    if raw != "max" and int(raw) < 1 << 60:
                        hard = int(raw)
                        headroom = (max(1 << 30, (hard * 15) // 100) if self.shared_reserve is None
                                    else self.shared_reserve)
                        limits.append(already_retained + (hard - use - headroom) // self.partners)
                if self.version == "v2":
                    row = dict(x.split() for x in (parent / event).read_text().splitlines())
                    events.append(sum(int(row.get(key, 0)) for key in ("high", "max", "oom", "oom_kill")))
                else:
                    events.append(int((parent / event).read_text()))
                    if parent == self.parents[0]:
                        stats = dict(x.split() for x in (parent / "memory.stat").read_text().splitlines())
                        hard = int(stats.get("hierarchical_memory_limit", 1 << 60))
                        if hard < 1 << 60:
                            headroom = (max(1 << 30, hard * 15 // 100) if self.shared_reserve is None
                                        else self.shared_reserve)
                            limits.append(already_retained + (hard - use - headroom) // self.partners)
            counters = tuple(events)
            if self.events is not None and any(a>b for a,b in zip(counters,self.events,strict=True)):
                self.frozen_until = self.clock()+5.0
            self.events = counters
            return max(0,min(limits)) if self.clock()>=self.frozen_until else 0
        except (OSError, KeyError, ValueError):
            return 0
