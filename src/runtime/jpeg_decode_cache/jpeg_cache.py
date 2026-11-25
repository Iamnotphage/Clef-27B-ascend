
"""Private opt-in JPEG data-URL decode cache. No model outputs are retained."""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import monotonic
import re

from PIL import Image, __version__ as PILLOW_VERSION

GIB = 1 << 30
HOST_NS = Path("/proc")
MAX_ENTRY = 48 << 20
DECLARED_IMAGE = "28d32486298b8d3692c074b8214c458074753d42bf789276edf5cd68c41f1512"
DECLARED_HASHER = "5de69819f793f987d836bfdef30380254f939a9a62ee4ec32a8e7802d1083b36"


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
    def __init__(self, requested, proc=HOST_NS, clock=monotonic):
        self.requested = requested
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
            limits = [self.requested, already_retained + host["MemAvailable"] - reserve]
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
                        headroom = max(1 << 30, (hard * 15) // 100)
                        limits.append(already_retained + hard - use - headroom)
                if self.version == "v2":
                    row = dict(x.split() for x in (parent / event).read_text().splitlines())
                    events.append(sum(int(row.get(key, 0)) for key in ("high", "max", "oom", "oom_kill")))
                else:
                    events.append(int((parent / event).read_text()))
                    if parent == self.parents[0]:
                        stats = dict(x.split() for x in (parent / "memory.stat").read_text().splitlines())
                        hard = int(stats.get("hierarchical_memory_limit", 1 << 60))
                        if hard < 1 << 60:
                            limits.append(already_retained + hard - use - max(1 << 30, hard * 15 // 100))
            counters = tuple(events)
            if self.events is not None and any(a>b for a,b in zip(counters,self.events,strict=True)):
                self.frozen_until = self.clock()+5.0
            self.events = counters
            return max(0,min(limits)) if self.clock()>=self.frozen_until else 0
        except (OSError, KeyError, ValueError):
            return 0


def immutable(value):
    if type(value) in (type(None),bool,str,bytes,int,float):
        return True
    return type(value) is tuple and all(immutable(v) for v in value)


@dataclass(frozen=True, slots=True)
class FrozenRGB:
    size: tuple
    format: str | None
    image_info: tuple
    io_config: tuple | None
    pixels: bytes
    charged_bytes: int
    weight: int

    @classmethod
    def freeze(cls, result, raw, available):
        image = result.media
        if type(image) is not Image.Image or image.mode!="RGB" or image.palette is not None:
            return None
        info=list(image.info.items())
        io=None if result.io_config is None else tuple(result.io_config.items())
        for record in (info,io or ()):
            if not all(type(k) is str and immutable(v) for k,v in record):
                return None
        metadata=sum(len(k)+len(v) if isinstance(v,(bytes,str)) else len(k)+64
                     for k,v in info+(list(io) if io else []))
        if metadata>64<<10:
            return None
        raw_pixels=image.width*image.height
        weight=4*raw_pixels+metadata+4096
        if weight>min(MAX_ENTRY,available):
            return None
        with Image.open(BytesIO(raw)) as wrapped:
            if wrapped.format!="JPEG" or raw_pixels!=wrapped.width*wrapped.height:
                return None
            charged=wrapped.width*wrapped.height*4
        return cls(image.size,image.format,tuple(info),io,image.tobytes(),charged,weight)

    def clone(self, raw, media_with_bytes):
        media=Image.frombytes("RGB",self.size,self.pixels)
        media.info=dict(self.image_info)
        media.format=self.format
        return media_with_bytes(media,raw,None if self.io_config is None else dict(self.io_config))


class DecodedLRU:
    def __init__(self,budget,clock=monotonic):
        self.budget=budget
        self.clock=clock
        self.update_by=0.0
        self.capacity=0
        self.retained=0
        self.peak=0
        self.lookups=0
        self.hits=0
        self.generation=0
        self.cache=OrderedDict()
        self.lock=Lock()

    def refresh(self):
        now=self.clock()
        if now>=self.update_by:
            self.capacity=self.budget(self.retained)
            self.update_by=now+1.0
            if not self.capacity:
                self.generation+=1
            self.trim(0)

    def trim(self,incoming):
        while self.cache and self.retained+incoming>self.capacity:
            _,removed=self.cache.popitem(last=False)
            self.retained-=removed.weight

    def get(self,key):
        with self.lock:
            self.refresh()
            found=self.cache.get(key)
            self.lookups+=1
            if found is not None:
                self.hits+=1
                self.cache.move_to_end(key)
            return found,self.generation,self.capacity

    def add(self,key,generation,item):
        with self.lock:
            self.refresh()
            if item.weight>self.capacity or generation!=self.generation or key in self.cache:
                return
            self.trim(item.weight)
            self.cache[key]=item
            self.retained+=item.weight
            self.peak=max(self.peak,self.retained)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.retained=0
            self.generation+=1


def install(module,cache):
    source=Path(module.__file__).resolve()
    assert sha256(source.read_bytes()).hexdigest()==DECLARED_IMAGE
    assert sha256((source.parent.parent/"hasher.py").read_bytes()).hexdigest()==DECLARED_HASHER
    version=(PILLOW_VERSION,DECLARED_IMAGE,
             sha256((source.parent.parent/"image.py").read_bytes()).hexdigest())
    load_base64=module.ImageMediaIO.load_base64
    load_bytes=module.ImageMediaIO.load_bytes

    def cached_base64(self,media_type,encoded):
        compatible=media_type in ("image/jpeg","image/jpg") and self.image_mode=="RGB"
        if not compatible or set(self.kwargs)-{"rgba_background_color"}:
            return load_base64(self,media_type,encoded)
        raw=module.pybase64.b64decode(encoded,validate=True)
        if not raw.startswith(b"\xff\xd8\xff"):
            return load_bytes(self,raw)
        key=(sha256(raw).digest(),len(raw),version,self.image_mode,
             self.rgba_background_color,int(module.envs.VLLM_MAX_IMAGE_PIXELS))
        entry,generation,limit=cache.get(key)
        if entry is not None:
            if self.decoded_budget is not None:
                self.decoded_budget.charge(entry.charged_bytes)
            return entry.clone(raw,module.MediaWithBytes)
        result=load_bytes(self,raw)
        if limit:
            try:
                item=FrozenRGB.freeze(result,raw,limit)
            except (TypeError,ValueError,OSError,MemoryError):
                item=None
            if item is not None:
                cache.add(key,generation,item)
        return result

    module.ImageMediaIO.load_base64=cached_base64
    return load_base64
