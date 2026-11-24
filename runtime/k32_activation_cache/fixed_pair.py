"""Ordered full-prefill pairs for the first post-norm batching probe."""
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from struct import pack

from activation_cache import text_key


@dataclass(frozen=True)
class PackedPair:
    items: tuple
    offsets: tuple[int, int, int]
    step_key: bytes
    keys: tuple[bytes, bytes]


def pack_pair(items, req_ids, offsets, token_ids, position_axes, strides, domain):
    """Bind a private numeric domain to the actual ordered pair and layout."""
    by_id = {item.request_id:item for item in items}
    if len(by_id) != 2 or len(req_ids) != 2 or set(req_ids) != by_id.keys():
        return None
    ordered = tuple(by_id[name] for name in req_ids)
    lengths = tuple(len(item.token_ids) for item in ordered)
    cuts = (0,lengths[0],sum(lengths))
    if any(n < 1 for n in lengths) or cuts[2] > 4096 or tuple(offsets) != cuts:
        return None
    if tuple(token_ids) != ordered[0].token_ids + ordered[1].token_ids:
        return None
    axes = len(position_axes)
    if (axes not in (1,3) or len(strides) != (1 if axes==1 else 2)
            or strides[-1] != 1 or (axes==3 and strides[0] < cuts[2])
            or any(len(x)!=cuts[2] for x in position_axes)):
        return None
    item_keys = []
    for i, item in enumerate(ordered):
        p0,p1 = cuts[i:i+2]
        wanted = tuple(range(lengths[i]))
        positions = tuple(x for axis in position_axes for x in axis[p0:p1])
        if positions != wanted * axes:
            return None
        shape = (lengths[i],) if axes==1 else (3,lengths[i])
        item_keys.append(text_key(domain,item.token_ids,positions,shape,strides))
    # The physical layout and each occurrence are numerical provenance in this
    # probe. Reordering or replacing a partner never reuses a prior entry.
    step_key = sha256(b"clef-k64-fixed-pair-v1\0"+pack("<II",*lengths)+b"".join(item_keys)).digest()
    return PackedPair(ordered,cuts,step_key,
                      tuple(sha256(step_key+bytes([i])+key).digest() for i,key in enumerate(item_keys)))


def run_fixed_pair(pair, *, cache, restore, capture, exchange, stock, enabled):
    """Use one ordered TP decision per branch, with a private packed restore."""
    digest = pair.step_key if pair else None
    restored = None
    if pair and enabled:
        with ExitStack() as leases:
            retained = {key:leases.enter_context(cache.lease(key)) for key in dict.fromkeys(pair.keys)}
            values = tuple(retained[key] for key in pair.keys)
            if all(value is not None for value in values):
                try:
                    restored = restore(pair,values)
                except (MemoryError,RuntimeError):
                    restored = None
    eligible, hit = exchange(digest,restored is not None)
    if hit:
        if restored is None:
            raise RuntimeError("paired TP resume without locally restored rows")
        return restored,True
    restored = None
    result = stock()
    if not eligible or not enabled or pair is None:
        return result,False

    for key in dict.fromkeys(pair.keys):
        cache.discard(key)
    prepared = []
    try:
        ready = False
        tickets = ()
        try:
            for i,key in enumerate(pair.keys):
                staged = capture(key,pair.offsets[i],pair.offsets[i+1],result)
                if staged is None:
                    break
                try:
                    prepared.append(staged)
                except BaseException:
                    cache.cancel(staged[0])
                    raise
            if len(prepared)==len(pair.keys):
                tickets = tuple(ticket for ticket,_,_ in prepared)
                ready = cache.prepare_many(prepared)
        except (MemoryError,OSError,RuntimeError):
            ready = False
        _, accepted = exchange(digest,ready)
        if accepted and (not ready or not cache.reveal_many(tickets)):
            raise RuntimeError("paired TP publication without both prepared entries")
    finally:
        for ticket,_,_ in prepared:
            cache.cancel(ticket)
    return result,False
