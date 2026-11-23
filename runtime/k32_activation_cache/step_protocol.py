"""A single, complete text pooling step and a bounded synchronous TP decision."""

from dataclasses import dataclass
import struct

from activation_cache import BOUNDARY, LAYERS, run_layers


@dataclass(frozen=True)
class TextStep:
    request_id: str
    token_ids: tuple[int, ...]
    manager_block_ids: tuple[tuple[int, ...], ...]


def fresh_text_step(scheduler_output, *, dummy_run=False, is_profile=False):
    """Admit only from fields distributed identically to every TP rank."""
    if dummy_run or is_profile or len(scheduler_output.scheduled_new_reqs) != 1:
        return None
    if (scheduler_output.scheduled_cached_reqs.req_ids
            or scheduler_output.scheduled_cached_reqs.resumed_req_ids
            or scheduler_output.scheduled_spec_decode_tokens
            or scheduler_output.scheduled_encoder_inputs
            or scheduler_output.encoder_input_updates
            or scheduler_output.preempted_req_ids
            or scheduler_output.kv_connector_metadata is not None
            or scheduler_output.has_sync_kv_loads):
        return None
    req = scheduler_output.scheduled_new_reqs[0]
    prompt = req.prompt_token_ids
    if (not prompt or req.mm_features or req.num_computed_tokens != 0
            or req.lora_request is not None or req.prompt_embeds is not None
            or req.replay_start != 0 or req.sampling_params is not None
            or getattr(req.pooling_params, "task", None) != "token_classify"
            or req.req_id in scheduler_output.finished_req_ids):
        return None
    if (req.prompt_is_token_ids is not None
            and (len(req.prompt_is_token_ids) != len(prompt)
                 or not all(req.prompt_is_token_ids))):
        return None
    if req.prefill_token_ids is not None and req.prefill_token_ids != prompt:
        return None
    if (scheduler_output.num_scheduled_tokens != {req.req_id: len(prompt)}
            or scheduler_output.total_num_scheduled_tokens != len(prompt)):
        return None
    return TextStep(req.req_id, tuple(prompt), tuple(tuple(ids) for ids in getattr(req, "block_ids", ())))


def vote_integers(key, ready):
    if key is not None and len(key) != struct.calcsize("32B"):
        raise ValueError("invalid TP content digest")
    return (int(key is not None), int(ready and key is not None), *(key or bytes(32)))


def vote_result(local, summed):
    """Both ranks observe the same count and must both match every key byte."""
    if len(local) != 34 or len(summed) != 34:
        raise ValueError("invalid TP vote vector")
    eligible = summed[0] == 2 and all(
        a * 2 == b for a, b in zip(local[2:], summed[2:], strict=True)
    )
    return eligible, eligible and summed[1] == 2


def run_candidate(layers, positions, input_pair, norm, *,
                  key, cache, restore, capture, exchange, stock, enabled=True,
                  observe_boundary=None, observed_layers=None):
    """Exchange readiness after the local copy, before the layer loop diverges.

    All tensor I/O is synchronous here. Restore must return private device
    tensors, or None on its recoverable failure path. Capture returns a
    reserved staged host value only when both copies completed.
    """
    if len(layers) != LAYERS:
        raise ValueError("K32 requires the audited 64-layer partition")
    loaded = None
    if enabled and key is not None:
        with cache.lease(key) as value:
            if value is not None:
                loaded = restore(value)
    eligible, resume = exchange(key, loaded is not None)
    if not eligible:
        loaded = None
        return stock(), False
    if not enabled:
        hidden, residual = run_layers(layers, positions, input_pair(), observe_boundary,
                                      observed=observed_layers)
        return norm(hidden, residual)[0], False
    if resume:
        if loaded is None:
            raise RuntimeError("positive TP vote without local restored tensors")
        hidden, residual = run_layers(layers, positions, loaded, start=BOUNDARY,
                                      observed=observed_layers)
        return norm(hidden, residual)[0], True
    # Release any local preallocated device copy before running the stock prefix.
    loaded = None
    cache.discard(key)

    # Every rank reaches the fill vote after layer 31. Publish after the
    # remaining layers and final norm finish successfully on this rank.
    staged = None
    accepted = False

    def tap(hidden, residual):
        nonlocal staged, accepted
        if key is not None:
            staged = capture(key, hidden, residual)
        _, accepted = exchange(key, staged is not None)

    try:
        hidden, residual = run_layers(layers, positions, input_pair(), tap, observed=observed_layers)
        output = norm(hidden, residual)[0]
        if accepted and staged is not None:
            cache.publish(*staged)
        return output, False
    finally:
        if staged is not None:
            cache.cancel(staged[0])
