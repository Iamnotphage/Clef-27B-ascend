"""Pure planning rules for the Tier 2A/2B cascade.

The engine hooks run exclusively on the NPU host, so every decision that can
be made without tensors lives here and is covered by offline unit tests:

* a **whole-batch** Tier 2A hit takes the representation-resume path;
* otherwise the step falls back to the existing Tier 2B cascade, and every
  request that will execute a full-length backbone pass becomes a Tier 2A
  capture candidate;
* a Tier 2A lease that was acquired locally but did not survive the TP2 vote
  is released, never silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

MODE_MISS = 0
MODE_POPULATE = 1
MODE_HIT = 2

PATH_REPRESENTATION = "representation"
PATH_STATE_RESUME = "state_resume"
PATH_FULL = "full"
PATH_CASCADE = "cascade"


@dataclass(frozen=True)
class BatchRepresentationPlan:
    """What one multi-request step should do with its Tier 2A candidates."""

    path: str
    capture_indices: tuple[int, ...] = ()
    demoted_hit_indices: tuple[int, ...] = ()


def choose_single_path(repr_mode: int | None, state_mode: int | None) -> str:
    """Cascade priority for one request.

    ``repr_mode``/``state_mode`` use the Tier 2 vote encoding: 2 hit, 1
    populate, 0 miss/populate-not-admitted, ``None`` ineligible.  A Tier 2A
    hit wins; otherwise a Tier 2B hit resumes the suffix; otherwise the full
    backbone runs.
    """
    if repr_mode == MODE_HIT:
        return PATH_REPRESENTATION
    if state_mode == MODE_HIT:
        return PATH_STATE_RESUME
    return PATH_FULL


def plan_batch(modes: Sequence[int | None]) -> BatchRepresentationPlan:
    """Plan one packed batch from the agreed per-request Tier 2A modes.

    ``modes`` is indexed by the runner's packed request order.  A single
    ineligible or non-hit request makes the whole step fall back to the Tier
    2B cascade: the representation-resume path is only bitwise-safe when every
    row of the packed step is restored at the same boundary.

    On the fallback path, locally-acquired hits are demoted and all requests
    with a populate/miss mode become capture candidates.  The caller removes
    the requests that end up taking a Tier 2B suffix resume, because that pass
    never materialises the full-length representation.
    """
    if not modes:
        return BatchRepresentationPlan(PATH_CASCADE)
    if all(mode == MODE_HIT for mode in modes):
        return BatchRepresentationPlan(PATH_REPRESENTATION)
    capture = tuple(index for index, mode in enumerate(modes)
                    if mode in (MODE_MISS, MODE_POPULATE))
    demoted = tuple(index for index, mode in enumerate(modes)
                    if mode == MODE_HIT)
    return BatchRepresentationPlan(PATH_CASCADE, capture, demoted)


def should_capture(repr_mode: int | None, state_path: str) -> bool:
    """A request is captured only when its full prompt runs end to end."""
    return (repr_mode in (MODE_MISS, MODE_POPULATE)
            and state_path != PATH_STATE_RESUME)
