"""Public types + state machine + pure helpers for :mod:`gatekeeper` (T-2.6).

Sanctioned split per PRD §11 D-GK-7 / constitution §2.2 ``extract logic``
exception — keeps ``shared/gatekeeper.py`` under the 150-LOC cap.
``GatekeeperError`` and ``QueueStatus`` live here too so they sit close
to the state shape they describe; ``gatekeeper.py`` re-exports them.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from on_prem_llm_lab.shared.rate_limit_config import ServiceLimits


class GatekeeperError(RuntimeError):
    """Hard failure (PRD §5.4). Reasons: ``queue_full`` | ``max_retries_exceeded`` | ``unknown_service``."""

    def __init__(
        self,
        *,
        reason: str,
        service: str,
        attempt: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(f"{reason} (service={service})")
        self.reason = reason
        self.service = service
        self.attempt = attempt
        self.cause = cause


@dataclass(frozen=True)
class QueueStatus:
    """Observable per-service state (PRD §5.3)."""

    service: str
    queue_depth: int
    in_flight: int
    requests_in_last_minute: int
    requests_in_last_hour: int


@dataclass
class _ServiceState:
    """Mutable state owned by one service entry."""

    limits: ServiceLimits
    lock: threading.Lock = field(default_factory=threading.Lock)
    queue: deque[threading.Event] = field(default_factory=deque)
    in_flight: int = 0
    minute_ts: deque[float] = field(default_factory=deque)
    hour_ts: deque[float] = field(default_factory=deque)


def prune_windows(state: _ServiceState, clock: Callable[[], float]) -> None:
    """Drop timestamps older than 60 s / 3600 s from the per-service windows.

    Boundary semantics: timestamps exactly ``window_size`` seconds old
    are pruned (``<=``). PRD §2.2 calls this out as the chosen
    fixed-window trade-off — the alternative ``<`` leaks one slot
    forever after the first request.
    """
    now = clock()
    m_cutoff = now - 60.0
    h_cutoff = now - 3600.0
    while state.minute_ts and state.minute_ts[0] <= m_cutoff:
        state.minute_ts.popleft()
    while state.hour_ts and state.hour_ts[0] <= h_cutoff:
        state.hour_ts.popleft()


def slot_available(state: _ServiceState) -> bool:
    """True if all three caps (concurrent, RPM, RPH) admit one more."""
    return (
        state.in_flight < state.limits.concurrent_max
        and len(state.minute_ts) < state.limits.requests_per_minute
        and len(state.hour_ts) < state.limits.requests_per_hour
    )


def take_slot(state: _ServiceState, clock: Callable[[], float]) -> None:
    """Reserve one slot: bump in-flight + append timestamps to both windows."""
    now = clock()
    state.in_flight += 1
    state.minute_ts.append(now)
    state.hour_ts.append(now)


def time_until_next_window(state: _ServiceState, clock: Callable[[], float]) -> float:
    """Seconds until at least one window expires enough to free a slot.

    Returns 0.0 when neither RPM nor RPH is the blocker — meaning the
    only thing holding us back is ``concurrent_max``, which clears on a
    release (event-signalled), not on time.
    """
    now = clock()
    candidates: list[float] = []
    if len(state.minute_ts) >= state.limits.requests_per_minute:
        candidates.append(state.minute_ts[0] + 60.0 - now)
    if len(state.hour_ts) >= state.limits.requests_per_hour:
        candidates.append(state.hour_ts[0] + 3600.0 - now)
    return max(candidates) if candidates else 0.0


__all__ = [
    "GatekeeperError",
    "QueueStatus",
    "_ServiceState",
    "prune_windows",
    "slot_available",
    "take_slot",
    "time_until_next_window",
]
