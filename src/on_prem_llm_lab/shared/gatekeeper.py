"""ApiGatekeeper — centralized rate-limit + FIFO queue + retry/backoff (T-2.6).

Contract: ``docs/PRD_api_gatekeeper.md`` v1.00 (approved 2026-06-30).
Public types (``GatekeeperError``, ``QueueStatus``) + state machine
live in ``gatekeeper_state.py`` (PRD D-GK-7); this module is just the
``ApiGatekeeper`` class plus structured-logging helper. ``clock`` and
``sleeper`` are injectable so tests advance a virtual clock.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from on_prem_llm_lab.shared.gatekeeper_state import (
    GatekeeperError,
    QueueStatus,
    _ServiceState,
    prune_windows,
    slot_available,
    take_slot,
    time_until_next_window,
)
from on_prem_llm_lab.shared.rate_limit_config import RateLimitConfig

T = TypeVar("T")
_DEFAULT_LOGGER = logging.getLogger("on_prem_llm_lab.gatekeeper")


class ApiGatekeeper:
    """Centralized gate for outbound API calls (PRD §5.5)."""

    def __init__(
        self,
        config: RateLimitConfig,
        *,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._states: dict[str, _ServiceState] = {}
        self._states_lock = threading.Lock()
        self._logger = logger or _DEFAULT_LOGGER
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep

    def _state_for(self, service: str) -> _ServiceState:
        try:
            limits = self._config.for_service(service)
        except KeyError as e:
            raise GatekeeperError(reason="unknown_service", service=service) from e
        with self._states_lock:
            if service not in self._states:
                self._states[service] = _ServiceState(limits=limits)
            return self._states[service]

    @contextlib.contextmanager
    def acquire(self, service: str) -> Iterator[None]:
        state = self._state_for(service)
        my_event: threading.Event | None = None
        with state.lock:
            prune_windows(state, self._clock)
            if not state.queue and slot_available(state):
                take_slot(state, self._clock)
                self._log("acquire", service=service, in_flight=state.in_flight)
            else:
                if len(state.queue) >= state.limits.queue_max_depth:
                    self._log("queue_full", service=service, queue_depth=len(state.queue))
                    raise GatekeeperError(reason="queue_full", service=service)
                my_event = threading.Event()
                state.queue.append(my_event)
                self._log("queue_wait", service=service, queue_depth=len(state.queue))
        if my_event is not None:
            self._wait_for_turn(state, my_event, service)
        try:
            yield
        finally:
            with state.lock:
                state.in_flight -= 1
                self._log("release", service=service, in_flight=state.in_flight)
                if state.queue:
                    state.queue[0].set()

    def _wait_for_turn(
        self, state: _ServiceState, my_event: threading.Event, service: str
    ) -> None:
        while True:
            with state.lock:
                prune_windows(state, self._clock)
                if state.queue[0] is my_event and slot_available(state):
                    state.queue.popleft()
                    take_slot(state, self._clock)
                    self._log("acquire", service=service, in_flight=state.in_flight)
                    return
                wait_s = time_until_next_window(state, self._clock)
            if wait_s > 0:
                self._sleeper(wait_s)
            else:
                my_event.wait()
                my_event.clear()

    def call(
        self, service: str, fn: Callable[..., T], *args: Any, **kwargs: Any,
    ) -> T:
        try:
            limits = self._config.for_service(service)
        except KeyError as e:
            raise GatekeeperError(reason="unknown_service", service=service) from e
        last_exc: Exception | None = None
        for attempt in range(limits.max_retries + 1):
            try:
                with self.acquire(service):
                    return fn(*args, **kwargs)
            except GatekeeperError:
                raise
            except Exception as e:  # noqa: BLE001
                last_exc = e
                self._log("retry", service=service, attempt=attempt, error=str(e))
                if attempt < limits.max_retries:
                    backoff = random.uniform(0.0, limits.retry_after_seconds * (2 ** attempt))
                    self._sleeper(backoff)
        raise GatekeeperError(
            reason="max_retries_exceeded", service=service,
            attempt=limits.max_retries, cause=last_exc,
        )

    def get_queue_status(self, service: str) -> QueueStatus:
        state = self._state_for(service)
        with state.lock:
            prune_windows(state, self._clock)
            return QueueStatus(
                service=service,
                queue_depth=len(state.queue),
                in_flight=state.in_flight,
                requests_in_last_minute=len(state.minute_ts),
                requests_in_last_hour=len(state.hour_ts),
            )

    def _log(self, event: str, **kwargs: Any) -> None:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event, **kwargs,
        }
        self._logger.info(json.dumps(payload))


__all__ = ["ApiGatekeeper", "GatekeeperError", "QueueStatus"]
