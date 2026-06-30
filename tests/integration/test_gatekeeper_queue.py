"""Integration tests for gatekeeper burst/queue behavior (T-2.7).

Maps to ``docs/PRD_api_gatekeeper.md`` §9.2 I-GK-1 + I-GK-2. Uses
injected ``clock`` + ``sleeper`` so the virtual clock advances on each
sleep — real test wall time stays < 100 ms while simulating minutes of
burst load. Marked ``integration`` because it exercises full
``RateLimitConfig → ApiGatekeeper`` end-to-end with the
real-shape `config/rate_limits.json` payload.
"""

from __future__ import annotations

import pytest

from on_prem_llm_lab.shared.gatekeeper import ApiGatekeeper
from on_prem_llm_lab.shared.rate_limit_config import RateLimitConfig


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _two_service_config() -> RateLimitConfig:
    """Mirror of `config/rate_limits.json` shape — both services configured."""
    return RateLimitConfig.from_dict({
        "version": "1.00",
        "services": {
            "huggingface_hub": {
                "requests_per_minute": 60, "requests_per_hour": 1000,
                "concurrent_max": 4, "retry_after_seconds": 15,
                "max_retries": 5, "queue_max_depth": 50,
            },
            "anthropic_messages": {
                "requests_per_minute": 20, "requests_per_hour": 400,
                "concurrent_max": 2, "retry_after_seconds": 30,
                "max_retries": 3, "queue_max_depth": 50,
            },
        },
    })


@pytest.mark.integration
class TestBurstBeyondRpm:
    """I-GK-1 — 60-call burst against `anthropic_messages` (rpm=20)."""

    def test_60_call_burst_drains_via_three_windows(self) -> None:
        vc = _VirtualClock()
        gk = ApiGatekeeper(
            _two_service_config(), clock=vc.clock, sleeper=vc.sleeper,
        )
        completion_order: list[int] = []

        for i in range(60):
            with gk.acquire("anthropic_messages"):
                completion_order.append(i)

        # FIFO trivially preserved in single-threaded submission — each call
        # returns before the next starts, and the gatekeeper's deque is the
        # belt-and-suspenders backup for multi-threaded callers (covered by
        # the unit test_gatekeeper_retry.TestConcurrentMax with real threads).
        assert completion_order == list(range(60))

        # Window math: rpm=20 -> 20 calls fit per window. 60 calls -> 3 windows
        # -> 2 inter-window waits. Each wait is exactly 60 s (window size).
        assert vc.sleeps == [60.0, 60.0]
        assert vc.now == pytest.approx(120.0)

        # Queue drained completely; no orphan state left behind.
        status = gk.get_queue_status("anthropic_messages")
        assert status.queue_depth == 0
        assert status.in_flight == 0

    def test_burst_does_not_raise_on_recovered_load(self) -> None:
        """SC-GK-1 spirit: no exception across the full burst."""
        vc = _VirtualClock()
        gk = ApiGatekeeper(
            _two_service_config(), clock=vc.clock, sleeper=vc.sleeper,
        )
        # Silence: no GatekeeperError should escape across 60 calls.
        for _ in range(60):
            with gk.acquire("anthropic_messages"):
                pass


@pytest.mark.integration
class TestPerServiceIsolation:
    """I-GK-2 — burst against one service does not leak into the other."""

    def test_hf_and_anthropic_counters_are_independent(self) -> None:
        vc = _VirtualClock()
        gk = ApiGatekeeper(
            _two_service_config(), clock=vc.clock, sleeper=vc.sleeper,
        )

        # Burst against HF until it's at the rpm cap (60). Nothing should
        # block because 60 calls exactly fits the window.
        for _ in range(60):
            with gk.acquire("huggingface_hub"):
                pass

        hf_full = gk.get_queue_status("huggingface_hub")
        ant_pristine = gk.get_queue_status("anthropic_messages")
        assert hf_full.requests_in_last_minute == 60
        # The critical isolation assertion: Anthropic counters untouched.
        assert ant_pristine.requests_in_last_minute == 0
        assert ant_pristine.queue_depth == 0
        assert ant_pristine.in_flight == 0

        # And now a single Anthropic call must NOT block — even though HF
        # is at its rpm cap, Anthropic has its own counter at 0.
        sleeps_before = len(vc.sleeps)
        with gk.acquire("anthropic_messages"):
            pass
        assert len(vc.sleeps) == sleeps_before  # no sleeper invocation

        # HF still at 60; Anthropic now at 1.
        hf_after = gk.get_queue_status("huggingface_hub")
        ant_after = gk.get_queue_status("anthropic_messages")
        assert hf_after.requests_in_last_minute == 60
        assert ant_after.requests_in_last_minute == 1
