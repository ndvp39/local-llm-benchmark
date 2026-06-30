"""Shared fixtures + helpers for ``shared/*`` tests (T-2.6)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from on_prem_llm_lab.shared.gatekeeper import ApiGatekeeper
from on_prem_llm_lab.shared.rate_limit_config import RateLimitConfig


class VirtualClock:
    """Tiny clock + sleeper pair that advances virtual time on each sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_config(**overrides: Any) -> RateLimitConfig:
    """One-service ``svc`` config with sensible defaults overridable by kwargs."""
    base: dict[str, Any] = {
        "requests_per_minute": 60, "requests_per_hour": 1000,
        "concurrent_max": 5, "retry_after_seconds": 1.0,
        "max_retries": 3, "queue_max_depth": 10,
    }
    base.update(overrides)
    return RateLimitConfig.from_dict(
        {"version": "1.00", "services": {"svc": base}}
    )


@pytest.fixture
def vc() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def gk_factory(vc: VirtualClock) -> Callable[..., ApiGatekeeper]:
    def _make(
        cfg: RateLimitConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> ApiGatekeeper:
        return ApiGatekeeper(
            cfg or make_config(), logger=logger,
            clock=vc.clock, sleeper=vc.sleeper,
        )
    return _make
