"""Core gatekeeper tests (T-2.6 / PRD §9 U-GK-5, 6, 10, 11, 13)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import pytest

from on_prem_llm_lab.shared.gatekeeper import (
    ApiGatekeeper,
    GatekeeperError,
    QueueStatus,
)

from .conftest import VirtualClock, make_config


class TestAcquireUnderLimit:
    # U-GK-5
    def test_returns_immediately_no_sleep(
        self, vc: VirtualClock, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory(make_config(requests_per_minute=5))
        with gk.acquire("svc"):
            pass
        assert vc.sleeps == []
        assert gk.get_queue_status("svc").in_flight == 0


class TestAcquireOverRpm:
    # U-GK-6
    def test_third_call_waits_for_window_to_expire(
        self, vc: VirtualClock, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory(make_config(requests_per_minute=2))
        with gk.acquire("svc"):
            pass
        with gk.acquire("svc"):
            pass
        # Both timestamps = 0; third must wait until t=60 to free a slot.
        with gk.acquire("svc"):
            pass
        assert len(vc.sleeps) == 1
        assert vc.sleeps[0] == pytest.approx(60.0)
        assert vc.now == pytest.approx(60.0)


class TestGetQueueStatus:
    # U-GK-10
    def test_reports_current_counters(
        self, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory()
        with gk.acquire("svc"):
            inside = gk.get_queue_status("svc")
            assert isinstance(inside, QueueStatus)
            assert inside.service == "svc"
            assert inside.in_flight == 1
            assert inside.queue_depth == 0
            assert inside.requests_in_last_minute == 1
        after = gk.get_queue_status("svc")
        assert after.in_flight == 0
        assert after.requests_in_last_minute == 1


class TestLogging:
    # U-GK-11
    def test_every_event_emits_one_valid_json_line(
        self,
        caplog: pytest.LogCaptureFixture,
        gk_factory: Callable[..., ApiGatekeeper],
    ) -> None:
        logger = logging.getLogger("test.gatekeeper")
        logger.setLevel(logging.INFO)
        gk = gk_factory(logger=logger)
        with caplog.at_level(logging.INFO, logger="test.gatekeeper"), gk.acquire("svc"):
            pass
        events = [json.loads(r.message) for r in caplog.records]
        kinds = [e["event"] for e in events]
        assert "acquire" in kinds and "release" in kinds
        for e in events:
            assert "ts" in e and e["service"] == "svc"


class TestUnknownService:
    # U-GK-13
    def test_acquire_unknown_service_raises(
        self, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory()
        with pytest.raises(GatekeeperError) as exc, gk.acquire("missing"):
            pass
        assert exc.value.reason == "unknown_service"
        assert exc.value.service == "missing"

    def test_call_unknown_service_raises_without_invoking_fn(
        self, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory()
        with pytest.raises(GatekeeperError) as exc:
            gk.call("missing", lambda: 1)
        assert exc.value.reason == "unknown_service"
