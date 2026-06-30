"""Queue-full / retry / concurrency tests (T-2.6 / PRD §9 U-GK-7, 8, 9, 12)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from on_prem_llm_lab.shared.gatekeeper import ApiGatekeeper, GatekeeperError

from .conftest import VirtualClock, make_config


class TestQueueFull:
    # U-GK-7
    def test_raises_when_queue_already_at_max_depth(
        self, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory(make_config(concurrent_max=1, queue_max_depth=2))
        # White-box: pre-populate the per-service state to skip multi-threaded
        # set-up — the queue-full branch is the bit under test.
        state = gk._state_for("svc")
        state.in_flight = 1
        state.queue.append(threading.Event())
        state.queue.append(threading.Event())
        with pytest.raises(GatekeeperError) as exc, gk.acquire("svc"):
            pass
        assert exc.value.reason == "queue_full"
        assert exc.value.service == "svc"


class TestCallRetry:
    # U-GK-8
    def test_retries_transient_then_returns_success(
        self, vc: VirtualClock, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory(make_config(max_retries=3, retry_after_seconds=0.5))
        calls = {"n": 0}

        def fn() -> int:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return 42

        assert gk.call("svc", fn) == 42
        assert calls["n"] == 3
        assert len(vc.sleeps) == 2  # 2 backoffs between 3 attempts

    # U-GK-9
    def test_max_retries_exceeded_raises_with_cause(
        self, gk_factory: Callable[..., ApiGatekeeper]
    ) -> None:
        gk = gk_factory(make_config(max_retries=2, retry_after_seconds=0.5))
        seen: list[Exception] = []

        def fn() -> int:
            err = RuntimeError(f"boom-{len(seen)}")
            seen.append(err)
            raise err

        with pytest.raises(GatekeeperError) as exc:
            gk.call("svc", fn)
        assert exc.value.reason == "max_retries_exceeded"
        assert exc.value.attempt == 2
        assert isinstance(exc.value.cause, RuntimeError)
        assert len(seen) == 3  # 1 initial + 2 retries


class TestConcurrentMax:
    # U-GK-12 — uses real threading (one of two cases the PRD calls out).
    def test_third_caller_blocks_until_release(self) -> None:
        gk = ApiGatekeeper(make_config(concurrent_max=2, queue_max_depth=10))
        barrier = threading.Barrier(3)  # 2 holders + main
        release = threading.Event()

        def hold() -> None:
            with gk.acquire("svc"):
                barrier.wait()
                release.wait()

        t1 = threading.Thread(target=hold, daemon=True)
        t2 = threading.Thread(target=hold, daemon=True)
        t1.start()
        t2.start()
        barrier.wait()  # both holders inside; in_flight == 2 == concurrent_max
        acquired = threading.Event()

        def third() -> None:
            with gk.acquire("svc"):
                acquired.set()

        t3 = threading.Thread(target=third, daemon=True)
        t3.start()
        # Deterministic wait for t3 to enter the queue.
        deadline = time.perf_counter() + 1.0
        while gk.get_queue_status("svc").queue_depth < 1:
            assert time.perf_counter() < deadline, "t3 never entered queue"
            time.sleep(0.005)
        assert not acquired.is_set()
        release.set()
        assert acquired.wait(timeout=1.0)
        for t in (t1, t2, t3):
            t.join(timeout=1.0)
