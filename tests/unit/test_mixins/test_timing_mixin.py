"""Unit tests for mixins/timing_mixin.py (T-2.2).

Drives a :class:`TimingMixin` instance with a fake clock that returns
preset timestamps; asserts ``ttft_ms`` / ``tpot_ms`` / ``n_tokens`` are
computed exactly. No real ``time.perf_counter`` reliance — the only
test that doesn't inject a clock just checks the default plumbing wires
through to a positive monotonic value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import FrozenInstanceError

import pytest

from on_prem_llm_lab.mixins.timing_mixin import TimingMeasurement, TimingMixin


class _Mixed(TimingMixin):
    """Minimal concrete subclass — TimingMixin is meant to be inherited."""


def _fake_clock(timestamps: list[float]) -> Callable[[], float]:
    """Return a callable that yields ``timestamps`` in order on each call."""
    iterator = iter(timestamps)

    def clock() -> float:
        return next(iterator)

    return clock


class TestTimeTokenStream:
    def test_empty_stream_returns_all_nones(self) -> None:
        # Only ``t0`` ever called (the ``next()`` raises StopIteration before
        # ``t_first`` is sampled).
        clock = _fake_clock([100.0])
        m = _Mixed().time_token_stream(iter([]), clock=clock)
        assert m == TimingMeasurement(ttft_ms=None, tpot_ms=None, n_tokens=0)

    def test_single_token_yields_ttft_only(self) -> None:
        # t0=1.0, t_first=1.5  ->  TTFT = 500 ms, TPOT undefined.
        clock = _fake_clock([1.0, 1.5])
        m = _Mixed().time_token_stream(iter([42]), clock=clock)
        assert m.ttft_ms == pytest.approx(500.0)
        assert m.tpot_ms is None
        assert m.n_tokens == 1

    def test_two_tokens_yields_ttft_and_tpot(self) -> None:
        # t0=10.0, t_first=10.4  ->  TTFT = 400 ms.
        # t_last=10.9, n-1=1     ->  TPOT = (10.9 - 10.4) * 1000 / 1 = 500 ms.
        clock = _fake_clock([10.0, 10.4, 10.9])
        m = _Mixed().time_token_stream(iter([1, 2]), clock=clock)
        assert m.ttft_ms == pytest.approx(400.0)
        assert m.tpot_ms == pytest.approx(500.0)
        assert m.n_tokens == 2

    def test_five_tokens_divides_by_n_minus_one(self) -> None:
        # t0=0, t_first=1, t_last=5 -> TTFT=1000 ms, TPOT=(5-1)*1000/4=1000 ms.
        clock = _fake_clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        m = _Mixed().time_token_stream(iter([1, 2, 3, 4, 5]), clock=clock)
        assert m.ttft_ms == pytest.approx(1000.0)
        assert m.tpot_ms == pytest.approx(1000.0)
        assert m.n_tokens == 5

    def test_consumes_a_lazy_generator(self) -> None:
        """Real back-ends return generators, not lists — they MUST work."""

        def gen() -> Iterable[int]:
            yield 1
            yield 2
            yield 3

        # t0=0.0, t_first=0.1 -> TTFT=100 ms; t_last=0.3 -> TPOT=(0.3-0.1)/2*1000=100 ms.
        clock = _fake_clock([0.0, 0.1, 0.2, 0.3])
        m = _Mixed().time_token_stream(gen(), clock=clock)
        assert m.n_tokens == 3
        assert m.ttft_ms == pytest.approx(100.0)
        assert m.tpot_ms == pytest.approx(100.0)

    def test_uneven_token_spacing_averages_over_decode_window(self) -> None:
        # Clock: t0=0.0, t_first=0.1, then 0.2, 0.5, 1.0.
        # TTFT = (0.1 - 0.0) * 1000 = 100 ms.
        # Decode window = t_last - t_first = 1.0 - 0.1 = 0.9 s.
        # n - 1 = 3 -> TPOT = 900 / 3 = 300 ms (per-token decode interval average).
        clock = _fake_clock([0.0, 0.1, 0.2, 0.5, 1.0])
        m = _Mixed().time_token_stream(iter([1, 2, 3, 4]), clock=clock)
        assert m.ttft_ms == pytest.approx(100.0)
        assert m.tpot_ms == pytest.approx(300.0)
        assert m.n_tokens == 4

    def test_default_clock_uses_perf_counter(self) -> None:
        """No ``clock=`` arg -> uses ``time.perf_counter`` and stays positive."""
        m = _Mixed().time_token_stream(iter([1, 2, 3]))
        assert m.n_tokens == 3
        assert m.ttft_ms is not None
        assert m.ttft_ms >= 0
        assert m.tpot_ms is not None
        assert m.tpot_ms >= 0


class TestTimingMeasurement:
    def test_is_frozen(self) -> None:
        m = TimingMeasurement(ttft_ms=1.0, tpot_ms=2.0, n_tokens=2)
        with pytest.raises(FrozenInstanceError):
            m.ttft_ms = 99.0  # type: ignore[misc]
