"""Unit tests for mixins/memory_sampling_mixin.py (T-2.3).

Drives :class:`MemorySamplingMixin` with a fake sampler that emits a
preset list of readings then signals exhaustion. The mixin's
``stop_memory_sampling`` does a thread ``join()``, which guarantees the
final reading is reflected in the peaks before the test reads them — so
the tests are deterministic without sleep-based polling.
"""

from __future__ import annotations

import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from on_prem_llm_lab.mixins.memory_sampling_mixin import (
    MemoryPeaks,
    MemoryReading,
    MemorySamplingMixin,
)


class _Mixed(MemorySamplingMixin):
    """Minimal concrete subclass."""


class _StepSampler:
    """Emit ``readings`` in order, then repeat the last one indefinitely.

    Sets :attr:`exhausted` once every preset reading has been emitted, so
    the test can wait for "all known readings consumed" without polling.
    """

    def __init__(self, readings: list[MemoryReading]) -> None:
        self._readings = readings
        self._idx = 0
        self.exhausted = threading.Event()

    def __call__(self) -> MemoryReading:
        if self._idx < len(self._readings):
            r = self._readings[self._idx]
            self._idx += 1
            if self._idx >= len(self._readings):
                self.exhausted.set()
            return r
        return self._readings[-1]


class TestValidation:
    def test_zero_hz_raises(self) -> None:
        with pytest.raises(ValueError, match="hz must be positive"):
            _Mixed().start_memory_sampling(
                hz=0.0, sampler=lambda: MemoryReading(ram_mb=0.0),
            )

    def test_negative_hz_raises(self) -> None:
        with pytest.raises(ValueError, match="hz must be positive"):
            _Mixed().start_memory_sampling(
                hz=-5.0, sampler=lambda: MemoryReading(ram_mb=0.0),
            )

    def test_start_twice_raises(self) -> None:
        m = _Mixed()
        m.start_memory_sampling(
            hz=100.0, sampler=lambda: MemoryReading(ram_mb=0.0),
        )
        try:
            with pytest.raises(RuntimeError, match="already running"):
                m.start_memory_sampling(
                    hz=100.0, sampler=lambda: MemoryReading(ram_mb=0.0),
                )
        finally:
            m.stop_memory_sampling()

    def test_stop_without_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not running"):
            _Mixed().stop_memory_sampling()


class TestSampling:
    def test_records_peak_ram_only(self) -> None:
        readings = [
            MemoryReading(ram_mb=100.0),
            MemoryReading(ram_mb=500.0),
            MemoryReading(ram_mb=300.0),
            MemoryReading(ram_mb=200.0),
        ]
        sampler = _StepSampler(readings)
        m = _Mixed()
        m.start_memory_sampling(hz=1000.0, sampler=sampler)
        assert sampler.exhausted.wait(timeout=2.0)
        peaks = m.stop_memory_sampling()
        assert peaks.peak_ram_mb == 500.0
        assert peaks.peak_vram_mb is None
        assert peaks.n_samples >= 4

    def test_records_peak_vram_when_present(self) -> None:
        readings = [
            MemoryReading(ram_mb=100.0, vram_mb=2000.0),
            MemoryReading(ram_mb=200.0, vram_mb=8000.0),
            MemoryReading(ram_mb=150.0, vram_mb=4000.0),
        ]
        sampler = _StepSampler(readings)
        m = _Mixed()
        m.start_memory_sampling(hz=1000.0, sampler=sampler)
        assert sampler.exhausted.wait(timeout=2.0)
        peaks = m.stop_memory_sampling()
        assert peaks.peak_ram_mb == 200.0
        assert peaks.peak_vram_mb == 8000.0

    def test_mixed_vram_some_none_some_value(self) -> None:
        """Peak VRAM = max of non-None readings; None readings ignored."""
        readings = [
            MemoryReading(ram_mb=100.0, vram_mb=None),
            MemoryReading(ram_mb=200.0, vram_mb=3000.0),
            MemoryReading(ram_mb=300.0, vram_mb=None),
            MemoryReading(ram_mb=400.0, vram_mb=5000.0),
        ]
        sampler = _StepSampler(readings)
        m = _Mixed()
        m.start_memory_sampling(hz=1000.0, sampler=sampler)
        assert sampler.exhausted.wait(timeout=2.0)
        peaks = m.stop_memory_sampling()
        assert peaks.peak_ram_mb == 400.0
        assert peaks.peak_vram_mb == 5000.0


class TestStopResponsiveness:
    def test_stop_returns_promptly_after_set(self) -> None:
        """At hz=10 (interval 100 ms), stop returns well under 500 ms."""
        m = _Mixed()
        m.start_memory_sampling(
            hz=10.0, sampler=lambda: MemoryReading(ram_mb=42.0),
        )
        time.sleep(0.05)
        t0 = time.perf_counter()
        peaks = m.stop_memory_sampling()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5
        assert peaks.n_samples >= 1


class TestDataclasses:
    def test_memory_reading_is_frozen(self) -> None:
        r = MemoryReading(ram_mb=100.0)
        with pytest.raises(FrozenInstanceError):
            r.ram_mb = 999.0  # type: ignore[misc]

    def test_memory_peaks_is_frozen(self) -> None:
        p = MemoryPeaks(peak_ram_mb=100.0, peak_vram_mb=None, n_samples=5)
        with pytest.raises(FrozenInstanceError):
            p.peak_ram_mb = 999.0  # type: ignore[misc]
