"""Background memory sampling mixin (T-2.3).

Spawns a daemon thread that calls a caller-supplied sampler at a
configurable Hz, accumulating peak RAM and (optionally) peak VRAM.
Thread-safe stop via :class:`threading.Event` so ``stop_memory_sampling``
returns within ``1/hz`` worst case.

The sampler is injectable so:

* unit tests drive deterministic readings without psutil/pynvml,
* back-ends plug in ``psutil.Process().memory_info().rss`` for RAM and
  ``pynvml`` for VRAM without the mixin caring.

State is held on the host instance under the ``_ms_`` prefix to avoid
attribute clashes with sibling mixins (``TimingMixin``,
``ManifestLoggingMixin``, ``EnergyAccountingMixin``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryReading:
    """One sample from the sampler — RAM mandatory, VRAM optional."""

    ram_mb: float
    vram_mb: float | None = None


@dataclass(frozen=True)
class MemoryPeaks:
    """Aggregate result returned by :meth:`stop_memory_sampling`."""

    peak_ram_mb: float
    peak_vram_mb: float | None
    n_samples: int


class MemorySamplingMixin:
    """Mixin: spawns a daemon thread to sample memory at fixed Hz.

    Lifecycle: ``start_memory_sampling(hz, sampler)`` →
    ``stop_memory_sampling()``. Both VRAM-equipped and CPU-only hosts
    are supported — the sampler returns ``vram_mb=None`` on CPU-only.
    """

    def start_memory_sampling(
        self,
        *,
        hz: float,
        sampler: Callable[[], MemoryReading],
    ) -> None:
        """Begin sampling at ``hz`` samples per second."""
        if hz <= 0:
            raise ValueError(f"hz must be positive, got {hz}")
        existing = getattr(self, "_ms_thread", None)
        if existing is not None and existing.is_alive():
            raise RuntimeError("Memory sampling already running")
        self._ms_stop = threading.Event()
        self._ms_peak_ram = 0.0
        self._ms_peak_vram: float | None = None
        self._ms_n_samples = 0
        self._ms_thread = threading.Thread(
            target=self._ms_loop,
            args=(hz, sampler),
            daemon=True,
        )
        self._ms_thread.start()

    def stop_memory_sampling(self) -> MemoryPeaks:
        """Signal the sampling thread to stop; return peaks."""
        thread = getattr(self, "_ms_thread", None)
        if thread is None:
            raise RuntimeError("Memory sampling not running")
        self._ms_stop.set()
        thread.join()
        peaks = MemoryPeaks(
            peak_ram_mb=self._ms_peak_ram,
            peak_vram_mb=self._ms_peak_vram,
            n_samples=self._ms_n_samples,
        )
        self._ms_thread = None
        return peaks

    def _ms_loop(
        self,
        hz: float,
        sampler: Callable[[], MemoryReading],
    ) -> None:
        """Sampling-thread body. Runs until ``_ms_stop`` is set."""
        interval = 1.0 / hz
        while not self._ms_stop.is_set():
            reading = sampler()
            self._ms_n_samples += 1
            if reading.ram_mb > self._ms_peak_ram:
                self._ms_peak_ram = reading.ram_mb
            if reading.vram_mb is not None and (
                self._ms_peak_vram is None
                or reading.vram_mb > self._ms_peak_vram
            ):
                self._ms_peak_vram = reading.vram_mb
            if self._ms_stop.wait(timeout=interval):
                return


__all__ = ["MemoryPeaks", "MemoryReading", "MemorySamplingMixin"]
