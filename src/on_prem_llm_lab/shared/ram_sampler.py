"""Default RAM sampler (T-2.10) — production callable for ``MemorySamplingMixin``.

Lazy-imports ``psutil`` inside the call so CLI ``--help`` and unit tests
that inject their own sampler don't pay the import cost.
"""

from __future__ import annotations

from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading

_BYTES_PER_MB = 1024 * 1024


def psutil_rss_sampler() -> MemoryReading:
    """Return current process RSS in MB (RAM only — VRAM = ``None``)."""
    import psutil  # noqa: PLC0415

    rss_bytes = psutil.Process().memory_info().rss
    return MemoryReading(ram_mb=rss_bytes / _BYTES_PER_MB)


__all__ = ["psutil_rss_sampler"]
