"""HardwareScanner data types — extracted to break a circular import.

The orchestrator (``hardware_scanner``), the detection helper
(``hardware_scanner_detect``), and the write helpers
(``hardware_scanner_writes``) all need these dataclasses. Per constitution
§2.2 ("extract model definitions to a separate file"), they live here.

Shape detail in PLAN §6.1 (``HardwareScanResult``) and PLAN §6.7
(``WriteReceipt`` per-file outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CpuInfo:
    model: str | None
    cores_physical: int
    cores_logical: int


@dataclass(frozen=True)
class RamInfo:
    total_gb: float
    available_gb: float


@dataclass(frozen=True)
class GpuInfo:
    present: bool
    model: str | None
    vram_gb: float | None


@dataclass(frozen=True)
class DiskInfo:
    free_gb: float
    fs: str | None
    kind: str  # "NVMe" | "SSD" | "HDD" | "unknown" — see ADR-005
    measured_at: str


@dataclass(frozen=True)
class WriteReceipt:
    """Per-file outcome of a HardwareScanner side-effect (ADR-015 step 2/3).

    status ∈ {"ok", "ok-noop", "skipped-no-markers", "skipped-file-missing", "fail"}.
    """

    status: str
    path: str
    bak: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class HardwareScanResult:
    captured_at: str
    os: str
    python: str
    cpu: CpuInfo
    ram: RamInfo
    gpu: GpuInfo
    disk: DiskInfo
    write_receipts: dict[str, WriteReceipt] = field(default_factory=dict)


__all__ = [
    "CpuInfo",
    "RamInfo",
    "GpuInfo",
    "DiskInfo",
    "WriteReceipt",
    "HardwareScanResult",
]
