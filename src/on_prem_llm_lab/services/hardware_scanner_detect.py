"""Default hardware detection backend for HardwareScanner (ADR-015 — step 1).

Reads ``config/setup.json`` to discover ``airllm.layer_shards_saving_path`` so
the disk-free measurement targets the drive AirLLM will actually write to
(per ADR-005). Falls back to the current working directory if the shard path
doesn't exist yet (AirLLM creates it lazily on first use).

Uses :mod:`psutil` (required dependency) plus :mod:`pynvml` (optional — its
absence is logged and treated as "no GPU detected"; never crashes the scan).
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
from pathlib import Path

import psutil

from on_prem_llm_lab.services.hardware_scanner_types import (
    CpuInfo,
    DiskInfo,
    GpuInfo,
    HardwareScanResult,
    RamInfo,
)

log = logging.getLogger(__name__)

_GB = 1024**3
_PLACEHOLDER_CAPTURED_AT = ""  # orchestrator overwrites this with its clock


def _detect_cpu() -> CpuInfo:
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    return CpuInfo(
        model=platform.processor() or None,
        cores_physical=int(physical),
        cores_logical=int(logical),
    )


def _detect_ram() -> RamInfo:
    vm = psutil.virtual_memory()
    return RamInfo(total_gb=vm.total / _GB, available_gb=vm.available / _GB)


def _detect_gpu() -> GpuInfo:
    try:
        import pynvml  # noqa: PLC0415
    except ImportError:
        log.debug("pynvml not installed; GPU detection skipped")
        return GpuInfo(present=False, model=None, vram_gb=None)
    try:
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return GpuInfo(present=False, model=None, vram_gb=None)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return GpuInfo(present=True, model=str(name), vram_gb=mem.total / _GB)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 — optional path, best-effort
        log.debug("GPU detection failed: %s", exc)
        return GpuInfo(present=False, model=None, vram_gb=None)


def _shard_path_from_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    raw = (cfg.get("airllm") or {}).get("layer_shards_saving_path")
    return Path(raw).expanduser() if raw else None


def _detect_disk(config_path: Path) -> DiskInfo:
    """Measure free space on the AirLLM shard drive (ADR-005), else cwd."""
    target = _shard_path_from_config(config_path)
    if target is not None:
        while not target.exists() and target.parent != target:
            target = target.parent
    if target is None or not target.exists():
        target = Path.cwd()
    usage = shutil.disk_usage(target)
    return DiskInfo(
        free_gb=usage.free / _GB,
        fs=None,
        kind="unknown",
        measured_at=str(target),
    )


def default_detect(config_path: Path) -> HardwareScanResult:
    """Probe the system; return a result with empty receipts + placeholder clock."""
    return HardwareScanResult(
        captured_at=_PLACEHOLDER_CAPTURED_AT,
        os=platform.platform(),
        python=sys.version.split()[0],
        cpu=_detect_cpu(),
        ram=_detect_ram(),
        gpu=_detect_gpu(),
        disk=_detect_disk(config_path),
        write_receipts={},
    )


__all__ = ["default_detect"]
