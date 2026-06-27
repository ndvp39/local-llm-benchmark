"""HardwareScanner — detect hardware, inject into config, patch docs (ADR-015).

Orchestrates the three side-effects defined in PLAN §6.7:

1. Probe CPU/RAM/GPU/VRAM/disk/OS/Python (via :mod:`.hardware_scanner_detect`).
2. Inject the payload into ``config/setup.json.hardware_constraints`` atomically.
3. Replace the placeholder block in every path listed in
   ``config.init.doc_targets`` (default: ``docs/PRD.md`` and ``README.md``).

Returns a :class:`HardwareScanResult` with per-file ``write_receipts``.
Idempotent modulo the injected ``captured_at`` clock — re-running with the
same clock produces byte-identical files (writes short-circuit at "ok-noop").
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from on_prem_llm_lab.services.hardware_scanner_detect import default_detect
from on_prem_llm_lab.services.hardware_scanner_types import (
    HardwareScanResult,
    WriteReceipt,
)
from on_prem_llm_lab.services.hardware_scanner_writes import (
    inject_into_setup_json,
    patch_doc_placeholders,
)

log = logging.getLogger(__name__)

DEFAULT_PLACEHOLDER_START = "<!-- HARDWARE_SPECS_PLACEHOLDER:START -->"
DEFAULT_PLACEHOLDER_END = "<!-- HARDWARE_SPECS_PLACEHOLDER:END -->"
DEFAULT_DOC_TARGETS: tuple[str, ...] = ("docs/PRD.md", "README.md")


def _utc_now_iso() -> str:
    """ISO-8601 UTC second precision, e.g. ``2026-06-26T10:00:00Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_markdown_table(r: HardwareScanResult) -> str:
    """Format the scan result as the Markdown table injected between placeholders."""
    cpu_line = (
        f"{r.cpu.model or 'unknown'} · "
        f"{r.cpu.cores_physical} physical / {r.cpu.cores_logical} logical"
    )
    ram_line = f"{r.ram.total_gb:.1f} GB total · {r.ram.available_gb:.1f} GB available"
    if r.gpu.present and r.gpu.vram_gb is not None:
        gpu_line = f"{r.gpu.model or 'unknown'} · {r.gpu.vram_gb:.1f} GB VRAM"
    else:
        gpu_line = "not detected (CPU-only run)"
    disk_line = (
        f"{r.disk.free_gb:.1f} GB free · {r.disk.fs or 'unknown'} · {r.disk.kind} "
        f"(measured at `{r.disk.measured_at}`)"
    )
    return (
        "| Component | Value |\n"
        "|-----------|-------|\n"
        f"| Captured at | {r.captured_at} |\n"
        f"| OS / Python | {r.os} / {r.python} |\n"
        f"| CPU | {cpu_line} |\n"
        f"| RAM | {ram_line} |\n"
        f"| GPU | {gpu_line} |\n"
        f"| Disk | {disk_line} |\n"
    )


class HardwareScanner:
    """Orchestrates detection + atomic config injection + doc patching (ADR-015)."""

    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        detector: Callable[[Path], HardwareScanResult] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.repo_root: Path = repo_root
        self.config_path: Path = config_path
        self._detect = detector or default_detect
        self._clock = clock or _utc_now_iso

    def scan(self) -> HardwareScanResult:
        """Run detect → inject → patch and return the populated result."""
        result = self._detect(self.config_path)
        result = replace(result, captured_at=self._clock())
        init = self._load_init_block()
        keep_bak = bool(init.get("keep_bak", True))
        receipts: dict[str, WriteReceipt] = {
            "config_setup_json": inject_into_setup_json(
                self.config_path, result, keep_bak=keep_bak
            ),
        }
        table_md = render_markdown_table(result)
        start = str(init.get("placeholder_start", DEFAULT_PLACEHOLDER_START))
        end = str(init.get("placeholder_end", DEFAULT_PLACEHOLDER_END))
        for doc_rel in init.get("doc_targets", DEFAULT_DOC_TARGETS):
            receipts[self._receipt_key(doc_rel)] = patch_doc_placeholders(
                self.repo_root / doc_rel, table_md, start, end, keep_bak=keep_bak
            )
        return replace(result, write_receipts=receipts)

    @staticmethod
    def _receipt_key(doc_rel: str) -> str:
        if doc_rel.endswith("PRD.md"):
            return "docs_prd_md"
        if doc_rel.endswith("README.md"):
            return "readme_md"
        return doc_rel.replace("/", "_").replace(".", "_")

    def _load_init_block(self) -> dict:
        if not self.config_path.exists():
            log.warning("config missing at %s; using defaults", self.config_path)
            return {}
        try:
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("config %s could not parse: %s", self.config_path, exc)
            return {}
        return cfg.get("init") or {}


__all__ = [
    "HardwareScanner",
    "render_markdown_table",
    "DEFAULT_PLACEHOLDER_START",
    "DEFAULT_PLACEHOLDER_END",
    "DEFAULT_DOC_TARGETS",
]
