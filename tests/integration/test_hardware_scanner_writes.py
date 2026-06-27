"""Integration: HardwareScanner mutates all three target files end-to-end (ADR-015).

T-1.13 unit-tested each side-effect in isolation with mocks. T-1.16's DoD asks
for an integration test that drives the *real* HardwareScanner against a real
on-disk fixture tree (tmp_path) and verifies the four outcomes in PLAN §6.7:

  1. ``setup.json.hardware_constraints`` is populated atomically.
  2. ``docs/PRD.md`` placeholder block is replaced with the rendered table.
  3. ``README.md`` placeholder block is replaced with the rendered table.
  4. Re-running with the same clock is byte-idempotent (no spurious writes).

The detector is stubbed so this test does not depend on the host's psutil
output — the scanner-under-test still does all its own atomic-write + marker-
parse + JSON-merge work against the real filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_prem_llm_lab.services.hardware_scanner import HardwareScanner
from on_prem_llm_lab.services.hardware_scanner_types import (
    CpuInfo,
    DiskInfo,
    GpuInfo,
    HardwareScanResult,
    RamInfo,
)

MARK_START = "<!-- HARDWARE_SPECS_PLACEHOLDER:START -->"
MARK_END = "<!-- HARDWARE_SPECS_PLACEHOLDER:END -->"


def _fake_detector(_config_path: Path) -> HardwareScanResult:
    return HardwareScanResult(
        captured_at="",
        os="TestOS-1.0",
        python="3.12.13",
        cpu=CpuInfo(model="TestCPU", cores_physical=4, cores_logical=8),
        ram=RamInfo(total_gb=16.0, available_gb=8.0),
        gpu=GpuInfo(present=True, model="TestGPU", vram_gb=12.0),
        disk=DiskInfo(free_gb=250.0, fs="NTFS", kind="NVMe", measured_at="/fixture"),
        write_receipts={},
    )


def _seed_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal repo tree with all three placeholder targets."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg_path = config_dir / "setup.json"
    cfg_path.write_text(
        json.dumps(
            {
                "version": "1.00",
                "init": {
                    "max_age_hours": 168,
                    "doc_targets": ["docs/PRD.md", "README.md"],
                    "placeholder_start": MARK_START,
                    "placeholder_end": MARK_END,
                    "keep_bak": True,
                },
                "hardware_constraints": None,
                "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    prd = docs / "PRD.md"
    prd.write_text(f"# PRD\n\n{MARK_START}\nOLD\n{MARK_END}\n\ntrailing\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"# README\n\n{MARK_START}\nOLD\n{MARK_END}\n", encoding="utf-8")
    return cfg_path, prd, readme


@pytest.mark.integration
def test_scanner_writes_all_three_targets_end_to_end(tmp_path: Path) -> None:
    """Single ``scan()`` call MUST inject config + patch PRD + patch README."""
    cfg_path, prd, readme = _seed_fixture(tmp_path)
    scanner = HardwareScanner(
        repo_root=tmp_path,
        config_path=cfg_path,
        detector=_fake_detector,
        clock=lambda: "2026-06-27T12:00:00Z",
    )

    result = scanner.scan()

    # All three write receipts ok.
    receipts = result.write_receipts
    assert receipts["config_setup_json"].status == "ok"
    assert receipts["docs_prd_md"].status == "ok"
    assert receipts["readme_md"].status == "ok"

    # Config now carries hardware_constraints sans write_receipts.
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["hardware_constraints"]["captured_at"] == "2026-06-27T12:00:00Z"
    assert cfg["hardware_constraints"]["cpu"]["cores_logical"] == 8
    assert "write_receipts" not in cfg["hardware_constraints"]

    # PRD + README markers preserved verbatim; content between them replaced.
    for path in (prd, readme):
        text = path.read_text(encoding="utf-8")
        assert text.count(MARK_START) == 1
        assert text.count(MARK_END) == 1
        between = text.split(MARK_START, 1)[1].split(MARK_END, 1)[0]
        assert "TestCPU" in between
        assert "TestGPU" in between
        assert "OLD" not in between

    # First write produced a .bak per file (keep_bak=true).
    assert (cfg_path.with_name(cfg_path.name + ".bak")).exists()
    assert (prd.with_name(prd.name + ".bak")).exists()
    assert (readme.with_name(readme.name + ".bak")).exists()


@pytest.mark.integration
def test_second_scan_is_idempotent_modulo_clock(tmp_path: Path) -> None:
    """Re-running with the same clock is byte-stable; receipts MUST be ``ok-noop``."""
    cfg_path, prd, readme = _seed_fixture(tmp_path)
    scanner = HardwareScanner(
        repo_root=tmp_path,
        config_path=cfg_path,
        detector=_fake_detector,
        clock=lambda: "2026-06-27T12:00:00Z",
    )

    scanner.scan()
    cfg_after_first = cfg_path.read_text(encoding="utf-8")
    prd_after_first = prd.read_text(encoding="utf-8")
    readme_after_first = readme.read_text(encoding="utf-8")

    second = scanner.scan()
    assert second.write_receipts["config_setup_json"].status == "ok-noop"
    assert second.write_receipts["docs_prd_md"].status == "ok-noop"
    assert second.write_receipts["readme_md"].status == "ok-noop"
    assert cfg_path.read_text(encoding="utf-8") == cfg_after_first
    assert prd.read_text(encoding="utf-8") == prd_after_first
    assert readme.read_text(encoding="utf-8") == readme_after_first


@pytest.mark.integration
def test_missing_doc_target_is_skipped_not_failed(tmp_path: Path) -> None:
    """A doc_target that doesn't exist MUST yield a ``skipped`` receipt, not ``fail``."""
    cfg_path, _, readme = _seed_fixture(tmp_path)
    # Delete PRD; README stays.
    (tmp_path / "docs" / "PRD.md").unlink()

    scanner = HardwareScanner(
        repo_root=tmp_path,
        config_path=cfg_path,
        detector=_fake_detector,
        clock=lambda: "2026-06-27T12:00:00Z",
    )
    result = scanner.scan()
    assert result.write_receipts["docs_prd_md"].status.startswith("skipped")
    assert result.write_receipts["readme_md"].status == "ok"
    assert result.write_receipts["config_setup_json"].status == "ok"
