"""Unit tests for HardwareScanner (T-1.13 / ADR-015).

Eight tests covering: detection (happy + error), config injection (happy + error),
doc patching (happy + 2 error variants), and the idempotency invariant. Real
:func:`default_detect` smoke-runs in test 1 only; the rest use a canned
:class:`HardwareScanResult` so the write paths are exercised deterministically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from on_prem_llm_lab.services.hardware_scanner import (
    DEFAULT_PLACEHOLDER_END,
    DEFAULT_PLACEHOLDER_START,
    HardwareScanner,
)
from on_prem_llm_lab.services.hardware_scanner_detect import default_detect
from on_prem_llm_lab.services.hardware_scanner_types import (
    CpuInfo,
    DiskInfo,
    GpuInfo,
    HardwareScanResult,
    RamInfo,
)
from on_prem_llm_lab.services.hardware_scanner_writes import (
    inject_into_setup_json,
    patch_doc_placeholders,
)


def _canned(captured_at: str = "2026-06-26T10:00:00Z") -> HardwareScanResult:
    return HardwareScanResult(
        captured_at=captured_at,
        os="Windows-10-10.0.19045",
        python="3.12.13",
        cpu=CpuInfo(model="Intel i7", cores_physical=8, cores_logical=16),
        ram=RamInfo(total_gb=32.0, available_gb=24.0),
        gpu=GpuInfo(present=True, model="RTX 3060", vram_gb=12.0),
        disk=DiskInfo(
            free_gb=250.0, fs="NTFS", kind="unknown",
            measured_at="D:/airllm_shards",
        ),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    setup = tmp_path / "config" / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "version": "1.00",
                "init": {
                    "doc_targets": ["docs/PRD.md", "README.md"],
                    "placeholder_start": DEFAULT_PLACEHOLDER_START,
                    "placeholder_end": DEFAULT_PLACEHOLDER_END,
                    "keep_bak": True,
                },
                "hardware_constraints": None,
                "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_detect_default_runs_without_crash(repo: Path) -> None:
    r = default_detect(repo / "config" / "setup.json")
    assert r.cpu.cores_logical >= 1
    assert r.ram.total_gb > 0
    assert isinstance(r.gpu.present, bool)
    assert r.disk.free_gb > 0
    assert r.os and r.python


def test_detect_handles_missing_pynvml(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # forces ImportError
    r = default_detect(repo / "config" / "setup.json")
    assert r.gpu.present is False
    assert r.gpu.vram_gb is None


def test_inject_into_setup_json_populates_hardware_constraints(repo: Path) -> None:
    receipt = inject_into_setup_json(
        repo / "config" / "setup.json", _canned(), keep_bak=True
    )
    assert receipt.status == "ok"
    data = json.loads((repo / "config" / "setup.json").read_text("utf-8"))
    assert data["hardware_constraints"]["cpu"]["cores_physical"] == 8
    assert data["hardware_constraints"]["gpu"]["model"] == "RTX 3060"


def test_inject_into_setup_json_reports_fail_when_file_missing(
    tmp_path: Path,
) -> None:
    receipt = inject_into_setup_json(
        tmp_path / "missing.json", _canned(), keep_bak=False
    )
    assert receipt.status == "fail"
    assert "not found" in (receipt.reason or "")


def test_patch_doc_replaces_between_markers(tmp_path: Path) -> None:
    doc = tmp_path / "PRD.md"
    doc.write_text(
        f"Intro\n\n{DEFAULT_PLACEHOLDER_START}\nold\n"
        f"{DEFAULT_PLACEHOLDER_END}\n\nOutro\n",
        encoding="utf-8",
    )
    receipt = patch_doc_placeholders(
        doc, "| OS | x |",
        DEFAULT_PLACEHOLDER_START, DEFAULT_PLACEHOLDER_END,
        keep_bak=False,
    )
    assert receipt.status == "ok"
    text = doc.read_text("utf-8")
    assert "| OS | x |" in text
    assert "old" not in text
    assert text.startswith("Intro")
    assert text.endswith("Outro\n")
    # Markers preserved verbatim for the next scan.
    assert DEFAULT_PLACEHOLDER_START in text
    assert DEFAULT_PLACEHOLDER_END in text


def test_patch_doc_skipped_when_marker_pair_missing(tmp_path: Path) -> None:
    doc = tmp_path / "PRD.md"
    doc.write_text("No markers here\n", encoding="utf-8")
    receipt = patch_doc_placeholders(
        doc, "table",
        DEFAULT_PLACEHOLDER_START, DEFAULT_PLACEHOLDER_END,
        keep_bak=False,
    )
    assert receipt.status == "skipped-no-markers"


def test_patch_doc_skipped_when_file_missing(tmp_path: Path) -> None:
    receipt = patch_doc_placeholders(
        tmp_path / "nope.md", "table",
        DEFAULT_PLACEHOLDER_START, DEFAULT_PLACEHOLDER_END,
        keep_bak=False,
    )
    assert receipt.status == "skipped-file-missing"


def _snap(root: Path) -> dict[str, bytes]:
    return {
        rel: (root / rel).read_bytes()
        for rel in ("docs/PRD.md", "README.md", "config/setup.json")
    }


def test_scan_is_idempotent_modulo_clock(repo: Path) -> None:
    (repo / "docs").mkdir()
    payload = f"{DEFAULT_PLACEHOLDER_START}\nold\n{DEFAULT_PLACEHOLDER_END}\n"
    (repo / "docs" / "PRD.md").write_text(payload, encoding="utf-8")
    (repo / "README.md").write_text(payload, encoding="utf-8")
    scanner = HardwareScanner(
        repo_root=repo,
        config_path=repo / "config" / "setup.json",
        detector=lambda _p: _canned(),
        clock=lambda: "2026-06-26T10:00:00Z",
    )
    scanner.scan()
    snap1 = _snap(repo)
    scanner.scan()  # same fake clock → must be byte-identical
    assert _snap(repo) == snap1
