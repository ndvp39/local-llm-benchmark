"""Tests for ``OnPremLlmSDK`` (T-1.12 constructor + T-1.14 scan/init)."""

from __future__ import annotations

import json
from pathlib import Path

import on_prem_llm_lab
from on_prem_llm_lab.sdk import InitEnvResult, OnPremLlmSDK
from on_prem_llm_lab.services.hardware_scanner import (
    DEFAULT_PLACEHOLDER_END,
    DEFAULT_PLACEHOLDER_START,
)
from on_prem_llm_lab.services.hardware_scanner_types import (
    CpuInfo,
    DiskInfo,
    GpuInfo,
    HardwareScanResult,
    RamInfo,
)


def _canned() -> HardwareScanResult:
    return HardwareScanResult(
        captured_at="placeholder",
        os="Linux-5.x", python="3.12.13",
        cpu=CpuInfo(model="cpu", cores_physical=4, cores_logical=8),
        ram=RamInfo(total_gb=16.0, available_gb=8.0),
        gpu=GpuInfo(present=False, model=None, vram_gb=None),
        disk=DiskInfo(free_gb=100.0, fs=None, kind="unknown", measured_at="/tmp"),
    )


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "setup.json").write_text(
        json.dumps({
            "version": "1.00",
            "init": {
                "doc_targets": [],
                "placeholder_start": DEFAULT_PLACEHOLDER_START,
                "placeholder_end": DEFAULT_PLACEHOLDER_END,
                "keep_bak": False,
            },
            "hardware_constraints": None,
            "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
        }, indent=2),
        encoding="utf-8",
    )
    return tmp_path


def test_sdk_constructor_accepts_path(tmp_path: Path) -> None:
    cfg = tmp_path / "setup.json"
    cfg.write_text("{}", encoding="utf-8")
    sdk = OnPremLlmSDK(config_path=cfg)
    assert sdk.config_path == cfg
    assert sdk.env == {}
    assert sdk.repo_root == Path.cwd()  # T-1.14 default


def test_sdk_constructor_accepts_str_and_env(tmp_path: Path) -> None:
    cfg_str = str(tmp_path / "x.json")
    sdk = OnPremLlmSDK(config_path=cfg_str, env={"HF_TOKEN": "fake"})
    assert sdk.config_path == Path(cfg_str)
    assert sdk.env["HF_TOKEN"] == "fake"


def test_sdk_constructor_accepts_repo_root(tmp_path: Path) -> None:
    sdk = OnPremLlmSDK(config_path=tmp_path / "x.json", repo_root=tmp_path)
    assert sdk.repo_root == tmp_path


def test_sdk_is_reexported_from_root() -> None:
    """Constitution §3.1: SDK is the single entry point — must be importable from root."""
    assert on_prem_llm_lab.OnPremLlmSDK is OnPremLlmSDK


def test_scan_hardware_returns_populated_result(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    sdk = OnPremLlmSDK(config_path=root / "config" / "setup.json", repo_root=root)
    result = sdk.scan_hardware(
        detector=lambda _p: _canned(),
        clock=lambda: "2026-06-26T10:00:00Z",
    )
    assert result.captured_at == "2026-06-26T10:00:00Z"
    assert "config_setup_json" in result.write_receipts
    assert result.write_receipts["config_setup_json"].status == "ok"


def test_initialize_environment_ok_when_all_writes_succeed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    sdk = OnPremLlmSDK(config_path=root / "config" / "setup.json", repo_root=root)
    result = sdk.initialize_environment(
        detector=lambda _p: _canned(),
        clock=lambda: "2026-06-26T10:00:00Z",
    )
    assert isinstance(result, InitEnvResult)
    assert result.ok is True
    assert result.failures == []


def test_initialize_environment_reports_fail_when_config_missing(tmp_path: Path) -> None:
    sdk = OnPremLlmSDK(
        config_path=tmp_path / "missing_setup.json", repo_root=tmp_path,
    )
    result = sdk.initialize_environment(
        detector=lambda _p: _canned(),
        clock=lambda: "2026-06-26T10:00:00Z",
    )
    assert result.ok is False
    assert any("config_setup_json" in f for f in result.failures)
