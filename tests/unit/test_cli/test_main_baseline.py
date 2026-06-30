"""CLI tests for ``run-baseline`` subcommand (T-2.10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from on_prem_llm_lab import OnPremLlmSDK
from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant
from on_prem_llm_lab.cli.main import app

_CONFIG: dict[str, Any] = {
    "version": "1.00",
    "init": {"max_age_hours": 168, "doc_targets": []},
    "hardware_constraints": {"captured_at": "2299-01-01T00:00:00Z"},
    "sampling": {"memory_hz": 100},
    "energy": {"assumed_watts_active": 180},
    "generation": {"seed": 42, "max_new_tokens": 16},
    "target_models": [
        {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "label": "llama3-8b-fp16", "quantization": "fp16",
            "loader": "direct",
        },
        {
            "id": "Qwen/Qwen2-7B-Instruct", "label": "qwen2-7b-q4",
            "quantization": "q4", "loader": "airllm",
        },
    ],
}


def _write_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "setup.json"
    cfg.write_text(json.dumps(_CONFIG), encoding="utf-8")
    return cfg


def _fake_result(label: str, log_path: str = "x.json") -> BackendRunResult:
    return BackendRunResult(
        run_id="r", started_at="2026-07-01T00:00:00Z",
        backend=BackendId.DIRECT, target_label=label,
        model_id="m", quantization=Quant.FP16,
        prompt_tokens=1, completion_tokens=1,
        ttft_ms=1.0, tpot_ms=1.0, throughput_tps=1.0,
        peak_ram_mb=1.0, wall_s=1.0, energy_wh=1.0,
        completion_text="ok", raw_log_path=log_path,
    )


def test_cli_help_lists_run_baseline() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-baseline" in result.stdout


def test_cli_run_baseline_specific_target_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_cfg(tmp_path)
    calls: list[str] = []

    def spy(self: OnPremLlmSDK, label: str, **kw: Any) -> BackendRunResult:
        calls.append(label)
        return _fake_result(label, log_path=str(tmp_path / f"baseline_{label}_x.json"))

    monkeypatch.setattr(OnPremLlmSDK, "run_baseline", spy)
    result = CliRunner().invoke(
        app, ["run-baseline", "llama3-8b-fp16",
              "--config", str(cfg), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == ["llama3-8b-fp16"]
    assert "OK: llama3-8b-fp16" in result.stdout


def test_cli_run_baseline_no_arg_iterates_all_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_cfg(tmp_path)
    calls: list[str] = []

    def spy(self: OnPremLlmSDK, label: str, **kw: Any) -> BackendRunResult:
        calls.append(label)
        return _fake_result(label)

    monkeypatch.setattr(OnPremLlmSDK, "run_baseline", spy)
    result = CliRunner().invoke(
        app, ["run-baseline", "--config", str(cfg), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == ["llama3-8b-fp16", "qwen2-7b-q4"]


def test_cli_run_baseline_failure_exits_nonzero_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_cfg(tmp_path)

    def raises(self: OnPremLlmSDK, label: str, **kw: Any) -> BackendRunResult:
        raise RuntimeError("boom-OOM")

    monkeypatch.setattr(OnPremLlmSDK, "run_baseline", raises)
    result = CliRunner().invoke(
        app, ["run-baseline", "llama3-8b-fp16",
              "--config", str(cfg), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 1, result.stdout
    assert "FAIL: llama3-8b-fp16" in result.stdout
    assert "RuntimeError" in result.stdout
    assert "boom-OOM" in result.stdout


def test_cli_run_baseline_unknown_target_label_is_bad_param(
    tmp_path: Path
) -> None:
    cfg = _write_cfg(tmp_path)
    result = CliRunner().invoke(
        app, ["run-baseline", "totally-bogus",
              "--config", str(cfg), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "Unknown target_label" in (result.stdout + (result.stderr or ""))
