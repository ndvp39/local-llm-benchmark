"""CLI tests for ``run-airllm`` subcommand (T-3.5a)."""

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
    "generation": {"seed": 42, "max_new_tokens": 8, "baseline_prompt": "hi."},
    "airllm": {"layer_shards_saving_path": "/tmp/shards"},
    "target_models": [
        {"id": "meta-llama/L", "label": "llama3-8b-fp16",
         "quantization": "fp16", "loader": "airllm"},
    ],
}


def _write_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "setup.json"
    cfg.write_text(json.dumps(_CONFIG), encoding="utf-8")
    return cfg


def _fake_result(label: str, log_path: str = "x.json") -> BackendRunResult:
    return BackendRunResult(
        run_id="r", started_at="2026-07-01T00:00:00Z",
        backend=BackendId.AIRLLM, target_label=label,
        model_id="m", quantization=Quant.Q4,
        prompt_tokens=1, completion_tokens=1,
        ttft_ms=1.0, tpot_ms=1.0, throughput_tps=1.0,
        peak_ram_mb=1.0, wall_s=1.0, energy_wh=1.0,
        completion_text="ok", raw_log_path=log_path,
    )


def test_cli_help_lists_run_airllm() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-airllm" in result.stdout


def test_cli_run_airllm_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cfg = _write_cfg(tmp_path)
    calls: list[tuple[str, dict]] = []

    def spy(self: OnPremLlmSDK, label: str, **kw: Any) -> BackendRunResult:
        calls.append((label, kw))
        return _fake_result(label, log_path=str(tmp_path / f"airllm_{label}_q4_x.json"))

    monkeypatch.setattr(OnPremLlmSDK, "run_airllm", spy)
    result = CliRunner().invoke(app, [
        "run-airllm", "llama3-8b-fp16",
        "--quantization", "q4", "--max-new-tokens", "64",
        "--config", str(cfg), "--repo-root", str(tmp_path),
    ])
    assert result.exit_code == 0, result.stdout
    assert calls[0][0] == "llama3-8b-fp16"
    assert calls[0][1]["quantization"] == "q4"
    assert calls[0][1]["max_new_tokens"] == 64
    assert "OK: llama3-8b-fp16" in result.stdout


def test_cli_run_airllm_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cfg = _write_cfg(tmp_path)

    def raises(self: OnPremLlmSDK, label: str, **kw: Any) -> BackendRunResult:
        raise RuntimeError("boom-AirLLM")

    monkeypatch.setattr(OnPremLlmSDK, "run_airllm", raises)
    result = CliRunner().invoke(app, [
        "run-airllm", "llama3-8b-fp16",
        "--config", str(cfg), "--repo-root", str(tmp_path),
    ])
    assert result.exit_code == 1, result.stdout
    assert "FAIL: llama3-8b-fp16" in result.stdout
    assert "boom-AirLLM" in result.stdout
