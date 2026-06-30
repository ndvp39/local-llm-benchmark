"""Unit tests for SDK.run_baseline wiring (T-2.10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab import EnvironmentNotInitializedError, OnPremLlmSDK
from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant

_FRESH_CONFIG: dict[str, Any] = {
    "version": "1.00",
    "init": {"max_age_hours": 168},
    "hardware_constraints": {
        "captured_at": "2299-01-01T00:00:00Z",  # never stale
        "cpu": {"cores_logical": 8},
    },
    "sampling": {"memory_hz": 100},
    "energy": {"assumed_watts_active": 180},
    "generation": {"seed": 42, "max_new_tokens": 16},
    "target_models": [
        {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "label": "llama3-8b-fp16", "quantization": "fp16",
            "loader": "direct",
        },
    ],
}


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(_FRESH_CONFIG), encoding="utf-8")
    return p


def _sample_result() -> BackendRunResult:
    return BackendRunResult(
        run_id="run-001", started_at="2026-07-01T00:00:00Z",
        backend=BackendId.DIRECT, target_label="llama3-8b-fp16",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        quantization=Quant.FP16,
        prompt_tokens=8, completion_tokens=4,
        ttft_ms=100.0, tpot_ms=50.0, throughput_tps=1.0,
        peak_ram_mb=100.0, wall_s=1.5, energy_wh=0.075,
        completion_text="hi",
        raw_log_path="dummy.json",
    )


class TestSdkRunBaseline:
    def test_delegates_to_baseline_service_with_full_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def spy(**kw: Any) -> BackendRunResult:
            captured.update(kw)
            return _sample_result()

        monkeypatch.setattr(
            "on_prem_llm_lab.sdk.sdk._run_baseline", spy,
        )
        cfg = _write_cfg(tmp_path)
        sdk = OnPremLlmSDK(config_path=cfg, repo_root=tmp_path)
        result = sdk.run_baseline(
            "llama3-8b-fp16", prompt="Hello", max_new_tokens=32,
        )
        assert isinstance(result, BackendRunResult)
        assert captured["target_label"] == "llama3-8b-fp16"
        assert captured["prompt"] == "Hello"
        assert captured["max_new_tokens"] == 32
        assert captured["results_dir"] == tmp_path / "results"
        assert captured["repo_root"] == tmp_path
        # config_snapshot is passed as a dict, not the raw file path.
        assert captured["config"]["version"] == "1.00"
        assert captured["config"]["target_models"][0]["label"] == "llama3-8b-fp16"

    def test_env_guard_fires_before_delegation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel: list[bool] = []

        def spy(**kw: Any) -> BackendRunResult:
            sentinel.append(True)
            return _sample_result()

        monkeypatch.setattr(
            "on_prem_llm_lab.sdk.sdk._run_baseline", spy,
        )
        # Null constraints -> env-init guard MUST fire first.
        cfg = tmp_path / "setup.json"
        cfg.write_text(
            json.dumps({
                "version": "1.00",
                "init": {"max_age_hours": 168},
                "hardware_constraints": None,
            }), encoding="utf-8",
        )
        sdk = OnPremLlmSDK(config_path=cfg, repo_root=tmp_path)
        with pytest.raises(EnvironmentNotInitializedError):
            sdk.run_baseline("llama3-8b-fp16")
        assert sentinel == [], "baseline service must NOT be called when env-init fails"
