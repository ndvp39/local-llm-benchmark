"""Unit tests for services/airllm_service.py (T-3.5a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant
from on_prem_llm_lab.services.airllm_service import run_airllm


def _cfg() -> dict[str, Any]:
    return {
        "version": "1.00",
        "sampling": {"memory_hz": 100},
        "energy": {"assumed_watts_active": 180},
        "generation": {"seed": 42, "max_new_tokens": 8, "baseline_prompt": "hi."},
        "airllm": {"layer_shards_saving_path": "/tmp/shards"},
        "target_models": [
            {"id": "meta-llama/L", "label": "llama3-8b-fp16",
             "quantization": "fp16", "loader": "airllm"},
            {"id": "Qwen/Q", "label": "qwen2-7b-q4",
             "quantization": "q4", "loader": "airllm"},
        ],
    }


def _sample() -> BackendRunResult:
    return BackendRunResult(
        run_id="r", started_at="2026-07-01T00:00:00Z",
        backend=BackendId.AIRLLM, target_label="llama3-8b-fp16",
        model_id="meta-llama/L", quantization=Quant.FP16,
        prompt_tokens=8, completion_tokens=8,
        ttft_ms=100.0, tpot_ms=50.0, throughput_tps=1.0,
        peak_ram_mb=100.0, wall_s=1.5, energy_wh=0.075,
        completion_text="ok",
    )


class _SpyBackend:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs


class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.results_dir = args[1] if len(args) > 1 else kwargs.get("results_dir")

    def run(self, backend: Any, **kwargs: Any) -> BackendRunResult:
        assert self.results_dir is not None
        p = Path(self.results_dir) / "run_r.json"
        p.write_text(json.dumps({"stub": True}), encoding="utf-8")
        return _sample_replace(raw_log_path=str(p))


class _RaisingRunner(_StubRunner):
    def run(self, backend: Any, **kwargs: Any) -> BackendRunResult:
        raise RuntimeError("boom-AirLLM")


def _sample_replace(**overrides: Any) -> BackendRunResult:
    import dataclasses
    return dataclasses.replace(_sample(), **overrides)


class TestUnknownTarget:
    def test_unknown_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown target_label"):
            run_airllm(
                target_label="missing", config=_cfg(), results_dir=tmp_path,
                backend_factory=_SpyBackend, runner_factory=_StubRunner,
                ram_sampler=lambda: __import__("on_prem_llm_lab.mixins.memory_sampling_mixin",
                                                fromlist=["MemoryReading"]).MemoryReading(ram_mb=0.0),
                clock=lambda: "20260701T000000Z",
            )


class TestHappyPath:
    def test_success_renames_manifest_to_airllm_convention(
        self, tmp_path: Path,
    ) -> None:
        from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
        result = run_airllm(
            target_label="llama3-8b-fp16", config=_cfg(),
            results_dir=tmp_path, quantization="q4",
            backend_factory=_SpyBackend, runner_factory=_StubRunner,
            ram_sampler=lambda: MemoryReading(ram_mb=0.0),
            clock=lambda: "20260701T000000Z",
        )
        new_path = tmp_path / "airllm_llama3-8b-fp16_q4_20260701T000000Z.json"
        assert new_path.exists()
        assert result.raw_log_path == str(new_path)
        assert not (tmp_path / "run_r.json").exists()

    def test_backend_receives_target_and_quantization_override(
        self, tmp_path: Path,
    ) -> None:
        from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
        captured: dict[str, Any] = {}

        def factory(**kw: Any) -> _SpyBackend:
            captured.update(kw)
            return _SpyBackend(**kw)

        run_airllm(
            target_label="qwen2-7b-q4", config=_cfg(),
            results_dir=tmp_path, quantization="q8",
            backend_factory=factory, runner_factory=_StubRunner,
            ram_sampler=lambda: MemoryReading(ram_mb=0.0),
            clock=lambda: "T",
        )
        assert captured["target_label"] == "qwen2-7b-q4"
        assert captured["model_id"] == "Qwen/Q"
        assert captured["quantization"] == "q8"  # override wins over target's "q4"

    def test_quantization_defaults_to_target_setting(
        self, tmp_path: Path,
    ) -> None:
        from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
        captured: dict[str, Any] = {}

        def factory(**kw: Any) -> _SpyBackend:
            captured.update(kw)
            return _SpyBackend(**kw)

        run_airllm(
            target_label="qwen2-7b-q4", config=_cfg(), results_dir=tmp_path,
            backend_factory=factory, runner_factory=_StubRunner,
            ram_sampler=lambda: MemoryReading(ram_mb=0.0),
            clock=lambda: "T",
        )
        assert captured["quantization"] == "q4"  # from target_models


class TestFailurePath:
    def test_failure_writes_structured_manifest_and_logs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
        with pytest.raises(RuntimeError, match="boom-AirLLM"):
            run_airllm(
                target_label="llama3-8b-fp16", config=_cfg(),
                results_dir=tmp_path, quantization="q4",
                backend_factory=_SpyBackend, runner_factory=_RaisingRunner,
                ram_sampler=lambda: MemoryReading(ram_mb=0.0),
                clock=lambda: "20260701T000000Z",
            )
        failure_path = tmp_path / (
            "airllm_llama3-8b-fp16_q4_20260701T000000Z_failure.json"
        )
        assert failure_path.exists()
        m = json.loads(failure_path.read_text(encoding="utf-8"))
        assert m["status"] == "failed"
        assert m["error_type"] == "RuntimeError"
        assert m["quantization"] == "q4"
        assert "boom-AirLLM" in m["error_message"]
        captured = capsys.readouterr()
        assert "[airllm] FAIL llama3-8b-fp16 @ q4" in captured.err
