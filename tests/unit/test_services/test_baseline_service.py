"""Unit tests for services/baseline_service.py (T-2.10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant
from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
from on_prem_llm_lab.services.baseline_service import run_baseline


def _make_config() -> dict[str, Any]:
    return {
        "version": "1.00",
        "sampling": {"memory_hz": 100},
        "energy": {"assumed_watts_active": 180},
        "generation": {"seed": 42, "max_new_tokens": 16},
        "target_models": [
            {
                "id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "label": "llama3-8b-fp16",
                "quantization": "fp16", "loader": "direct",
            },
            {
                "id": "Qwen/Qwen2-7B-Instruct", "label": "qwen2-7b-q4",
                "quantization": "q4", "loader": "airllm",
            },
        ],
    }


def _sample_result(**overrides: Any) -> BackendRunResult:
    base: dict[str, Any] = {
        "run_id": "run-001", "started_at": "2026-07-01T00:00:00Z",
        "backend": BackendId.DIRECT, "target_label": "llama3-8b-fp16",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quantization": Quant.FP16,
        "prompt_tokens": 8, "completion_tokens": 4,
        "ttft_ms": 100.0, "tpot_ms": 50.0, "throughput_tps": 1.0,
        "peak_ram_mb": 100.0, "wall_s": 1.5, "energy_wh": 0.075,
        "completion_text": "hi",
    }
    base.update(overrides)
    return BackendRunResult(**base)


class _SpyBackend:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs


class _StubRunner:
    """Pretends to run the backend; writes a fake manifest and returns a result."""

    def __init__(
        self, config: Any, results_dir: Path, *,
        ram_sampler: Any, repo_root: Any,
    ) -> None:
        self.results_dir = results_dir

    def run(self, backend: Any, **kwargs: Any) -> BackendRunResult:
        manifest_path = self.results_dir / "run_run-001.json"
        manifest_path.write_text(json.dumps({"stub": True}), encoding="utf-8")
        return _sample_result(raw_log_path=str(manifest_path))


class _RaisingRunner(_StubRunner):
    def run(self, backend: Any, **kwargs: Any) -> BackendRunResult:
        raise RuntimeError("boom-OOM")


class TestUnknownTarget:
    def test_unknown_target_label_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unknown target_label"):
            run_baseline(
                target_label="missing", config=_make_config(),
                results_dir=tmp_path,
                backend_factory=_SpyBackend, runner_factory=_StubRunner,
                ram_sampler=lambda: MemoryReading(ram_mb=0.0),
                clock=lambda: "20260701T000000Z",
            )


class TestHappyPath:
    def test_success_renames_manifest_to_baseline_label_ts_json(
        self, tmp_path: Path
    ) -> None:
        result = run_baseline(
            target_label="llama3-8b-fp16", config=_make_config(),
            results_dir=tmp_path, prompt="Hello",
            backend_factory=_SpyBackend, runner_factory=_StubRunner,
            ram_sampler=lambda: MemoryReading(ram_mb=0.0),
            clock=lambda: "20260701T000000Z",
        )
        new_path = tmp_path / "baseline_llama3-8b-fp16_20260701T000000Z.json"
        assert new_path.exists()
        assert result.raw_log_path == str(new_path)
        # Old run_<id>.json is gone (renamed, not copied).
        assert not (tmp_path / "run_run-001.json").exists()

    def test_passes_target_fields_to_backend_factory(
        self, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}

        def factory(**kw: Any) -> _SpyBackend:
            captured.update(kw)
            return _SpyBackend(**kw)

        run_baseline(
            target_label="qwen2-7b-q4", config=_make_config(),
            results_dir=tmp_path,
            backend_factory=factory, runner_factory=_StubRunner,
            ram_sampler=lambda: MemoryReading(ram_mb=0.0),
            clock=lambda: "20260701T000000Z",
        )
        assert captured["target_label"] == "qwen2-7b-q4"
        assert captured["model_id"] == "Qwen/Qwen2-7B-Instruct"
        assert captured["quantization"] == "q4"


class TestFailurePath:
    def test_failure_writes_failure_manifest_and_logs_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(RuntimeError, match="boom-OOM"):
            run_baseline(
                target_label="llama3-8b-fp16", config=_make_config(),
                results_dir=tmp_path,
                backend_factory=_SpyBackend, runner_factory=_RaisingRunner,
                ram_sampler=lambda: MemoryReading(ram_mb=0.0),
                clock=lambda: "20260701T000000Z",
            )
        failure_path = tmp_path / (
            "baseline_llama3-8b-fp16_20260701T000000Z_failure.json"
        )
        assert failure_path.exists()
        manifest = json.loads(failure_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["error_type"] == "RuntimeError"
        assert "boom-OOM" in manifest["error_message"]
        assert manifest["target_label"] == "llama3-8b-fp16"
        # SC-1 — failure surfaced on stderr.
        captured = capsys.readouterr()
        assert "[baseline] FAIL llama3-8b-fp16" in captured.err
        assert "RuntimeError" in captured.err
