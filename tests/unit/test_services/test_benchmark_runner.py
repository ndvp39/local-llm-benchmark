"""Unit tests for services/benchmark_runner.py (T-2.9)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.backends.base import (
    BackendId,
    BackendRunResult,
    InferenceBackend,
    Quant,
)
from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
from on_prem_llm_lab.services.benchmark_runner import BenchmarkRunner


def _sample_result(**overrides: Any) -> BackendRunResult:
    base: dict[str, Any] = {
        "run_id": "run-001", "started_at": "2026-07-01T00:00:00Z",
        "backend": BackendId.DIRECT, "target_label": "t",
        "model_id": "m", "quantization": Quant.FP16,
        "prompt_tokens": 8, "completion_tokens": 4,
        "ttft_ms": 100.0, "tpot_ms": 50.0, "throughput_tps": 1.0,
        "peak_ram_mb": 0.0, "wall_s": 1.5, "energy_wh": 0.0,
        "completion_text": "hi",
    }
    base.update(overrides)
    return BackendRunResult(**base)


class _MockBackend(InferenceBackend):
    def __init__(
        self,
        wait_for: threading.Event | None = None,
        result: BackendRunResult | None = None,
    ) -> None:
        self.events: list[str] = []
        self._wait = wait_for
        self._result = result or _sample_result()

    def load(self) -> None:
        self.events.append("load")

    def generate(
        self, prompt: str, *, max_new_tokens: int, params: dict[str, Any]
    ) -> BackendRunResult:
        self.events.append("generate")
        if self._wait is not None:
            self._wait.wait(timeout=1.0)
        return self._result

    def unload(self) -> None:
        self.events.append("unload")


def _step_sampler(
    values: list[float],
) -> tuple[Callable[[], MemoryReading], threading.Event]:
    """Emit ``values`` in order; set ``done`` after the last is returned."""
    idx = {"i": 0}
    done = threading.Event()

    def sampler() -> MemoryReading:
        i = idx["i"]
        idx["i"] += 1
        v = values[i] if i < len(values) else values[-1]
        if idx["i"] == len(values):
            done.set()
        return MemoryReading(ram_mb=v)

    return sampler, done


def _make_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "sampling": {"memory_hz": 1000},
        "energy": {"assumed_watts_active": 180},
        "generation": {"seed": 42},
    }
    base.update(overrides)
    return base


class TestLifecycleOrdering:
    def test_load_generate_unload_in_order(self, tmp_path: Path) -> None:
        sampler, done = _step_sampler([100.0, 200.0])
        backend = _MockBackend(wait_for=done)
        runner = BenchmarkRunner(
            _make_config(), tmp_path, ram_sampler=sampler,
        )
        runner.run(backend, prompt="hi", max_new_tokens=4)
        assert backend.events == ["load", "generate", "unload"]

    def test_unload_called_even_on_generate_failure(self, tmp_path: Path) -> None:
        class _ErroringBackend(_MockBackend):
            def generate(
                self, prompt: str, *, max_new_tokens: int,
                params: dict[str, Any],
            ) -> BackendRunResult:
                self.events.append("generate")
                raise RuntimeError("boom")

        sampler, _ = _step_sampler([100.0])
        backend = _ErroringBackend()
        runner = BenchmarkRunner(
            _make_config(), tmp_path, ram_sampler=sampler,
        )
        with pytest.raises(RuntimeError, match="boom"):
            runner.run(backend, prompt="hi", max_new_tokens=4)
        assert "unload" in backend.events


class TestResultEnrichment:
    def test_peak_ram_from_sampler_max(self, tmp_path: Path) -> None:
        sampler, done = _step_sampler([100.0, 500.0, 300.0])
        backend = _MockBackend(wait_for=done)
        runner = BenchmarkRunner(
            _make_config(), tmp_path, ram_sampler=sampler,
        )
        result = runner.run(backend, prompt="hi", max_new_tokens=4)
        assert result.peak_ram_mb == 500.0

    def test_energy_wh_from_wattage_times_wall_s(self, tmp_path: Path) -> None:
        # wall_s=1.5 * watts=180 / 3600 = 0.075 Wh
        sampler, done = _step_sampler([100.0])
        backend = _MockBackend(
            wait_for=done, result=_sample_result(wall_s=1.5),
        )
        runner = BenchmarkRunner(
            _make_config(), tmp_path, ram_sampler=sampler,
        )
        result = runner.run(backend, prompt="hi", max_new_tokens=4)
        assert result.energy_wh == pytest.approx(0.075)

    def test_raw_log_path_set_to_manifest_path(self, tmp_path: Path) -> None:
        sampler, done = _step_sampler([100.0])
        backend = _MockBackend(wait_for=done)
        runner = BenchmarkRunner(
            _make_config(), tmp_path, ram_sampler=sampler,
        )
        result = runner.run(backend, prompt="hi", max_new_tokens=4)
        assert result.raw_log_path is not None
        assert Path(result.raw_log_path).exists()


class TestManifest:
    def test_manifest_carries_seed_prompt_config_snapshot(
        self, tmp_path: Path
    ) -> None:
        sampler, done = _step_sampler([100.0])
        backend = _MockBackend(wait_for=done)
        config = _make_config()
        runner = BenchmarkRunner(config, tmp_path, ram_sampler=sampler)
        result = runner.run(backend, prompt="Hello", max_new_tokens=8)
        manifest = json.loads(Path(result.raw_log_path).read_text())
        assert manifest["seed"] == 42
        assert manifest["prompt"] == "Hello"
        assert manifest["max_new_tokens"] == 8
        assert manifest["target_label"] == "t"
        assert manifest["model_id"] == "m"
        assert manifest["backend"] == "direct"
        assert manifest["quantization"] == "fp16"
        assert manifest["config_snapshot"]["sampling"]["memory_hz"] == 1000
        assert manifest["run_result"]["peak_ram_mb"] == 100.0
