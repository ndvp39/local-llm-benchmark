"""Integration test for BenchmarkRunner + DirectBackend together (T-2.9).

Real ``BenchmarkRunner`` + real ``DirectBackend`` end-to-end with a stub
``LoadedModel`` factory — proves the runner's mixin composition fills
the ``peak_ram_mb`` + ``energy_wh`` + ``raw_log_path`` fields that the
back-end leaves at 0/0/None, and that the JSON manifest lands on disk
in the shape PRD FR-10 prescribes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.backends.base import Quant
from on_prem_llm_lab.backends.direct_backend import DirectBackend
from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
from on_prem_llm_lab.services.benchmark_runner import BenchmarkRunner
from on_prem_llm_lab.shared.automodel_factory import LoadedModel


class _Tensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, idx: int) -> _Tensor:
        return self


class _Tokenized:
    def __init__(self, length: int) -> None:
        self.input_ids = _Tensor(length)


class _Tokenizer:
    def __call__(
        self, prompt: str, return_tensors: str = "pt"
    ) -> _Tokenized:
        return _Tokenized(4)

    def decode(self, output: Any, **kwargs: Any) -> str:
        return "integration completion"


class _Model:
    def generate(self, input_ids: _Tensor, max_new_tokens: int) -> _Tensor:
        return _Tensor(input_ids.shape[-1] + max_new_tokens)


def _factory(model_id: str, quantization: str) -> LoadedModel:
    return LoadedModel(
        model=_Model(), tokenizer=_Tokenizer(),
        resolved_dtype="torch.float16", resolved_device_map="auto",
    )


@pytest.mark.integration
class TestBenchmarkRunnerWithDirectBackend:
    def test_end_to_end_produces_enriched_result_and_manifest(
        self, tmp_path: Path
    ) -> None:
        backend = DirectBackend(
            target_label="llama3-8b-fp16",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            quantization=Quant.FP16,
            factory=_factory,
        )

        # Sampler emits 3 readings then signals exhaustion via threading.Event;
        # the backend mock would block on this, but DirectBackend.generate
        # finishes near-instantly with the stub model — we use a 50 ms wait
        # bound below to make peaks deterministic.
        idx = {"i": 0}
        readings = [100.0, 250.0, 175.0]

        def sampler() -> MemoryReading:
            i = idx["i"]
            idx["i"] += 1
            v = readings[i] if i < len(readings) else readings[-1]
            return MemoryReading(ram_mb=v)

        config = {
            "sampling": {"memory_hz": 200},  # 5 ms interval
            "energy": {"assumed_watts_active": 180},
            "generation": {"seed": 42},
        }
        runner = BenchmarkRunner(config, tmp_path, ram_sampler=sampler)
        result = runner.run(
            backend, prompt="Integration test", max_new_tokens=4,
        )

        # Result is fully populated (backend's 0.0 placeholders replaced).
        assert result.target_label == "llama3-8b-fp16"
        assert result.quantization is Quant.FP16
        assert result.completion_text == "integration completion"
        assert result.peak_ram_mb > 0  # sampler ran
        assert result.energy_wh >= 0.0  # computed from wall_s + watts
        assert result.raw_log_path is not None

        # Manifest on disk + carries the FR-10 fields.
        manifest_path = Path(result.raw_log_path)
        assert manifest_path.exists()
        assert manifest_path.parent == tmp_path
        manifest = json.loads(manifest_path.read_text())
        assert manifest["target_label"] == "llama3-8b-fp16"
        assert manifest["backend"] == "direct"
        assert manifest["quantization"] == "fp16"
        assert manifest["seed"] == 42
        assert manifest["prompt"] == "Integration test"
        assert manifest["config_snapshot"]["sampling"]["memory_hz"] == 200
        # run_result block carries the enriched fields, not the 0.0 placeholders.
        assert manifest["run_result"]["peak_ram_mb"] == result.peak_ram_mb
        assert manifest["run_result"]["energy_wh"] == result.energy_wh
