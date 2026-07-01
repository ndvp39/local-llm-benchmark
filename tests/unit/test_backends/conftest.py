"""Shared stubs + fixtures for ``tests/unit/test_backends/`` (T-3.1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from on_prem_llm_lab.backends.airllm_backend import AirLLMBackend
from on_prem_llm_lab.backends.base import Quant


class Tensor:
    """Tiny tensor-shaped stub: ``shape`` attribute + indexable."""

    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, idx: int) -> Tensor:
        return self


class Tokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> Any:
        return SimpleNamespace(input_ids=Tensor(8))

    def decode(self, output: Any, **kwargs: Any) -> str:
        return "airllm stub completion"


class StubAirLLMModel:
    """Minimal ``airllm.AutoModel``-shaped stub for AirLLMBackend tests."""

    layers = list(range(35))

    def __init__(self) -> None:
        self.tokenizer = Tokenizer()

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        return Tensor(input_ids.shape[-1] + max_new_tokens)


class StepClock:
    def __init__(self, timestamps: list[float]) -> None:
        self._it = iter(timestamps)

    def __call__(self) -> float:
        return next(self._it)


@pytest.fixture
def make_airllm() -> Any:
    """Factory fixture: returns ``make(tmp_path, calls=None, **overrides)``."""

    def _make(
        tmp_path: Path,
        calls: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> AirLLMBackend:
        rec = calls if calls is not None else []

        def factory(model_id: str, **kwargs: Any) -> StubAirLLMModel:
            rec.append({"model_id": model_id, **kwargs})
            return StubAirLLMModel()

        kwargs: dict[str, Any] = {
            "target_label": "test-target", "model_id": "test/model",
            "quantization": Quant.FP16, "layer_shards_saving_path": tmp_path,
            "min_free_disk_gb": 0.001, "factory": factory,
        }
        kwargs.update(overrides)
        return AirLLMBackend(**kwargs)

    return _make
