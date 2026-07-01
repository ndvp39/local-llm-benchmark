"""Unit tests for backends/direct_backend.py (T-2.8).

A tiny stub model + tokenizer pair stands in for the real
``AutoModelForCausalLM`` so unit tests never trigger an HF download.
The injected ``factory`` + ``clock`` seams keep the back-end's
behaviour fully deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from on_prem_llm_lab.backends.base import (
    BackendId,
    BackendRunResult,
    Quant,
)
from on_prem_llm_lab.backends.direct_backend import DirectBackend
from on_prem_llm_lab.shared.automodel_factory import LoadedModel


class _Tensor:
    """Minimal tensor-shaped object: ``shape`` attribute + indexable."""

    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, idx: int) -> _Tensor:
        return self


class _Tokenized:
    def __init__(self, prompt_len: int) -> None:
        self.input_ids = _Tensor(prompt_len)


class _StubTokenizer:
    def __init__(self) -> None:
        self.decode_calls = 0

    def __call__(self, prompt: str, return_tensors: str = "pt") -> _Tokenized:
        return _Tokenized(prompt_len=8)

    def decode(self, output: Any, **kwargs: Any) -> str:
        self.decode_calls += 1
        return "stub completion"


class _StubModel:
    def __init__(self) -> None:
        self.generate_calls: list[int] = []

    def generate(self, input_ids: _Tensor, max_new_tokens: int) -> _Tensor:
        self.generate_calls.append(max_new_tokens)
        return _Tensor(input_ids.shape[-1] + max_new_tokens)


def _stub_loaded() -> LoadedModel:
    return LoadedModel(
        model=_StubModel(), tokenizer=_StubTokenizer(),
        resolved_dtype="torch.float16", resolved_device_map="auto",
    )


def _factory_recording(calls: list[tuple[str, str]]) -> Any:
    def _factory(model_id: str, quantization: str, **kwargs: Any) -> LoadedModel:
        calls.append((model_id, quantization))
        return _stub_loaded()
    return _factory


class _StepClock:
    """Returns preset timestamps in order on each call."""

    def __init__(self, timestamps: list[float]) -> None:
        self._it = iter(timestamps)

    def __call__(self) -> float:
        return next(self._it)


class TestConstruction:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [("fp16", Quant.FP16), (Quant.Q4, Quant.Q4)],
    )
    def test_quantization_accepted_as_str_or_enum(
        self, given: Any, expected: Quant
    ) -> None:
        be = DirectBackend(
            target_label="t", model_id="m", quantization=given,
            factory=_factory_recording([]),
        )
        assert be._quantization is expected


class TestLoad:
    def test_calls_factory_with_model_id_and_quant_label(self) -> None:
        calls: list[tuple[str, str]] = []
        be = DirectBackend(
            target_label="synthetic-fp16",
            model_id="ndvp/synthetic-tiny",  # not in HF cache -> pre-flight skips
            quantization=Quant.FP16,
            factory=_factory_recording(calls),
        )
        be.load()
        assert calls == [("ndvp/synthetic-tiny", "fp16")]

    def test_load_is_idempotent(self) -> None:
        calls: list[tuple[str, str]] = []
        be = DirectBackend(
            target_label="t", model_id="m", quantization=Quant.FP16,
            factory=_factory_recording(calls),
        )
        be.load()
        be.load()
        assert len(calls) == 1


class TestGenerate:
    def test_raises_before_load(self) -> None:
        be = DirectBackend(
            target_label="t", model_id="m", quantization=Quant.FP16,
            factory=_factory_recording([]),
        )
        with pytest.raises(RuntimeError, match="before load"):
            be.generate("hi", max_new_tokens=8, params={})

    def test_returns_well_shaped_backend_run_result(self) -> None:
        # t0=0, t_first=0.1, t_last=1.1 -> TTFT=100, wall=1.1.
        clock = _StepClock([0.0, 0.1, 1.1])
        be = DirectBackend(
            target_label="synthetic-fp16",
            model_id="ndvp/synthetic-tiny",  # not in HF cache -> pre-flight skips
            quantization=Quant.FP16,
            factory=_factory_recording([]), clock=clock,
        )
        be.load()
        result = be.generate("Hello", max_new_tokens=4, params={})
        assert isinstance(result, BackendRunResult)
        assert result.backend is BackendId.DIRECT
        assert result.target_label == "synthetic-fp16"
        assert result.model_id == "ndvp/synthetic-tiny"
        assert result.quantization is Quant.FP16
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4
        assert result.ttft_ms == pytest.approx(100.0)
        # tpot = (1.1 - 0.1) * 1000 / (4 - 1) = 333.33...
        assert result.tpot_ms == pytest.approx(1000.0 / 3.0)
        assert result.wall_s == pytest.approx(1.1)
        assert result.throughput_tps == pytest.approx(4 / 1.1)
        assert result.completion_text == "stub completion"
        assert result.peak_ram_mb == 0.0  # runner fills this in
        assert result.energy_wh == 0.0  # runner fills this in

    def test_records_two_generate_calls_with_correct_token_budgets(self) -> None:
        clock = _StepClock([0.0, 0.1, 1.1])
        factory_calls: list[tuple[str, str]] = []
        stub_model = _StubModel()

        def factory(model_id: str, quantization: str, **kwargs: Any) -> LoadedModel:
            factory_calls.append((model_id, quantization))
            return LoadedModel(
                model=stub_model, tokenizer=_StubTokenizer(),
                resolved_dtype="torch.float16", resolved_device_map="auto",
            )

        be = DirectBackend(
            target_label="t", model_id="m", quantization=Quant.FP16,
            factory=factory, clock=clock,
        )
        be.load()
        be.generate("hi", max_new_tokens=4, params={})
        assert stub_model.generate_calls == [1, 4]


class TestUnload:
    def test_releases_loaded_model_and_is_idempotent(self) -> None:
        be = DirectBackend(
            target_label="t", model_id="m", quantization=Quant.FP16,
            factory=_factory_recording([]),
        )
        be.load()
        assert be._loaded is not None
        be.unload()
        be.unload()  # second call must not error
        assert be._loaded is None
