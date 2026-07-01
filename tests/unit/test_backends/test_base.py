"""Unit tests for backends/base.py (T-2.1).

Covers Quant + BackendId enum integrity, BackendRunResult construction
and frozen semantics, and the InferenceBackend ABC contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from on_prem_llm_lab.backends.base import (
    BackendId,
    BackendRunResult,
    InferenceBackend,
    Quant,
)
from on_prem_llm_lab.shared.automodel_factory import SUPPORTED_QUANTIZATIONS


class TestQuant:
    def test_factory_supported_is_subset_of_quant_enum(self) -> None:
        """DP-3 — the Direct factory supports a SUBSET of the Quant enum
        (only fp32 + fp16 after the DP-3 shrinkage). Q4/Q8 are handled by
        AirLLM; Q2/NF4 are placeholder labels per FR-Q-5."""
        assert set(SUPPORTED_QUANTIZATIONS) <= {q.value for q in Quant}
        assert set(SUPPORTED_QUANTIZATIONS) == {"fp32", "fp16"}

    def test_quant_is_string_subclass(self) -> None:
        """str-inheritance makes ``json.dumps(Quant.FP16) == '"fp16"'``."""
        assert isinstance(Quant.FP16, str)
        assert Quant.FP16 == "fp16"

    @pytest.mark.parametrize(
        "label", ["fp32", "fp16", "q8", "q4", "q2", "nf4"]
    )
    def test_each_label_round_trips(self, label: str) -> None:
        assert Quant(label).value == label


class TestBackendId:
    def test_three_members_only(self) -> None:
        assert {b.value for b in BackendId} == {"direct", "airllm", "api"}

    def test_is_string_subclass(self) -> None:
        assert BackendId.API == "api"
        assert isinstance(BackendId.AIRLLM, str)


def _sample_result(**overrides: Any) -> BackendRunResult:
    base: dict[str, Any] = {
        "run_id": "run-001",
        "started_at": "2026-06-30T20:00:00Z",
        "backend": BackendId.AIRLLM,
        "target_label": "llama3-8b-fp16",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quantization": Quant.FP16,
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "ttft_ms": 367_282.43,
        "tpot_ms": 368_350.11,
        "throughput_tps": 0.0027,
        "peak_ram_mb": 1286.99,
        "wall_s": 1103.99,
        "energy_wh": 35.5,
        "completion_text": "Hi!",
    }
    base.update(overrides)
    return BackendRunResult(**base)


class TestBackendRunResult:
    def test_construct_with_required_fields_only(self) -> None:
        r = _sample_result()
        assert r.run_id == "run-001"
        assert r.backend is BackendId.AIRLLM
        assert r.quantization is Quant.FP16
        assert r.peak_vram_mb is None
        assert r.raw_log_path is None

    def test_construct_with_optional_fields(self) -> None:
        r = _sample_result(
            peak_vram_mb=7610.0,
            raw_log_path="results/logs/x.jsonl",
        )
        assert r.peak_vram_mb == 7610.0
        assert r.raw_log_path == "results/logs/x.jsonl"

    def test_is_frozen(self) -> None:
        r = _sample_result()
        with pytest.raises(FrozenInstanceError):
            r.run_id = "tampered"  # type: ignore[misc]

    def test_kw_only_rejects_positional(self) -> None:
        """``kw_only=True`` means positional construction must fail."""
        with pytest.raises(TypeError):
            BackendRunResult("run-002")  # type: ignore[call-arg,misc]


class TestInferenceBackend:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            InferenceBackend()  # type: ignore[abstract]

    def test_subclass_missing_method_is_abstract(self) -> None:
        class Incomplete(InferenceBackend):
            def load(self) -> None: ...
            # missing ``generate`` + ``unload``

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates_and_returns_result(self) -> None:
        class Fake(InferenceBackend):
            def load(self) -> None: ...

            def generate(
                self,
                prompt: str,
                *,
                max_new_tokens: int,
                params: dict[str, Any],
            ) -> BackendRunResult:
                return _sample_result(completion_text=prompt)

            def unload(self) -> None: ...

        b = Fake()
        r = b.generate("hi", max_new_tokens=2, params={})
        assert isinstance(r, BackendRunResult)
        assert r.completion_text == "hi"
