"""Unit tests for the DP-3 quantization support matrix (T-3.3).

Covers U-Q-1..10 from ``docs/PRD_quantization.md`` §9.1 — the
``is_supported`` / ``supported_matrix`` / ``require_supported`` trio
in ``shared/automodel_factory.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from on_prem_llm_lab.backends.base import BackendId, Quant
from on_prem_llm_lab.backends.direct_backend import DirectBackend
from on_prem_llm_lab.shared import automodel_factory as amf
from on_prem_llm_lab.shared.automodel_factory import LoadedModel


def _stub_loaded() -> LoadedModel:
    return LoadedModel(
        model=object(), tokenizer=object(),
        resolved_dtype="stub", resolved_device_map="stub",
    )


def _rec(calls: list[tuple[str, str]]) -> Any:
    def _f(model_id: str, quantization: str, **kwargs: Any) -> LoadedModel:
        calls.append((model_id, quantization))
        return _stub_loaded()
    return _f


class TestIsSupported:
    # U-Q-1
    @pytest.mark.parametrize(
        ("backend", "quantization"),
        [
            (BackendId.DIRECT, Quant.FP32),
            (BackendId.DIRECT, Quant.FP16),
            (BackendId.AIRLLM, Quant.FP16),
            (BackendId.AIRLLM, Quant.Q4),
            (BackendId.AIRLLM, Quant.Q8),
            (BackendId.API, Quant.FP16),
        ],
    )
    def test_matrix_positives(
        self, backend: BackendId, quantization: Quant,
    ) -> None:
        assert amf.is_supported(backend, quantization) is True

    # U-Q-2
    def test_direct_rejects_q4(self) -> None:
        assert amf.is_supported(BackendId.DIRECT, Quant.Q4) is False

    # U-Q-3
    def test_airllm_rejects_fp32(self) -> None:
        assert amf.is_supported(BackendId.AIRLLM, Quant.FP32) is False

    # U-Q-4
    @pytest.mark.parametrize(
        "backend", [BackendId.DIRECT, BackendId.AIRLLM, BackendId.API],
    )
    def test_q2_unsupported_everywhere(self, backend: BackendId) -> None:
        assert amf.is_supported(backend, Quant.Q2) is False

    # U-Q-5
    @pytest.mark.parametrize(
        "backend", [BackendId.DIRECT, BackendId.AIRLLM, BackendId.API],
    )
    def test_nf4_unsupported_everywhere(self, backend: BackendId) -> None:
        assert amf.is_supported(backend, Quant.NF4) is False

    # U-Q-9 — str + Quant parametrized
    @pytest.mark.parametrize(
        ("backend", "quantization", "expected"),
        [
            ("direct", "fp16", True),
            ("direct", "q4", False),
            (BackendId.DIRECT, "fp16", True),
            ("direct", Quant.FP16, True),
            (BackendId.AIRLLM, Quant.Q4, True),
        ],
    )
    def test_accepts_str_or_enum(
        self, backend: object, quantization: object, expected: bool,
    ) -> None:
        assert amf.is_supported(backend, quantization) is expected


class TestSupportedMatrix:
    # U-Q-6
    def test_three_backend_keys(self) -> None:
        m = amf.supported_matrix()
        assert set(m) == {"direct", "airllm", "api"}

    def test_returned_dict_is_a_copy(self) -> None:
        """Mutating the returned dict MUST NOT affect subsequent calls."""
        m = amf.supported_matrix()
        m.pop("direct", None)
        assert "direct" in amf.supported_matrix()

    def test_matrix_contents_v1_00(self) -> None:
        """DP-3 §5.1 — canonical v1.00 policy shape."""
        m = amf.supported_matrix()
        assert m["direct"] == frozenset({"fp32", "fp16"})
        assert m["airllm"] == frozenset({"fp16", "q4", "q8"})
        assert m["api"] == frozenset({"fp16"})


class TestRequireSupported:
    # U-Q-7
    def test_no_op_for_supported(self) -> None:
        amf.require_supported(BackendId.DIRECT, Quant.FP16)  # must not raise

    # U-Q-8
    def test_raises_with_backend_and_allowed_in_message(self) -> None:
        with pytest.raises(
            amf.UnsupportedQuantizationError,
        ) as exc_info:
            amf.require_supported(BackendId.DIRECT, Quant.Q4)
        msg = str(exc_info.value)
        assert "direct" in msg
        assert "q4" in msg
        # allowed set (sorted) present in the message
        assert "fp16" in msg
        assert "fp32" in msg

    def test_raises_for_unknown_backend(self) -> None:
        """A backend not in the matrix should be treated as 'nothing allowed'."""
        with pytest.raises(amf.UnsupportedQuantizationError):
            amf.require_supported("noexist", Quant.FP16)


class TestDtypeByQuantShrinkage:
    # U-Q-10
    def test_dtype_by_quant_contains_only_fp32_and_fp16(self) -> None:
        """DP-3 D-Q-1 — silent-fp16 bug fixed by the shrinkage."""
        assert {
            "fp32": torch.float32, "fp16": torch.float16,
        } == amf._DTYPE_BY_QUANT

    def test_supported_quantizations_matches_dtype_map(self) -> None:
        assert set(amf.SUPPORTED_QUANTIZATIONS) == set(amf._DTYPE_BY_QUANT)


class TestDirectBackendDp3Integration:
    """SC-Q-3 + SC-Q-6 — DirectBackend.load() wires ``require_supported``."""

    @pytest.mark.parametrize("q", [Quant.Q4, Quant.Q8, Quant.Q2, Quant.NF4])
    def test_load_rejects_unsupported_with_backend_name_in_message(
        self, q: Quant,
    ) -> None:
        calls: list[tuple[str, str]] = []
        be = DirectBackend(
            target_label="t", model_id="m", quantization=q, factory=_rec(calls),
        )
        with pytest.raises(amf.UnsupportedQuantizationError, match="direct"):
            be.load()
        assert calls == []

    def test_load_accepts_fp32(self) -> None:
        calls: list[tuple[str, str]] = []
        be = DirectBackend(
            target_label="t", model_id="m", quantization=Quant.FP32,
            factory=_rec(calls),
        )
        be.load()
        assert calls == [("m", "fp32")]
