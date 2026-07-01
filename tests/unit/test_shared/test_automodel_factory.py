"""Tests for ``shared/automodel_factory.py`` (T-1.15 · ADR-009).

External ``transformers`` calls are monkeypatched so no real download occurs.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from on_prem_llm_lab.shared import automodel_factory as amf


class _StubTok:
    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = kwargs


class _StubModel:
    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = kwargs


@pytest.fixture
def patched_hf(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[str, dict]]]:
    """Replace AutoModel*/AutoTokenizer with capturing stubs."""
    calls: dict[str, list[tuple[str, dict]]] = {"model": [], "tokenizer": []}

    class _ModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _StubModel:
            calls["model"].append((model_id, kwargs))
            return _StubModel(model_id, **kwargs)

    class _TokFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _StubTok:
            calls["tokenizer"].append((model_id, kwargs))
            return _StubTok(model_id, **kwargs)

    monkeypatch.setattr(amf, "AutoModelForCausalLM", _ModelFactory)
    monkeypatch.setattr(amf, "AutoTokenizer", _TokFactory)
    return calls


def test_unsupported_quantization_raises() -> None:
    """PRD FR-8 — unknown quantization labels MUST raise UnsupportedQuantizationError."""
    with pytest.raises(amf.UnsupportedQuantizationError, match="int3"):
        amf.load_causal_lm("any/model", quantization="int3")


@pytest.mark.parametrize("q", ["q8", "q4", "q2", "nf4"])
def test_dp3_q_labels_no_longer_silently_coerce_to_fp16(q: str) -> None:
    """DP-3 D-Q-1 — the factory now REJECTS q*/nf4 labels instead of silently
    coercing them to fp16 (the sweep-poisoning bug this PRD fixed)."""
    with pytest.raises(amf.UnsupportedQuantizationError):
        amf.load_causal_lm("any/model", quantization=q)


def test_happy_path_uses_automodel_factories(
    patched_hf: dict[str, list[tuple[str, dict]]],
) -> None:
    """ADR-009 — happy path resolves dtype + calls AutoModel*/AutoTokenizer once each."""
    out = amf.load_causal_lm(
        "ndvp/tiny-test",
        quantization="fp16",
        device_map="cpu",
        trust_remote_code=False,
    )

    assert isinstance(out, amf.LoadedModel)
    assert isinstance(out.model, _StubModel)
    assert isinstance(out.tokenizer, _StubTok)
    assert out.resolved_device_map == "cpu"
    assert out.resolved_dtype == str(torch.float16)

    assert len(patched_hf["tokenizer"]) == 1
    tok_id, tok_kwargs = patched_hf["tokenizer"][0]
    assert tok_id == "ndvp/tiny-test"
    assert tok_kwargs == {"trust_remote_code": False}

    assert len(patched_hf["model"]) == 1
    model_id, model_kwargs = patched_hf["model"][0]
    assert model_id == "ndvp/tiny-test"
    assert model_kwargs["torch_dtype"] is torch.float16
    assert model_kwargs["device_map"] == "cpu"
    assert model_kwargs["low_cpu_mem_usage"] is True


@pytest.mark.parametrize(
    ("quant", "expected_dtype"),
    [("fp32", torch.float32), ("fp16", torch.float16)],
)
def test_dtype_resolution_per_supported_quantization(
    patched_hf: dict[str, list[tuple[str, dict]]],
    quant: str,
    expected_dtype: torch.dtype,
) -> None:
    """FR-7 / DP-3 — the two levels the Direct factory actually supports map
    to the documented torch dtype. q*/nf4 are covered by the raises-test above."""
    amf.load_causal_lm("ndvp/tiny-test", quantization=quant)
    _, model_kwargs = patched_hf["model"][-1]
    assert model_kwargs["torch_dtype"] is expected_dtype
