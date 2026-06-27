"""AutoModel* factory — the only allowed entry point for HF model loading (ADR-009).

Centralises ``torch_dtype`` / ``device_map`` resolution per quantization label,
then delegates to ``transformers.AutoModelForCausalLM`` and ``AutoTokenizer``.
Back-ends MUST go through here so the static AST guard (T-2.0) can prove that
no concrete ``*ForCausalLM`` class is ever imported under ``backends/``.

Supported quantizations (PRD FR-7): ``fp32 | fp16 | q8 | q4 | q2 | nf4``.
Anything else raises :class:`UnsupportedQuantizationError` (PRD FR-8 — the
``Error`` suffix follows the project's exception-naming convention, matching
``GatekeeperError`` / ``PlumbingStageError`` / ``EnvironmentNotInitializedError``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_QUANTIZATIONS: tuple[str, ...] = ("fp32", "fp16", "q8", "q4", "q2", "nf4")

_DTYPE_BY_QUANT: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "q8": torch.float16,
    "q4": torch.float16,
    "q2": torch.float16,
    "nf4": torch.float16,
}


class UnsupportedQuantizationError(ValueError):
    """Raised when a quantization label is not in :data:`SUPPORTED_QUANTIZATIONS`."""


@dataclass(frozen=True)
class LoadedModel:
    """Output of :func:`load_causal_lm` — HF objects plus the resolved knobs."""

    model: Any
    tokenizer: Any
    resolved_dtype: str
    resolved_device_map: str


def load_causal_lm(
    model_id: str,
    quantization: str,
    *,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load model + tokenizer via the ``AutoModel*`` factories (ADR-009)."""
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise UnsupportedQuantizationError(
            f"quantization {quantization!r} not in {SUPPORTED_QUANTIZATIONS}"
        )
    dtype = _DTYPE_BY_QUANT[quantization]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        resolved_dtype=str(dtype),
        resolved_device_map=device_map,
    )


__all__ = [
    "SUPPORTED_QUANTIZATIONS",
    "LoadedModel",
    "UnsupportedQuantizationError",
    "load_causal_lm",
]
