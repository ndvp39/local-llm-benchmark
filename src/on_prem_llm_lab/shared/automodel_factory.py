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
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = True,
    max_memory: dict[str, str] | None = None,
) -> LoadedModel:
    """Load model + tokenizer via the ``AutoModel*`` factories (ADR-009).

    ``device_map`` + ``low_cpu_mem_usage`` + ``max_memory`` are exposed
    knobs (T-2.11 fix) so the Direct backend can force the naive
    RAM-constrained load path that PRD SC-1's "baseline must fail
    visibly" narrative depends on. Leaving the defaults gives the smart
    accelerate path (auto disk-offload, unbounded CPU RAM) appropriate
    for AirLLM-style callers.

    Passing ``max_memory={"cpu": "6GB"}`` without an offload folder
    forces accelerate to raise ``ValueError`` (Python-level, catchable)
    if the model won't fit — the honest baseline failure mode on
    under-resourced hardware. Without this constraint on Windows the
    naive load can OS-segfault before Python catches anything.
    """
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise UnsupportedQuantizationError(
            f"quantization {quantization!r} not in {SUPPORTED_QUANTIZATIONS}"
        )
    dtype = _DTYPE_BY_QUANT[quantization]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        resolved_dtype=str(dtype),
        resolved_device_map=str(device_map),
    )


__all__ = [
    "SUPPORTED_QUANTIZATIONS",
    "LoadedModel",
    "UnsupportedQuantizationError",
    "load_causal_lm",
]
