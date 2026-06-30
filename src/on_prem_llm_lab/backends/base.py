"""Inference back-end abstraction (T-2.1).

ADR-002 establishes a single :class:`InferenceBackend` Strategy ABC with
three concrete implementations: ``DirectBackend`` (T-2.8),
``AirLLMBackend`` (T-3.1), and ``ApiBackend`` (T-4.2). This module hosts
the ABC and its supporting types:

* :class:`Quant` — quantization labels used in config + result rows
  (PRD FR-7). ``str``-inherited so ``json.dumps(Quant.FP16) -> '"fp16"'``.
* :class:`BackendId` — backend identifier for ``SweepRunner`` + manifest.
* :class:`BackendRunResult` — frozen schema returned by ``generate()``
  (PLAN §6.2).

Concrete subclasses live in ``backends/*_backend.py``. None of those
files may import a concrete ``*ForCausalLM`` class from ``transformers``
— the T-2.0 AST guard enforces ADR-009; subclasses route through
:func:`shared.automodel_factory.load_causal_lm` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Quant(StrEnum):
    """Quantization labels (PRD FR-7).

    ``StrEnum`` makes JSON serialization automatic. Values mirror
    :data:`on_prem_llm_lab.shared.automodel_factory.SUPPORTED_QUANTIZATIONS`;
    a cross-check test keeps the two in sync.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    Q8 = "q8"
    Q4 = "q4"
    Q2 = "q2"
    NF4 = "nf4"


class BackendId(StrEnum):
    """Backend identifier (PLAN ADR-002).

    Three members only: ``DIRECT`` (transformers full-load baseline),
    ``AIRLLM`` (layer-by-layer mmap), ``API`` (Anthropic — ADR-011).
    """

    DIRECT = "direct"
    AIRLLM = "airllm"
    API = "api"


@dataclass(frozen=True, kw_only=True)
class BackendRunResult:
    """Frozen schema returned by :meth:`InferenceBackend.generate`.

    Mirrors PLAN §6.2. Numeric units: durations in ms, throughput in
    tokens/s, memory in MB, wall time in s, energy in Wh.
    ``peak_vram_mb`` and ``raw_log_path`` are optional — CPU-only runs
    have no VRAM to sample, and the raw-log path is only known once
    :class:`ManifestLoggingMixin` (T-2.4) has written it.
    """

    run_id: str
    started_at: str
    backend: BackendId
    target_label: str
    model_id: str
    quantization: Quant
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    throughput_tps: float
    peak_ram_mb: float
    wall_s: float
    energy_wh: float
    completion_text: str
    peak_vram_mb: float | None = None
    raw_log_path: str | None = None


class InferenceBackend(ABC):
    """Strategy ABC (PLAN §2.4 / ADR-002).

    Concrete subclasses bind ``model_id``, ``quantization``,
    ``backend_config``, ``gatekeeper``, and a memory sampler at
    construction; the lifecycle is :py:meth:`load` ->
    :py:meth:`generate` -> :py:meth:`unload`, orchestrated by the
    ``BenchmarkRunner`` (T-2.9). The ABC stays parameter-free so each
    concrete ``__init__`` signature can vary.
    """

    @abstractmethod
    def load(self) -> None:
        """Load weights + tokenizer via the subclass-specific path."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        params: dict[str, Any],
    ) -> BackendRunResult:
        """Run inference; emit a per-run :class:`BackendRunResult`."""

    @abstractmethod
    def unload(self) -> None:
        """Release memory; safe to call multiple times."""


__all__ = [
    "BackendId",
    "BackendRunResult",
    "InferenceBackend",
    "Quant",
]
