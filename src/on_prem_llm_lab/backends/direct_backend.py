"""DirectBackend (T-2.8) — baseline transformers loader (PRD FR-4 / SC-1).

Loads the full model into RAM via :func:`shared.automodel_factory.load_causal_lm`
(ADR-009 — the only sanctioned path; the T-2.0 AST guard polices this
file for concrete ``*ForCausalLM`` imports). Two ``generate()`` calls
capture TTFT separately from the full-run TPOT — the same pattern
used in ``services/plumbing_default_stages.py``.

On the deliberately-oversized targets this back-end is **expected to
fail or crawl** (PRD SC-1 — that failure is the observation the report
narrates, not a bug). Memory + energy + manifest concerns belong to
the runner (T-2.9) that composes the cross-cutting mixins around
``generate()``; the back-end stays focused on load → generate → unload.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from on_prem_llm_lab.backends.base import (
    BackendId,
    BackendRunResult,
    InferenceBackend,
    Quant,
)
from on_prem_llm_lab.shared.automodel_factory import LoadedModel, load_causal_lm

_DirectFactory = Callable[[str, str], LoadedModel]


class DirectBackend(InferenceBackend):
    """Baseline back-end (PRD FR-4): ``AutoModelForCausalLM`` full load.

    Setup parameters are kw-only — explicit at call sites in the runner.
    ``factory`` and ``clock`` are injectable seams so unit tests never
    trigger an HF download or wall-sleep.
    """

    BACKEND_ID = BackendId.DIRECT

    def __init__(
        self,
        *,
        target_label: str,
        model_id: str,
        quantization: Quant | str,
        factory: _DirectFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._target_label = target_label
        self._model_id = model_id
        self._quantization = Quant(quantization)
        self._factory = factory or load_causal_lm
        self._clock = clock or time.perf_counter
        self._loaded: LoadedModel | None = None

    def load(self) -> None:
        """Idempotent load via the ``AutoModel*`` factory (ADR-009)."""
        if self._loaded is not None:
            return
        self._loaded = self._factory(self._model_id, self._quantization.value)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        params: dict[str, Any],
    ) -> BackendRunResult:
        if self._loaded is None:
            raise RuntimeError("DirectBackend.generate called before load()")
        model = self._loaded.model
        tokenizer = self._loaded.tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_tokens = int(input_ids.shape[-1])
        t0 = self._clock()
        model.generate(input_ids, max_new_tokens=1)
        t_first = self._clock()
        full_output = model.generate(input_ids, max_new_tokens=max_new_tokens)
        t_last = self._clock()
        completion_tokens = int(full_output.shape[-1]) - prompt_tokens
        completion_text = tokenizer.decode(full_output[0])
        ttft_ms = (t_first - t0) * 1000.0
        wall_s = t_last - t0
        decode_steps = max(1, max_new_tokens - 1)
        tpot_ms = ((t_last - t_first) * 1000.0) / decode_steps
        throughput_tps = completion_tokens / wall_s if wall_s > 0 else 0.0
        return BackendRunResult(
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            backend=self.BACKEND_ID,
            target_label=self._target_label,
            model_id=self._model_id,
            quantization=self._quantization,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            throughput_tps=throughput_tps,
            peak_ram_mb=0.0,
            wall_s=wall_s,
            energy_wh=0.0,
            completion_text=completion_text,
        )

    def unload(self) -> None:
        """Release the loaded model — safe to call multiple times."""
        self._loaded = None


__all__ = ["DirectBackend"]
