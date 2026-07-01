"""AirLLMBackend (T-3.1) — layer-by-layer mmap via the ``airllm`` library.

Contract: ``docs/PRD_airllm_integration.md`` v1.00 (approved 2026-07-01).
Loads a HF model via ``airllm.AutoModel``, which streams per-layer
safetensors shards from ``layer_shards_saving_path`` into RAM one at a
time — the LLM analog of OS virtual-memory paging (L08 §8). Suitable
for oversized targets the Direct back-end cannot naive-load (see M2b
``results/witness_baseline_*.json`` for the "Direct fails on this box"
evidence, and M2a ``results/plumbing_20260630T194154Z.json`` for the
"AirLLM runs on this box" evidence).

Compatibility surface (M2a T-2a.5, prompts_book §6–§10): sharded
multi-file safetensors + separate ``lm_head`` + supported architecture.
CPU device pin (FR-AL-5) is mandatory: AirLLM defaults to ``cuda:0``
and crashes at init on CPU-only torch. Compression map (FR-AL-6):
``q4 -> "4bit"``, ``q8 -> "8bit"``, ``fp16 -> no kwarg (default)``.
No ``*ForCausalLM`` imports (ADR-009 + T-2.0 AST guard).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.base import (
    BackendId,
    BackendRunResult,
    InferenceBackend,
    Quant,
)
from on_prem_llm_lab.shared.automodel_factory import UnsupportedQuantizationError

_COMPRESSION: dict[str, str | None] = {"fp16": None, "q4": "4bit", "q8": "8bit"}


class AirLLMConfigError(ValueError):
    """Missing / empty ``layer_shards_saving_path`` config field (FR-AL-3)."""


class AirLLMDiskError(RuntimeError):
    """Insufficient free disk on the shard-cache drive (FR-AL-4)."""


def _default_factory(model_id: str, **kwargs: Any) -> Any:
    """Production default — real ``airllm.AutoModel.from_pretrained``."""
    from airllm import AutoModel  # type: ignore[import-untyped]  # noqa: PLC0415

    return AutoModel.from_pretrained(model_id, **kwargs)


def _resolve_compression(quantization: str) -> str | None:
    if quantization not in _COMPRESSION:
        raise UnsupportedQuantizationError(
            f"AirLLM does not support quantization {quantization!r}; "
            f"supported: {list(_COMPRESSION)}"
        )
    return _COMPRESSION[quantization]


class AirLLMBackend(InferenceBackend):
    """AirLLM back-end (PRD FR-AL-1..10)."""

    BACKEND_ID = BackendId.AIRLLM

    def __init__(
        self,
        *,
        target_label: str,
        model_id: str,
        quantization: Quant | str,
        layer_shards_saving_path: Path | str,
        hf_token: str | None = None,
        min_free_disk_gb: float = 25.0,
        factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not layer_shards_saving_path:
            raise AirLLMConfigError("layer_shards_saving_path is required")
        self._target_label = target_label
        self._model_id = model_id
        self._quantization = Quant(quantization)
        self._compression = _resolve_compression(self._quantization.value)
        self._layer_shards_path = Path(layer_shards_saving_path)
        self._hf_token = hf_token
        self._min_free_disk_gb = min_free_disk_gb
        self._factory = factory or _default_factory
        self._clock = clock or time.perf_counter
        self._loaded: Any = None
        self._tokenizer: Any = None
        self._n_layers: int | None = None

    def load(self) -> None:
        """Idempotent load via AirLLM (ADR-009: no ``*ForCausalLM`` imports)."""
        if self._loaded is not None:
            return
        self._preflight_disk_check()
        token = self._hf_token if self._hf_token is not None else os.environ.get("HF_TOKEN")
        kwargs: dict[str, Any] = {
            "layer_shards_saving_path": str(self._layer_shards_path),
            "device": self._resolve_device(),
        }
        if token:
            kwargs["hf_token"] = token
        if self._compression is not None:
            kwargs["compression"] = self._compression
        model = self._factory(self._model_id, **kwargs)
        self._loaded = model
        self._tokenizer = getattr(model, "tokenizer", None)
        layers = getattr(model, "layers", None)
        self._n_layers = len(layers) if layers is not None else None

    def generate(
        self, prompt: str, *, max_new_tokens: int, params: dict[str, Any],
    ) -> BackendRunResult:
        if self._loaded is None:
            raise RuntimeError("AirLLMBackend.generate called before load()")
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        prompt_tokens = int(input_ids.shape[-1])
        t0 = self._clock()
        self._loaded.generate(input_ids, max_new_tokens=1)
        t_first = self._clock()
        full_output = self._loaded.generate(input_ids, max_new_tokens=max_new_tokens)
        t_last = self._clock()
        completion_tokens = int(full_output.shape[-1]) - prompt_tokens
        completion_text = self._tokenizer.decode(full_output[0])
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
        self._tokenizer = None

    def _preflight_disk_check(self) -> None:
        import psutil  # noqa: PLC0415

        self._layer_shards_path.mkdir(parents=True, exist_ok=True)
        free_bytes = psutil.disk_usage(str(self._layer_shards_path)).free
        free_gib = free_bytes / (1024 ** 3)
        if free_gib < self._min_free_disk_gb:
            raise AirLLMDiskError(
                f"AirLLM pre-flight: shard cache path {self._layer_shards_path} "
                f"has {free_gib:.2f} GiB free < required {self._min_free_disk_gb} GiB."
            )

    def _resolve_device(self) -> str:
        import torch  # noqa: PLC0415

        return "cuda:0" if torch.cuda.is_available() else "cpu"


__all__ = ["AirLLMBackend", "AirLLMConfigError", "AirLLMDiskError"]
