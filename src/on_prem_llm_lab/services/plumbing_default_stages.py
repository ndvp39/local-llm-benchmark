"""Default stage callables for :class:`PlumbingTestRunner` (T-2a.2).

Bridges the abstract StageCallable contract to concrete production behavior
(HF Hub download → AirLLM mmap load → time + RAM sampling around a tiny
``generate``). Heavy deps (``huggingface_hub``, ``airllm``, ``psutil``) are
**lazy-imported inside each closure** so:

* unit tests stub stages out without paying any of those import costs;
* CLI startup (``--help``, ``initialize``) doesn't trigger AirLLM import.

Refactor targets: T-3.x replaces ``download`` with a proper ``ModelAcquirer``
Building Block and ``mmap_allocation`` + ``metric_collection`` with an
``AirLLMBackend`` Building Block; T-2.3 replaces the inline psutil RSS read
with the threaded MemorySamplingMixin. The plumbing test will keep working
through those refactors because its contract is the StageCallable signature,
not these implementations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from on_prem_llm_lab.services.plumbing_test_runner import StageCallable

_DEFAULT_PROMPT = "Hello, world. Briefly:"
_DEFAULT_MAX_NEW = 32
_BYTES_PER_MB = 1024 * 1024


def build_default_stages(
    config: Mapping[str, Any],
    results_dir: Path,
    *,
    hf_token: str | None = None,
) -> dict[str, StageCallable]:
    """Compose the three pre-stages from runtime config.

    State is shared via a dict captured by all three closures, so
    ``mmap_allocation`` sees ``download``'s local path and
    ``metric_collection`` sees the model loaded by ``mmap_allocation``.
    """
    state: dict[str, Any] = {}
    model_cfg = config["plumbing_test_model"]
    airllm_cfg = config.get("airllm") or {}
    gen_cfg = config.get("generation") or {}

    def download() -> dict[str, Any]:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local = snapshot_download(
            repo_id=model_cfg["id"],
            token=hf_token or os.environ.get("HF_TOKEN"),
        )
        state["model_path"] = str(local)
        return {"path": str(local)}

    def mmap_allocation() -> dict[str, Any]:
        from airllm import AutoModel  # type: ignore[import-untyped]  # noqa: PLC0415

        kwargs: dict[str, Any] = {}
        shards = airllm_cfg.get("layer_shards_saving_path")
        if shards:
            kwargs["layer_shards_saving_path"] = str(Path(shards).expanduser())
        model = AutoModel.from_pretrained(state["model_path"], **kwargs)
        state["model"] = model
        layers = getattr(model, "layers", None)
        return {"layers": len(layers) if layers is not None else None}

    def metric_collection() -> dict[str, Any]:
        import time  # noqa: PLC0415

        import psutil  # noqa: PLC0415

        model = state["model"]
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AirLLM model has no tokenizer attached")
        prompt = str(gen_cfg.get("plumbing_prompt", _DEFAULT_PROMPT))
        max_new = int(gen_cfg.get("plumbing_max_new_tokens", _DEFAULT_MAX_NEW))
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        proc = psutil.Process()
        rss_before = proc.memory_info().rss
        t0 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=1)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=max_new)
        tpot_ms = ((time.perf_counter() - t1) * 1000.0) / max(1, max_new)
        peak_ram_mb = max(rss_before, proc.memory_info().rss) / _BYTES_PER_MB
        return {
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "peak_ram_mb": round(peak_ram_mb, 2),
            "tokens_generated": max_new,
        }

    return {
        "download": download,
        "mmap_allocation": mmap_allocation,
        "metric_collection": metric_collection,
    }


__all__ = ["build_default_stages"]
