"""Default stage callables for :class:`PlumbingTestRunner` (T-2a.2).

Bridges the abstract StageCallable contract to concrete production behavior
(HF Hub download -> model load -> time + RAM sampling around a tiny
``generate`` -> manifest). Heavy deps (``huggingface_hub``, ``transformers``,
``airllm``, ``psutil``) are **lazy-imported inside each closure** so:

* unit tests stub stages out without paying any of those import costs;
* CLI startup (``--help``, ``initialize``) doesn't trigger AirLLM import.

The plumbing model can use either loader. AirLLM is the right choice for
oversized targets but requires a sharded multi-file layout + separate
``lm_head`` (no small mainstream HF model satisfies both). Transformers'
``AutoModelForCausalLM`` is the right choice for a true smoke-test plumbing
model (small, fast, any layout, any architecture). Picked via the
``plumbing_test_model.loader`` config field (default ``"transformers"``).
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
_AIRLLM_COMPRESSION: dict[str, str] = {"q4": "4bit", "q8": "8bit"}


def _load_via_airllm(
    model_cfg: Mapping[str, Any],
    airllm_cfg: Mapping[str, Any],
    hf_token: str | None,
) -> tuple[Any, Any, int | None]:
    """Load a model via AirLLM. Returns (model, tokenizer, n_layers)."""
    import torch  # noqa: PLC0415
    from airllm import AutoModel  # type: ignore[import-untyped]  # noqa: PLC0415

    kwargs: dict[str, Any] = {}
    shards = airllm_cfg.get("layer_shards_saving_path")
    if shards:
        kwargs["layer_shards_saving_path"] = str(Path(shards).expanduser())
    if hf_token:
        kwargs["hf_token"] = hf_token
    compression = _AIRLLM_COMPRESSION.get(str(model_cfg.get("quantization", "fp16")).lower())
    if compression is not None:
        kwargs["compression"] = compression
    # AirLLM defaults to device="cuda:0" and tries to allocate a CUDA stream
    # at init even when CUDA isn't available, crashing with "Torch not
    # compiled with CUDA enabled" on CPU-only torch wheels. Pin the device
    # to "cpu" explicitly so the prefetch stream is never created.
    kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    # AirLLM requires the HF repo_id (not the local path) so it can manage
    # its own download + layer-split cycle; HF cache hit makes it fast.
    model = AutoModel.from_pretrained(model_cfg["id"], **kwargs)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AirLLM model has no tokenizer attached")
    layers = getattr(model, "layers", None)
    return model, tokenizer, (len(layers) if layers is not None else None)


def _load_via_transformers(
    model_path: str,
) -> tuple[Any, Any, int | None]:
    """Load a model via transformers.AutoModelForCausalLM (no quantization).

    Plumbing-test loader for small (≲ 7B) models that AirLLM cannot handle
    (single-file safetensors, tied embeddings, etc.). Loads at fp16 with
    ``low_cpu_mem_usage=True`` to keep RAM peak ~= 2x param count in GB.
    Quantization at this layer is GPU-bound (bitsandbytes 4-bit needs CUDA);
    the M3 sweep handles CPU quantization via AirLLM compression.
    """
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    return model, tokenizer, n_layers


def build_default_stages(
    config: Mapping[str, Any],
    results_dir: Path,
    *,
    hf_token: str | None = None,
) -> dict[str, StageCallable]:
    """Compose the three pre-stages from runtime config.

    State is shared via a dict captured by all three closures: ``download``
    stores ``model_path``, ``mmap_allocation`` stores ``model`` +
    ``tokenizer``, ``metric_collection`` reads both.
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
        loader = str(model_cfg.get("loader", "transformers")).lower()
        token = hf_token or os.environ.get("HF_TOKEN")
        if loader == "airllm":
            model, tokenizer, n_layers = _load_via_airllm(model_cfg, airllm_cfg, token)
        elif loader == "transformers":
            model, tokenizer, n_layers = _load_via_transformers(state["model_path"])
        else:
            raise ValueError(f"Unsupported plumbing loader: {loader!r}")
        state["model"] = model
        state["tokenizer"] = tokenizer
        return {"layers": n_layers, "loader": loader}

    def metric_collection() -> dict[str, Any]:
        import time  # noqa: PLC0415

        import psutil  # noqa: PLC0415

        model = state["model"]
        tokenizer = state["tokenizer"]
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
