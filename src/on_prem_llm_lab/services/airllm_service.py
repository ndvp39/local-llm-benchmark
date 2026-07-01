"""AirLLM orchestration (T-3.5a) — AirLLMBackend through BenchmarkRunner.

Single-cell AirLLM run analogous to :mod:`baseline_service` but for the
AirLLM back-end. Renames the runner's manifest to
``results/airllm_<label>_<quantization>_<timestamp>.json`` so the M3
sweep's hero measurement is easy to locate. On failure writes a
parallel ``airllm_<label>_<quantization>_<ts>_failure.json`` + logs to
stderr (same discipline as the baseline path).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.airllm_backend import AirLLMBackend
from on_prem_llm_lab.backends.base import BackendRunResult
from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
from on_prem_llm_lab.services.benchmark_runner import BenchmarkRunner
from on_prem_llm_lab.shared.ram_sampler import psutil_rss_sampler


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_airllm(
    *,
    target_label: str,
    config: dict[str, Any],
    results_dir: Path,
    prompt: str | None = None,
    quantization: str | None = None,
    max_new_tokens: int | None = None,
    repo_root: Path | None = None,
    backend_factory: Callable[..., Any] | None = None,
    runner_factory: Callable[..., BenchmarkRunner] | None = None,
    ram_sampler: Callable[[], MemoryReading] | None = None,
    clock: Callable[[], str] | None = None,
) -> BackendRunResult:
    """Run one AirLLM cell; rename manifest to ``airllm_<label>_<q>_<ts>.json``."""
    targets = {t["label"]: t for t in config.get("target_models", [])}
    if target_label not in targets:
        raise ValueError(
            f"Unknown target_label: {target_label!r}; known: {sorted(targets)}"
        )
    target = targets[target_label]
    quant = quantization or target["quantization"]
    gen_cfg = config.get("generation") or {}
    airllm_cfg = config.get("airllm") or {}
    prompt_value = str(
        prompt or gen_cfg.get("airllm_prompt")
        or gen_cfg.get("baseline_prompt") or "Hello, world."
    )
    max_new = int(max_new_tokens or gen_cfg.get("max_new_tokens", 128))
    shards_path = airllm_cfg.get("layer_shards_saving_path", "airllm_shards")

    backend = (backend_factory or AirLLMBackend)(
        target_label=target["label"],
        model_id=target["id"],
        quantization=quant,
        layer_shards_saving_path=shards_path,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = (runner_factory or BenchmarkRunner)(
        config, results_dir,
        ram_sampler=ram_sampler or psutil_rss_sampler, repo_root=repo_root,
    )
    ts = (clock or _utc_ts)()

    try:
        result = runner.run(backend, prompt=prompt_value, max_new_tokens=max_new)
    except Exception as exc:  # noqa: BLE001 — surface every failure honestly
        failure_path = (
            results_dir / f"airllm_{target_label}_{quant}_{ts}_failure.json"
        )
        failure_path.write_text(
            json.dumps({
                "target_label": target_label,
                "model_id": target["id"],
                "quantization": quant,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "captured_at": ts,
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"[airllm] FAIL {target_label} @ {quant}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    old_path = Path(result.raw_log_path) if result.raw_log_path else None
    if old_path is not None and old_path.exists():
        new_path = results_dir / f"airllm_{target_label}_{quant}_{ts}.json"
        old_path.rename(new_path)
        result = dataclasses.replace(result, raw_log_path=str(new_path))
    return result


__all__ = ["run_airllm"]
