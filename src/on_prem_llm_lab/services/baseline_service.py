"""Baseline orchestration (T-2.10) — DirectBackend through BenchmarkRunner.

Wraps one ``BenchmarkRunner.run`` call against a single target from
``config.target_models``, renames the resulting manifest to the
PRD baseline convention ``results/baseline_<label>_<timestamp>.json``,
and on failure writes a parallel ``baseline_<label>_<ts>_failure.json``
+ logs to stderr (PRD SC-1 — failures MUST surface, not hang silently).

Pure function, not a class — the SDK calls it from one place. The
``backend_factory`` / ``runner_factory`` / ``ram_sampler`` seams keep
unit tests from triggering an HF download or starting a real psutil
thread; production callers leave them ``None``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.base import BackendRunResult
from on_prem_llm_lab.backends.direct_backend import DirectBackend
from on_prem_llm_lab.mixins.memory_sampling_mixin import MemoryReading
from on_prem_llm_lab.services.benchmark_runner import BenchmarkRunner
from on_prem_llm_lab.shared.ram_sampler import psutil_rss_sampler


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_baseline(
    *,
    target_label: str,
    config: dict[str, Any],
    results_dir: Path,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    repo_root: Path | None = None,
    backend_factory: Callable[..., Any] | None = None,
    runner_factory: Callable[..., BenchmarkRunner] | None = None,
    ram_sampler: Callable[[], MemoryReading] | None = None,
    clock: Callable[[], str] | None = None,
) -> BackendRunResult:
    """Run one baseline; rename manifest to ``baseline_<label>_<ts>.json``."""
    targets = {t["label"]: t for t in config.get("target_models", [])}
    if target_label not in targets:
        raise ValueError(
            f"Unknown target_label: {target_label!r}; "
            f"known: {sorted(targets)}"
        )
    target = targets[target_label]
    gen_cfg = config.get("generation") or {}
    prompt_value = prompt or str(gen_cfg.get("baseline_prompt", "Hello, world."))
    max_new = int(max_new_tokens or gen_cfg.get("max_new_tokens", 128))

    backend = (backend_factory or DirectBackend)(
        target_label=target["label"],
        model_id=target["id"],
        quantization=target["quantization"],
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = (runner_factory or BenchmarkRunner)(
        config, results_dir,
        ram_sampler=ram_sampler or psutil_rss_sampler,
        repo_root=repo_root,
    )
    ts = (clock or _utc_ts)()

    try:
        result = runner.run(
            backend, prompt=prompt_value, max_new_tokens=max_new,
        )
    except Exception as exc:  # noqa: BLE001 — SC-1 mandates surfacing every failure
        failure_path = results_dir / f"baseline_{target_label}_{ts}_failure.json"
        failure_path.write_text(
            json.dumps({
                "target_label": target_label,
                "model_id": target["id"],
                "quantization": target["quantization"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "captured_at": ts,
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"[baseline] FAIL {target_label}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    old_path = Path(result.raw_log_path) if result.raw_log_path else None
    if old_path is not None and old_path.exists():
        new_path = results_dir / f"baseline_{target_label}_{ts}.json"
        old_path.rename(new_path)
        result = dataclasses.replace(result, raw_log_path=str(new_path))
    return result


__all__ = ["run_baseline"]
