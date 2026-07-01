"""Sweep orchestration — wires SweepRunner with real backends.

Builds a production ``cell_runner`` that constructs the right backend
(DirectBackend / AirLLMBackend) per cell, wraps it in a BenchmarkRunner,
and returns the resulting BackendRunResult. The SDK calls this from one
place; T-3.5 SweepRunner iterates the matrix with the injected runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.airllm_backend import AirLLMBackend
from on_prem_llm_lab.backends.base import BackendRunResult
from on_prem_llm_lab.backends.direct_backend import DirectBackend
from on_prem_llm_lab.services.benchmark_runner import BenchmarkRunner
from on_prem_llm_lab.services.sweep_runner import CellRunner, SweepRunner
from on_prem_llm_lab.shared.ram_sampler import psutil_rss_sampler


def build_default_cell_runner(
    config: dict[str, Any], results_dir: Path, repo_root: Path | None,
    hf_token: str | None = None,
) -> CellRunner:
    """Return a ``cell_runner`` that builds a backend per cell + runs it."""
    airllm_cfg = config.get("airllm") or {}
    shards_path = airllm_cfg.get("layer_shards_saving_path", "airllm_shards")

    def _run(
        target_label: str, model_id: str, quantization: str,
        backend: str, max_new_tokens: int, prompt: str,
    ) -> BackendRunResult:
        if backend == "airllm":
            be: Any = AirLLMBackend(
                target_label=target_label, model_id=model_id,
                quantization=quantization,
                layer_shards_saving_path=shards_path,
                hf_token=hf_token,
            )
        elif backend == "direct":
            be = DirectBackend(
                target_label=target_label, model_id=model_id,
                quantization=quantization,
            )
        else:
            raise ValueError(f"Unsupported backend for sweep: {backend!r}")
        runner = BenchmarkRunner(
            config, results_dir,
            ram_sampler=psutil_rss_sampler, repo_root=repo_root,
        )
        return runner.run(be, prompt=prompt, max_new_tokens=max_new_tokens)

    return _run


def run_sweep(
    *, config: dict[str, Any], results_dir: Path,
    backends: list[str], repo_root: Path | None = None,
    hf_token: str | None = None,
    cell_runner: CellRunner | None = None,
) -> Path:
    """Iterate target × quantization × backend; return the CSV path."""
    cr = cell_runner or build_default_cell_runner(
        config, results_dir, repo_root, hf_token,
    )
    runner = SweepRunner(config, results_dir, cell_runner=cr)
    return runner.run(backends)


__all__ = ["build_default_cell_runner", "run_sweep"]
