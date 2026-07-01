"""SweepRunner (T-3.5) — iterate target × quantization × backend matrix.

Implementation of ``docs/PRD_benchmarking_methodology.md`` v1.00 (approved
2026-07-01). Respects the DP-3 quantization support matrix (skips
unsupported cells with ``skip_reason``) and DP-4's warm-up / repeats /
statistics rules. Writes ``results/sweep_<ts>.csv`` + companion manifest.

Cell execution is delegated to an injected ``cell_runner`` callable so
tests can drive the matrix without spawning real backends; the SDK
wires a production ``cell_runner`` that builds
``DirectBackend`` / ``AirLLMBackend`` + a ``BenchmarkRunner`` around the
matrix cells.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.base import BackendRunResult
from on_prem_llm_lab.services.sweep_stats import (
    METRICS,
    NAN_STATS,
    MetricStats,
    SweepRow,
    aggregate,
    has_visible_second_prefill,
    write_csv,
    write_manifest,
)
from on_prem_llm_lab.shared.automodel_factory import is_supported

_QUANTIZATIONS: tuple[str, ...] = ("fp32", "fp16", "q4", "q8")
_LOG = logging.getLogger("on_prem_llm_lab.sweep_runner")

# Cell runner: (target_label, model_id, quantization, backend, max_new_tokens, prompt)
CellRunner = Callable[[str, str, str, str, int, str], BackendRunResult]


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class SweepRunner:
    """Iterate target × quantization × backend (DP-4)."""

    def __init__(
        self, config: Mapping[str, Any], results_dir: Path, *,
        cell_runner: CellRunner, clock: Callable[[], str] | None = None,
    ) -> None:
        self._config = dict(config)
        self._results_dir = Path(results_dir)
        self._cell_runner = cell_runner
        self._clock = clock or _utc_ts

    def run(self, backends: list[str]) -> Path:
        """Iterate every cell; write CSV + manifest; return the CSV path."""
        gen = self._config.get("generation") or {}
        sampling = self._config.get("sampling") or {}
        prompt = str(
            gen.get("sweep_prompt") or gen.get("baseline_prompt") or "Hello, world."
        )
        max_new = int(gen.get("max_new_tokens", 128))
        seed = int(gen.get("seed", 42))
        repeat = int(sampling.get("repeat", 5))
        warmup = int(sampling.get("warmup_repeats", 1))
        targets = list(self._config.get("target_models", []))
        rows: list[SweepRow] = [
            self._run_cell(
                target=t, quantization=q, backend=be, prompt=prompt,
                max_new_tokens=max_new, seed=seed, repeat=repeat, warmup=warmup,
            )
            for t in targets for q in _QUANTIZATIONS for be in backends
        ]
        ts = self._clock()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._results_dir / f"sweep_{ts}.csv"
        write_csv(csv_path, rows)
        write_manifest(
            self._results_dir / f"sweep_{ts}.json", self._config, rows, ts,
        )
        return csv_path

    def _run_cell(
        self, *, target: Mapping[str, Any], quantization: str, backend: str,
        prompt: str, max_new_tokens: int, seed: int, repeat: int, warmup: int,
    ) -> SweepRow:
        base = {
            "target_label": target["label"], "backend": backend,
            "quantization": quantization, "seed": seed,
            "prompt_tokens": 0, "max_new_tokens": max_new_tokens,
            "completion_tokens": 0, "repeat": repeat, "warmup_repeats": warmup,
        }
        nan: dict[str, MetricStats] = dict.fromkeys(METRICS, NAN_STATS)
        if not is_supported(backend, quantization):
            return SweepRow(
                **base, n_success=0, n_failed=0,
                skip_reason="unsupported_quantization",
                method_note="", stats=nan,
            )
        try:
            for _ in range(warmup):
                self._cell_runner(
                    target["label"], target["id"], quantization, backend,
                    max_new_tokens, prompt,
                )
        except Exception as exc:  # noqa: BLE001 -- C-BM-5 warm-up fatal
            _LOG.warning("sweep warmup failed for %s: %s", base, exc)
            return SweepRow(
                **base, n_success=0, n_failed=0,
                skip_reason=f"warmup_failed: {type(exc).__name__}",
                method_note="", stats=nan,
            )
        results: list[BackendRunResult] = []
        n_failed = 0
        for _ in range(repeat):
            try:
                results.append(self._cell_runner(
                    target["label"], target["id"], quantization, backend,
                    max_new_tokens, prompt,
                ))
            except Exception as exc:  # noqa: BLE001 -- FR-BM-7 log+continue
                _LOG.warning("sweep run failed for %s: %s", base, exc)
                n_failed += 1
        n_success = len(results)
        stats = (
            {m: aggregate([getattr(r, m) for r in results]) for m in METRICS}
            if n_success else nan
        )
        if not n_success:
            _LOG.warning(
                "sweep cell %s produced zero successes; NaN row emitted", base,
            )
        method_note = ""
        if n_success and has_visible_second_prefill(
            stats["ttft_ms"].mean, stats["tpot_ms"].mean, max_new_tokens,
        ):
            method_note = "second-prefill overhead visible (DP-4 FR-BM-10)"
        first = results[0] if results else None
        return SweepRow(
            **{**base,
               "prompt_tokens": first.prompt_tokens if first else 0,
               "completion_tokens": first.completion_tokens if first else 0},
            n_success=n_success, n_failed=n_failed, skip_reason=None,
            method_note=method_note, stats=stats,
        )


__all__ = ["CellRunner", "SweepRunner"]
