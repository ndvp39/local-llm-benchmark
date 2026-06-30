"""BenchmarkRunner (T-2.9) — orchestrates one backend run end-to-end.

Composes the cross-cutting mixins (MemorySamplingMixin,
EnergyAccountingMixin, ManifestLoggingMixin) around the
:class:`InferenceBackend` lifecycle::

    load -> start sampling -> generate -> stop sampling -> unload
    -> compute energy -> enrich result -> write manifest

The back-end returns a :class:`BackendRunResult` with placeholder
``peak_ram_mb=0`` / ``energy_wh=0`` (per PRD FR-4/9 the back-end stays
focused on inference); the runner fills those fields in from the
sampler peaks + the configured wattage knob, then sets
``raw_log_path`` to the manifest it just wrote.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.base import BackendRunResult, InferenceBackend
from on_prem_llm_lab.mixins.energy_accounting_mixin import EnergyAccountingMixin
from on_prem_llm_lab.mixins.manifest_logging_mixin import (
    ManifestLoggingMixin,
    resolve_git_hash,
)
from on_prem_llm_lab.mixins.memory_sampling_mixin import (
    MemoryReading,
    MemorySamplingMixin,
)


class BenchmarkRunner(
    MemorySamplingMixin, ManifestLoggingMixin, EnergyAccountingMixin
):
    """Orchestrate one back-end run + write the manifest."""

    def __init__(
        self,
        config: Mapping[str, Any],
        results_dir: Path,
        *,
        ram_sampler: Callable[[], MemoryReading],
        repo_root: Path | None = None,
    ) -> None:
        self._config = dict(config)
        self._results_dir = Path(results_dir)
        self._ram_sampler = ram_sampler
        self._repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    def run(
        self,
        backend: InferenceBackend,
        *,
        prompt: str,
        max_new_tokens: int,
        params: dict[str, Any] | None = None,
    ) -> BackendRunResult:
        """Drive ``backend`` through one full benchmark cycle."""
        sampling_cfg = self._config.get("sampling") or {}
        energy_cfg = self._config.get("energy") or {}
        gen_cfg = self._config.get("generation") or {}
        hz = float(sampling_cfg.get("memory_hz", 5))
        watts = float(energy_cfg.get("assumed_watts_active", 180))
        seed = int(gen_cfg.get("seed", 42))

        backend.load()
        self.start_memory_sampling(hz=hz, sampler=self._ram_sampler)
        try:
            raw = backend.generate(
                prompt, max_new_tokens=max_new_tokens, params=params or {}
            )
        finally:
            peaks = self.stop_memory_sampling()
            backend.unload()

        energy_wh = self.compute_energy_wh(watts=watts, wall_s=raw.wall_s)
        enriched = dataclasses.replace(
            raw,
            peak_ram_mb=peaks.peak_ram_mb,
            peak_vram_mb=peaks.peak_vram_mb,
            energy_wh=energy_wh,
        )
        manifest = self.build_manifest(
            run_id=enriched.run_id,
            target_label=enriched.target_label,
            model_id=enriched.model_id,
            backend=enriched.backend.value,
            quantization=enriched.quantization.value,
            seed=seed,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            config_snapshot=self._config,
            git_hash=resolve_git_hash(self._repo_root),
            run_result=dataclasses.asdict(enriched),
        )
        path = self.write_manifest(manifest, self._results_dir)
        return dataclasses.replace(enriched, raw_log_path=str(path))


__all__ = ["BenchmarkRunner"]
