"""OnPremLlmSDK facade — single entry point for all business logic.

Constitution §3.1 / ADR-002: CLI, notebooks, and any future GUI MUST go
through this facade and are forbidden to import :mod:`services`, :mod:`backends`,
or :mod:`shared` directly. Per-method history (T-1.12 / T-1.14 / T-1.16 / T-2a.*)
lives in ``docs/TODO.md``; every downstream method opens with the env-init guard.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from on_prem_llm_lab.backends.base import BackendRunResult
from on_prem_llm_lab.sdk._future_stubs import _FutureStubsMixin
from on_prem_llm_lab.services.baseline_service import run_baseline as _run_baseline
from on_prem_llm_lab.services.hardware_scanner import HardwareScanner
from on_prem_llm_lab.services.hardware_scanner_types import HardwareScanResult
from on_prem_llm_lab.services.plumbing_default_stages import build_default_stages
from on_prem_llm_lab.services.plumbing_test_runner import (
    PlumbingResult,
    PlumbingStageError,
    PlumbingTestRunner,
    StageCallable,
)
from on_prem_llm_lab.shared.env_guard import (
    EnvironmentNotInitializedError,
    require_initialized_env,
)
from on_prem_llm_lab.shared.plumbing_guard import (
    PlumbingNotRunError,
    require_current_plumbing,
)


@dataclass(frozen=True)
class InitEnvResult:
    """ADR-016 — outcome of ``uv run init_env.py`` / CLI ``initialize``."""

    scan: HardwareScanResult
    ok: bool
    failures: list[str]


class OnPremLlmSDK(_FutureStubsMixin):
    """Single entry point for all business logic. See PLAN §3.1 for full surface."""

    def __init__(
        self,
        config_path: Path | str,
        env: Mapping[str, str] | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self.config_path: Path = Path(config_path)
        self.env: Mapping[str, str] = env if env is not None else {}
        self.repo_root: Path = Path(repo_root) if repo_root else Path.cwd()

    def scan_hardware(
        self,
        *,
        detector: Callable[[Path], HardwareScanResult] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> HardwareScanResult:
        """Run HardwareScanner (ADR-015) and return the populated result."""
        scanner = HardwareScanner(
            repo_root=self.repo_root,
            config_path=self.config_path,
            detector=detector,
            clock=clock,
        )
        return scanner.scan()

    def initialize_environment(
        self,
        *,
        detector: Callable[[Path], HardwareScanResult] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> InitEnvResult:
        """ADR-016 bootstrap — scan + compose overall env-init outcome."""
        scan = self.scan_hardware(detector=detector, clock=clock)
        failures = [
            f"{key}: {r.status} ({r.reason or 'no reason given'})"
            for key, r in scan.write_receipts.items()
            if r.status == "fail"
        ]
        return InitEnvResult(scan=scan, ok=not failures, failures=failures)

    def _require_initialized_env(self) -> None:
        """ADR-016 guard — call from every downstream method as its first action."""
        require_initialized_env(self.config_path)

    def _require_current_plumbing(self) -> Path:
        """ADR-010 guard for ``run_sweep`` — caller MUST handle ``skip_plumbing``."""
        return require_current_plumbing(self.repo_root / "results")

    def run_plumbing_test(
        self,
        *,
        stages: Mapping[str, StageCallable] | None = None,
    ) -> PlumbingResult:
        """ADR-010 pre-flight on the small/Q2 model (T-2a.2).

        ``stages`` is an injection seam for tests; production callers leave it
        ``None`` so :func:`build_default_stages` wires real HF + AirLLM + psutil
        closures from config. The runner writes ``results/plumbing_<ts>.json``
        and raises :class:`PlumbingStageError` on any stage failure.
        """
        self._require_initialized_env()
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        results_dir = self.repo_root / "results"
        built_stages = stages or build_default_stages(
            cfg, results_dir, hf_token=self.env.get("HF_TOKEN")
        )
        runner = PlumbingTestRunner(
            plumbing_test_model=cfg["plumbing_test_model"],
            stages=built_stages,
            results_dir=results_dir,
        )
        return runner.run()

    def run_baseline(
        self,
        target_label: str,
        *,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        skip_preflight: bool = False,
    ) -> BackendRunResult:
        """Direct back-end baseline run on one oversized target (T-2.10 / SC-1)."""
        self._require_initialized_env()
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        return _run_baseline(
            target_label=target_label,
            config=cfg,
            results_dir=self.repo_root / "results",
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repo_root=self.repo_root,
            skip_preflight=skip_preflight,
        )

__all__ = [
    "EnvironmentNotInitializedError",
    "InitEnvResult",
    "OnPremLlmSDK",
    "PlumbingNotRunError",
    "PlumbingResult",
    "PlumbingStageError",
]
