"""OnPremLlmSDK facade — single entry point for all business logic.

Constitution §3.1 / ADR-002: CLI, notebooks, and any future GUI MUST go
through this facade and are forbidden to import :mod:`services`, :mod:`backends`,
or :mod:`shared` directly.

T-1.14 wired :meth:`scan_hardware` and :meth:`initialize_environment` on top
of the original T-1.12 stub. T-1.16 (ADR-016) adds the env-init precondition
guard and stubs the seven downstream methods listed in PLAN §3.1 so the guard
is enforced uniformly *before* their real bodies arrive in M2a..M6. Future
tasks replace each ``NotImplementedError`` body in place; the guard line at
the top of every method stays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from on_prem_llm_lab.services.hardware_scanner import HardwareScanner
from on_prem_llm_lab.services.hardware_scanner_types import HardwareScanResult
from on_prem_llm_lab.shared.env_guard import (
    EnvironmentNotInitializedError,
    require_initialized_env,
)


@dataclass(frozen=True)
class InitEnvResult:
    """ADR-016 — outcome of ``uv run init_env.py`` / CLI ``initialize``."""

    scan: HardwareScanResult
    ok: bool
    failures: list[str]


class OnPremLlmSDK:
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

    def run_plumbing_test(self) -> NoReturn:
        """ADR-010 pre-flight on small/Q2 model. Implementation: T-2a.1 (M2a)."""
        self._require_initialized_env()
        raise NotImplementedError("T-2a.1 lands in M2a.")

    def run_baseline(self, target_label: str, prompt: str, **kwargs: Any) -> NoReturn:
        """Direct back-end run on an oversized target. Implementation: T-2.10 (M2b)."""
        self._require_initialized_env()
        raise NotImplementedError("T-2.10 lands in M2b.")

    def run_airllm(self, target_label: str, prompt: str, **kwargs: Any) -> NoReturn:
        """AirLLM back-end run. Implementation: T-3.1 (M3)."""
        self._require_initialized_env()
        raise NotImplementedError("T-3.1 lands in M3.")

    def run_sweep(self, prompts: list[str], **kwargs: Any) -> NoReturn:
        """target × quant × backend sweep. Implementation: T-3.5 (M3)."""
        self._require_initialized_env()
        raise NotImplementedError("T-3.5 lands in M3.")

    def run_qlora_finetune(
        self, target_label: str, dataset_path: Path, lora_config: Any
    ) -> NoReturn:
        """QLoRA fine-tune (ADR-014). Implementation: T-5.2 (M5, blocked on T-5.0)."""
        self._require_initialized_env()
        raise NotImplementedError("T-5.2 lands in M5 (after T-5.0 PRD_qlora.md).")

    def economic_analysis(self, sweep: Any) -> NoReturn:
        """3-curve break-even (ADR-012). Implementation: T-4.1 (M4)."""
        self._require_initialized_env()
        raise NotImplementedError("T-4.1 lands in M4.")

    def assemble_readme(self, *, dry_run: bool = False) -> NoReturn:
        """Auto-assemble README from results/ (ADR-007). Implementation: T-6.1 (M6)."""
        self._require_initialized_env()
        raise NotImplementedError("T-6.1 lands in M6.")


__all__ = ["EnvironmentNotInitializedError", "InitEnvResult", "OnPremLlmSDK"]
