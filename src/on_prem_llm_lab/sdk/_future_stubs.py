"""Future-method stubs split out of :mod:`sdk` (size-cap exception).

Each method raises ``NotImplementedError`` until the relevant milestone
implements it. ``_require_initialized_env`` + ``_require_current_plumbing``
live on the concrete ``OnPremLlmSDK`` class and are accessed through
the inherited ``self`` so the env-init guard and ADR-010 plumbing
precondition still fire from each stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn


class _FutureStubsMixin:
    """Carries SDK methods awaiting later milestone implementation."""

    def run_sweep(
        self, prompts: list[str], *,
        skip_plumbing: bool = False, **kwargs: Any,
    ) -> NoReturn:
        """target × quant × backend sweep. T-3.5 (M3)."""
        self._require_initialized_env()  # type: ignore[attr-defined]
        if not skip_plumbing:
            self._require_current_plumbing()  # type: ignore[attr-defined]
        raise NotImplementedError("T-3.5 lands in M3.")

    def run_qlora_finetune(
        self, target_label: str, dataset_path: Path, lora_config: Any,
    ) -> NoReturn:
        """QLoRA fine-tune (ADR-014). T-5.2 (M5, blocked on T-5.0)."""
        self._require_initialized_env()  # type: ignore[attr-defined]
        raise NotImplementedError("T-5.2 lands in M5 (after T-5.0 PRD_qlora.md).")

    def economic_analysis(self, sweep: Any) -> NoReturn:
        """3-curve break-even (ADR-012). T-4.1 (M4)."""
        self._require_initialized_env()  # type: ignore[attr-defined]
        raise NotImplementedError("T-4.1 lands in M4.")

    def assemble_readme(self, *, dry_run: bool = False) -> NoReturn:
        """Auto-assemble README from results/ (ADR-007). T-6.1 (M6)."""
        self._require_initialized_env()  # type: ignore[attr-defined]
        raise NotImplementedError("T-6.1 lands in M6.")


__all__ = ["_FutureStubsMixin"]
