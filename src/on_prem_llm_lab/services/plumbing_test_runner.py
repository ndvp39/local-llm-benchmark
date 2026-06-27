"""Pre-flight pipeline verification (ADR-010 / FR-PT-1..3 / PRD §3.10).

Stages: ``download`` → ``mmap_allocation`` → ``metric_collection`` → ``manifest_write``.
The first three are injected as zero-arg callables (Building Block setup
per constitution §15) so the runner is testable today without depending on
M2b / M3 services that will supply them later. Stage 4 is the runner's own.

On failure (FR-PT-3): stop at the first failed stage (rest marked ``skipped``),
still persist the partial manifest, then raise :class:`PlumbingStageError`
carrying the partial result for T-2a.4's sweep precondition to surface.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

StageCallable = Callable[[], Mapping[str, Any]]

STAGE_ORDER: tuple[str, ...] = ("download", "mmap_allocation", "metric_collection")
DEFAULT_REMEDIATION: dict[str, str] = {
    "download": "Check HF_TOKEN, network, and disk free space.",
    "mmap_allocation": "Check airllm.layer_shards_saving_path exists with free space (ADR-005).",
    "metric_collection": "Check sampler/generation params; retry with a smaller max_new_tokens.",
    "manifest_write": "Check results/ exists and is writable.",
}


@dataclass(frozen=True)
class StageOutcome:
    """Per-stage record persisted into the manifest (matches PRD §6.3)."""

    status: str  # "ok" | "fail" | "skipped"
    duration_s: float = 0.0
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.duration_s:
            out["duration_s"] = round(self.duration_s, 4)
        if self.error:
            out["error"] = self.error
        out.update(self.extras)
        return out


@dataclass(frozen=True)
class PlumbingResult:
    """ADR-010 outcome bundle (matches PRD §6.3 — plus ``manifest_path`` for callers)."""

    captured_at: str
    plumbing_test_model: Mapping[str, Any]
    stages: dict[str, StageOutcome]
    overall: str  # "ok" | "fail"
    remediation_hint: str | None
    manifest_path: Path | None


class PlumbingStageError(RuntimeError):
    """FR-PT-3 — raised when any plumbing stage fails. Carries the partial result."""

    def __init__(self, stage: str, message: str, result: PlumbingResult) -> None:
        super().__init__(f"plumbing stage {stage!r} failed: {message}")
        self.stage = stage
        self.result = result


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class PlumbingTestRunner:
    """Building Block (constitution §15). See module docstring for the contract."""

    def __init__(
        self,
        plumbing_test_model: Mapping[str, Any],
        stages: Mapping[str, StageCallable],
        results_dir: Path,
        clock: Callable[[], str] | None = None,
    ) -> None:
        missing = [s for s in STAGE_ORDER if s not in stages]
        if missing:
            raise ValueError(f"missing stage callables: {missing}")
        self.plumbing_test_model = dict(plumbing_test_model)
        self.stages = dict(stages)
        self.results_dir = results_dir
        self._clock = clock or _utc_now_iso

    def run(self) -> PlumbingResult:
        captured_at = self._clock()
        manifest_path = self.results_dir / f"plumbing_{captured_at.replace(':', '').replace('-', '')}.json"
        outcomes, first_fail = self._run_pre_stages()
        outcomes["manifest_write"] = StageOutcome(
            status="ok", extras={"path": str(manifest_path)}
        )
        hint = DEFAULT_REMEDIATION.get(first_fail[0]) if first_fail else None
        payload = self._payload(captured_at, outcomes, first_fail is None, hint)
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            outcomes["manifest_write"] = StageOutcome(
                status="fail", error=str(exc), extras={"path": str(manifest_path)}
            )
            first_fail = first_fail or ("manifest_write", str(exc))
            payload = self._payload(
                captured_at, outcomes, False, DEFAULT_REMEDIATION["manifest_write"]
            )
        result = PlumbingResult(
            captured_at=captured_at,
            plumbing_test_model=self.plumbing_test_model,
            stages=outcomes,
            overall=payload["overall"],
            remediation_hint=payload.get("remediation_hint"),
            manifest_path=manifest_path,
        )
        if first_fail is not None:
            raise PlumbingStageError(first_fail[0], first_fail[1], result)
        return result

    def _run_pre_stages(self) -> tuple[dict[str, StageOutcome], tuple[str, str] | None]:
        outcomes: dict[str, StageOutcome] = {}
        first_fail: tuple[str, str] | None = None
        for name in STAGE_ORDER:
            if first_fail is not None:
                outcomes[name] = StageOutcome(status="skipped")
                continue
            t0 = time.perf_counter()
            try:
                payload = self.stages[name]() or {}
                outcomes[name] = StageOutcome(
                    status="ok",
                    duration_s=time.perf_counter() - t0,
                    extras=dict(payload),
                )
            except Exception as exc:  # noqa: BLE001 — stage callables are caller-supplied
                outcomes[name] = StageOutcome(
                    status="fail",
                    duration_s=time.perf_counter() - t0,
                    error=str(exc),
                )
                first_fail = (name, str(exc))
        return outcomes, first_fail

    def _payload(
        self,
        captured_at: str,
        outcomes: Mapping[str, StageOutcome],
        ok: bool,
        hint: str | None,
    ) -> dict[str, Any]:
        return {
            "captured_at": captured_at,
            "plumbing_test_model": self.plumbing_test_model,
            "stages": {k: v.to_jsonable() for k, v in outcomes.items()},
            "overall": "ok" if ok else "fail",
            "remediation_hint": hint,
        }


__all__ = [
    "DEFAULT_REMEDIATION",
    "STAGE_ORDER",
    "PlumbingResult",
    "PlumbingStageError",
    "PlumbingTestRunner",
    "StageCallable",
    "StageOutcome",
]
