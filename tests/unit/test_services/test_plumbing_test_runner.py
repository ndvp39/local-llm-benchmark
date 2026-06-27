"""Tests for ``services/plumbing_test_runner.py`` (T-2a.1 · ADR-010)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.services.plumbing_test_runner import (
    STAGE_ORDER,
    PlumbingResult,
    PlumbingStageError,
    PlumbingTestRunner,
)

_MODEL = {
    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "quantization": "q2",
    "label": "tiny-q2-plumbing",
}


def _fixed_clock() -> str:
    return "2026-06-27T12:00:00Z"


def _ok_stages(**overrides: Any) -> dict[str, Any]:
    """Default stage set; pass keyword args to swap individual stages."""
    base: dict[str, Any] = {
        "download": lambda: {"duration_s": 12.4},
        "mmap_allocation": lambda: {},
        "metric_collection": lambda: {"ttft_ms": 1200, "tpot_ms": 95},
    }
    base.update(overrides)
    return base


def test_constructor_rejects_missing_stages(tmp_path: Path) -> None:
    """A stages dict missing any required key MUST raise ValueError."""
    with pytest.raises(ValueError, match="missing stage callables"):
        PlumbingTestRunner(_MODEL, {}, results_dir=tmp_path)


def test_happy_path_writes_manifest_and_returns_ok(tmp_path: Path) -> None:
    """All four stages ok → overall='ok', manifest on disk, no exception."""
    runner = PlumbingTestRunner(_MODEL, _ok_stages(), results_dir=tmp_path, clock=_fixed_clock)
    result = runner.run()

    assert result.overall == "ok"
    assert result.remediation_hint is None
    assert result.captured_at == "2026-06-27T12:00:00Z"
    for name in STAGE_ORDER:
        assert result.stages[name].status == "ok"
    assert result.stages["manifest_write"].status == "ok"

    assert result.manifest_path is not None
    on_disk = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert on_disk["overall"] == "ok"
    assert on_disk["plumbing_test_model"]["id"] == _MODEL["id"]
    assert on_disk["stages"]["metric_collection"]["ttft_ms"] == 1200


def test_download_failure_aborts_with_structured_error(tmp_path: Path) -> None:
    """Stage 1 raises → manifest persisted, error raised, later stages skipped."""

    def boom() -> dict[str, Any]:
        raise RuntimeError("network down")

    runner = PlumbingTestRunner(
        _MODEL, _ok_stages(download=boom), results_dir=tmp_path, clock=_fixed_clock,
    )
    with pytest.raises(PlumbingStageError) as exc:
        runner.run()
    assert exc.value.stage == "download"
    assert "network down" in str(exc.value)

    result: PlumbingResult = exc.value.result
    assert result.overall == "fail"
    assert result.stages["download"].status == "fail"
    assert result.stages["mmap_allocation"].status == "skipped"
    assert result.stages["metric_collection"].status == "skipped"
    assert result.manifest_path is not None and result.manifest_path.exists()
    on_disk = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert on_disk["stages"]["download"]["error"] == "network down"
    assert on_disk["remediation_hint"] is not None


@pytest.mark.parametrize("failing_stage", ["mmap_allocation", "metric_collection"])
def test_mid_pipeline_failure_carries_stage_name(
    tmp_path: Path, failing_stage: str
) -> None:
    """Stage 2 or 3 failure → PlumbingStageError.stage matches the failed stage."""

    def boom() -> dict[str, Any]:
        raise RuntimeError(f"{failing_stage} broke")

    runner = PlumbingTestRunner(
        _MODEL,
        _ok_stages(**{failing_stage: boom}),
        results_dir=tmp_path,
        clock=_fixed_clock,
    )
    with pytest.raises(PlumbingStageError) as exc:
        runner.run()
    assert exc.value.stage == failing_stage
    assert exc.value.result.stages[failing_stage].status == "fail"


def test_manifest_write_failure_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the manifest write itself fails, stage='manifest_write' surfaces in the error."""
    runner = PlumbingTestRunner(_MODEL, _ok_stages(), results_dir=tmp_path, clock=_fixed_clock)

    def fail_write(self: Path, *args: Any, **kwargs: Any) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(PlumbingStageError) as exc:
        runner.run()
    assert exc.value.stage == "manifest_write"
    assert exc.value.result.stages["manifest_write"].status == "fail"
    assert exc.value.result.overall == "fail"


def test_results_dir_is_created_if_missing(tmp_path: Path) -> None:
    """The runner MUST create results_dir if it doesn't already exist."""
    nested = tmp_path / "deeply" / "nested" / "results"
    runner = PlumbingTestRunner(_MODEL, _ok_stages(), results_dir=nested, clock=_fixed_clock)
    runner.run()
    assert nested.exists()
    assert any(nested.glob("plumbing_*.json"))
