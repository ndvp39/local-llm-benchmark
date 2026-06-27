"""Tests for ``OnPremLlmSDK.run_plumbing_test`` wiring (T-2a.2 · ADR-010).

Split out of ``test_sdk.py`` to keep both files under the 150 LOC cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_prem_llm_lab.sdk import (
    EnvironmentNotInitializedError,
    OnPremLlmSDK,
    PlumbingStageError,
)


def _make_initialized_setup(tmp_path: Path) -> Path:
    """Setup with non-null hardware_constraints + plumbing_test_model (T-2a.2 happy)."""
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps({
            "version": "1.00",
            "init": {"max_age_hours": 99999},
            "hardware_constraints": {
                "captured_at": "2026-06-27T10:00:00Z",
                "cpu": {"cores_logical": 8},
            },
            "plumbing_test_model": {
                "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "quantization": "q2",
                "label": "tiny-q2",
            },
            "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
        }),
        encoding="utf-8",
    )
    return setup


def test_run_plumbing_test_passes_through_runner_with_injected_stages(tmp_path: Path) -> None:
    """Injected stages let us exercise the SDK wiring without an HF download."""
    setup = _make_initialized_setup(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    result = sdk.run_plumbing_test(stages={
        "download": lambda: {"path": "/fake/local"},
        "mmap_allocation": lambda: {"layers": 22},
        "metric_collection": lambda: {"ttft_ms": 100.0, "tpot_ms": 10.0, "peak_ram_mb": 500.0},
    })
    assert result.overall == "ok"
    assert (tmp_path / "results").exists()
    assert result.manifest_path is not None and result.manifest_path.exists()


def test_run_plumbing_test_raises_plumbing_stage_error_on_failure(tmp_path: Path) -> None:
    """Stage failure from inside an injected callable bubbles as PlumbingStageError."""
    setup = _make_initialized_setup(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)

    def boom() -> dict:
        raise RuntimeError("network down")

    with pytest.raises(PlumbingStageError) as exc:
        sdk.run_plumbing_test(stages={
            "download": boom,
            "mmap_allocation": lambda: {},
            "metric_collection": lambda: {},
        })
    assert exc.value.stage == "download"


def test_run_plumbing_test_guard_fires_before_runner(tmp_path: Path) -> None:
    """Env-init guard MUST raise before any stage callable is invoked."""
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps({"version": "1.00", "hardware_constraints": None}),
        encoding="utf-8",
    )
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    sentinel = {"called": False}

    def must_not_run() -> dict:
        sentinel["called"] = True
        return {}

    with pytest.raises(EnvironmentNotInitializedError):
        sdk.run_plumbing_test(stages={
            "download": must_not_run,
            "mmap_allocation": must_not_run,
            "metric_collection": must_not_run,
        })
    assert sentinel["called"] is False
