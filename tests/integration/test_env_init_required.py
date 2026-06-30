"""Integration: every guarded SDK method refuses to run without env-init (ADR-016).

The DoD for T-1.16 calls out two cases:
  1. ``config.hardware_constraints`` is ``null`` (fresh checkout, init_env.py never ran).
  2. ``config.hardware_constraints.captured_at`` is older than ``init.max_age_hours``.

In both cases the SDK MUST raise :class:`EnvironmentNotInitializedError` with a
remediation hint that points back to ``uv run init_env.py``. This test exercises
the real SDK class through all seven guarded stub methods (run_plumbing_test,
run_baseline, run_airllm, run_sweep, run_qlora_finetune, economic_analysis,
assemble_readme) — proving the guard is wired into each one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from on_prem_llm_lab import EnvironmentNotInitializedError, OnPremLlmSDK

_NULL_CONSTRAINTS: dict[str, object] = {
    "version": "1.00",
    "init": {"max_age_hours": 168},
    "hardware_constraints": None,
}

_STALE_CONSTRAINTS: dict[str, object] = {
    "version": "1.00",
    "init": {"max_age_hours": 1},
    "hardware_constraints": {
        "captured_at": "2020-01-01T00:00:00Z",
        "cpu": {"cores_logical": 8},
    },
}


def _write_cfg(tmp_path: Path, payload: dict[str, object]) -> Path:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _invocations(sdk: OnPremLlmSDK) -> list[Callable[[], object]]:
    """One zero-arg call per guarded SDK method — pass through the guard, then raise."""
    return [
        sdk.run_plumbing_test,
        lambda: sdk.run_baseline("llama3-8b-fp16", prompt="hello"),
        lambda: sdk.run_airllm("llama3-8b-fp16", "hello"),
        lambda: sdk.run_sweep(["hello"]),
        lambda: sdk.run_qlora_finetune("llama3-8b-fp16", Path("ds"), {}),
        lambda: sdk.economic_analysis({}),
        sdk.assemble_readme,
    ]


@pytest.mark.integration
def test_missing_constraints_blocks_every_guarded_method(tmp_path: Path) -> None:
    """Case 1: hardware_constraints=null — every guarded method MUST refuse."""
    cfg = _write_cfg(tmp_path, _NULL_CONSTRAINTS)
    sdk = OnPremLlmSDK(config_path=cfg, repo_root=tmp_path)
    for call in _invocations(sdk):
        with pytest.raises(EnvironmentNotInitializedError) as exc:
            call()
        msg = str(exc.value)
        assert "null" in msg
        assert "init_env.py" in msg


@pytest.mark.integration
def test_stale_constraints_blocks_every_guarded_method(tmp_path: Path) -> None:
    """Case 2: captured_at older than max_age_hours — every guarded method MUST refuse."""
    cfg = _write_cfg(tmp_path, _STALE_CONSTRAINTS)
    sdk = OnPremLlmSDK(config_path=cfg, repo_root=tmp_path)
    for call in _invocations(sdk):
        with pytest.raises(EnvironmentNotInitializedError) as exc:
            call()
        msg = str(exc.value)
        assert "stale" in msg
        assert "init_env.py" in msg


@pytest.mark.integration
def test_bootstrap_methods_are_exempt_from_guard(tmp_path: Path) -> None:
    """``scan_hardware`` and ``initialize_environment`` MUST NOT trip the guard.

    They are the only path that populates ``hardware_constraints``; guarding them
    would be a chicken-and-egg deadlock (ADR-016 explicitly exempts them).
    """
    cfg = _write_cfg(tmp_path, _NULL_CONSTRAINTS)
    sdk = OnPremLlmSDK(config_path=cfg, repo_root=tmp_path)
    # Real psutil call — keep this trivial. Just assert no EnvironmentNotInitializedError.
    result = sdk.scan_hardware()
    assert result.cpu.cores_logical > 0
