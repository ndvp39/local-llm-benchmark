"""Tests for ``OnPremLlmSDK.run_sweep`` precondition wiring (T-2a.4 · ADR-010).

These are SDK-level guard tests — they assert the SDK calls the env + plumbing
guards in the right order and respects ``skip_plumbing``. Logic-level coverage
of the plumbing guard itself lives in ``test_shared/test_plumbing_guard.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_prem_llm_lab.sdk import (
    EnvironmentNotInitializedError,
    OnPremLlmSDK,
    PlumbingNotRunError,
)


def _initialised_setup(tmp_path: Path) -> Path:
    """setup.json with non-null hardware_constraints and a generous max age."""
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps({
            "version": "1.00",
            "init": {"max_age_hours": 99999},
            "hardware_constraints": {
                "captured_at": "2026-06-27T10:00:00Z",
                "cpu": {"cores_logical": 8},
            },
        }),
        encoding="utf-8",
    )
    return setup


def _write_ok_manifest(repo_root: Path, stamp: str = "20260627T120000Z") -> Path:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"plumbing_{stamp}.json"
    path.write_text(json.dumps({"overall": "ok"}), encoding="utf-8")
    return path


def test_run_sweep_raises_plumbing_not_run_when_no_manifest(tmp_path: Path) -> None:
    """Env initialised but no plumbing manifest → PlumbingNotRunError."""
    setup = _initialised_setup(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    with pytest.raises(PlumbingNotRunError, match="No plumbing manifest"):
        sdk.run_sweep(prompts=["hi"])


def test_run_sweep_skip_plumbing_bypasses_the_guard(tmp_path: Path) -> None:
    """``skip_plumbing=True`` MUST bypass the plumbing check (still hits T-3.5 stub)."""
    setup = _initialised_setup(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    with pytest.raises(NotImplementedError, match="T-3.5"):
        sdk.run_sweep(prompts=["hi"], skip_plumbing=True)


def test_run_sweep_passes_guards_when_ok_manifest_present(tmp_path: Path) -> None:
    """Env initialised + ok plumbing manifest → reaches NotImplementedError (T-3.5)."""
    setup = _initialised_setup(tmp_path)
    _write_ok_manifest(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    with pytest.raises(NotImplementedError, match="T-3.5"):
        sdk.run_sweep(prompts=["hi"])


def test_run_sweep_env_guard_fires_before_plumbing_guard(tmp_path: Path) -> None:
    """If env is uninitialised, EnvironmentNotInitializedError MUST be the error."""
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps({"version": "1.00", "hardware_constraints": None}),
        encoding="utf-8",
    )
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    # No plumbing manifest either — but env guard takes precedence.
    with pytest.raises(EnvironmentNotInitializedError):
        sdk.run_sweep(prompts=["hi"])
