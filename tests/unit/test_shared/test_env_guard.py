"""Tests for ``shared/env_guard.py`` (T-1.16 · ADR-016)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from on_prem_llm_lab.shared.env_guard import (
    REMEDIATION,
    EnvironmentNotInitializedError,
    require_initialized_env,
)


def _write_cfg(tmp_path: Path, **overrides: object) -> Path:
    """Build a minimal setup.json with overridable top-level keys."""
    base: dict[str, object] = {
        "version": "1.00",
        "init": {"max_age_hours": 168},
        "hardware_constraints": None,
    }
    base.update(overrides)
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_missing_config_file_raises() -> None:
    """No setup.json on disk → guard MUST raise with the remediation hint."""
    with pytest.raises(EnvironmentNotInitializedError, match="not found"):
        require_initialized_env(Path("nonexistent.json"))


def test_null_hardware_constraints_raises(tmp_path: Path) -> None:
    """Fresh checkout / never-initialised env raises with hint pointing at init_env.py."""
    cfg = _write_cfg(tmp_path, hardware_constraints=None)
    with pytest.raises(EnvironmentNotInitializedError) as exc:
        require_initialized_env(cfg)
    assert "null" in str(exc.value)
    assert REMEDIATION in str(exc.value)


def test_fresh_constraints_pass_through(tmp_path: Path) -> None:
    """Captured_at within the freshness window → guard returns the dict (no raise)."""
    cfg = _write_cfg(
        tmp_path,
        init={"max_age_hours": 168},
        hardware_constraints={"captured_at": "2026-06-27T10:00:00Z", "cpu": {"cores_logical": 8}},
    )
    out = require_initialized_env(cfg, now=lambda: datetime(2026, 6, 27, 11, 0, tzinfo=UTC))
    assert out["cpu"]["cores_logical"] == 8


def test_stale_constraints_raise(tmp_path: Path) -> None:
    """Captured_at older than max_age_hours → guard MUST raise with 'stale' in message."""
    cfg = _write_cfg(
        tmp_path,
        init={"max_age_hours": 1},
        hardware_constraints={"captured_at": "2020-01-01T00:00:00Z"},
    )
    with pytest.raises(EnvironmentNotInitializedError, match="stale"):
        require_initialized_env(cfg, now=lambda: datetime(2026, 6, 27, tzinfo=UTC))


def test_constraints_without_captured_at_raise(tmp_path: Path) -> None:
    """Malformed constraints (no captured_at) → guard refuses; doesn't crash on KeyError."""
    cfg = _write_cfg(tmp_path, hardware_constraints={"cpu": {"cores_logical": 8}})
    with pytest.raises(EnvironmentNotInitializedError, match="captured_at"):
        require_initialized_env(cfg)
