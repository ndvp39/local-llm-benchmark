"""Env-init precondition guard (ADR-016).

Every SDK method *except* ``scan_hardware`` and ``initialize_environment``
MUST call :func:`require_initialized_env` as its first action. Without this
guard, a caller who forgets to run ``uv run init_env.py`` (or whose previous
scan has gone stale beyond ``init.max_age_hours``) would crash deep inside
the pipeline instead of receiving an actionable error.

The guard reads ``config/setup.json`` directly — no SDK dependency — so it
can be invoked from anywhere (SDK methods, tests, the ``init_env.py``
bootstrap script itself).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

REMEDIATION = "Run `uv run init_env.py` to populate config.hardware_constraints."
DEFAULT_MAX_AGE_HOURS = 168


class EnvironmentNotInitializedError(RuntimeError):
    """Raised when config.hardware_constraints is null, missing, or stale (ADR-016)."""


def _parse_captured_at(value: str) -> datetime:
    """Accept the HardwareScanner clock format ``YYYY-MM-DDTHH:MM:SSZ`` (UTC)."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def require_initialized_env(
    config_path: Path,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict:
    """Return ``hardware_constraints`` if present & fresh; else raise.

    ``now`` is an injectable seam for unit tests; production callers leave
    it ``None`` so wall-clock UTC is used. Returns the loaded constraints
    dict so downstream callers don't need to re-parse the config.
    """
    if not config_path.exists():
        raise EnvironmentNotInitializedError(
            f"setup config not found at {config_path}. {REMEDIATION}"
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    constraints = cfg.get("hardware_constraints")
    if constraints is None:
        raise EnvironmentNotInitializedError(
            f"config.hardware_constraints is null. {REMEDIATION}"
        )
    max_age_hours = int((cfg.get("init") or {}).get("max_age_hours", DEFAULT_MAX_AGE_HOURS))
    captured_raw = constraints.get("captured_at")
    if not captured_raw:
        raise EnvironmentNotInitializedError(
            f"config.hardware_constraints has no captured_at. {REMEDIATION}"
        )
    captured = _parse_captured_at(captured_raw)
    current = (now or (lambda: datetime.now(UTC)))()
    if current - captured > timedelta(hours=max_age_hours):
        raise EnvironmentNotInitializedError(
            f"config.hardware_constraints is stale "
            f"(captured {captured_raw}, max_age_hours={max_age_hours}). {REMEDIATION}"
        )
    return constraints


__all__ = [
    "DEFAULT_MAX_AGE_HOURS",
    "REMEDIATION",
    "EnvironmentNotInitializedError",
    "require_initialized_env",
]
