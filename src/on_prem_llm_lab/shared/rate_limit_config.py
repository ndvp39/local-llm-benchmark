"""Typed reader for ``config/rate_limits.json`` (T-2.6, PRD §5.1-5.2).

Frozen dataclasses validated at load time. The dict ``services`` is a
mutable Python container, but the dataclass instance itself is frozen
so callers can pass it around without defensive copies — the policy
that the gatekeeper enforces is fixed at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SUPPORTED_VERSION = "1.00"


@dataclass(frozen=True)
class ServiceLimits:
    """One service's quota (PRD_api_gatekeeper §5.1)."""

    requests_per_minute: int
    requests_per_hour: int
    concurrent_max: int
    retry_after_seconds: float
    max_retries: int
    queue_max_depth: int


@dataclass(frozen=True)
class RateLimitConfig:
    """Full rate-limit policy keyed by service name (PRD §5.2)."""

    version: str
    services: dict[str, ServiceLimits]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RateLimitConfig:
        version = str(payload.get("version", ""))
        if version != _SUPPORTED_VERSION:
            raise ValueError(
                f"Unsupported rate_limits version: {version!r} "
                f"(expected {_SUPPORTED_VERSION!r})"
            )
        services_raw = payload.get("services", {})
        if not isinstance(services_raw, dict):
            raise ValueError("rate_limits.services must be a JSON object")
        services = {
            name: ServiceLimits(**spec) for name, spec in services_raw.items()
        }
        return cls(version=version, services=services)

    def for_service(self, name: str) -> ServiceLimits:
        """Return the limits for ``name``; raises ``KeyError`` if missing."""
        try:
            return self.services[name]
        except KeyError as e:
            raise KeyError(f"Unknown rate-limit service: {name!r}") from e


__all__ = ["RateLimitConfig", "ServiceLimits"]
