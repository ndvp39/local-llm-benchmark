"""Unit tests for shared/rate_limit_config.py (T-2.6 / PRD §9 U-GK-1..4)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from on_prem_llm_lab.shared.rate_limit_config import RateLimitConfig, ServiceLimits


def _valid_payload() -> dict:
    return {
        "version": "1.00",
        "services": {
            "default": {
                "requests_per_minute": 30,
                "requests_per_hour": 500,
                "concurrent_max": 5,
                "retry_after_seconds": 30,
                "max_retries": 3,
                "queue_max_depth": 100,
            },
            "huggingface_hub": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "concurrent_max": 4,
                "retry_after_seconds": 15,
                "max_retries": 5,
                "queue_max_depth": 50,
            },
        },
    }


class TestFromDict:
    def test_happy_path(self) -> None:
        # U-GK-1
        cfg = RateLimitConfig.from_dict(_valid_payload())
        assert cfg.version == "1.00"
        assert set(cfg.services) == {"default", "huggingface_hub"}
        assert cfg.services["default"].requests_per_minute == 30
        assert cfg.services["huggingface_hub"].queue_max_depth == 50

    def test_version_mismatch_raises(self) -> None:
        # U-GK-2
        payload = _valid_payload()
        payload["version"] = "0.99"
        with pytest.raises(ValueError, match="Unsupported rate_limits version"):
            RateLimitConfig.from_dict(payload)

    def test_missing_services_block_yields_empty(self) -> None:
        cfg = RateLimitConfig.from_dict({"version": "1.00"})
        assert cfg.services == {}

    def test_services_must_be_object(self) -> None:
        with pytest.raises(ValueError, match="services must be a JSON object"):
            RateLimitConfig.from_dict({"version": "1.00", "services": []})


class TestForService:
    def test_returns_limits(self) -> None:
        cfg = RateLimitConfig.from_dict(_valid_payload())
        limits = cfg.for_service("huggingface_hub")
        assert limits.requests_per_minute == 60

    def test_unknown_service_raises_keyerror(self) -> None:
        # U-GK-3
        cfg = RateLimitConfig.from_dict(_valid_payload())
        with pytest.raises(KeyError, match="Unknown rate-limit service"):
            cfg.for_service("missing")


class TestFrozen:
    def test_service_limits_is_frozen(self) -> None:
        # U-GK-4
        limits = ServiceLimits(
            requests_per_minute=1, requests_per_hour=1, concurrent_max=1,
            retry_after_seconds=1.0, max_retries=1, queue_max_depth=1,
        )
        with pytest.raises(FrozenInstanceError):
            limits.requests_per_minute = 99  # type: ignore[misc]

    def test_rate_limit_config_is_frozen(self) -> None:
        cfg = RateLimitConfig.from_dict(_valid_payload())
        with pytest.raises(FrozenInstanceError):
            cfg.version = "tampered"  # type: ignore[misc]
