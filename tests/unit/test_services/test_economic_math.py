"""Tests for `services/economic_math.py` (T-4.1 pure functions per DP-5)."""

from __future__ import annotations

import pytest

from on_prem_llm_lab.services.economic_math import (
    AssumptionsSnapshot,
    break_even_volume,
    cost_per_req_anthropic,
    cost_per_req_cloud_gpu,
    cost_per_req_on_prem,
)


def _snap(**overrides) -> AssumptionsSnapshot:  # type: ignore[no-untyped-def]
    defaults = {
        "capex_usd": 2500.0,
        "lifetime_hours": 5000.0,
        "watts_active": 180.0,
        "electricity_price_per_kwh_usd": 0.16,
        "cloud_gpu_hourly_usd": 0.6,
        "cloud_gpu_speedup_over_cpu": 1000.0,
        "anthropic_in_per_million_usd": 1.0,
        "anthropic_out_per_million_usd": 5.0,
        "anthropic_model": "claude-haiku-4-5-20251001",
        "anthropic_prices_captured_at": "2026-07-01",
    }
    defaults.update(overrides)
    return AssumptionsSnapshot(**defaults)


class TestCostPerReqOnPrem:
    def test_zero_wall_time(self) -> None:
        assert cost_per_req_on_prem(0.0, _snap()) == 0.0

    def test_typical_hero_wall_3300s(self) -> None:
        # Expected: (2500/5000) × (3300/3600) hardware
        # + 180 × (3300/3600) × (0.16/1000) electricity
        s = _snap()
        c = cost_per_req_on_prem(3300, s)
        expected = (2500 / 5000) * (3300 / 3600) + 180 * (3300 / 3600) * (0.16 / 1000)
        assert c == pytest.approx(expected, rel=1e-9)
        assert 0.4 < c < 0.6  # sanity: ~$0.49

    def test_linear_in_wall(self) -> None:
        s = _snap()
        assert cost_per_req_on_prem(7200, s) == pytest.approx(
            2 * cost_per_req_on_prem(3600, s), rel=1e-9,
        )


class TestCostPerReqAnthropic:
    def test_zero_tokens_zero_cost(self) -> None:
        assert cost_per_req_anthropic(0, 0, _snap()) == 0.0

    def test_typical_5in_8out(self) -> None:
        # (5 × 1 + 8 × 5) / 1M = 45 / 1M = 4.5e-5
        c = cost_per_req_anthropic(5, 8, _snap())
        assert c == pytest.approx(45 / 1_000_000, rel=1e-9)

    def test_million_tokens_is_price(self) -> None:
        c = cost_per_req_anthropic(1_000_000, 0, _snap())
        assert c == pytest.approx(1.0, rel=1e-9)


class TestCostPerReqCloudGpu:
    def test_1000x_speedup_default(self) -> None:
        # 3300s CPU / 1000 = 3.3s GPU × ($0.6/3600) = $5.5e-4
        c = cost_per_req_cloud_gpu(3300, _snap())
        assert c == pytest.approx(0.6 * (3300 / 1000) / 3600, rel=1e-9)

    def test_slower_gpu_higher_cost(self) -> None:
        fast = cost_per_req_cloud_gpu(3300, _snap(cloud_gpu_speedup_over_cpu=100))
        slow = cost_per_req_cloud_gpu(3300, _snap(cloud_gpu_speedup_over_cpu=10))
        assert slow > fast


class TestBreakEvenVolume:
    def test_competitor_cheaper_returns_none(self) -> None:
        # capex=2500, opex=0.03, competitor=0.001 (Anthropic-like) — competitor is cheaper
        assert break_even_volume(2500, 0.03, 0.001) is None

    def test_zero_delta_returns_none(self) -> None:
        assert break_even_volume(2500, 0.5, 0.5) is None

    def test_valid_break_even(self) -> None:
        # capex=2500, opex=0.01, competitor=0.50 → ceil(2500 / 0.49) = 5103
        n = break_even_volume(2500, 0.01, 0.50)
        assert n == 5103

    def test_ceil_rounds_up(self) -> None:
        # capex=100, opex=0, competitor=3 → 100/3 = 33.33 → ceil = 34
        assert break_even_volume(100, 0, 3) == 34
