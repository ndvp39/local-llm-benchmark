"""Economic math (T-4.1 per DP-5) — pure functions, no I/O.

Three cost formulas + break-even volume solver. See
`docs/PRD_economic_analysis.md` §2.2 + §5.2 for derivations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class AssumptionsSnapshot:
    capex_usd: float
    lifetime_hours: float
    watts_active: float
    electricity_price_per_kwh_usd: float
    cloud_gpu_hourly_usd: float
    cloud_gpu_speedup_over_cpu: float
    anthropic_in_per_million_usd: float
    anthropic_out_per_million_usd: float
    anthropic_model: str
    anthropic_prices_captured_at: str | None


def cost_per_req_on_prem(wall_s: float, s: AssumptionsSnapshot) -> float:
    """`(capex/lifetime_hours) × wall_hours + watts × wall_hours × elec/1000`."""
    hours = wall_s / 3600.0
    amort_per_hour = s.capex_usd / s.lifetime_hours
    hardware = amort_per_hour * hours
    electricity = s.watts_active * hours * (s.electricity_price_per_kwh_usd / 1000.0)
    return hardware + electricity


def cost_per_req_anthropic(
    prompt_tokens: int, completion_tokens: int, s: AssumptionsSnapshot,
) -> float:
    """`(in × in_price + out × out_price) / 1M`."""
    return (
        prompt_tokens * s.anthropic_in_per_million_usd
        + completion_tokens * s.anthropic_out_per_million_usd
    ) / 1_000_000.0


def cost_per_req_cloud_gpu(wall_s_cpu: float, s: AssumptionsSnapshot) -> float:
    """`hourly_usd × (wall_s_cpu / speedup) / 3600`."""
    wall_s_gpu = wall_s_cpu / s.cloud_gpu_speedup_over_cpu
    return s.cloud_gpu_hourly_usd * wall_s_gpu / 3600.0


def break_even_volume(
    capex_usd: float, opex_per_req: float, competitor_cost_per_req: float,
) -> int | None:
    """Volume `N` at which `capex + N × opex = N × competitor`. None if never."""
    delta = competitor_cost_per_req - opex_per_req
    if delta <= 0:
        return None
    return math.ceil(capex_usd / delta)


__all__ = [
    "AssumptionsSnapshot",
    "break_even_volume",
    "cost_per_req_anthropic",
    "cost_per_req_cloud_gpu",
    "cost_per_req_on_prem",
]
