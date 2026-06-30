"""Energy accounting mixin (T-2.5).

Computes wall-time energy in watt-hours from a caller-supplied wattage
and wall-clock duration. Pure math — no state, no I/O, no clock.

Formula (PRD FR-9): ``Wh = (watts * wall_s) / 3600``. Watts and seconds
arrive from runtime knobs; the caller decides which wattage to plug in
— typically ``config.energy.assumed_watts_active`` during inference and
``config.energy.assumed_watts_idle`` for the wall-time gap between
runs. The break-even chart (M4 / T-4.1) consumes the resulting Wh.
"""

from __future__ import annotations

_SECONDS_PER_HOUR: float = 3600.0


class EnergyAccountingMixin:
    """Mixin: convert wattage * wall-seconds to watt-hours.

    Kept deliberately tiny so back-ends can compose it alongside the
    other cross-cutting mixins (timing, memory_sampling,
    manifest_logging) without ordering concerns.
    """

    def compute_energy_wh(self, *, watts: float, wall_s: float) -> float:
        """Return wall-clock energy in Wh.

        Raises ``ValueError`` on negative inputs — both wattage and
        elapsed time are non-negative by definition; a negative value
        is almost always a measurement bug worth surfacing loudly.
        """
        if watts < 0:
            raise ValueError(f"watts must be non-negative, got {watts}")
        if wall_s < 0:
            raise ValueError(f"wall_s must be non-negative, got {wall_s}")
        return (watts * wall_s) / _SECONDS_PER_HOUR


__all__ = ["EnergyAccountingMixin"]
