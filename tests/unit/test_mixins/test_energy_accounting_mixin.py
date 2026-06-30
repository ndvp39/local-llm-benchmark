"""Unit tests for mixins/energy_accounting_mixin.py (T-2.5)."""

from __future__ import annotations

import pytest

from on_prem_llm_lab.mixins.energy_accounting_mixin import EnergyAccountingMixin


class _Mixed(EnergyAccountingMixin):
    """Minimal concrete subclass."""


class TestComputeEnergyWh:
    def test_one_hour_at_one_watt_is_one_wh(self) -> None:
        """100 W for 3600 s -> 100 Wh (the textbook unit identity)."""
        assert _Mixed().compute_energy_wh(watts=100.0, wall_s=3600.0) == pytest.approx(100.0)

    def test_active_inference_at_180w_for_m2a_wall_time(self) -> None:
        """180 W * 1103.99 s / 3600 = 55.1995 Wh — sanity-checks the M2a
        Llama-3 AirLLM run (`results/plumbing_20260630T194154Z.json`)."""
        wh = _Mixed().compute_energy_wh(watts=180.0, wall_s=1103.99)
        assert wh == pytest.approx(55.1995, abs=1e-4)

    def test_idle_30w_for_one_minute(self) -> None:
        """30 W * 60 s / 3600 = 0.5 Wh."""
        assert _Mixed().compute_energy_wh(watts=30.0, wall_s=60.0) == pytest.approx(0.5)

    def test_zero_wall_seconds_yields_zero_wh(self) -> None:
        assert _Mixed().compute_energy_wh(watts=180.0, wall_s=0.0) == 0.0

    def test_zero_watts_yields_zero_wh(self) -> None:
        assert _Mixed().compute_energy_wh(watts=0.0, wall_s=1234.5) == 0.0

    def test_fractional_inputs(self) -> None:
        """0.5 W * 7200 s / 3600 = 1.0 Wh."""
        assert _Mixed().compute_energy_wh(watts=0.5, wall_s=7200.0) == pytest.approx(1.0)

    def test_negative_watts_raises(self) -> None:
        with pytest.raises(ValueError, match="watts must be non-negative"):
            _Mixed().compute_energy_wh(watts=-5.0, wall_s=10.0)

    def test_negative_wall_seconds_raises(self) -> None:
        with pytest.raises(ValueError, match="wall_s must be non-negative"):
            _Mixed().compute_energy_wh(watts=100.0, wall_s=-1.0)

    def test_kw_only_signature_rejects_positional_call(self) -> None:
        """The ``*,`` in the signature forces kw-only; positional must fail."""
        with pytest.raises(TypeError):
            _Mixed().compute_energy_wh(100.0, 3600.0)  # type: ignore[misc]
