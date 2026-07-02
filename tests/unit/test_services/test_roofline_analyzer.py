"""Tests for `services/roofline_analyzer.py` (T-3.7c pure per DP-6)."""

from __future__ import annotations

import math

import pytest

from on_prem_llm_lab.services.roofline_analyzer import (
    RooflineCeilings,
    analyze_row,
    arith_intensity_for_bit_width,
    attained_gflops_from_tpot,
    bit_width_of,
    regime_for,
    ridge_intensity,
    wall_share_disk_pct,
)


def _ceilings() -> RooflineCeilings:
    return RooflineCeilings(
        peak_compute_gflops=50.0,
        peak_dram_bandwidth_gbps=30.0,
        peak_disk_bandwidth_mbps=100.0,
    )


class TestBitWidth:
    @pytest.mark.parametrize(
        ("quant", "expected"),
        [("fp32", 32), ("fp16", 16), ("q8", 8), ("q4", 4)],
    )
    def test_supported(self, quant: str, expected: int) -> None:
        assert bit_width_of(quant) == expected

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported bit width"):
            bit_width_of("bogus")


class TestArithIntensity:
    def test_fp16(self) -> None:
        assert arith_intensity_for_bit_width(16) == 1.0

    def test_q4(self) -> None:
        assert arith_intensity_for_bit_width(4) == 4.0

    def test_q8(self) -> None:
        assert arith_intensity_for_bit_width(8) == 2.0


class TestRidgeIntensity:
    def test_dram_ridge(self) -> None:
        # 50 GFLOPS / 30 GB/s = 1.67 FLOPs/byte
        r = ridge_intensity(50.0, 30e9)
        assert r == pytest.approx(50e9 / 30e9, rel=1e-9)
        assert 1.6 < r < 1.7

    def test_disk_ridge_much_higher(self) -> None:
        # 50 GFLOPS / 100 MB/s = 500 FLOPs/byte
        r = ridge_intensity(50.0, 100e6)
        assert r == pytest.approx(500.0, rel=1e-9)


class TestAttainedGflops:
    def test_llama_8b_tpot_371s(self) -> None:
        # 2 × 8e9 / 371 / 1e9 = ~0.0431 GFLOPS
        g = attained_gflops_from_tpot(8.0, 371.0)
        assert g == pytest.approx((2 * 8e9 / 371) / 1e9, rel=1e-9)
        assert 0.04 < g < 0.05

    def test_zero_tpot_returns_nan(self) -> None:
        assert math.isnan(attained_gflops_from_tpot(8.0, 0.0))

    def test_nan_tpot_returns_nan(self) -> None:
        assert math.isnan(attained_gflops_from_tpot(8.0, math.nan))


class TestWallShareDisk:
    def test_llama_fp16_43pct(self) -> None:
        # bytes = 8e9 × 16/8 = 16 GB per token
        # disk_seconds = 16e9 / 100e6 = 160 s
        # wall_share = 100 × 160 / 371 = 43.13%
        share = wall_share_disk_pct(8.0, 16, 100e6, 371.0)
        assert 40 < share < 45


class TestRegimeFor:
    def test_memory_bound_when_below_ridge(self) -> None:
        assert regime_for(1.0, 500.0) == "memory_bound"

    def test_compute_bound_when_above_ridge(self) -> None:
        assert regime_for(1000.0, 500.0) == "compute_bound"

    def test_boundary_is_memory(self) -> None:
        assert regime_for(500.0, 500.0) == "memory_bound"


class TestAnalyzeRow:
    def test_nan_tpot_returns_none(self) -> None:
        row = {
            "target_label": "llama", "quantization": "fp16", "backend": "airllm",
            "tpot_ms_mean": math.nan, "wall_s_mean": 1.0,
        }
        assert analyze_row(row, 8.0, _ceilings()) is None

    def test_llama_fp16_airllm(self) -> None:
        row = {
            "target_label": "llama3-8b-fp16", "quantization": "fp16",
            "backend": "airllm", "tpot_ms_mean": 371000.0, "wall_s_mean": 371.0,
        }
        p = analyze_row(row, 8.0, _ceilings())
        assert p is not None
        assert p.arith_intensity == 1.0
        assert p.regime == "memory_bound"  # 1.0 vs ridge_disk=500
        assert p.wall_share_disk_pct is not None and 40 < p.wall_share_disk_pct < 45

    def test_direct_backend_uses_dram_ridge(self) -> None:
        row = {
            "target_label": "small", "quantization": "fp16",
            "backend": "direct", "tpot_ms_mean": 100.0, "wall_s_mean": 1.0,
        }
        p = analyze_row(row, 1.0, _ceilings())
        assert p is not None
        # ridge_dram = 50/30 = 1.67; I=1.0 → memory_bound
        assert p.regime == "memory_bound"
        assert p.wall_share_disk_pct is None  # only computed for airllm
