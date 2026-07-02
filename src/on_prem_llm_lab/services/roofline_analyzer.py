"""Roofline analyzer (T-3.7c per DP-6) — pure math + CSV/manifest emit.

For each sweep-CSV row, computes: `arith_intensity` (16/bit_width),
`attained_gflops` from TPOT, `ceiling_gflops` per backend's β_effective
(DRAM for direct, DISK for airllm), and `regime` label
(`compute_bound` | `memory_bound`).

See `docs/PRD_roofline.md` §5 for the full formulas.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, kw_only=True)
class RooflineCeilings:
    peak_compute_gflops: float
    peak_dram_bandwidth_gbps: float
    peak_disk_bandwidth_mbps: float


@dataclass(frozen=True, kw_only=True)
class RooflinePoint:
    target_label: str
    quantization: str
    backend: str
    n_params_billion: float
    bit_width: int
    tpot_seconds: float
    arith_intensity: float
    attained_gflops: float
    ceiling_gflops: float
    regime: str
    ridge_intensity: float
    wall_share_disk_pct: float | None


_BIT_BY_QUANT: dict[str, int] = {"fp32": 32, "fp16": 16, "q8": 8, "q4": 4}


def bit_width_of(quantization: str) -> int:
    if quantization not in _BIT_BY_QUANT:
        raise ValueError(f"unsupported bit width for quantization={quantization!r}")
    return _BIT_BY_QUANT[quantization]


def arith_intensity_for_bit_width(bit_width: int) -> float:
    return 16.0 / bit_width


def ridge_intensity(peak_gflops: float, peak_bandwidth_bytes_per_s: float) -> float:
    return (peak_gflops * 1e9) / peak_bandwidth_bytes_per_s


def attained_gflops_from_tpot(n_params_billion: float, tpot_seconds: float) -> float:
    if tpot_seconds <= 0 or math.isnan(tpot_seconds):
        return math.nan
    return (2.0 * n_params_billion * 1e9) / tpot_seconds / 1e9


def wall_share_disk_pct(
    n_params_billion: float, bit_width: int,
    peak_disk_bytes_per_s: float, wall_s: float,
) -> float:
    if wall_s <= 0 or math.isnan(wall_s):
        return math.nan
    bytes_per_token = n_params_billion * 1e9 * bit_width / 8.0
    disk_seconds = bytes_per_token / peak_disk_bytes_per_s
    return 100.0 * disk_seconds / wall_s


def regime_for(arith_intensity_val: float, ridge_intensity_val: float) -> str:
    return "compute_bound" if arith_intensity_val > ridge_intensity_val else "memory_bound"


def analyze_row(
    row: dict[str, Any], n_params_billion: float, ceilings: RooflineCeilings,
) -> RooflinePoint | None:
    """Compute a RooflinePoint for one CSV row. Returns None if row is NaN."""
    tpot_ms = row.get("tpot_ms_mean")
    if tpot_ms is None or (isinstance(tpot_ms, float) and math.isnan(tpot_ms)):
        return None
    quant = str(row["quantization"])
    backend = str(row["backend"])
    bit = bit_width_of(quant)
    intensity = arith_intensity_for_bit_width(bit)
    tpot_s = float(tpot_ms) / 1000.0
    attained = attained_gflops_from_tpot(n_params_billion, tpot_s)
    if backend == "airllm":
        beta = ceilings.peak_disk_bandwidth_mbps * 1e6
    else:
        beta = ceilings.peak_dram_bandwidth_gbps * 1e9
    ridge = ridge_intensity(ceilings.peak_compute_gflops, beta)
    ceiling = min(ceilings.peak_compute_gflops, beta / 1e9 * intensity)
    wall_share = None
    wall_s = row.get("wall_s_mean")
    if backend == "airllm" and wall_s is not None:
        try:
            wall_share = wall_share_disk_pct(
                n_params_billion, bit,
                ceilings.peak_disk_bandwidth_mbps * 1e6,
                float(wall_s),
            )
        except (ValueError, ZeroDivisionError):
            wall_share = None
    return RooflinePoint(
        target_label=str(row["target_label"]), quantization=quant, backend=backend,
        n_params_billion=n_params_billion, bit_width=bit, tpot_seconds=tpot_s,
        arith_intensity=intensity, attained_gflops=attained, ceiling_gflops=ceiling,
        regime=regime_for(intensity, ridge), ridge_intensity=ridge,
        wall_share_disk_pct=wall_share,
    )


def write_analysis_csv(path: Path, points: list[RooflinePoint]) -> None:
    cols = [
        "target_label", "quantization", "backend", "n_params_billion", "bit_width",
        "tpot_seconds", "arith_intensity", "attained_gflops", "ceiling_gflops",
        "regime", "ridge_intensity", "wall_share_disk_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))


def write_manifest(
    path: Path, ceilings: RooflineCeilings, points: list[RooflinePoint],
    sweep_csv_path: Path,
) -> None:
    path.write_text(json.dumps({
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sweep_csv_path": str(sweep_csv_path),
        "ceilings": asdict(ceilings),
        "n_points": len(points),
        "points": [asdict(p) for p in points],
    }, indent=2), encoding="utf-8")


__all__ = [
    "RooflineCeilings",
    "RooflinePoint",
    "analyze_row",
    "arith_intensity_for_bit_width",
    "attained_gflops_from_tpot",
    "bit_width_of",
    "regime_for",
    "ridge_intensity",
    "wall_share_disk_pct",
    "write_analysis_csv",
    "write_manifest",
]
