"""Roofline chart (T-3.7c per DP-6) — plot ceilings + data points + ridges.

log-log plot: arithmetic intensity (FLOPs/byte) on X, attained perf
(GFLOPs/sec) on Y. Two sloped memory-bandwidth ceilings (DRAM + disk) +
one horizontal compute ceiling. Points annotated with (target, quant).

See `docs/PRD_roofline.md` §5 and FR-RL-7 for the chart specification.
"""

from __future__ import annotations

from pathlib import Path

from on_prem_llm_lab.services.chart_style import (
    CHART_DPI,
    TARGET_COLORS,
    TARGET_MARKERS,
)
from on_prem_llm_lab.services.roofline_analyzer import (
    RooflineCeilings,
    RooflinePoint,
)


def render(
    points: list[RooflinePoint], ceilings: RooflineCeilings, out_path: Path,
) -> Path:
    """Render Roofline chart with ceilings + points + ridge annotations."""
    import matplotlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(9, 6))

    # X range covers observed intensities (with padding)
    i_min = min(0.1, min((p.arith_intensity for p in points), default=0.1) / 2)
    i_max = max(1000.0, max((p.arith_intensity for p in points), default=1000.0) * 2)
    intensities = np.geomspace(i_min, i_max, 200)

    peak_c = ceilings.peak_compute_gflops
    beta_dram = ceilings.peak_dram_bandwidth_gbps  # GB/s
    beta_disk_gbps = ceilings.peak_disk_bandwidth_mbps / 1000.0  # GB/s

    # DRAM roof: min(peak_c, beta_dram * I)
    dram_ceiling = np.minimum(peak_c, beta_dram * intensities)
    disk_ceiling = np.minimum(peak_c, beta_disk_gbps * intensities)

    ax.plot(intensities, dram_ceiling, color="#666", linestyle="--", linewidth=1.5,
            label=f"DRAM roof ({beta_dram:.0f} GB/s)")
    ax.plot(intensities, disk_ceiling, color="#a0522d", linestyle="-", linewidth=1.8,
            label=f"USB HDD roof ({ceilings.peak_disk_bandwidth_mbps:.0f} MB/s)")
    ax.axhline(y=peak_c, color="#444", linestyle=":", linewidth=1.2,
               label=f"Peak compute ({peak_c:.0f} GFLOPS)")

    # Ridge points — where roofs meet the compute ceiling
    ridge_dram = peak_c / beta_dram
    ridge_disk = peak_c / beta_disk_gbps
    ax.axvline(x=ridge_dram, color="#666", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(x=ridge_disk, color="#a0522d", linestyle=":", linewidth=0.8, alpha=0.5)

    for p in points:
        color = TARGET_COLORS.get(p.target_label, "#000")
        marker = TARGET_MARKERS.get(p.target_label, "^")
        ax.scatter([p.arith_intensity], [p.attained_gflops],
                   color=color, marker=marker, s=110, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(
            f"{p.quantization}", xy=(p.arith_intensity, p.attained_gflops),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Attained performance (GFLOPs / sec)")
    ax.set_title("Roofline — AirLLM CPU-only vs peak-compute / disk-bandwidth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = ["render"]
