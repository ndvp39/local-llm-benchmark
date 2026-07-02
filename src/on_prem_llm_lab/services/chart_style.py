"""Chart style constants (T-3.7) — colors, markers, axes labels.

Kept as constants-only so every chart helper looks visually consistent.
Deliberately tiny: matplotlib rc is *not* set here — plot helpers pass
explicit kwargs so the module has zero side effects at import time.
"""

from __future__ import annotations

# Per-target color + marker (both models must render distinctly)
TARGET_COLORS: dict[str, str] = {
    "llama3-8b-fp16": "#1f77b4",   # blue
    "qwen2-7b-q4":    "#d62728",   # red
}
TARGET_MARKERS: dict[str, str] = {
    "llama3-8b-fp16": "o",  # circle
    "qwen2-7b-q4":    "s",  # square
}

# Ordered x-axis positions for quantization levels (compute-density order)
QUANT_ORDER: tuple[str, ...] = ("fp32", "fp16", "q8", "q4")
QUANT_LABEL: dict[str, str] = {
    "fp32": "FP32 (32-bit)",
    "fp16": "FP16 (16-bit)",
    "q8":   "int8 (8-bit)",
    "q4":   "NF4 (4-bit)",
}

# Axis labels — match DP-4 metric names
METRIC_LABELS: dict[str, str] = {
    "ttft_ms":       "TTFT — time to first token (ms)",
    "tpot_ms":       "TPOT — per output token (ms)",
    "throughput_tps": "Throughput (tokens / s)",
    "peak_ram_mb":   "Peak RAM (MB)",
    "wall_s":        "Wall time (s)",
    "energy_wh":     "Energy (Wh)",
}

CHART_DPI: int = 150

__all__ = [
    "CHART_DPI",
    "METRIC_LABELS",
    "QUANT_LABEL",
    "QUANT_ORDER",
    "TARGET_COLORS",
    "TARGET_MARKERS",
]
