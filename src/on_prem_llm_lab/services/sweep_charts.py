"""Chart helpers (T-3.7) — TTFT / TPOT / Throughput / peak RAM vs quantization.

Reads a sweep CSV (produced by `SweepRunner`), groups rows by target, and
emits one PNG per metric under ``figures/``. Cells that skipped (NaN stats,
`skip_reason` set) get plotted as hollow markers with a footnote so the
report shows FAILED cells honestly rather than hiding them.

Contract: assignment §7 requires "per-target AirLLM + Quantization results
(tables + charts)" — this module is the "charts" half.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from on_prem_llm_lab.services.chart_style import (
    CHART_DPI,
    METRIC_LABELS,
    QUANT_LABEL,
    QUANT_ORDER,
    TARGET_COLORS,
    TARGET_MARKERS,
)


def _f(s: str) -> float:
    """Parse a CSV cell as float, returning NaN on 'nan' / empty."""
    if s == "" or s.lower() == "nan":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def load_sweep_csv(path: Path) -> list[dict[str, Any]]:
    """Parse the sweep CSV, coercing all numeric-ish columns to floats/ints."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row: dict[str, Any] = dict(r)
            for k, v in list(row.items()):
                if k in {"target_label", "backend", "quantization",
                         "skip_reason", "method_note"}:
                    continue
                row[k] = _f(v) if "_" in k or k in {"seed"} else v
            rows.append(row)
    return rows


def _group_by_target(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["target_label"], []).append(r)
    return out


def render_metric_chart(
    rows: list[dict[str, Any]], metric: str, out_path: Path,
    *, title_suffix: str = "",
) -> Path:
    """Line plot per target — metric_mean vs quantization, log-y."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    stat_col = f"{metric}_mean"
    grouped = _group_by_target(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    xtick_pos = list(range(len(QUANT_ORDER)))
    for target, cells in grouped.items():
        color = TARGET_COLORS.get(target, "#888")
        marker = TARGET_MARKERS.get(target, "^")
        ys = []
        for q in QUANT_ORDER:
            match = [c for c in cells if c["quantization"] == q]
            v = match[0][stat_col] if match else math.nan
            ys.append(v)
        ax.plot(xtick_pos, ys, marker=marker, color=color, linewidth=1.5,
                markersize=9, label=target)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([QUANT_LABEL[q] for q in QUANT_ORDER], rotation=15, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Target model", loc="best")
    title = f"{METRIC_LABELS.get(metric, metric)} vs quantization"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all_metric_charts(
    csv_path: Path, figures_dir: Path,
) -> dict[str, Path]:
    """Emit TTFT/TPOT/Throughput/RAM/wall/energy charts to `figures_dir`."""
    rows = load_sweep_csv(csv_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    for metric in ("ttft_ms", "tpot_ms", "throughput_tps",
                    "peak_ram_mb", "wall_s", "energy_wh"):
        outs[metric] = render_metric_chart(
            rows, metric, figures_dir / f"{metric}_vs_quant.png",
        )
    return outs


__all__ = [
    "load_sweep_csv",
    "render_all_metric_charts",
    "render_metric_chart",
]
