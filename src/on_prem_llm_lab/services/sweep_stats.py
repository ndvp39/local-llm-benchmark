"""Statistics + row schema + CSV/manifest serialization for :mod:`sweep_runner`.

Split out of ``sweep_runner.py`` to keep the runner file under the 150-LOC
cap (constitution §2.2 sanctioned). Pure helpers — no state; only the
CSV/manifest writers do file I/O.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NAN = math.nan

METRICS: tuple[str, ...] = (
    "ttft_ms", "tpot_ms", "throughput_tps",
    "peak_ram_mb", "wall_s", "energy_wh",
)
_STAT_FIELDS: tuple[str, ...] = ("mean", "median", "std", "min", "max", "p95")


@dataclass(frozen=True)
class MetricStats:
    """Aggregate statistics for one metric across N measurement runs."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p95: float


NAN_STATS: MetricStats = MetricStats(
    mean=_NAN, median=_NAN, std=_NAN, min=_NAN, max=_NAN, p95=_NAN,
)


@dataclass(frozen=True, kw_only=True)
class SweepRow:
    """One CSV row (DP-4 §5.1). Nested stats flatten via :meth:`flatten`."""

    target_label: str
    backend: str
    quantization: str
    seed: int
    prompt_tokens: int
    max_new_tokens: int
    completion_tokens: int
    repeat: int
    warmup_repeats: int
    n_success: int
    n_failed: int
    skip_reason: str | None
    method_note: str
    stats: dict[str, MetricStats] = field(default_factory=dict)


def aggregate(values: list[float]) -> MetricStats:
    """Compute mean+median+std+min+max+p95 over ``values``.

    Empty list -> ``NAN_STATS``. Single value -> std=0, all others match.
    < 20 values -> p95 collapses to max (per DP-4 §5.2 default).
    """
    if not values:
        return NAN_STATS
    m = statistics.mean(values)
    med = statistics.median(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    lo = min(values)
    hi = max(values)
    p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else hi
    return MetricStats(mean=m, median=med, std=std, min=lo, max=hi, p95=p95)


def has_visible_second_prefill(
    ttft_ms: float, tpot_ms: float, max_new_tokens: int,
) -> bool:
    """FR-BM-10 — the two-generate second-prefill overhead is 'visible'."""
    if max_new_tokens < 2 or tpot_ms <= 0:
        return False
    return ttft_ms / (max_new_tokens - 1) > 0.1 * tpot_ms


def csv_columns() -> list[str]:
    """FR-BM-6 / §5.3 canonical CSV column order."""
    base = [
        "target_label", "backend", "quantization", "seed",
        "prompt_tokens", "max_new_tokens", "completion_tokens",
        "repeat", "warmup_repeats", "n_success", "n_failed",
        "skip_reason", "method_note",
    ]
    for m in METRICS:
        base.extend(f"{m}_{f}" for f in _STAT_FIELDS)
    return base


def row_to_csv_dict(row: SweepRow) -> dict[str, Any]:
    d: dict[str, Any] = {
        k: v for k, v in dataclasses.asdict(row).items() if k != "stats"
    }
    for m in METRICS:
        s = row.stats.get(m, NAN_STATS)
        for f in _STAT_FIELDS:
            d[f"{m}_{f}"] = getattr(s, f)
    return d


def write_csv(path: Path, rows: list[SweepRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns())
        w.writeheader()
        for r in rows:
            w.writerow(row_to_csv_dict(r))


def write_manifest(
    path: Path, config: Mapping[str, Any], rows: list[SweepRow], ts: str,
) -> None:
    path.write_text(json.dumps({
        "captured_at": ts, "config_snapshot": dict(config),
        "n_cells": len(rows),
        "n_supported": sum(1 for r in rows if r.skip_reason is None),
        "n_skipped": sum(1 for r in rows if r.skip_reason),
    }, indent=2), encoding="utf-8")


__all__ = [
    "METRICS",
    "NAN_STATS",
    "MetricStats",
    "SweepRow",
    "aggregate",
    "csv_columns",
    "has_visible_second_prefill",
    "row_to_csv_dict",
    "write_csv",
    "write_manifest",
]
