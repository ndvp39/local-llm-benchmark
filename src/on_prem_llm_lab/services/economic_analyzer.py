"""Economic analyzer (T-4.1 per DP-5) — CSV I/O + break-even CSV emit.

Reads a sweep CSV + `config.economic` + `config/api_pricing.json`, emits
per-cell per-option cost points to `results/economic_analysis_<ts>.csv`
and a manifest under `results/economic_manifest_<ts>.json`. The chart
helper (`break_even_chart.py`) consumes the analyzer's output.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.services.economic_math import (
    AssumptionsSnapshot,
    break_even_volume,
    cost_per_req_anthropic,
    cost_per_req_cloud_gpu,
    cost_per_req_on_prem,
)


@dataclass(frozen=True, kw_only=True)
class CostPoint:
    target_label: str
    quantization: str
    backend: str
    option: str
    wall_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_per_req_usd: float
    break_even_volume_vs_anthropic: int | None
    break_even_volume_vs_cloud_gpu: int | None


def load_pricing(pricing_path: Path) -> dict[str, Any]:
    return json.loads(pricing_path.read_text(encoding="utf-8"))


def build_assumptions(
    config: dict[str, Any], pricing: dict[str, Any],
) -> AssumptionsSnapshot:
    econ = config.get("economic") or {}
    energy = config.get("energy") or {}
    cloud = econ.get("cloud_gpu") or {}
    anth = (pricing.get("providers") or {}).get("anthropic") or {}
    return AssumptionsSnapshot(
        capex_usd=float(econ.get("hardware_capex_usd", 2500)),
        lifetime_hours=float(econ.get("lifetime_hours", 5000)),
        watts_active=float(energy.get("assumed_watts_active", 180)),
        electricity_price_per_kwh_usd=float(
            energy.get("electricity_price_per_kwh_usd", 0.16),
        ),
        cloud_gpu_hourly_usd=float(cloud.get("hourly_usd", 0.6)),
        cloud_gpu_speedup_over_cpu=float(cloud.get("speedup_over_cpu", 1000.0)),
        anthropic_in_per_million_usd=float(anth.get("in_per_million_usd", 1.0)),
        anthropic_out_per_million_usd=float(anth.get("out_per_million_usd", 5.0)),
        anthropic_model=str(anth.get("model", "unknown")),
        anthropic_prices_captured_at=anth.get("captured_at"),
    )


def _f(s: str) -> float:
    if s == "" or s.lower() == "nan":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _load_sweep_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out: dict[str, Any] = dict(r)
            for k, v in list(r.items()):
                if k in {"target_label", "backend", "quantization",
                         "skip_reason", "method_note"}:
                    continue
                out[k] = _f(v)
            rows.append(out)
    return rows


def _points_for_row(
    row: dict[str, Any], s: AssumptionsSnapshot,
) -> list[CostPoint]:
    wall_s_mean = row.get("wall_s_mean")
    if wall_s_mean is None or (isinstance(wall_s_mean, float) and math.isnan(wall_s_mean)):
        return []
    wall_s = float(wall_s_mean)
    prompt_tokens = int(row.get("prompt_tokens") or 0)
    completion_tokens = int(row.get("completion_tokens") or 0)
    base_kwargs = {
        "target_label": str(row["target_label"]),
        "quantization": str(row["quantization"]),
        "backend": str(row["backend"]),
        "wall_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    on_prem_opex = cost_per_req_on_prem(wall_s, s)
    on_prem_electricity_only = s.watts_active * (wall_s / 3600.0) * (
        s.electricity_price_per_kwh_usd / 1000.0
    )
    anth = cost_per_req_anthropic(prompt_tokens, completion_tokens, s)
    cloud = cost_per_req_cloud_gpu(wall_s, s)
    on_prem_be_vs_anth = break_even_volume(s.capex_usd, on_prem_electricity_only, anth)
    on_prem_be_vs_cloud = break_even_volume(s.capex_usd, on_prem_electricity_only, cloud)
    return [
        CostPoint(**base_kwargs, option="on_prem",
                  cost_per_req_usd=on_prem_opex,
                  break_even_volume_vs_anthropic=on_prem_be_vs_anth,
                  break_even_volume_vs_cloud_gpu=on_prem_be_vs_cloud),
        CostPoint(**base_kwargs, option="anthropic", cost_per_req_usd=anth,
                  break_even_volume_vs_anthropic=None,
                  break_even_volume_vs_cloud_gpu=None),
        CostPoint(**base_kwargs, option="cloud_gpu", cost_per_req_usd=cloud,
                  break_even_volume_vs_anthropic=None,
                  break_even_volume_vs_cloud_gpu=None),
    ]


def analyze_sweep(
    sweep_csv_path: Path, config: dict[str, Any], pricing: dict[str, Any],
) -> tuple[list[CostPoint], AssumptionsSnapshot]:
    snapshot = build_assumptions(config, pricing)
    rows = _load_sweep_csv(sweep_csv_path)
    points: list[CostPoint] = []
    for r in rows:
        points.extend(_points_for_row(r, snapshot))
    return points, snapshot


def write_analysis_csv(path: Path, points: list[CostPoint]) -> None:
    cols = [
        "target_label", "quantization", "backend", "option", "wall_s",
        "prompt_tokens", "completion_tokens", "cost_per_req_usd",
        "break_even_volume_vs_anthropic", "break_even_volume_vs_cloud_gpu",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))


def write_manifest(
    path: Path, snapshot: AssumptionsSnapshot,
    points: list[CostPoint], sweep_csv_path: Path,
) -> None:
    """Serialise assumptions + rows to a JSON manifest."""
    m = {"captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "sweep_csv_path": str(sweep_csv_path),
         "assumptions": asdict(snapshot), "n_points": len(points),
         "rows": [asdict(p) for p in points]}
    path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")


__all__ = [
    "CostPoint",
    "analyze_sweep",
    "build_assumptions",
    "load_pricing",
    "write_analysis_csv",
    "write_manifest",
]
