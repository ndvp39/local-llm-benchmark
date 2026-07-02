"""One-shot post-sweep analyzer — runs all M3/M4 downstream helpers.

Reads the latest ``results/sweep_<ts>.csv`` + per-run manifests + config,
then produces every derived artifact the README needs:

* Quality matrix (T-3.6) → ``results/quality_matrix.md``
* Metric charts (T-3.7) → ``figures/{ttft,tpot,throughput,peak_ram,wall,energy}_vs_quant.png``
* Roofline analysis + chart (T-3.7c / DP-6) →
  ``results/roofline_analysis_<ts>.csv`` +
  ``results/roofline_manifest_<ts>.json`` + ``figures/roofline.png``
* Economic analysis + break-even chart (T-4.1 / T-4.2 / DP-5) →
  ``results/economic_analysis_<ts>.csv`` +
  ``results/economic_manifest_<ts>.json`` + ``figures/break_even.png``

Usage::

    uv run python tools/analyze_sweep.py [SWEEP_CSV]

If ``SWEEP_CSV`` is omitted, the newest ``results/sweep_*.csv`` is used.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import on_prem_llm_lab  # noqa: F401 — side-effect: load .env + patch airllm
from on_prem_llm_lab.services.break_even_chart import render as render_be_chart
from on_prem_llm_lab.services.economic_analyzer import (
    analyze_sweep,
    load_pricing,
)
from on_prem_llm_lab.services.economic_analyzer import (
    write_analysis_csv as write_econ_csv,
)
from on_prem_llm_lab.services.economic_analyzer import (
    write_manifest as write_econ_manifest,
)
from on_prem_llm_lab.services.quality_matrix import write_quality_matrix
from on_prem_llm_lab.services.roofline_analyzer import (
    RooflineCeilings,
)
from on_prem_llm_lab.services.roofline_analyzer import (
    analyze_row as roofline_analyze_row,
)
from on_prem_llm_lab.services.roofline_analyzer import (
    write_analysis_csv as write_roofline_csv,
)
from on_prem_llm_lab.services.roofline_analyzer import (
    write_manifest as write_roofline_manifest,
)
from on_prem_llm_lab.services.roofline_chart import render as render_roofline
from on_prem_llm_lab.services.sweep_charts import (
    load_sweep_csv,
    render_all_metric_charts,
)


def _pick_latest_sweep_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("sweep_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no sweep_*.csv in {results_dir}")
    return candidates[-1]


def _n_params_for_target(target_label: str, config: dict) -> float:
    for t in config.get("target_models", []):
        if t.get("label") == target_label:
            return float(t.get("n_params_billion", 8.0))
    return 8.0


def main() -> int:
    repo_root = Path.cwd()
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    sweep_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else _pick_latest_sweep_csv(results_dir)
    print(f"[analyze_sweep] sweep CSV: {sweep_csv}")

    config = json.loads((repo_root / "config" / "setup.json").read_text(encoding="utf-8"))
    pricing = load_pricing(repo_root / "config" / "api_pricing.json")
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prompt = (config.get("generation") or {}).get("sweep_prompt") or (
        config.get("generation") or {}).get("baseline_prompt") or "Hello, world."

    print("[analyze_sweep] 1. Quality matrix")
    qm_path = write_quality_matrix(results_dir, prompt=prompt)
    print(f"   -> {qm_path}")

    print("[analyze_sweep] 2. Metric charts")
    metric_outs = render_all_metric_charts(sweep_csv, figures_dir)
    for m, p in metric_outs.items():
        print(f"   -> {m}: {p}")

    print("[analyze_sweep] 3. Roofline analysis")
    rows = load_sweep_csv(sweep_csv)
    roofline_cfg = config.get("roofline") or {}
    ceilings = RooflineCeilings(
        peak_compute_gflops=float(roofline_cfg.get("peak_compute_gflops", 50.0)),
        peak_dram_bandwidth_gbps=float(roofline_cfg.get("peak_dram_bandwidth_gbps", 30.0)),
        peak_disk_bandwidth_mbps=float(roofline_cfg.get("peak_disk_bandwidth_mbps", 100.0)),
    )
    points = []
    for r in rows:
        n_p = _n_params_for_target(r["target_label"], config)
        p = roofline_analyze_row(r, n_p, ceilings)
        if p is not None:
            points.append(p)
    roof_csv = results_dir / f"roofline_analysis_{ts}.csv"
    roof_mf = results_dir / f"roofline_manifest_{ts}.json"
    write_roofline_csv(roof_csv, points)
    write_roofline_manifest(roof_mf, ceilings, points, sweep_csv)
    print(f"   -> {roof_csv}")
    print(f"   -> {roof_mf}")
    if points:
        roof_png = figures_dir / "roofline.png"
        render_roofline(points, ceilings, roof_png)
        print(f"   -> {roof_png}")
    else:
        print("   -> no successful roofline points (all NaN); chart skipped")

    print("[analyze_sweep] 4. Economic analysis + break-even chart")
    cost_points, snapshot = analyze_sweep(sweep_csv, config, pricing)
    econ_csv = results_dir / f"economic_analysis_{ts}.csv"
    econ_mf = results_dir / f"economic_manifest_{ts}.json"
    write_econ_csv(econ_csv, cost_points)
    write_econ_manifest(econ_mf, snapshot, cost_points, sweep_csv)
    print(f"   -> {econ_csv}")
    print(f"   -> {econ_mf}")
    if cost_points:
        be_png = figures_dir / "break_even.png"
        render_be_chart(rows, snapshot, be_png)
        print(f"   -> {be_png}")
    else:
        print("   -> no successful sweep rows; break-even chart skipped")

    print("[analyze_sweep] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
