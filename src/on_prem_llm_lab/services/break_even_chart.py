"""Break-even chart (T-4.2 per DP-5) — 3-curve cost-per-request vs volume.

Per DP-5 §2.4: X-axis is cumulative request volume (log); Y-axis is
cost-per-request. On-Prem is `(capex / N) + opex_per_req` — hyperbolic;
Anthropic + Cloud GPU are flat horizontal lines. Crossovers annotated.

Uses ONE representative cell per DP-5 Q-EA-2 (the hero cell —
`llama3-8b-fp16` × `q4` × `airllm`). Falls back to the first successful
row if that specific cell has NaN stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from on_prem_llm_lab.services.economic_math import (
    AssumptionsSnapshot,
    cost_per_req_anthropic,
    cost_per_req_cloud_gpu,
    cost_per_req_on_prem,
)


def _pick_hero(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for target, quant in (("llama3-8b-fp16", "q4"), ("llama3-8b-fp16", "fp16")):
        for r in rows:
            if r["target_label"] == target and r["quantization"] == quant:
                w = r.get("wall_s_mean")
                if isinstance(w, float) and w > 0:
                    return r
    for r in rows:
        w = r.get("wall_s_mean")
        if isinstance(w, float) and w > 0:
            return r
    return None


def render(
    sweep_rows: list[dict[str, Any]], snapshot: AssumptionsSnapshot, out_path: Path,
) -> Path:
    """Plot 3-curve break-even chart + annotations to `out_path`."""
    import matplotlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    hero = _pick_hero(sweep_rows)
    if hero is None:
        raise ValueError("break_even_chart: no successful sweep row found")

    wall_s = float(hero["wall_s_mean"])
    prompt_tokens = int(hero.get("prompt_tokens") or 0)
    completion_tokens = int(hero.get("completion_tokens") or 0)

    _ = cost_per_req_on_prem(wall_s, snapshot)  # computed elsewhere; kept for API symmetry
    electricity_only_op = snapshot.watts_active * (wall_s / 3600.0) * (
        snapshot.electricity_price_per_kwh_usd / 1000.0
    )
    anth = cost_per_req_anthropic(prompt_tokens, completion_tokens, snapshot)
    cloud = cost_per_req_cloud_gpu(wall_s, snapshot)

    n_axis = np.geomspace(1, 1_000_000, 400)
    cost_on_prem_curve = (snapshot.capex_usd / n_axis) + electricity_only_op
    cost_anth = np.full_like(n_axis, anth)
    cost_cloud = np.full_like(n_axis, cloud)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(n_axis, cost_on_prem_curve, color="#1f77b4", linewidth=2.0,
            label=f"On-Prem CPU laptop ({snapshot.watts_active:.0f} W)")
    ax.plot(n_axis, cost_anth, color="#2ca02c", linestyle="--", linewidth=2.0,
            label=f"Anthropic API ({snapshot.anthropic_model})")
    ax.plot(n_axis, cost_cloud, color="#d62728", linestyle=":", linewidth=2.0,
            label=(
                f"Cloud GPU ({snapshot.cloud_gpu_hourly_usd:.2f}/hr, "
                f"{snapshot.cloud_gpu_speedup_over_cpu:.0f}x speedup)"
            ))

    for competitor_cost, color, name in (
        (anth, "#2ca02c", "vs Anthropic"),
        (cloud, "#d62728", "vs Cloud GPU"),
    ):
        delta = competitor_cost - electricity_only_op
        if delta > 0:
            n_break = snapshot.capex_usd / delta
            if 1 <= n_break <= 1_000_000:
                ax.axvline(x=n_break, color=color, alpha=0.3, linewidth=1)
                ax.annotate(
                    f"break-even {name}\nN ~= {int(n_break):,}",
                    xy=(n_break, competitor_cost),
                    xytext=(5, 20), textcoords="offset points",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.5},
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative request volume (log)")
    ax.set_ylabel("Cost per request (USD, log)")
    ax.set_title(
        f"Break-even: On-Prem vs Anthropic vs Cloud GPU\n"
        f"(hero cell: {hero['target_label']} x {hero['quantization']} x "
        f"{hero['backend']}, wall={wall_s:.0f}s)",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Assumptions box
    assumptions_text = "\n".join([
        f"CAPEX: ${snapshot.capex_usd:.0f}",
        f"Lifetime: {snapshot.lifetime_hours:.0f} hr",
        f"Electricity: ${snapshot.electricity_price_per_kwh_usd:.2f}/kWh",
        f"Watts (active): {snapshot.watts_active:.0f}",
        f"Cloud GPU: ${snapshot.cloud_gpu_hourly_usd:.2f}/hr",
        f"Cloud speedup: {snapshot.cloud_gpu_speedup_over_cpu:.0f}x",
        f"Anthropic in: ${snapshot.anthropic_in_per_million_usd:.2f}/M",
        f"Anthropic out: ${snapshot.anthropic_out_per_million_usd:.2f}/M",
    ])
    ax.text(
        0.02, 0.02, assumptions_text, transform=ax.transAxes,
        fontsize=7.5, verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "#f5f5f5", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = ["render"]
