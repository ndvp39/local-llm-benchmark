# PRD — Economic Analysis (3-Curve Break-Even: On-Prem vs Anthropic API vs Cloud GPU)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-5 — **MANDATORY**.
> **Blocks:** **T-4.1** code (`services/economic_analyzer.py`) + **T-4.2** break-even chart helper — the break-even math + config schema + chart specification are not code-guessable; this PRD defines them. Approval required before T-4.1 implementation begins.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf` §5, §7 (economic comparison mandatory) → `L08-summary-Lora-AirLLM.pdf` §1.1 Table 1 (three deployment options) → `docs/PRD.md` v1.10 G5 (economic break-even 3 curves) + K7 (chart with crossovers) + FR-12 (Anthropic $/req) + FR-13 (On-Prem $/req) + FR-13a (Cloud GPU $/req) + FR-14 (matplotlib chart) + D-4 (Cloud GPU curve mandatory) + ADR-011 (Anthropic sole third-party) + ADR-012 (3-curve chart).
> **Empirical anchor:** the M3 sweep manifests (once landed) — `results/sweep_<ts>.csv` + 18 `results/run_<uuid>.json` per-run manifests provide the `wall_s`, `energy_wh`, `prompt_tokens`, `completion_tokens` values that all three cost formulas consume. Verified end-to-end by M2a's first per-run manifest (`results/run_21821fde-ac8c-4bf7-98a6-fe408cacb442.json`, 2026-07-01) — same 10 FR-9 fields present.
> **Document version:** 1.00 — 2026-07-01.
> **Status:** **APPROVED 2026-07-01** — user (ndvp39@gmail.com) confirmed the three open-question defaults: Q-EA-1 = 1000× Cloud GPU speedup, Q-EA-2 = hero cell (Llama-3-8B × q4 × airllm) as chart representative, Q-EA-3 = sensitivity chart optional. T-4.1 code cleared to start once the T-3.6b sweep CSV lands.

---

## 1. What and why

### 1.1 What
Formalise the **economic-analysis policy** — how the SDK converts raw benchmark data (`wall_s`, `energy_wh`, `prompt_tokens`, `completion_tokens`) into a **three-curve cost-per-request-vs-request-volume chart**, saved as `figures/break_even.png`. Covers:

* **Three cost formulas** — On-Prem CPU laptop (amortised CAPEX + electrical OPEX), Anthropic API (per-token pricing from `config/api_pricing.json`), Cloud GPU (rented hourly).
* **Volume axis definition** — request volume from `1` to `N_max`, log-scaled where crossovers demand it, with each curve's break-even point vs the others clearly marked.
* **Assumptions table** — electricity price, hardware CAPEX, lifetime, wattage, cloud hourly rate, Anthropic prices — every input published inline next to the chart per FR-14.
* **Sensitivity handling** — the chart uses point estimates from config; sensitivity ranges (e.g., ±20% electricity price) are captured as an optional companion figure (`figures/break_even_sensitivity.png`), not required by K7.
* **Uncertainty declaration** — CPU-only inference on 7.8 GB RAM is the reference workload; the report explicitly states which curve dominates in each regime and where the empirical measurements sit on the curves.

This PRD does NOT introduce new benchmarking. It defines how the ALREADY-CAPTURED sweep data feeds an economic model, and specifies the chart the report must show.

### 1.2 Why
* **Assignment G5 / K7.** The 3-curve chart is one of two headline deliverables (the other is the AirLLM/quantization sweep results). Without it, the report fails K7 outright.
* **Assignment §5 rigour.** Comparing "run locally vs pay for API" is the assignment's central economic question. The chart is the one visual that ties on-prem measurements to the "should you buy a laptop or use Anthropic?" business question.
* **Reproducibility.** Numbers change over time (electricity prices, Anthropic pricing, cloud GPU pricing). A dedicated PRD that names every input as a config field (never a magic number) means re-running with new prices needs zero code changes.
* **L08 §1.1 Table 1 alignment.** The lecture's three-way comparison (API / Cloud GPU / On-Prem) is exactly the three curves this chart shows — the report's L08 §1.1 discussion depends on the chart.

### 1.3 Who consumes this policy
| Consumer | Where the policy applies | Landing task |
|---|---|---|
| `services/economic_analyzer.py` | Reads sweep CSV + `config.economic` + `config/api_pricing.json`; emits per-curve cost points | T-4.1 |
| `services/break_even_chart.py` | Consumes per-curve points from T-4.1; plots 3 curves + crossovers to `figures/break_even.png` | T-4.2 |
| SDK method `economic_analysis(sweep)` | Public entry point (currently a stub in `sdk/_future_stubs.py`) | T-4.1 wire |
| CLI `economic-analysis` command | Delegates to SDK; produces figure + assumptions table | T-4.3 |
| Report assembler (T-6.1) | Embeds `figures/break_even.png` + assumptions table into README §Economic Analysis | T-6.1 |

---

## 2. Theoretical background & definitions

### 2.1 The three deployment options (L08 §1.1 Table 1)

| Option | CAPEX | OPEX per request | Failure mode |
|---|---|---|---|
| **On-Prem** (student's CPU laptop) | Hardware purchase, one-time | Electricity + wear (approximated as CAPEX amortisation over `lifetime_hours`) | Slow wall time (physics — see PRD_benchmarking_methodology §11) |
| **Anthropic API** (managed) | Zero | Per-token: `input_price × prompt_tokens + output_price × completion_tokens` | Rate limits + latency to provider |
| **Cloud GPU** (rent-a-box) | Zero (or trivial spin-up) | Hourly billing × wall seconds | Cold-start latency + GPU availability |

**Break-even question:** at what request volume does each option's cumulative cost overtake / undercut the others?

### 2.2 Cost-per-request formulas

Each formula computes the **cost of ONE request** at a given wall time and token count. The chart aggregates: over N requests, total cost is `N × cost_per_request` (linear for API/Cloud), plus a one-time CAPEX offset (for On-Prem).

**On-Prem** (FR-13):
```
cost_per_req_on_prem_usd(wall_s, capex_usd, lifetime_hours, watts_active, elec_price_per_kwh) =
    (capex_usd / lifetime_hours) × (wall_s / 3600)     # amortised hardware
    + watts_active × (wall_s / 3600) × (elec_price_per_kwh / 1000)  # electricity
```

**Anthropic API** (FR-12):
```
cost_per_req_anthropic_usd(prompt_tokens, completion_tokens, in_price_per_M, out_price_per_M) =
    (prompt_tokens × in_price_per_M + completion_tokens × out_price_per_M) / 1_000_000
```

**Cloud GPU** (FR-13a):
```
cost_per_req_cloud_gpu_usd(wall_s, hourly_usd) =
    hourly_usd × wall_s / 3600
```

**Design choice — Cloud GPU wall time source.** For the Cloud GPU curve, `wall_s` MUST NOT be the on-prem CPU measurement (that would inflate the curve absurdly). Two acceptable strategies:

1. **Scaled estimate (default).** Use a fixed speedup factor `cloud_gpu.speedup_over_cpu` (config, default `1000×`) so `wall_s_gpu = wall_s_cpu / speedup`. This reflects the ~1000× GPU speedup discussed in `docs/PRD_airllm_integration.md` and the report's L08 §1.1 discussion.
2. **Empirical (optional).** If a real cloud-GPU measurement is captured (bonus experiment on Colab or similar), use its `wall_s` directly. Overrides the scaled estimate for that cell.

**Chosen default: scaled estimate at 1000×.** Empirical override is a future enhancement — not required for K7. Config-controlled so the assumption is documented + adjustable.

### 2.3 Volume-cost curve construction

For each option, the **cumulative cost at request volume N** is:

```
cumulative_on_prem(N) = capex_usd + N × opex_per_req_on_prem
                     = capex_usd + N × (watts_active × wall_s / 3600 × elec_price / 1000)

cumulative_anthropic(N) = N × cost_per_req_anthropic
cumulative_cloud_gpu(N) = N × cost_per_req_cloud_gpu
```

**Break-even between On-Prem and Anthropic:**
```
N_break_on_prem_vs_anthropic = capex_usd / (cost_per_req_anthropic - opex_per_req_on_prem)
```
(only valid when denominator > 0; otherwise Anthropic is always cheaper per request and On-Prem never breaks even — a legitimate outcome that the chart shows honestly).

Similar formulas for On-Prem vs Cloud GPU and Anthropic vs Cloud GPU.

### 2.4 Cost-per-request axis vs cumulative axis

The chart uses **cost-per-request on the Y axis** (not cumulative), because that's how the "at what volume does buying overtake renting?" story reads naturally:

* Y-axis: `cost_per_request_usd`
* X-axis: `cumulative_request_volume` (log-scaled if crossovers span >2 orders of magnitude)
* On-Prem curve: `(capex_usd / N) + opex_per_req` — decreasing hyperbolic (amortisation kicks in as N grows)
* Anthropic curve: flat horizontal line at `cost_per_req_anthropic`
* Cloud GPU curve: flat horizontal line at `cost_per_req_cloud_gpu`

**Crossover points are where the On-Prem hyperbola dips below each flat line** — visually intuitive + matches assignment §5 economic-question framing.

---

## 3. What this PRD is NOT

* **Not a Roofline analysis.** The Roofline chart (`figures/roofline.png`) is a *performance* analysis (Prefill compute-bound vs Decode memory-bound), owned by T-3.7. This PRD only produces the *economic* chart.
* **Not the quality matrix.** Side-by-side text-output comparison is owned by T-3.6 + FR-11 + `results/quality_matrix.md`.
* **Not QLoRA cost.** Training economics is a distinct discussion in the M5 QLoRA milestone; break-even chart is inference-only per K7.
* **Not power measurement.** `energy_wh` in the manifests is `watts_active × wall_s / 3600` (configured wattage, not measured). The chart uses this same estimate for consistency with FR-9.

---

## 4. Building block: `services/economic_analyzer.py` + `services/break_even_chart.py`

### 4.1 Input
* `results/sweep_<ts>.csv` — 6 rows (2 targets × 3 quantizations × 1 airllm backend). Per row: `wall_s_mean`, `energy_wh_mean`, `prompt_tokens`, `completion_tokens`, `target_label`, `quantization`, `backend`.
* `config/setup.json.economic` — `hardware_capex_usd`, `lifetime_hours`, `cloud_gpu.hourly_usd`, `cloud_gpu.include_in_chart`, `cloud_gpu.speedup_over_cpu` (new, default 1000).
* `config/setup.json.energy` — `assumed_watts_active`, `electricity_price_per_kwh_usd`.
* `config/api_pricing.json.providers.anthropic` — `in_per_million_usd`, `out_per_million_usd`, `model`, `captured_at`.

### 4.2 Output
* `results/economic_analysis_<ts>.csv` — per-cell per-curve cost points: `target_label`, `quantization`, `backend`, `option` (on_prem|anthropic|cloud_gpu), `wall_s`, `cost_per_req_usd`, `break_even_volume_vs_anthropic` (int|null), `break_even_volume_vs_cloud_gpu` (int|null), `assumptions_snapshot` (nested dict — config values used at compute time for reproducibility).
* `figures/break_even.png` — 3-curve chart with crossovers marked.
* `results/economic_manifest_<ts>.json` — machine-readable summary: which sweep CSV was consumed, which config values were used, per-cell headline numbers.

### 4.3 Setup
* `numpy` (already a transitive dependency of torch/transformers) for volume-axis vectorisation.
* `matplotlib` (already installed for T-3.7) for chart rendering.
* No new heavy dependency required.

---

## 5. The T-4.1/T-4.2 API surface

### 5.1 Data types

```python
@dataclass(frozen=True, kw_only=True)
class CostPoint:
    target_label: str
    quantization: str
    backend: str
    option: str                # "on_prem" | "anthropic" | "cloud_gpu"
    wall_s: float
    cost_per_req_usd: float
    break_even_volume_vs_anthropic: int | None
    break_even_volume_vs_cloud_gpu: int | None

@dataclass(frozen=True, kw_only=True)
class AssumptionsSnapshot:
    capex_usd: float
    lifetime_hours: float
    watts_active: float
    electricity_price_per_kwh_usd: float
    cloud_gpu_hourly_usd: float
    cloud_gpu_speedup_over_cpu: float
    anthropic_in_per_million_usd: float
    anthropic_out_per_million_usd: float
    anthropic_model: str
    anthropic_prices_captured_at: str | None
```

### 5.2 Pure functions

```python
def cost_per_req_on_prem(wall_s: float, snapshot: AssumptionsSnapshot) -> float:
    amort_per_hour = snapshot.capex_usd / snapshot.lifetime_hours
    hours = wall_s / 3600
    hardware = amort_per_hour * hours
    electricity = snapshot.watts_active * hours * (
        snapshot.electricity_price_per_kwh_usd / 1000
    )
    return hardware + electricity

def cost_per_req_anthropic(prompt_tokens: int, completion_tokens: int,
                            snapshot: AssumptionsSnapshot) -> float:
    return (prompt_tokens * snapshot.anthropic_in_per_million_usd
            + completion_tokens * snapshot.anthropic_out_per_million_usd) / 1_000_000

def cost_per_req_cloud_gpu(wall_s_cpu: float, snapshot: AssumptionsSnapshot) -> float:
    wall_s_gpu = wall_s_cpu / snapshot.cloud_gpu_speedup_over_cpu
    return snapshot.cloud_gpu_hourly_usd * wall_s_gpu / 3600

def break_even_volume(capex_usd: float, opex_per_req: float,
                      competitor_cost_per_req: float) -> int | None:
    delta = competitor_cost_per_req - opex_per_req
    if delta <= 0:
        return None  # competitor is cheaper per request — never breaks even
    return math.ceil(capex_usd / delta)
```

### 5.3 CSV columns for `economic_analysis_<ts>.csv`
```
target_label,quantization,backend,option,wall_s,cost_per_req_usd,
break_even_volume_vs_anthropic,break_even_volume_vs_cloud_gpu
```

### 5.4 Manifest columns for `economic_manifest_<ts>.json`
```json
{
  "captured_at": "...",
  "sweep_csv_path": "results/sweep_<ts>.csv",
  "api_pricing_path": "config/api_pricing.json",
  "assumptions": { ... },
  "rows": [ ...one per cell × 3 options... ],
  "chart_path": "figures/break_even.png"
}
```

---

## 6. Functional Requirements

* **FR-EA-1.** `economic_analyzer` MUST read the latest `results/sweep_<ts>.csv` (or take an explicit path argument) and produce one `CostPoint` per `(cell, option)` combination — 6 cells × 3 options = **18 rows** in `economic_analysis_<ts>.csv`.
* **FR-EA-2.** The Cloud GPU curve MUST be included if `config.economic.cloud_gpu.include_in_chart == true` (default true per D-4). Setting to `false` reduces the chart to 2 curves + a note.
* **FR-EA-3.** `AssumptionsSnapshot` MUST be persisted in the manifest — every config value used at compute time is recorded so the report can quote them + so a future re-run reveals if assumptions drifted.
* **FR-EA-4.** Break-even volumes MUST be computed for each cell × each competitor pair. Volumes where the denominator is ≤ 0 MUST record `null` and emit a `WARN` log line.
* **FR-EA-5.** `figures/break_even.png` MUST plot: (a) 3 curves in distinct colors + line styles, (b) x-axis auto-log-scaled if crossovers span >2 orders of magnitude, (c) crossover annotations at each break-even point, (d) an inline assumptions table (or below-chart caption listing every value).
* **FR-EA-6.** The chart MUST use ONE representative cell for its curves (the "hero" cell — `target=llama3-8b-fp16`, `quantization=q4`, `backend=airllm` per T-3.6b). Cell selection is a config-controllable field: `config.economic.chart_representative_cell`. Other cells feed the CSV; the chart shows one line per option for the hero cell.
* **FR-EA-7.** `figures/break_even_sensitivity.png` (OPTIONAL) MAY be produced showing the chart with ±20% electricity price, ±20% Anthropic pricing, ±50% cloud GPU speedup — as light grey uncertainty bands. Not required by K7 but useful for the report methodology.
* **FR-EA-8.** SDK method `economic_analysis(sweep_csv_path: Path | None = None)` MUST replace the current `_future_stubs` stub — env guard first, then delegate to the analyzer.
* **FR-EA-9.** CLI `uv run on-prem-llm economic-analysis` MUST invoke the SDK method + print `OK: economic_analysis_<ts>.csv + figures/break_even.png` on success.
* **FR-EA-10.** If `config/api_pricing.json.providers.anthropic.captured_at` is `null` OR older than 90 days from today, `economic_analyzer` MUST emit a WARN log line + set `assumptions.anthropic_prices_stale = true` in the manifest. Does not abort — but flags the risk.

## 7. Non-Functional Requirements

* **NFR-EA-1.** Zero hard-coded numeric constants — every price, wattage, hour, dollar comes from `config/setup.json` or `config/api_pricing.json`.
* **NFR-EA-2.** Pure math functions (cost_per_req_*, break_even_volume) MUST be unit-testable without `matplotlib`, `numpy`, or file I/O.
* **NFR-EA-3.** Chart rendering MUST be side-effect-free at import time (matplotlib backend set to 'Agg' inside the plotting function, not at module top).
* **NFR-EA-4.** File-size cap: every new source file ≤ 150 LOC per constitution §2.2. `economic_analyzer.py` will need a split between pure math (`economic_math.py`) and I/O (`economic_analyzer.py`) — mirrors T-3.5's `sweep_stats.py` + `sweep_runner.py` split.
* **NFR-EA-5.** No new heavy dependency — matplotlib + numpy already installed.

---

## 8. Success Criteria (SC-EA)

* **SC-EA-1.** `results/economic_analysis_<ts>.csv` exists after `uv run on-prem-llm economic-analysis`; has 18 rows (6 cells × 3 options) or 12 rows (6 cells × 2 options if Cloud GPU disabled).
* **SC-EA-2.** `figures/break_even.png` exists + is a valid PNG + non-empty (≥ 10 KB).
* **SC-EA-3.** `AssumptionsSnapshot` in the manifest matches config values byte-identically — reproducibility check.
* **SC-EA-4.** For the M3 sweep hero cell (Llama-3-8B × q4 × airllm), given empirical `wall_s ≈ 3300 s` + `capex=$2500` + `lifetime_hours=5000`:
  * `cost_per_req_on_prem ≈ $2500/5000 × (3300/3600) + 180W × (3300/3600) × ($0.16/kWh / 1000) ≈ $0.46 + $0.026 ≈ $0.49 per request`
  * `cost_per_req_anthropic ≈ (5 tokens × $1/M + 8 tokens × $5/M) / 1M ≈ $45 per million ≈ $0.000045 per request` (essentially free per call at these token counts)
  * `cost_per_req_cloud_gpu ≈ $0.6 × (3300/1000)/3600 ≈ $0.000550 per request` (assuming 1000× speedup)
  * **Predicted break-even: On-Prem NEVER breaks even vs Anthropic at this workload size.** That is a real, honest, publishable finding — small workloads favor API, large workloads favor GPUs, on-prem CPU is neither.
* **SC-EA-5.** Chart labels are legible at 300 DPI PDF export (the format the README will use).
* **SC-EA-6.** Ruff clean, ≤ 150 LOC per file, ≥ 85% coverage on new modules, tests exercise every pure math function + at least one end-to-end golden-file test.

---

## 9. Test Scenarios

* **U-EA-1** — `cost_per_req_on_prem` — 3300 s wall + $2500 capex + 5000 hrs lifetime + 180 W active + $0.16/kWh returns ~$0.4859.
* **U-EA-2** — `cost_per_req_anthropic` — 5 in + 8 out at $1/M + $5/M returns $4.5e-5.
* **U-EA-3** — `cost_per_req_cloud_gpu` — 3300 s CPU wall + 1000× speedup + $0.6/hr returns ~$5.5e-4.
* **U-EA-4** — `break_even_volume` — capex $2500 + on-prem opex $0.026 + Anthropic cost $4.5e-5 returns `None` (Anthropic cheaper).
* **U-EA-5** — `break_even_volume` — capex $2500 + hypothetical on-prem opex $0.01 + competitor $0.50 returns `⌈2500/0.49⌉ = 5103`.
* **U-EA-6** — CSV column order matches FR-EA-3 spec.
* **U-EA-7** — `include_in_chart: false` reduces CSV rows to 12 (6 × 2 options), manifest records `cloud_gpu_included: false`.
* **U-EA-8** — `chart_representative_cell` targets the CORRECT cell by matching all three of `target_label`, `quantization`, `backend`.
* **U-EA-9** — FR-EA-10 stale-price warn fires when `captured_at` is `null` OR > 90 days old.
* **U-EA-10** — Manifest `AssumptionsSnapshot` matches config JSON byte-identically for a controlled fixture.
* **I-EA-1** — golden-file end-to-end test: fixture sweep CSV + fixture config → known-good `economic_analysis_<ts>.csv` + `figures/break_even.png` byte-identical to the golden output.

---

## 10. Alternatives Considered

| # | Alternative | Rejected because |
|---|---|---|
| A-1 | Compute break-even by binary search over volume | Closed-form formula (§2.3) is faster + deterministic + testable |
| A-2 | Include an "AirLLM" curve as a fourth line | K7 says 3 curves. AirLLM is a strategy inside the On-Prem curve, not a separate deployment option |
| A-3 | Use REAL cloud GPU wall time by spinning up Colab | Optional (FR-EA-7 sensitivity chart); not required by K7. Would extend the assignment beyond CPU-only scope |
| A-4 | Include Ollama / Direct-backend cells in the economic analysis | Already covered by On-Prem curve (same amortisation formula); adding Direct would duplicate rows without new information — Direct's `wall_s` is captured but the report caveats it as "the failure mode" |
| A-5 | Compute cost using ACTUAL power draw (via powermetrics/nvidia-smi) | Configured wattage is FR-9's chosen abstraction — real power sampling is a future NFR-10 enhancement |
| A-6 | Amortise Anthropic per month instead of per request | Adds complexity + no assignment requirement; per-request is the L08 §1.1 convention |
| A-7 | Use Anthropic Opus / Sonnet pricing instead of Haiku | Haiku is the cheapest published tier — represents the API's best case for the break-even. Report can add Opus/Sonnet as sensitivity variants |

---

## 11. Open Questions — RESOLVED 2026-07-01

* **Q-EA-1 (Cloud GPU speedup factor). RESOLVED = 1000×.** Config key: `config.economic.cloud_gpu.speedup_over_cpu` (new field, default 1000). Represents mid-tier discrete GPU (e.g. A10G / RTX 3090) inference speedup vs the 4-core CPU-only reference laptop. Adjustable per PRD_airllm_integration.md's GPU discussion; 100× (T4-conservative) or 5000× (A100-aggressive) are the practical range.
* **Q-EA-2 (Chart representative cell). RESOLVED = hero cell** (`target_label="llama3-8b-fp16"`, `quantization="q4"`, `backend="airllm"`). Config key: `config.economic.chart_representative_cell` (new field). Same cell the M=32 hero uses — makes the chart tell the "our best on-prem measurement vs the market" story. Other 5 cells still feed the CSV; the chart focuses on one line per option.
* **Q-EA-3 (Sensitivity chart). RESOLVED = optional.** FR-EA-7 stays optional; SC-EA-6 tests only the main `figures/break_even.png`. If time permits after M4, the `figures/break_even_sensitivity.png` with ±20% electricity + ±20% Anthropic + ±50% cloud GPU speedup is a nice-to-have.

**Approver:** user (ndvp39@gmail.com), inline chat approval, 2026-07-01.

---

## 12. Landing plan (informative, not gated by this PRD)

* **T-4.0** — This PRD authored + approved (2026-07-01).
* **T-4.1** — `services/economic_math.py` (pure math) + `services/economic_analyzer.py` (I/O + CSV/manifest write) + unit tests.
* **T-4.2** — `services/break_even_chart.py` (matplotlib chart) + golden-file test.
* **T-4.3** — SDK method `economic_analysis()` (replaces `_future_stubs` stub) + CLI `economic-analysis` command + integration test.
* **T-4.4** — Real Anthropic price verification: user checks `anthropic.com/pricing`, updates `config/api_pricing.json` if drift observed, sets `captured_at` to today's date.

Every step maps to constitution §1.3's SDLC phases: **PRD → SDK method + tests → CLI wrap → integration + docs**.

---

## 13. Ready to review?

Everything numeric in this PRD's success criteria (§8) uses the concrete config values already in `config/setup.json`, `config/api_pricing.json`, and the sweep manifest schema. The 3 open questions (§11) each have a default proposed. Approval unblocks T-4.1 code.

**Approver:** user (ndvp39@gmail.com) — signs off inline in a chat reply.
