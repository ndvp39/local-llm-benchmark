# PRD — Roofline Analysis (Prefill Compute-bound vs Decode Memory-bound Attribution)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-6 — **MANDATORY**.
> **Blocks:** **T-3.7c** Roofline chart helper (`services/roofline_analyzer.py` + `services/roofline_chart.py`) — the Roofline math, axes, and attribution rule are not code-guessable. Approval required before T-3.7c implementation begins.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf` §3 (Prefill vs Decode headline analysis) → `L08-summary-Lora-AirLLM.pdf` §3 (Roofline model = compute-vs-memory-bandwidth ceiling; ridge point separates compute-bound from memory-bound regimes) + §8 (AirLLM's layer streaming as OS-paging analog — disk becomes the effective memory-bandwidth ceiling) → `docs/PRD.md` v1.10 **G3** ("Roofline-style analysis attributing each measured slowdown to either compute-bound (Prefill) or memory-bound (Decode), backed by data, not opinion") + FR-14b (implicit — G3 mandates a chart-form attribution).
> **Empirical anchor:** M2a plumbing `results/plumbing_20260701T165601Z.json` (Llama-3-8B fp16 AirLLM: TTFT=370 s, TPOT=371 s, peak RAM=1287 MB, 2 tokens) — the near-equality of TTFT and TPOT is THE Roofline observation: this workload sits FAR below the ridge point, dominated by disk-streaming memory bandwidth, not compute. T-3.6b's sweep manifests will provide 6 more data points across quantization × target for the same chart.
> **Document version:** 1.00 — 2026-07-01.
> **Status:** **APPROVED 2026-07-01** — user (ndvp39@gmail.com) confirmed all 4 open-question defaults: Q-RL-1 = 50 GFLOPs peak compute, Q-RL-2 = 30 GB/s DRAM, Q-RL-3 = 100 MB/s disk, Q-RL-4 = combined chart layout. T-3.7c code cleared to start once the T-3.6b sweep CSV lands.

---

## 1. What and why

### 1.1 What
Formalise the **Roofline attribution policy** — how the SDK converts the M3 sweep's `(TTFT, TPOT, wall_s, peak_ram_mb, prompt_tokens, completion_tokens)` per cell into per-cell **arithmetic intensity + attained performance points on a Roofline chart**, with a hard rule for calling each cell "compute-bound" or "memory-bound". Covers:

* **Roofline axes** — arithmetic intensity (FLOPs/byte) on X, attained performance (FLOPs/sec) on Y, log-log scale.
* **Ceilings** — peak compute (horizontal) and peak memory bandwidth (sloped) lines, ridge point where they intersect.
* **Effective memory bandwidth choice** — DRAM bandwidth for Direct-backend rows, DISK bandwidth for AirLLM rows (L08 §8 layer-streaming = the memory-bandwidth analog is disk read).
* **Attribution rule** — a cell is `memory_bound` if its point sits to the LEFT of the ridge point (arithmetic intensity < ridge intensity), `compute_bound` if to the RIGHT.
* **Chart specification** — one facet per target (Llama-3-8B, Qwen2-7B), or a single combined chart with distinct markers per target.
* **Report narrative** — the sweep's empirical points will (per the plumbing evidence) all sit far below the memory-bandwidth ceiling AND to the left of the ridge point — proving the L08 §8 headline that AirLLM CPU-only inference is bounded by disk-streaming bandwidth, not compute.

This PRD does introduce new code (`services/roofline_analyzer.py` + `services/roofline_chart.py`) — neither exists yet.

### 1.2 Why
* **Assignment G3.** Roofline attribution is a headline deliverable of the report — the assignment explicitly asks for compute-bound vs memory-bound labels, backed by data. Without the chart, G3 fails.
* **L08 §3 curriculum linkage.** The lecture's Roofline diagram is THE central abstraction it introduces for reasoning about inference performance. The report's §Theory section directly references L08 §3; the empirical chart lets the report say "this is that diagram, drawn from OUR measurements."
* **L08 §8 curriculum linkage.** The lecture's OS-paging analogy for AirLLM ("layer streaming from disk is a virtual-memory page fault") is only intellectually satisfying if a Roofline chart PROVES the ridge point shifts drastically leftward when the effective bandwidth is disk (100 MB/s USB HDD) instead of DRAM (~40 GB/s).
* **Report narrative closure.** Without Roofline, the sweep results table + break-even chart tell WHAT the numbers are but not WHY. Roofline is the "why" — every cell's slowness is attributed to a physical ceiling, not to hand-waving.
* **Reproducibility.** Every ceiling value (peak GFLOPS, DRAM GB/s, disk MB/s) comes from a config field — re-running on a different laptop just needs the peaks updated.

### 1.3 Who consumes this policy
| Consumer | Where the policy applies | Landing task |
|---|---|---|
| `services/roofline_analyzer.py` | Reads sweep CSV + `config.roofline` ceilings; emits per-cell `RooflinePoint` | T-3.7c |
| `services/roofline_chart.py` | Consumes RooflinePoints; plots ceilings + points + ridge annotations to `figures/roofline.png` | T-3.7c |
| SDK method `roofline_analysis(sweep_csv_path)` | Wraps analyzer + chart | T-3.7c wire |
| CLI `make-report` (T-6.1) | Invokes the SDK method; produces figure inline | T-6.1 |
| Report assembler (T-6.1) | Embeds `figures/roofline.png` + attribution table into README §Roofline | T-6.1 |

---

## 2. Theoretical background & definitions

### 2.1 The Roofline model — L08 §3

Given a processor with:
* **π** = peak compute (FLOPs/sec) — hardware's maximum arithmetic throughput.
* **β** = peak memory bandwidth (bytes/sec) — hardware's maximum rate of moving bytes between memory tiers.

A kernel with **arithmetic intensity `I`** (FLOPs per byte transferred) achieves at most:
```
attained_flops(I) = min(π, β × I)
```
The plot: X-axis is `I` (log), Y-axis is `attained_flops(I)` (log). Two lines: horizontal at `π`, sloped at `β × I`. **Ridge point** where they meet: `I_ridge = π / β`.

* `I < I_ridge` → **memory-bound**. Doubling compute buys nothing; only faster memory helps.
* `I > I_ridge` → **compute-bound**. Doubling memory bandwidth buys nothing; only more/faster FLOPs help.

### 2.2 Applying Roofline to LLM inference — L08 §3

* **FLOPs per token** ≈ `2 × N_params` (each token forward pass = one matmul per layer, and 2 FLOPs per fused multiply-add, and N_params of them). This is the L08 §3 canonical estimate; ignores softmax and layer-norm which are ~1% of total.
* **Bytes per token (weights)** = `N_params × bit_width / 8`. During Decode, ALL layer weights are read from memory per token (Prefill can amortize with batching, but our two-generate pattern doesn't).
* **Arithmetic intensity per token** = `flops / bytes = (2 × N_params) / (N_params × bit_width / 8) = 16 / bit_width`.
  * fp16: I = 1.0 FLOPs/byte
  * q8: I = 2.0 FLOPs/byte
  * q4: I = 4.0 FLOPs/byte
* **Insight:** quantization INCREASES arithmetic intensity → moves the operating point to the RIGHT on the Roofline chart. If the ridge point is well to the RIGHT of your operating point (typical for on-prem CPU + AirLLM disk-streaming), quantization stays memory-bound but moves you visibly closer to the ridge, matching the wall-time speedup observation.

### 2.3 Effective memory bandwidth — the L08 §8 twist

For Direct backend on CPU:
* β_effective = DRAM bandwidth (~20-40 GB/s on DDR4-3200 dual-channel).

For **AirLLM backend on CPU** (L08 §8 OS-paging analog):
* Layer weights DO NOT fit in RAM (7.8 GB total, 15 GB fp16 weights). AirLLM streams shards from D: drive per token.
* β_effective = **DISK bandwidth**, not DRAM bandwidth.
* Your D: drive was empirically confirmed as USB HDD (`WD Elements 1078`), ~100-150 MB/s sequential.
* β_effective_airllm = ~100 MB/s = 1e8 bytes/sec.

**The ridge point shifts dramatically:**
* CPU peak compute (single-threaded LINPACK-ish estimate): ~50 GFLOPs (i.e., 5e10 FLOPs/sec).
* Direct backend on DDR4: `I_ridge = 5e10 / 3e10 = 1.67 FLOPs/byte` — close to fp16's 1.0.
* AirLLM backend on USB HDD: `I_ridge = 5e10 / 1e8 = 500 FLOPs/byte` — **far to the right of every quantization level we can pick**.

**Consequence:** AirLLM sweep points ALL sit far to the LEFT of the AirLLM ridge point → 100% memory-bound → wall time attribution goes 100% to disk streaming, matching L08 §8's headline.

### 2.4 Attained-performance computation from the sweep

For each cell row:
```
tokens_generated = completion_tokens
wall_seconds     = wall_s_mean
tpot_seconds     = tpot_ms_mean / 1000
n_params_billion = 8 for Llama-3-8B, 7 for Qwen2-7B  (from config.roofline.model_params_billion)
flops_per_token  = 2 × n_params × 1e9
attained_flops   = flops_per_token / tpot_seconds
arith_intensity  = 16 / bit_width_for_quantization
```

Then `attained_flops` vs `arith_intensity` is the (Y, X) coordinate of the cell on the chart.

Attribution:
```
regime = "compute_bound" if arith_intensity > ridge_intensity else "memory_bound"
```

---

## 3. What this PRD is NOT

* **Not the sweep runner.** DP-4 owns TTFT/TPOT/RAM measurement.
* **Not the economic chart.** DP-5 owns cost-per-request curves.
* **Not real hardware-peak benchmarking.** Peak GFLOPs + DRAM GB/s come from `config.roofline.peaks` (with defaults from the CPU/RAM datasheet, calibratable). Running an actual LINPACK / STREAM benchmark on the reference machine is a NICE-TO-HAVE future extension, not required for G3.
* **Not multi-batch analysis.** Batch=1 assumption baked in per our two-generate pattern; the report's Roofline discussion notes that batching would move points rightward (higher arithmetic intensity), which is why real serving systems batch aggressively.
* **Not FLOP-counting per-layer profiling.** The `2 × N_params` estimate is the L08 §3 canonical simplification.

---

## 4. Building block: `services/roofline_analyzer.py` + `services/roofline_chart.py`

### 4.1 Input
* `results/sweep_<ts>.csv` — 6 rows (2 targets × 3 quantizations × 1 airllm backend).
* `config/setup.json.roofline` — new config subtree (see §5.3).
* `config/setup.json.target_models[]` — for `n_params_billion` per label.

### 4.2 Output
* `results/roofline_analysis_<ts>.csv` — one row per cell with `target_label`, `quantization`, `backend`, `arith_intensity_flops_per_byte`, `attained_gflops`, `regime` (`compute_bound` | `memory_bound`), `ridge_intensity_flops_per_byte`, `ceiling_dram_gflops_at_this_I`, `ceiling_disk_gflops_at_this_I`, `wall_share_disk_pct` (estimate of what fraction of wall time went to disk streaming).
* `results/roofline_manifest_<ts>.json` — machine-readable summary + assumptions snapshot.
* `figures/roofline.png` — main Roofline chart with ceilings + data points + ridge annotations.
* `figures/roofline_faceted.png` (optional per FR-RL-8) — 2-facet version (one panel per target).

### 4.3 Setup
* `matplotlib` (already installed for T-3.7).
* `numpy` (already installed).
* No new heavy dependency.

---

## 5. The T-3.7c API surface

### 5.1 Data types

```python
@dataclass(frozen=True, kw_only=True)
class RooflineCeilings:
    peak_compute_gflops: float          # π (config)
    peak_dram_bandwidth_gbps: float     # β for Direct
    peak_disk_bandwidth_mbps: float     # β for AirLLM
    ridge_intensity_dram: float         # π / β_dram
    ridge_intensity_disk: float         # π / β_disk

@dataclass(frozen=True, kw_only=True)
class RooflinePoint:
    target_label: str
    quantization: str
    backend: str
    n_params_billion: float
    bit_width: int
    tpot_seconds: float
    arith_intensity: float              # 16 / bit_width
    attained_gflops: float              # (2 × n_params × 1e9 / tpot_seconds) / 1e9
    ceiling_gflops: float               # min(π, β × I) using backend-appropriate β
    regime: str                         # "compute_bound" | "memory_bound"
    wall_share_disk_pct: float | None   # AirLLM only; None for Direct
```

### 5.2 Pure functions

```python
def arith_intensity_for_bit_width(bit_width: int) -> float:
    return 16.0 / bit_width

def ridge_intensity(peak_gflops: float, peak_bandwidth_bytes_per_s: float) -> float:
    return (peak_gflops * 1e9) / peak_bandwidth_bytes_per_s

def attained_gflops_from_tpot(n_params_billion: float, tpot_seconds: float) -> float:
    return (2 * n_params_billion * 1e9) / tpot_seconds / 1e9

def regime_for(arith_intensity: float, ridge_intensity: float) -> str:
    return "compute_bound" if arith_intensity > ridge_intensity else "memory_bound"

def wall_share_disk_pct(n_params_billion: float, bit_width: int,
                        peak_disk_bytes_per_s: float, wall_s: float) -> float:
    """Estimate of wall_time spent streaming from disk for one AirLLM token."""
    bytes_per_token = n_params_billion * 1e9 * bit_width / 8
    disk_seconds = bytes_per_token / peak_disk_bytes_per_s
    return 100.0 * disk_seconds / wall_s
```

### 5.3 Config schema — `config/setup.json.roofline` (new)

```json
"roofline": {
    "peak_compute_gflops": 50.0,
    "peak_dram_bandwidth_gbps": 30.0,
    "peak_disk_bandwidth_mbps": 100.0,
    "ceilings_source": "cpu-and-drive-datasheet + M2a plumbing evidence",
    "ceilings_captured_at": "2026-07-01",
    "chart_layout": "combined"
}
```

Plus in each `target_models[]` entry, add `n_params_billion` (informative):
```json
{"id": "meta-llama/Meta-Llama-3-8B-Instruct", "label": "llama3-8b-fp16", "n_params_billion": 8.03, ...}
{"id": "Qwen/Qwen2-7B-Instruct", "label": "qwen2-7b-q4", "n_params_billion": 7.62, ...}
```

### 5.4 CSV columns for `roofline_analysis_<ts>.csv`

```
target_label,quantization,backend,n_params_billion,bit_width,tpot_seconds,
arith_intensity_flops_per_byte,attained_gflops,ceiling_gflops,regime,
ridge_intensity_flops_per_byte,wall_share_disk_pct
```

---

## 6. Functional Requirements (FR-RL)

* **FR-RL-1.** `roofline_analyzer` MUST read the latest `results/sweep_<ts>.csv` (or an explicit path arg) and produce one `RooflinePoint` per row — 6 points total for the T-3.6b sweep.
* **FR-RL-2.** Arithmetic intensity MUST be computed as `16 / bit_width` where `bit_width` is 4 (q4), 8 (q8), 16 (fp16), 32 (fp32) — no other bit-widths supported.
* **FR-RL-3.** Attained performance MUST be computed from `tpot_ms_mean` (DECODE phase, not TTFT). TTFT is compute-bound Prefill and belongs in a separate discussion; the Roofline chart plots the Decode operating point where the memory-bound story lives.
* **FR-RL-4.** For AirLLM cells, `β_effective` MUST equal `peak_disk_bandwidth_mbps × 1e6` bytes/sec (disk bandwidth). For Direct-backend cells, `β_effective` MUST equal `peak_dram_bandwidth_gbps × 1e9` bytes/sec (DRAM bandwidth). Ridge point per backend.
* **FR-RL-5.** Regime attribution MUST be a hard binary: `arith_intensity > ridge_intensity` → `compute_bound`, else `memory_bound`. No fuzzy or intermediate labels.
* **FR-RL-6.** `wall_share_disk_pct` MUST be reported per cell — it's the report's headline number for L08 §8 ("X% of wall time went to disk streaming, not compute").
* **FR-RL-7.** `figures/roofline.png` MUST plot: (a) horizontal peak-compute ceiling, (b) two sloped memory-bandwidth ceilings (DRAM + disk), (c) two ridge points labelled, (d) every data point as a marker with `(target_label, quantization)` annotation, (e) legend + axes labels + title.
* **FR-RL-8.** OPTIONAL `figures/roofline_faceted.png` MAY be produced when `config.roofline.chart_layout == "faceted"` — one panel per target label, each panel identical ceilings but only that target's data points. Not required by G3.
* **FR-RL-9.** SDK method `roofline_analysis(sweep_csv_path: Path | None = None)` — new SDK entry point (not previously in `_future_stubs.py`; adding at T-3.7c).
* **FR-RL-10.** The manifest MUST persist `RooflineCeilings` snapshot AT compute time — future re-runs with changed config values are traceable to the analysis they produced.

## 7. Non-Functional Requirements (NFR-RL)

* **NFR-RL-1.** Zero hard-coded numeric constants — every ceiling (peak_compute_gflops, peak_dram_bandwidth_gbps, peak_disk_bandwidth_mbps) comes from `config.roofline`.
* **NFR-RL-2.** Pure math functions (arith_intensity, ridge_intensity, attained_gflops_from_tpot, regime_for, wall_share_disk_pct) MUST be unit-testable without matplotlib or file I/O.
* **NFR-RL-3.** Chart rendering MUST be side-effect-free at import time (matplotlib backend set to 'Agg' inside the plotting function).
* **NFR-RL-4.** File-size cap: `roofline_analyzer.py` + `roofline_chart.py` each ≤ 150 LOC per constitution §2.2. Mirrors the T-3.5 sweep_runner + sweep_stats split.
* **NFR-RL-5.** Log-scale on both axes; axis labels + tick formats readable at 300 DPI PDF export.

---

## 8. Success Criteria (SC-RL)

* **SC-RL-1.** `results/roofline_analysis_<ts>.csv` exists after `uv run on-prem-llm roofline-analysis`; has 6 rows matching the sweep cells.
* **SC-RL-2.** `figures/roofline.png` exists + is a valid PNG + ≥ 20 KB.
* **SC-RL-3.** `RooflineCeilings` snapshot in the manifest matches `config.roofline` byte-identically.
* **SC-RL-4.** For every AirLLM cell in the sweep, `regime == "memory_bound"` (the L08 §8 headline prediction — all our cells sit far left of the disk ridge point).
* **SC-RL-5.** For every AirLLM cell, `wall_share_disk_pct` is in the range **80-99%** (proves the OS-paging analog claim empirically). E.g., Llama-3-8B fp16 at 100 MB/s disk: `bytes_per_token = 8e9 × 16 / 8 = 16 GB` per token → `disk_seconds = 16e9 / 1e8 = 160 s per token → wall_share ≈ 160 / 371 ≈ 43%`. **Wait** — that's below 80%; the ceiling assumption of 100 MB/s is optimistic for random reads across 35 files. Effective bandwidth is likely 40-60 MB/s. **The SC target of 80-99% is a hypothesis the analysis TESTS against reality — publishable either way.** The report frames this as "predicted vs measured disk share of wall time".
* **SC-RL-6.** Attribution table in the report reads:

| target × quant × backend | regime | I (FLOPs/byte) | GFLOPs attained | wall_share_disk |
|---|---|---|---|---|
| Llama-3-8B × fp16 × airllm | memory_bound | 1.0 | 0.043 | 43% (measured) |
| Llama-3-8B × q4 × airllm | memory_bound | 4.0 | ~0.086 (predicted) | ~10-20% |
| ... etc |

* **SC-RL-7.** Ruff clean, ≤ 150 LOC per file, ≥ 85% coverage on the pure math + one golden-file test on the chart output.

---

## 9. Test Scenarios

* **U-RL-1** — `arith_intensity_for_bit_width(4)` → 4.0. `(8)` → 2.0. `(16)` → 1.0. `(32)` → 0.5.
* **U-RL-2** — `arith_intensity_for_bit_width(5)` raises `ValueError` (unsupported).
* **U-RL-3** — `ridge_intensity(peak_gflops=50, peak_bandwidth_bytes_per_s=1e8)` → 500.
* **U-RL-4** — `ridge_intensity(peak_gflops=50, peak_bandwidth_bytes_per_s=3e10)` → 1.67.
* **U-RL-5** — `attained_gflops_from_tpot(n_params_billion=8, tpot_seconds=371)` → ~0.0431.
* **U-RL-6** — `regime_for(arith_intensity=1.0, ridge_intensity=500)` → `"memory_bound"`.
* **U-RL-7** — `regime_for(arith_intensity=1000, ridge_intensity=500)` → `"compute_bound"`.
* **U-RL-8** — `wall_share_disk_pct(n_params_billion=8, bit_width=16, peak_disk_bytes_per_s=1e8, wall_s=371)` → ~43%.
* **U-RL-9** — CSV columns exactly match §5.4 canonical order.
* **U-RL-10** — Ceilings snapshot in manifest matches config JSON byte-identically for a fixture.
* **I-RL-1** — Golden-file end-to-end: fixture sweep CSV + fixture config → known-good `roofline_analysis_<ts>.csv` + `figures/roofline.png` byte-identical to the golden output.

---

## 10. Alternatives Considered

| # | Alternative | Rejected because |
|---|---|---|
| A-RL-1 | Compute FLOPs per layer explicitly via model introspection | Overkill; L08 §3's `2 × N_params` estimate is the canonical simplification and matches the pedagogy |
| A-RL-2 | Use TTFT instead of TPOT for the operating point | TTFT is Prefill (batch-parallel, compute-bound); TPOT is Decode (memory-bound). The Roofline story lives in Decode. TTFT can appear as a SECOND point per cell (future extension). |
| A-RL-3 | Include KV-cache bytes in the byte-count for arithmetic intensity | Simplification per L08 §3: KV cache is small vs weights for our token counts; ignored |
| A-RL-4 | Include activations in the FLOP count | `2 × N_params` is the dominant term; softmax + layer-norm + activation FLOPs are ~1% |
| A-RL-5 | Run actual LINPACK / STREAM to measure peak_compute + peak_bandwidth | Nice-to-have; datasheet-derived defaults are fine for G3. Optional extension. |
| A-RL-6 | Plot a separate Roofline for Prefill (compute-bound side) | Report can describe verbally — chart focuses on the Decode story where the L08 §8 OS-paging analog dominates |
| A-RL-7 | Add batch-size axis to show batching's rightward shift | Batch=1 baked in per §5.2; batching is a discussion point in the report, not a chart dimension |

---

## 11. Open Questions (blocking approval)

* **Q-RL-1 (Peak compute value).**
  * Option A: **datasheet-derived** — Intel Core i7-1165G7 (or similar Tiger Lake / 11th gen) single-precision peak ≈ 200 GFLOPs across 4 cores, or ~50 GFLOPs per core in FP16-AVX2. Default: **`peak_compute_gflops = 50.0`** (single-thread conservative; layer-stream is single-threaded per token). Adjustable in config.
  * Option B: **empirically measured** via LINPACK before the chart runs (adds a one-time ~1 min benchmark step).
  * Author's suggested default: **Option A (datasheet, 50 GFLOPs).** LINPACK is a nice-to-have extension.
* **Q-RL-2 (Peak DRAM bandwidth).**
  * Option A: **datasheet** — DDR4-3200 dual-channel ≈ 51.2 GB/s peak; realistic effective ~30 GB/s. Default: **30.0 GB/s.**
  * Option B: **STREAM-benchmarked.**
  * Author's suggested default: **Option A (30 GB/s).** Direct-backend Roofline is not the primary story anyway (all our sweep is AirLLM).
* **Q-RL-3 (Peak disk bandwidth — the AirLLM ridge point driver).**
  * Option A: **datasheet USB HDD** — WD Elements 1078 USB 3.0 HDD, ~100-150 MB/s sequential, ~40-60 MB/s random. Default: **`peak_disk_bandwidth_mbps = 100.0`** (optimistic sequential — report notes actual effective is likely lower).
  * Option B: **measure** via a one-time `dd`-style benchmark of a 15 GB file read from D:.
  * Author's suggested default: **Option A (100 MB/s optimistic).** Report explicitly says "SC-RL-5's predicted 43% disk share may under-shoot reality if effective bandwidth is closer to 50 MB/s" — the analysis is publishable either way, and the honest gap between predicted and measured IS a report finding.
* **Q-RL-4 (Chart layout — combined vs faceted).**
  * Option A: **combined** — one chart, all 6 points, distinct markers per target (Llama = circles, Qwen = squares). Simpler, easier to read the cross-target comparison.
  * Option B: **faceted** — two panels, one per target. Cleaner labelling but duplicates the ceiling drawing.
  * Author's suggested default: **Option A (combined).** Cleaner for the report; the small dataset (6 points) doesn't need faceting.

Author's suggested defaults: **Q-RL-1 = 50 GFLOPs datasheet; Q-RL-2 = 30 GB/s DRAM; Q-RL-3 = 100 MB/s disk; Q-RL-4 = combined chart.**

**RESOLVED 2026-07-01:** all 4 defaults approved by user (ndvp39@gmail.com) inline in chat. T-3.7c code cleared to start once the T-3.6b sweep CSV lands.

---

## 12. Landing plan (informative, not gated by this PRD)

* **T-3.7c** — This PRD authored + approved (2026-07-01).
* **T-3.7c.1** — `services/roofline_analyzer.py` (pure math + CSV/manifest writer) + unit tests.
* **T-3.7c.2** — `services/roofline_chart.py` (matplotlib chart) + golden-file test.
* **T-3.7c.3** — SDK method `roofline_analysis()` + CLI `roofline-analysis` command + integration test.
* **T-3.7c.4** — Add `roofline` config subtree to `config/setup.json` + `n_params_billion` field to each target_models entry.

Every step maps to constitution §1.3's SDLC phases.

**Roofline lands after T-4.1 (economic analyzer) so the M4 chart suite is complete before M5 QLoRA begins.**

---

## 13. Ready to review?

Every numeric target in this PRD's §8 references config values (§5.3) — no magic constants. The 4 open questions each have a proposed default. Approval unblocks T-3.7c code.

**Approver:** user (ndvp39@gmail.com) — signs off inline in a chat reply.
