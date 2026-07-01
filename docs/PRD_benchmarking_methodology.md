# PRD — Benchmarking Methodology (TTFT / TPOT / Throughput / peak memory / energy)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-4 — **MANDATORY**.
> **Blocks:** **T-3.5** code (`services/sweep_runner.py`) — the sweep runner's warm-up / repetitions / aggregate-statistics behaviour is not code-guessable; it is a policy this PRD defines. Approval required before T-3.5 implementation begins.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf.pdf` §3, §5 (measurement rigor) → `L08-summary-Lora-AirLLM.pdf` §3 (Roofline: TTFT proxies Prefill, TPOT proxies Decode) → `docs/PRD.md` v1.10 FR-9 (records TTFT / TPOT / Throughput / peak RAM / peak VRAM / wall / energy / token counts) + FR-10 (repeatable, seed + prompt + max_new_tokens + model_id + quantization in manifest) + K4 (≥ 5 repetitions) + K5 (TPOT over ≥ 64 tokens) + K6 (memory sampling ≥ 2 Hz).
> **Empirical anchor:** M2a `results/plumbing_20260630T194154Z.json` (AirLLM Llama-3-8B, 2 tokens, TTFT 367 282 ms / TPOT 368 350 ms / peak RAM 1287 MB / wall 1104 s) + M2b `results/baseline_llama3-8b-fp16_20260630T221156Z.json` (Direct-with-offload, 128 tok, TTFT 63 708 ms / TPOT 35 139 ms / peak RAM 3190 MB / wall 4526 s). Both manifests already carry the fields this PRD formalises — the schema is measured, not proposed.
> **Document version:** 1.00 — 2026-07-01.
> **Status:** DRAFT, awaiting user approval before T-3.5 code is written.

---

## 1. What and why

### 1.1 What
The project-wide **benchmarking methodology** — precise definitions of every number the sweep, quality matrix, and M6 report will quote. Covers:

* **Timing metrics** — TTFT, TPOT, ITL, Throughput, wall time. What each *counts*, when the timer starts, when it stops, how they interact with the two-generate pattern the backends already use.
* **Memory metrics** — peak RSS, peak VRAM. Sampling rate, sample window, aggregation rule.
* **Energy metric** — Wh from configured wattage × wall seconds.
* **Repetition policy** — warm-up runs (excluded from stats), N measurement runs (included), aggregation into mean / std / median / p95.
* **Prompt policy** — fixed benchmark prompt(s), token-count normalisation.
* **Failure handling** — retries on transient errors, structured stats when N runs land partial failures.

This PRD does NOT introduce a new component. It retroactively documents:

* `on_prem_llm_lab.mixins.timing_mixin.TimingMixin` (T-2.2) — TTFT + TPOT + n_tokens on any streaming token iterator.
* `on_prem_llm_lab.mixins.memory_sampling_mixin.MemorySamplingMixin` (T-2.3) — daemon-thread sampler at configurable Hz with peak tracking.
* `on_prem_llm_lab.mixins.energy_accounting_mixin.EnergyAccountingMixin` (T-2.5) — `Wh = watts × wall_s / 3600`.
* `on_prem_llm_lab.services.benchmark_runner.BenchmarkRunner` (T-2.9) — the runner that composes the mixins around one backend cycle.
* `on_prem_llm_lab.backends.direct_backend.DirectBackend.generate` (T-2.8) + `on_prem_llm_lab.backends.airllm_backend.AirLLMBackend.generate` (T-3.1) — the two-generate TTFT/TPOT pattern.

… and defines how **T-3.5's SweepRunner** composes those pieces to produce a `results/sweep_<timestamp>.csv` with N repetitions per `(target, quantization, backend)` cell + aggregated statistics.

### 1.2 Why
* **Reproducibility.** The report's tradeoff tables (e.g. "AirLLM 6 min/tok vs Direct+offload 35 s/tok") are only meaningful if every reader can trace each number to a defined counting rule + a measurement recipe. FR-10 mandates it.
* **Comparability.** SC-2 requires per-target comparisons across quantization levels. Comparing TTFT across (target=Llama-3-8B, q=fp16, backend=airllm) and (target=Llama-3-8B, q=q4, backend=airllm) is only fair if the timer starts + stops at the exact same event in both.
* **L08 §3 Roofline.** The lecture separates Prefill (compute-bound proxy = TTFT) from Decode (memory-bound proxy = TPOT). Producing that plot correctly requires each metric to actually measure what it claims — the plumbing manifest's TTFT ≈ TPOT under AirLLM is the L08 §3 headline observation *only* if TTFT and TPOT were measured to identical rules.
* **K4/K5/K6 compliance.** FR-9's K-goals mandate ≥ 5 reps, TPOT over ≥ 64 tokens, sampling ≥ 2 Hz. This PRD encodes those into T-3.5's config defaults + assertions.

### 1.3 Who consumes this policy
| Consumer | Where the policy applies | Landing task |
|---|---|---|
| `services/sweep_runner.py` | Reads warm-up / repeats / prompt / max_new_tokens from config; writes aggregated CSV rows | T-3.5 |
| `services/plumbing_default_stages.py` | Already uses the two-generate pattern; regression-check against this PRD | already live (M2a) |
| Quality matrix harness | Same prompt across every supported `(target, quantization)`; captures completion texts | T-3.6 |
| Report assembler | Turns the sweep CSV into narrative + charts | T-6.1 |
| The M6 README | Quotes every number with the field name + manifest path | T-6.2 |

---

## 2. Theoretical background & definitions

### 2.1 The two-generate pattern (already shipped)

Every backend's `generate()` runs `model.generate(input_ids, max_new_tokens=1)` first to time the prefill / first-token step, then `model.generate(input_ids, max_new_tokens=N)` for the full completion. The two-timer measurement:

```
t0     = clock() at generate() start
model.generate(max_new_tokens=1)    # prefill + first token
t_first = clock()
model.generate(max_new_tokens=N)    # prefill again + N tokens (full completion)
t_last  = clock()
```

This is deliberately not a streaming/TextIteratorStreamer setup — it maximises portability across DirectBackend, AirLLMBackend, and future ApiBackend at the cost of counting a second prefill against the TPOT window. This PRD accepts that trade-off and defines TPOT accordingly (see 2.3).

### 2.2 TTFT (Time To First Token, ms)

**Definition:** `ttft_ms = (t_first - t0) × 1000`.
**What it counts:** the wall time from the moment `generate()` is called with `max_new_tokens=1` to the moment that single-token call returns. Includes prompt tokenization, prefill (attention pass over the input tokens), and the single-token forward pass.
**What it excludes:** the tokenization done to compute `prompt_tokens` (that runs BEFORE the timer starts), model load, memory sampler start-up.
**Prefill proxy:** TTFT is the L08 §3 Prefill regime's wall-time proxy — dominated by compute on large prompts, dominated by disk streaming under AirLLM.

### 2.3 TPOT (Time Per Output Token, ms/tok)

**Definition:** `tpot_ms = (t_last - t_first) × 1000 / max(1, max_new_tokens - 1)`.
**What it counts:** wall time of the SECOND `generate()` (full run) divided by `max_new_tokens - 1` decode steps.
**What it INcludes** (deliberate honesty):
* A second prefill (the second `generate()` reprocesses the prompt from scratch) — this inflates TPOT by roughly `TTFT / (max_new_tokens - 1)` under fast decoders and is negligible when TPOT ≫ TTFT / N (the AirLLM case).
* Sampling overhead (whatever the backend defaults do — greedy in our config).
**Rationale for the trade-off:** the M6 report's headline observation *is* that under AirLLM per-layer streaming, TTFT ≈ TPOT (see M2a plumbing manifest, 367 s vs 368 s — 0.3 % apart), meaning both regimes collapse to the disk-streaming rate. That observation is preserved by this measurement — the second prefill contributes only ~365 s / 128 ≈ 3 s per decode step, well within the noise floor of a 6 min/tok run.
**When to prefer streaming ITL instead:** for models where TPOT is expected to be < 1 s/tok AND max_new_tokens is small (< 32), the two-generate pattern's second-prefill overhead becomes visible. In that regime, T-3.5 SHOULD emit a warning in the sweep output row (`method_note = "two-generate; second-prefill overhead ≈ N ms/tok"`) so the report can flag it. See §11 D-BM-3.

### 2.4 ITL (Inter-Token Latency) — NOT what we report

**Definition (industry standard):** `itl_ms = time between consecutive output tokens` under a streaming decoder. This project does NOT report ITL under this name because the two-generate pattern's second-prefill overhead is baked into every observation — the number would be systematically higher than a streaming ITL. `TPOT` in our CSV is the label; consumers reading the M6 report should interpret it as "amortised per-token wall time from the second-generate call, second-prefill overhead included".

### 2.5 Throughput (tokens/s)

**Definition:** `throughput_tps = completion_tokens / wall_s`.
**What it counts:** total generated tokens divided by the full two-generate wall time (`t_last - t0`).
**What it excludes:** tokenizer setup, model load, memory sampler start-up.

### 2.6 Wall time (s)

**Definition:** `wall_s = t_last - t0`.
**What it counts:** the total two-generate call duration.
**What it excludes:** everything BEFORE `generate()` was called (load, pre-flight, tokenization) and everything AFTER `generate()` returned (unload, manifest write). Memory sampling runs concurrently with wall time — the sampler is joined AFTER `generate()` returns so no overhead is counted.

### 2.7 Peak RAM (MB)

**Definition:** `peak_ram_mb = max(sample.ram_mb for sample in samples)` where `samples` is the ordered list of `MemoryReading` values the `MemorySamplingMixin` daemon thread produced between `start_memory_sampling()` and `stop_memory_sampling()`.
**Sampling rate:** `sampling.memory_hz` from `config/setup.json` (default 5 Hz per config, meets K6 ≥ 2 Hz).
**Sample source:** `psutil.Process().memory_info().rss` for the current Python process (from `shared/ram_sampler.psutil_rss_sampler`). Includes the loaded model + tokenizer + activations + KV cache, excludes the AirLLM shard cache on disk.
**What it does NOT capture:** the memory the OS pages in during AirLLM's per-layer streaming — that shows up as pressure on the *system* available RAM (M2b witness manifests captured that separately). If a future revision needs to distinguish "in-process RSS" vs "system memory pressure", add a second sampler that reads `psutil.virtual_memory().available` alongside. Out of scope for v1.00.

### 2.8 Peak VRAM (MB)

**Definition:** `peak_vram_mb = max(sample.vram_mb for sample in samples if sample.vram_mb is not None)`.
**Sample source:** `pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)` when a GPU is present; `None` on CPU-only hosts (M2b reference machine, `config.hardware_constraints.gpu.present == False`).
**Reported as `None` in `BackendRunResult`** when no GPU sample was ever taken (all samples had `vram_mb=None`).

### 2.9 Energy (Wh)

**Definition:** `energy_wh = watts × wall_s / 3600` per `EnergyAccountingMixin.compute_energy_wh`.
**Watts source:** `config.energy.assumed_watts_active` (default 180 W). This is a nameplate estimate, NOT a physical measurement — the M6 report must call that out. A future revision may add real RAPL / IPMI measurement on Linux machines.
**Rationale for keeping it simple:** the break-even chart (M4) plots cost-per-request against volume; the exact wattage is a sensitivity variable. Any value in the 50–300 W range produces a plot with the same shape.

---

## 3. Functional requirements (FR-BM-*)

* **FR-BM-1.** SweepRunner MUST perform `warmup_repeats` warm-up runs of `generate()` before the measurement runs. Warm-ups populate the OS page cache (AirLLM shard streaming) + torch's kernel-selection heuristics and are EXCLUDED from stats. Default `warmup_repeats = 1`.
* **FR-BM-2.** SweepRunner MUST perform `repeat` measurement runs and aggregate them into `mean`, `median`, `std`, `min`, `max`, `p95` for TTFT, TPOT, Throughput, peak RAM, wall, and energy. `repeat` reads from `config.sampling.repeat` (default 5 per current config, meets K4).
* **FR-BM-3.** SweepRunner MUST NOT interleave measurement runs for different `(target, quantization, backend)` cells — each cell runs its warm-ups + repeats to completion before the next cell starts. This gives each cell a clean disk-cache state for the warm-up and stable RSS baseline for the samplers.
* **FR-BM-4.** Every measurement run MUST use the SAME prompt (read from `config.generation.sweep_prompt`, defaulting to `config.generation.baseline_prompt` if absent). Prompt length affects TTFT; keeping it constant across cells makes cells comparable.
* **FR-BM-5.** Every measurement run MUST use `max_new_tokens = config.generation.max_new_tokens` (default 128), which satisfies K5 (TPOT over ≥ 64 tokens as long as the default is ≥ 65).
* **FR-BM-6.** SweepRunner MUST include per-row: `target_label`, `model_id`, `quantization`, `backend`, `seed`, `prompt_tokens`, `max_new_tokens`, `repeat`, `warmup_repeats`, plus the six aggregated statistics per metric. The CSV columns are: `target_label, backend, quantization, ttft_ms_{mean,median,std,min,max,p95}, tpot_ms_{...}, throughput_tps_{...}, peak_ram_mb_{...}, wall_s_{...}, energy_wh_{...}, prompt_tokens, completion_tokens, seed, repeat, warmup_repeats, n_success, n_failed`.
* **FR-BM-7.** A single measurement run that raises MUST be recorded in `n_failed` with the error captured in a per-run raw log (`results/logs/sweep_<ts>_<cell>_<run>.jsonl`). Statistics are computed over the `n_success` runs; if `n_success == 0` for a cell the row's stats are `NaN` and a WARNING is logged.
* **FR-BM-8.** SweepRunner MUST honour ADR-010 (plumbing precondition) — refuses to start unless a successful plumbing manifest exists under `results/`, unless `skip_plumbing=True` (already implemented in `sdk.run_sweep`, T-2a.4).
* **FR-BM-9.** SweepRunner MUST honour DP-3 (quantization matrix) — cells for unsupported `(backend, quantization)` combos are SKIPPED with a WARNING, not attempted. The skip is recorded as a CSV row with `n_success=0`, `n_failed=0`, `skip_reason="unsupported quantization"`.
* **FR-BM-10.** The two-generate pattern's second-prefill overhead MUST be flagged in an inline `method_note` column when TPOT-as-measured exceeds a heuristic threshold that says "second prefill is visible" — specifically when `ttft_ms / (max_new_tokens - 1) > 0.1 × tpot_ms`. Otherwise `method_note = ""`.

## 4. Non-functional requirements (NFR-BM-*)

* **NFR-BM-1.** SweepRunner ≤ 150 LOC per constitution. Statistics helpers may live in a sibling module if needed (constitution §2.2 sanctioned).
* **NFR-BM-2.** No hard-coded numerics — `warmup_repeats`, `repeat`, `max_new_tokens`, `memory_hz`, wattage all come from `config/setup.json`.
* **NFR-BM-3.** Coverage ≥ 85 % on SweepRunner + statistics helpers. Target ≥ 95 % since the surface is pure aggregation.
* **NFR-BM-4.** The CSV row format MUST be stable across sweep runs so downstream chart helpers (T-3.7) don't need to change when the sweep runs again.
* **NFR-BM-5.** No new heavy dependency — statistics come from the standard library (`statistics.mean`, `stdev`, `median`, `quantiles`).

---

## 5. I/O Contract

### 5.1 `SweepRow` (frozen dataclass — the row that lands in the CSV)

```python
@dataclass(frozen=True, kw_only=True)
class SweepRow:
    target_label: str
    backend: BackendId
    quantization: Quant
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
    # Aggregated stats — one struct per metric
    ttft_ms: MetricStats
    tpot_ms: MetricStats
    throughput_tps: MetricStats
    peak_ram_mb: MetricStats
    wall_s: MetricStats
    energy_wh: MetricStats


@dataclass(frozen=True)
class MetricStats:
    mean: float
    median: float
    std: float
    min: float
    max: float
    p95: float
```

### 5.2 Statistics helpers (module-level, `services/sweep_stats.py`)

```python
def aggregate(values: list[float]) -> MetricStats: ...
    # mean = statistics.mean(values)
    # median = statistics.median(values)
    # std = statistics.stdev(values) if len(values) > 1 else 0.0
    # p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)


def has_visible_second_prefill(ttft_ms: float, tpot_ms: float, max_new_tokens: int) -> bool:
    """FR-BM-10 heuristic."""
    if max_new_tokens < 2 or tpot_ms == 0:
        return False
    return ttft_ms / (max_new_tokens - 1) > 0.1 * tpot_ms
```

### 5.3 CSV shape (flattened rows)

Column order fixed:
```
target_label,backend,quantization,seed,prompt_tokens,max_new_tokens,
completion_tokens,repeat,warmup_repeats,n_success,n_failed,skip_reason,method_note,
ttft_ms_mean,ttft_ms_median,ttft_ms_std,ttft_ms_min,ttft_ms_max,ttft_ms_p95,
tpot_ms_mean,tpot_ms_median,tpot_ms_std,tpot_ms_min,tpot_ms_max,tpot_ms_p95,
throughput_tps_mean,throughput_tps_median,throughput_tps_std,throughput_tps_min,throughput_tps_max,throughput_tps_p95,
peak_ram_mb_mean,peak_ram_mb_median,peak_ram_mb_std,peak_ram_mb_min,peak_ram_mb_max,peak_ram_mb_p95,
wall_s_mean,wall_s_median,wall_s_std,wall_s_min,wall_s_max,wall_s_p95,
energy_wh_mean,energy_wh_median,energy_wh_std,energy_wh_min,energy_wh_max,energy_wh_p95
```

### 5.4 Sweep manifest (`results/sweep_<ts>.json`)

Companion to the CSV. Records the full config snapshot + repo git hash + wall-clock start/end + list of processed cells (with per-cell manifest paths). Same shape as the existing `BenchmarkRunner.write_manifest` output but keyed by cell.

---

## 6. Constraints

* **C-BM-1.** File-size cap 150 LOC per file.
* **C-BM-2.** No hard-coded numerics.
* **C-BM-3.** Statistics MUST use stdlib `statistics` module — no `numpy`/`scipy` for this.
* **C-BM-4.** The two-generate pattern is FIXED across all backends. Adding a streaming path is a separate PRD.
* **C-BM-5.** Warm-up run failures are FATAL — if the warm-up raises, the cell aborts with no CSV row (only a manifest with `skip_reason="warmup_failed"`). This is stricter than measurement-run failures (FR-BM-7). Rationale: a failing warm-up means the cell is broken; running the measurement passes anyway wastes time.

---

## 7. Alternatives considered

| # | Option | Reason rejected |
|---|---|---|
| A-BM-1 | Use streaming `TextIteratorStreamer` for TTFT/ITL instead of the two-generate pattern | The M2a plumbing test proved AirLLM does not surface a streamer object we can drop into `TimingMixin.time_token_stream`. The two-generate pattern works uniformly across DirectBackend + AirLLMBackend + ApiBackend at the cost of a documented second-prefill overhead (FR-BM-10 flags when it's visible). Keeping the codepath uniform outweighs the overhead. |
| A-BM-2 | Report only mean + std, drop median / p95 | K4 says ≥ 5 reps. For N=5 the mean+std are noisy; adding median gives outlier resistance and p95 flags stragglers. Free (stdlib `statistics.median`, `quantiles`). Keep them. |
| A-BM-3 | Sample memory at 20 Hz for finer granularity | Overkill on a CPU-only box where the sampler thread already competes with the model thread for CPU cycles. 5 Hz meets K6 and stays out of the measurement's way. Configurable via `sampling.memory_hz` if a future run needs finer sampling. |
| A-BM-4 | Real energy measurement via Intel RAPL / IPMI | Portable across Linux only; the reference box is Windows. Nameplate wattage is an accepted approximation for break-even shape. A future revision can add real measurement if a Linux GPU box appears. |
| A-BM-5 | Interleave cells to average out background load | Adds bookkeeping complexity and makes the CSV rows harder to reason about (each row spans all cells). Non-interleaved is simpler and the report is comparing steady-state behaviour, not detecting drift. |
| A-BM-6 | Adaptive repetitions ("keep repeating until std < X %") | Adds runtime unpredictability — the sweep could run for hours instead of the planned window. Fixed N is more honest about the noise floor and keeps the run bounded. |

---

## 8. Success criteria

* **SC-BM-1.** SweepRunner writes exactly one CSV row per `(target, quantization, backend)` cell in `config.target_models × supported_matrix`; unsupported combos land as `skip_reason` rows (SC-Q-3 backwards).
* **SC-BM-2.** Each supported cell's row carries mean+median+std+min+max+p95 for all six metrics, computed from `n_success` measurement runs.
* **SC-BM-3.** Warm-up runs are excluded from statistics (asserted by unit test: warm-up returns 1000 ms TTFT, measurement returns 500 ms; stat mean == 500 ms).
* **SC-BM-4.** A failed measurement run does NOT abort the cell — it's counted in `n_failed`, its raw log lands under `results/logs/`, and the row's stats use the remaining `n_success` runs.
* **SC-BM-5.** `n_success == 0` for a cell produces a row with `NaN` stats and a WARNING in the runner's log.
* **SC-BM-6.** `has_visible_second_prefill` fires correctly on the fast-decoder edge case (unit test with synthetic TTFT=100 ms + TPOT=50 ms + max_new_tokens=8 → True; TTFT=6100 s + TPOT=6100 s + max_new_tokens=2 → False).
* **SC-BM-7.** The CSV round-trips through Python's `csv.DictReader` cleanly (unit test).
* **SC-BM-8.** The sweep manifest JSON validates as strict JSON and carries the git hash + config snapshot (SC-BM-8 mirrors T-2.4's manifest shape).

---

## 9. Test scenarios (informs T-3.5 unit suite)

### 9.1 Unit tests (in `tests/unit/test_services/test_sweep_stats.py` — new)
| ID | Scenario |
|----|----------|
| U-BM-1 | `aggregate([1.0])` returns `MetricStats(mean=1, median=1, std=0, min=1, max=1, p95=1)` |
| U-BM-2 | `aggregate([1.0, 2.0, 3.0, 4.0, 5.0])` returns mean=3, median=3, std≈1.58, min=1, max=5, p95=5 (< 20 values → p95 = max) |
| U-BM-3 | `aggregate([...20 values...])` returns p95 = the actual 95th percentile via `statistics.quantiles` |
| U-BM-4 | `has_visible_second_prefill(100, 50, 8)` → True |
| U-BM-5 | `has_visible_second_prefill(367_282, 368_350, 2)` → False (AirLLM case) |
| U-BM-6 | `has_visible_second_prefill(0, 100, 2)` → False (edge) |

### 9.2 Unit tests (in `tests/unit/test_services/test_sweep_runner.py` — new)
| ID | Scenario |
|----|----------|
| U-BM-7 | `run_sweep` iterates every `(target × quantization × backend)` cell in the input |
| U-BM-8 | Unsupported cells emit a skip row and do NOT invoke the backend |
| U-BM-9 | Warm-up runs excluded from stats — 1 warmup at 1000 ms + 3 measurements at 500 ms → mean=500 ms |
| U-BM-10 | A measurement run raising `RuntimeError` bumps `n_failed`, does not abort the cell |
| U-BM-11 | `n_success == 0` produces a NaN-stats row + WARNING |
| U-BM-12 | `method_note` populated when the fast-decoder second-prefill condition fires |
| U-BM-13 | CSV written to `results/sweep_<ts>.csv` with the FR-BM-6 column order |
| U-BM-14 | Sweep manifest JSON written alongside the CSV with the T-2.4 shape |
| U-BM-15 | ADR-010 plumbing precondition — no plumbing manifest → refuse to start (verified via `require_current_plumbing` mock) |

### 9.3 Integration
| ID | Scenario |
|----|----------|
| I-BM-1 | End-to-end sweep with a mock backend (2 targets × 3 quantizations × 2 backends = 12 cells, ~4 supported per DP-3 matrix). Assert CSV row count = 12, aggregate stats present on supported rows, `skip_reason` on unsupported. |

---

## 10. Out of scope

* **Streaming TTFT/ITL via `TextIteratorStreamer`.** Different codepath, different overhead accounting. Post-v1.00 revision if the fast-decoder case becomes important.
* **Real energy measurement (RAPL / IPMI).** Nameplate wattage stays until a Linux GPU box is in scope.
* **Adaptive repetition counts.** Fixed N per cell; predictability wins.
* **Cross-cell interleaving.** See A-BM-5.
* **VRAM sampling beyond `pynvml`.** No GPU on the reference machine; the sampler is code-ready but never fires today.
* **Prompt sweeps (multiple prompts per cell).** One prompt per sweep run for v1.00. If the report needs per-prompt breakdowns, add a `prompts: list[str]` config field in a future revision.

---

## 11. Decisions taken in this PRD

| ID | Decision | Why |
|----|----------|-----|
| D-BM-1 | Keep the two-generate pattern | Portability across DirectBackend + AirLLMBackend + future ApiBackend beats a streaming path that only works for one backend. |
| D-BM-2 | TPOT includes the second prefill's overhead (FR-BM-10 flags visibility) | Alternative would be to subtract TTFT from wall then divide — but that assumes the second prefill takes the same time as the first, which is only true when disk cache is warm. Reporting the raw two-generate wall is more honest. |
| D-BM-3 | Non-adaptive N with stats (mean+median+std+min+max+p95) | K4 = 5, stdlib does the rest. Adaptive N adds runtime unpredictability. |
| D-BM-4 | Warm-up = 1 pass by default | AirLLM's shard streaming benefits from OS page cache warm-up on the second pass. torch's kernel selection heuristics also settle after the first invocation. 1 is enough; configurable via `config.sampling.warmup_repeats`. |
| D-BM-5 | Failed measurement runs do NOT abort the cell (FR-BM-7) but failed WARM-UPS do (C-BM-5) | Warm-up failures are structural; measurement failures are noise. Different response. |
| D-BM-6 | CSV column order is FIXED per §5.3 | Downstream chart helpers (T-3.7) index by column name; changing order later breaks charts silently. |

---

## 12. Open questions for user

1. **Warm-up repeats default: 1 or 2?** My draft says 1 (D-BM-4). AirLLM's disk cache is the main beneficiary; 1 pass gets the shards into OS cache. 2 passes doubles the warm-up time (a real cost on 6 min/tok cells). Confirming 1.
2. **`n_success == 0` — NaN stats or omit the row entirely?** My draft says NaN row + WARNING (FR-BM-7 / SC-BM-5). Omitting silently would hide broken cells from the CSV. Keeping the row surfaces the failure to the report. Confirming NaN.
3. **`sweep_prompt` config field.** My draft falls back to `baseline_prompt` if the sweep-specific prompt is absent. Alternative: require an explicit `sweep_prompt` (fail loudly if missing). My draft is more forgiving — confirming that's OK.

---

## 13. Approval

This PRD MUST be approved by the user before any code is written for T-3.5. Approval flips DP-4 status in `docs/TODO.md` §7 to "MANDATORY · authored + approved".

> Approval status: ☑ **Approved 2026-07-01** by user — *"approved, 1 warmup, NaN row, fall back to baseline_prompt"*. Answers to the 3 open questions in §12 all match the draft: (Q1) `warmup_repeats = 1`; (Q2) `n_success == 0` → NaN-stats row + WARNING (FR-BM-7 / SC-BM-5); (Q3) `sweep_prompt` falls back to `baseline_prompt` if absent.
> Implementation status: ☑ **Implemented 2026-07-01** by T-3.5 — `services/sweep_stats.py` (118 LOC, all pure helpers: `MetricStats` + `SweepRow` + `aggregate` + `has_visible_second_prefill` + `csv_columns` + `row_to_csv_dict` + `write_csv` + `write_manifest`, 100 % coverage) + `services/sweep_runner.py` (135 LOC, `SweepRunner` class + `CellRunner` type alias, 99 % coverage — the single uncovered line is the `_utc_ts` production wall-clock fallback since tests always inject `clock=`). 21 new tests across 2 files covering U-BM-1..15 + SC-BM-7/8 + C-BM-5 + the Q3 fallback path. Split from a single 164-LOC candidate into the 2-file layout per constitution §2.2. Contract fulfilled verbatim — no deviations from v1.00.
