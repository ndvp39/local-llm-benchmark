# PRD — On-Premises LLM Benchmark Lab (`on_prem_llm_lab`)

> **Document type:** Product Requirements Document (SDLC Phase 1).
> **Project code:** `on_prem_llm_lab` (working name).
> **Source authority:** `SOFTWARE_PROJECT_GUIDELINES.md` (Dr. Yoram Segal, v3.00) — binding constitution.
> **Assignment source:** `ex05-AirLLM.pdf.pdf` (Assignment 05 — Running a Massive LLM Locally), `L08-summary-Lora-AirLLM.pdf` (lecture L08).
> **Document version:** 1.10 — 2026-06-26.
> **Status:** REVISED DRAFT — incorporates user decisions on the four open questions. Awaiting approval before TODO is locked and code begins.

> **Changelog (v1.00 → v1.10).** Resolved all four open questions: target models are **Llama-3-8B** and **Qwen-7B-Q4** (no 70B); economic comparison uses **Anthropic** as the sole third-party provider; §5.7 extension is **QLoRA training** (triggers mandatory `docs/PRD_qlora.md`); the **Cloud GPU** curve is included in the break-even chart. Added two cross-cutting rules: (i) target models are defined as a JSON **array** in `config/setup.json` and iterated by the SDK; (ii) a **mandatory pipeline plumbing test** runs first against a small model + aggressive quantization (Q2) before any oversized run. Initialization technical rule (use `AutoModel*` factories) is captured in PLAN.

---

## 1. Project Overview & Context

### 1.1 What we are building
An **engineering-research lab** that proves, in a measured and reasoned way, that we can run Large Language Models that are **deliberately too big** for the available local hardware, by combining two complementary techniques:

1. **AirLLM** — layer-by-layer execution with memory-mapped (`mmap`) weight loading, exploiting the OS virtual-memory paging mechanism to keep only one transformer layer "hot" in RAM/VRAM at a time.
2. **Quantization** — reducing per-weight bit-width (FP32 → FP16 → FP8 → Q4 → Q2 and, where relevant, NF4 / QLoRA) to compress the model footprint.

The deliverable is **not a product for end users**. It is an **engineering deep-dive technical report** (which, per assignment §7, MUST be the project `README.md` itself) backed by reproducible code, experiments, plots, and an economic analysis (On-Prem vs Anthropic API vs Cloud GPU).

### 1.2 Problem statement
Modern open-weight LLMs (e.g., Llama-3, Qwen, Mistral families at the 7B–70B+ scale) routinely exceed the VRAM of consumer GPUs and even the RAM of typical workstations. Three deployment options are available — external **API** (e.g., Anthropic, OpenAI), **Cloud GPU**, or **On-Premises** — each with different trade-offs in cost, privacy, latency and operational complexity (see L08 §1.1, Table 1).

The practitioner therefore faces three concrete unknowns:

- **Bottleneck identification:** when direct inference fails or crawls, is the limiting resource **memory** (RAM/VRAM bandwidth and capacity) or **compute** (FLOPs)? How do we *prove* this from measurements rather than guess?
- **Optimization payoff:** how much does AirLLM + Quantization actually buy us in latency, throughput and peak memory, and at what cost in output quality?
- **Economic crossover:** at what request volume does paying CAPEX+OPEX for local hardware become cheaper than paying a third-party API (Anthropic) per token, and where does Cloud GPU sit relative to both curves?

### 1.3a Detected hardware (auto-populated by `init_env.py` — do not hand-edit)

<!-- HARDWARE_SPECS_PLACEHOLDER:START -->

| Component | Value |
|-----------|-------|
| Captured at | 2026-06-30T17:22:50Z |
| OS / Python | Windows-10-10.0.19045-SP0 / 3.12.13 |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 4 physical / 8 logical |
| RAM | 7.8 GB total · 2.9 GB available |
| GPU | not detected (CPU-only run) |
| Disk | 392.8 GB free · unknown · unknown (measured at `D:\AI_agents_course\airllm_shards`) |


<!-- HARDWARE_SPECS_PLACEHOLDER:END -->

### 1.3 Target audience
- **Primary:** the course instructor / reviewer, who will grade on technical depth, measurement rigor, and economic analysis.
- **Secondary (downstream):** ML practitioners and architects evaluating local-vs-API deployment for new projects; the report should be reusable as a reference write-up.
- **Tertiary:** future Self — the project will be cited from the student's portfolio / GitHub.

### 1.4 Market & landscape (brief)
Local-LLM tooling exploded in 2023–2025: **Ollama** (GGUF runtime with an OpenAI-compatible local API), **llama.cpp**, **AirLLM** (CPU/low-VRAM layered execution), **vLLM / PagedAttention** (SOSP 2023), **FlexGen** (ICML 2023), **LLM in a Flash** (Apple, 2024), **QLoRA / NF4** (Dettmers et al., NeurIPS 2023), **Disaggregated Serving** (Splitwise ISCA 2024, DistServe OSDI 2024). The L08 lecture is the curricular anchor; this project operationalizes a focused subset of it (AirLLM + Quantization + measurement + economics + QLoRA fine-tune extension).

---

## 2. Goals, KPIs & Acceptance Criteria

### 2.1 Measurable project goals

| # | Goal | Definition of success |
|---|------|-----------------------|
| G1 | **Pipeline plumbing verified.** | A small model + Q2 quantization completes the full end-to-end pipeline (download → AirLLM mmap allocation → generate → metric collection → manifest write) before any oversized run is attempted. |
| G2 | **Run oversized models locally.** | **Each** model in `config.target_models` (initially Llama-3-8B and Qwen-7B-Q4) completes a fixed benchmark prompt set successfully via the AirLLM path. |
| G3 | **Identify the true bottleneck.** | The report contains a Roofline-style analysis attributing each measured slowdown to either *compute-bound* (Prefill) or *memory-bound* (Decode), backed by data, not opinion. |
| G4 | **Quantify the optimization payoff.** | Side-by-side measurements across at least 3 quantization levels (e.g., FP16, Q8, Q4 — Q2 reserved for the plumbing test and the aggressive-end of the sweep) with TTFT, TPOT, Throughput, peak RAM+VRAM, electrical cost, and a qualitative output-quality assessment. |
| G5 | **Compute the economic break-even (3 curves).** | A break-even chart (cost vs request volume) comparing **On-Prem**, **Anthropic API**, and **Cloud GPU**; assumption table inline; crossover points called out. |
| G6 | **QLoRA fine-tune demonstration (§5.7 extension).** | A small LoRA / QLoRA fine-tune of one target model on a tiny dataset succeeds; the report shows VRAM cost of training is 3–5× inference (L08 §7.1) and explains NF4 / Paged Optimizers (L08 §5.1). |
| G7 | **Deliver a publishable deep-dive report.** | `README.md` IS the technical report: hardware spec, methodology, plumbing test, per-target oversized runs, AirLLM + Quantization sweep, economic analysis (3 curves), QLoRA extension, all plots/tables/screenshots inline. |
| G8 | **Meet the constitution's quality bar.** | Ruff zero violations, test coverage ≥ 85 %, all business logic behind the SDK, no hard-coded values, `uv` only, every file ≤ 150 lines of code. |

### 2.2 KPIs (numeric targets)

| KPI | Target | Measured how |
|-----|--------|--------------|
| K1 — Plumbing test | small-model + Q2 run completes end-to-end before any oversized run | Captured manifest `results/plumbing_<timestamp>.json` |
| K2 — Successful oversized AirLLM runs | 1 fixed prompt produces a complete completion **for every entry** in `config.target_models` | Captured `stdout` + saved completion artifacts |
| K3 — Quantization sweep coverage | ≥ 3 distinct bit-widths benchmarked per model | Result table in `results/quantization_sweep.csv` + chart in `figures/` |
| K4 — TTFT measurement accuracy | Reported with ≥ 5 repetitions, mean ± std | Stats in CSV |
| K5 — TPOT / ITL measurement | Reported per token over a fixed-length completion (≥ 64 generated tokens) | Stats in CSV |
| K6 — Peak memory tracking | Continuous RAM + VRAM sampling at ≥ 2 Hz during a run | Time series in `results/` |
| K7 — Break-even curves | 3 cost curves (On-Prem, Anthropic API, Cloud GPU) in one chart with marked crossover(s) | `figures/break_even.png` |
| K8 — QLoRA extension | LoRA/QLoRA fine-tune completes; train-vs-inference VRAM ratio reported | Result + chart in `results/qlora_*` |
| K9 — Ruff violations | **0** | `uv run ruff check src/ tests/` |
| K10 — Test coverage | **≥ 85 %** | `uv run pytest --cov`; `fail_under = 85` |
| K11 — Files over 150 LOC | **0** | CI check (see TODO infra task) |
| K12 — Secrets in code | **0** | Manual review + scan; `.env-example` committed |
| K13 — Package manager violations | **0** uses of `pip`, `venv`, `python -m`, `virtualenv` | grep in CI |
| K14 — Hard-coded model IDs in source | **0** — every model id comes from `config.target_models` | grep in CI; reviewed manually |

### 2.3 Acceptance criteria (DoD for the *project as a whole*)
The project is "Done" when **all** of the following hold:
- [ ] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` exist and are approved by the user.
- [ ] Every "central mechanism" PRD listed in §7 of `TODO.md` exists in `docs/` (including the **mandatory** `docs/PRD_qlora.md` since the §5.7 extension is QLoRA training).
- [ ] All eight goals G1–G8 demonstrably met.
- [ ] All KPIs K1–K14 verified by the CI pipeline.
- [ ] README contains: hardware spec, methodology, plumbing-test write-up, baseline failure narrative, **per-target** AirLLM + Quantization results (tables + charts), 3-curve economic break-even chart (with Cloud GPU), qualitative quality assessment, connection to Prefill/Decode + Roofline + Paging theory, QLoRA extension section, "what we extended originally" linkage to assignment §5.7.

---

## 3. Functional Requirements (FR)

> Numbering: **FR-x** = top-level requirement. `MUST` / `SHOULD` per RFC 2119.

### 3.1 Hardware & environment introspection
- **FR-1.** The system **MUST** detect and report: CPU model + core count, total RAM, GPU model + VRAM (if present), free disk + filesystem type (NVMe / SSD / HDD), OS, Python version. Output is machine-readable JSON saved to `results/hardware_profile.json` and human-readable in the README.

### 3.2 Model acquisition & dynamic target-model configuration
- **FR-2.** The system **MUST** download a Hugging Face model identified **only** via `config/setup.json`, into a configured cache directory, **without** hard-coding the model ID, the cache path, or the HF token in source code.
- **FR-2a.** **Target models MUST be defined as a JSON array** under `config.target_models[]`, each entry carrying at minimum: `{ "id": "...", "quantization": "fp16|q8|q4|q2|nf4", "label": "..." }`. The SDK iterates this array; adding or removing a model requires **no code change** (constitution §6.2).
- **FR-2b.** A separate `config.plumbing_test_model` entry **MUST** also be defined for the §3.10 pipeline plumbing test. The entry carries an explicit `loader` field (`"transformers"` | `"airllm"`). **Final shape (v1.50+, installed 2026-06-30; prompts_book §10):** `plumbing_test_model` points at `config.target_models[0]` byte-identically (currently `meta-llama/Meta-Llama-3-8B-Instruct` fp16 with `loader: "airllm"`) and is constrained to **`config.generation.plumbing_max_new_tokens` (default 2)** instead of `max_new_tokens`. Rationale: §6–§9 demonstrated there is no AirLLM-compatible model below ~7B (the floor is hard — sharded multi-file safetensors + separate `lm_head`), so a *separate small* plumbing model can only validate the transformers loader, not the production AirLLM loader. Collapsing plumbing into "production loader on production model with a tighter token budget" exercises the *exact* code path M2b/M3 will run, while a 2-token cap keeps wall-time ≤ ~20 min on the under-resourced reference hardware. **The plumbing test therefore now validates**: HF download + token + gate, HF cache redirect, dependency stack (AirLLM + optimum + transformers pins), AirLLM layer-split + shard write to D:, mmap allocation, AirLLM tokenizer attachment, AirLLM `generate()`, RAM sampling, manifest write — i.e., the full FR-PT-1 surface. Earlier defaults (v1.10 Q2-of-TinyLlama, v1.40 TinyLlama-fp16-via-transformers) are preserved as historical context in prompts_book §6–§9.
- **FR-3.** The HF token, **if any**, **MUST** be read exclusively from the environment via `os.environ.get("HF_TOKEN")`; a `.env-example` placeholder MUST exist. The HF token MUST NOT be logged.

### 3.3 Inference back-ends
The SDK **MUST** expose three interchangeable inference back-ends behind a single abstract interface (`InferenceBackend`):
- **FR-4.** **Direct** back-end — load the full model via `transformers` / `Ollama` for the *baseline* run (expected to fail or crawl on the oversized model — this is the desired observation).
- **FR-5.** **AirLLM** back-end — layer-by-layer execution with `mmap`-backed SafeTensors shards; the layer shard cache path is **configurable** (assignment §6.1 Do — `layer_shards_saving_path`).
- **FR-6.** **API** back-end — used for the economic comparison; calls **Anthropic** (sole third-party provider for this project). Model name and endpoint come from config; key (`ANTHROPIC_API_KEY`) from env.

### 3.4 Quantization
- **FR-7.** The SDK **MUST** support at least three quantization levels selectable by config (e.g., `fp16`, `q8`, `q4`); `q2` is mandatory for the plumbing test and SHOULD also be measured at the aggressive end of the sweep; `nf4` is required by the QLoRA extension (FR-20).
- **FR-8.** Each quantization level **MUST** be runnable through both the Direct and AirLLM back-ends (subject to back-end capability — gracefully refuse where unsupported with a structured `UnsupportedQuantization` error).

### 3.5 Measurement
- **FR-9.** The SDK **MUST** record per-run: TTFT (ms), TPOT / ITL (ms per generated token), total Throughput (tokens / s), peak RAM (MB), peak VRAM (MB if GPU), wall-clock duration (s), estimated electrical energy (Wh = power × time, with configurable assumed wattage), and tokens-in / tokens-out counts.
- **FR-10.** Each run **MUST** be repeatable; `seed`, prompt, max_new_tokens, model id, quantization, back-end identifier, and software versions are recorded in the run manifest (`results/run_<timestamp>.json`).
- **FR-11.** A **qualitative output-quality** comparison harness **MUST** exist: same prompt, all quantization levels, per target model, side-by-side outputs captured in `results/quality_matrix.md`.

### 3.6 Economic analysis (3 curves)
- **FR-12.** The SDK **MUST** compute, from `config/api_pricing.json`, the per-request cost on the **Anthropic API** path: `(prompt_tokens × in_price_per_M + completion_tokens × out_price_per_M) / 1_000_000`.
- **FR-13.** The SDK **MUST** compute the **On-Prem** amortized cost: `(CAPEX / lifetime_requests) + OPEX_per_request`, with electricity rate, hardware CAPEX, expected lifetime hours, and idle/active power from config.
- **FR-13a.** The SDK **MUST** compute the **Cloud GPU** per-request cost: `(hourly_usd × wall_seconds) / 3600`, with `hourly_usd` from config.
- **FR-14.** The SDK **MUST** produce a **break-even chart** (matplotlib) plotting cost-per-request vs cumulative request volume for **{Anthropic API, On-Prem, Cloud GPU}** — all three curves — with marked crossover point(s). Saved to `figures/break_even.png` and referenced from README. Assumptions table (electricity rate, CAPEX, lifetime, wattage, cloud hourly rate, Anthropic prices) MUST appear next to the chart.

### 3.7 API Gatekeeper (mandatory per constitution §4)
- **FR-15.** **All** outbound API calls (HF Hub download, Anthropic API inference, any future telemetry) **MUST** be routed through the centralized `ApiGatekeeper`. No bypass is allowed.
- **FR-16.** The gatekeeper **MUST** enforce per-service rate limits loaded from `config/rate_limits.json` (versioned), with FIFO overflow queueing and configurable retry policy. **No** numeric limit may be hard-coded in source.

### 3.8 Entry points
- **FR-17.** A **CLI** (`uv run on-prem-llm <subcommand>`) **MUST** expose: `profile-hardware`, `download-model`, `run-plumbing-test`, `run-baseline`, `run-airllm`, `run-sweep`, `run-qlora`, `economic-analysis`, `make-report`. All subcommands **delegate to the SDK only** — zero business logic in the CLI module (constitution §3.1).
- **FR-18.** A minimal `notebooks/analysis.ipynb` reproduces the headline plots from CSVs in `results/`.

### 3.9 Reporting
- **FR-19.** A `make-report` action **MUST** assemble README sections that depend on numeric results (tables, image references) from `results/` and `figures/` — so re-running experiments and re-running the assembler updates the README without manual edits to numbers.

### 3.10 Pipeline plumbing test (mandatory pre-flight)
- **FR-PT-1.** Before any run against a model listed in `config.target_models`, the SDK **MUST** execute a **plumbing test** against `config.plumbing_test_model`, loaded via the loader indicated by the `loader` config field, with generation capped at `config.generation.plumbing_max_new_tokens` (default 2). **In the v1.50+ final shape (see FR-2b)**, `plumbing_test_model` mirrors `target_models[0]` — `meta-llama/Meta-Llama-3-8B-Instruct` fp16 via AirLLM — so the plumbing test exercises the exact production loader path at a 2-token budget. The full pipeline tested: HF download (with cache hit) → AirLLM layer-split + shard write → mmap allocation → tokenizer attachment → metric collection (TTFT, TPOT, peak RAM) → manifest write → cleanup. Reference wall-time on the 4-core / 7.8 GB / CPU-only hardware: **~18 minutes / 2 tokens** (TTFT ≈ TPOT ≈ 6 min/token under AirLLM per-layer streaming — see prompts_book §10).
- **FR-PT-2.** The plumbing test **MUST** be runnable standalone (`uv run on-prem-llm run-plumbing-test`) and **MUST** also be invoked automatically as a precondition by `run-sweep` unless explicitly skipped with `--skip-plumbing` (which logs a warning).
- **FR-PT-3.** Plumbing-test failure **MUST** abort the sweep with a structured error pointing at the failed stage (download / mmap / sample / manifest). The test result is saved as `results/plumbing_<timestamp>.json`.

### 3.11 QLoRA fine-tune extension (assignment §5.7 — committed)
- **FR-20.** The SDK **MUST** expose `sdk.run_qlora_finetune(target_label: str, dataset_path: Path, lora_config: LoraConfig)` that fine-tunes one of the target models with **QLoRA** (NF4 quantization + LoRA adapters per L08 §5.1, §7.3, §7.5).
- **FR-21.** The QLoRA run **MUST** record both training and inference VRAM peaks so the report can show the 3–5× ratio observation from L08 §7.1.
- **FR-22.** The QLoRA mechanism MUST have its own dedicated requirements document at **`docs/PRD_qlora.md`** before implementation begins (constitution §1.3).

---

## 4. Non-Functional Requirements (NFR)

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | **Code quality** | `ruff check` MUST return zero violations under the rule set defined in `pyproject.toml` (constitution §6.1). |
| NFR-2 | **Testability** | Test coverage MUST be ≥ 85 % (constitution §5.2). All public SDK methods MUST have ≥ 1 unit test. |
| NFR-3 | **File size** | No source file MUST exceed 150 lines of code (constitution §2.2). |
| NFR-4 | **Configurability** | Zero hard-coded API URLs, model IDs, rate limits, timeouts, prices, paths, cloud hourly rates, electricity prices (constitution §6.2). |
| NFR-5 | **Security** | Zero secrets in source; `.env-example` present; `.env`, `*.pem`, `*.key`, `credentials.json` in `.gitignore` (constitution §6.4). |
| NFR-6 | **Package manager** | `uv` only; `pip`, `venv`, `virtualenv`, `python -m` strictly forbidden; `uv.lock` committed (constitution §7.4). |
| NFR-7 | **Reproducibility** | Every run produces a manifest containing seed, versions, config snapshot, hardware fingerprint. |
| NFR-8 | **Observability** | Structured logging (JSON-line) for every API call, every back-end invocation, every gatekeeper queue event. |
| NFR-9 | **Portability** | Project MUST install and run on Windows 10/11 and Linux (the student's machine is Windows 10). Path handling MUST use `pathlib`. |
| NFR-10 | **Performance** | Measurement overhead MUST be < 5 % of measured wall-clock (verified by null-op benchmark). |
| NFR-11 | **Versioning** | Initial code version `1.00` in `src/<pkg>/shared/version.py`; config files carry `"version": "1.00"`; runtime validates config compatibility (constitution §7.1). |
| NFR-12 | **Documentation** | Every public class / function has a docstring explaining inputs, outputs, setup, side-effects (constitution §2.3). |
| NFR-13 | **Modularity (Building Blocks)** | Each domain block declares Input / Output / Setup (constitution §15) — see PLAN.md. |
| NFR-14 | **Thread safety** | The gatekeeper, the measurement sampler, and the run manifest writer MUST be thread-safe (constitution §14.2). |
| NFR-15 | **Auditability** | Prompts Book maintained per constitution §7.3 (file `docs/prompts_book.md`). |
| NFR-16 | **Robust initialization** | Models MUST be loaded through the **generic** `transformers.AutoModel` / `AutoModelForCausalLM` factories so the right concrete class is resolved from the model's `config.json` and **Class-mismatch errors are avoided** (assignment §6.1 Do; see PLAN ADR-009). |

---

## 5. User Stories & Scenarios

### 5.1 Personas
- **P1 — Sarah, the Student-Engineer:** runs the experiments on her own laptop, needs reproducibility and clear pass/fail signals.
- **P2 — Dr. Yoram, the Reviewer:** opens `README.md` first; expects to find the deep-dive technical report with plots and tables inline.
- **P3 — Maya, the ML Architect (future reader):** evaluating local-vs-API for her own team; needs the economic break-even (3 curves) and the qualitative quality matrix to make a recommendation.

### 5.2 User stories
- **US-1.** *As Sarah,* I want one command (`uv run on-prem-llm run-sweep`) to execute the full quantization sweep over **every** model in `config.target_models`, after the plumbing test passes, so I do not have to orchestrate runs by hand.
- **US-2.** *As Sarah,* I want the sweep to refuse to start unless the plumbing test has passed recently, with a clear remediation message.
- **US-3.** *As Dr. Yoram,* I want the README to open with the bottom-line table (per target model: TTFT / TPOT / Throughput / peak memory) so I can grade rigor in one minute, and to find the 3-curve break-even chart at the top of the economics section.
- **US-4.** *As Maya,* I want a single chart that shows where On-Prem and Cloud GPU each overtake the Anthropic API in cost-per-request, with all input assumptions explicit and adjustable.
- **US-5.** *As Sarah,* I want to add a third target model (e.g., Mistral-7B) by editing `config/setup.json` only — no code changes — so I can compare model families if time permits.
- **US-6.** *As Sarah,* I want a `run-qlora` command that fine-tunes a target model with QLoRA on a tiny dataset and emits the train-vs-inference VRAM-ratio chart for the report.

### 5.3 Critical scenarios
- **SC-0 — Plumbing test passes first.** Running `run-plumbing-test` with the small model + Q2 succeeds end-to-end before any oversized run. Failure aborts the sweep with a stage-specific error.
- **SC-1 — Baseline must fail visibly.** Running an oversized target model through the Direct back-end on the bare hardware MUST surface the failure (OOM, swap thrash, or unacceptably long latency) with a captured diagnostic, *not* a silent hang. This failure is a required deliverable, not a bug.
- **SC-2 — AirLLM rescues the run for every target.** Each target model + prompt completes via the AirLLM back-end; the report explains the *why* using Paging / mmap analogy from L08 §8.
- **SC-3 — Quantization quality cliff.** At some bit-width the output degrades unacceptably; the report documents *where* the "red line" is, per target model.
- **SC-4 — Three-way break-even crossover plotted.** For sustained low volume, API wins; for sustained high volume, On-Prem and/or Cloud GPU win. The chart visibly contains the crossover points and labels them.
- **SC-5 — Gatekeeper protects against rate-limit abuse.** A burst of Anthropic API calls beyond the configured `requests_per_minute` is queued (FIFO), not dropped or crashed (constitution §4.3).
- **SC-6 — QLoRA training succeeds and emits VRAM ratio.** A tiny QLoRA run completes; the report shows train-VRAM ≈ 3–5× inference-VRAM and discusses NF4 + Paged Optimizers.

---

## 6. Assumptions, Dependencies, Constraints

### 6.1 Assumptions
- A-1. The student's machine has at minimum: 16 GB RAM, ≥ 50 GB free disk (assignment §6.1 Do). Exact values are profiled at runtime, not assumed in code.
- A-2. Outbound internet access is available for Hugging Face downloads and Anthropic API calls.
- A-3. **Anthropic** is the only third-party provider invoked at runtime (Q2 resolved). Per-token pricing for the selected Anthropic model is published and recorded in `config/api_pricing.json` with the capture date.
- A-4. The course timeline allows ≈ 6.5–11 wall-hours, 2–3 active hours of focused work (assignment §11.5 — Realistic Time Estimation table); the QLoRA extension consumes the §5.7 budget.
- A-5. The two named target models (Llama-3-8B, Qwen-7B-Q4) are the *initial* contents of `config.target_models`; additional models can be appended via the JSON array without code changes.

### 6.2 Dependencies (runtime / dev)
| Layer | Dependency | Why |
|-------|------------|-----|
| Runtime | `transformers`, `accelerate`, `torch` (CPU build acceptable) | Direct back-end (via `AutoModelForCausalLM`) and tokenization. |
| Runtime | `airllm` | AirLLM back-end. |
| Runtime | `safetensors` | Layer shard format (per L08 §4.3, §4.4). |
| Runtime | `huggingface_hub` | Model download. |
| Runtime | `psutil`, `pynvml` (optional) | RAM / VRAM sampling. |
| Runtime | `anthropic` | API back-end (sole third-party provider). |
| Runtime | `peft`, `bitsandbytes` | QLoRA fine-tune extension (NF4, LoRA adapters). |
| Runtime | `datasets` | Tiny fine-tune dataset loading. |
| Runtime | `matplotlib`, `pandas` | Charts and tabular results. |
| Runtime | `pydantic` or `dataclasses` | Typed configs. |
| Dev | `uv` | Sole package manager (constitution §7.4). |
| Dev | `ruff` | Linter (constitution §6.1). |
| Dev | `pytest`, `pytest-cov` | TDD harness + coverage (constitution §5). |
| Dev | `hypothesis` (optional) | Property-based tests for measurement math. |

### 6.3 Constraints
- C-1. **No business logic in CLI / GUI layers** — constitution §3.1.
- C-2. **All outbound calls through `ApiGatekeeper`** — constitution §4.
- C-3. **Files ≤ 150 LOC** — constitution §2.2.
- C-4. **No hard-coded values** — constitution §6.2. Includes target model IDs (FR-2a), Anthropic model name and prices, cloud GPU hourly rate, electricity price, all wattage assumptions.
- C-5. **`uv` only** — constitution §7.4.
- C-6. **Assignment guardrails (§6.2 Don't):**
  - MUST NOT pick a 70B-class model with no chance of running even under AirLLM (Q1 resolved — capped at 8B-class).
  - MUST NOT commit HF token or Anthropic API key.
  - MUST NOT present raw numbers without analysis / charts / theory linkage.
  - MUST NOT skip economic analysis.
  - MUST NOT balloon scope into a "graduation project".
- C-7. **Plumbing-first rule (FR-PT-1..3).** No oversized run may execute until a plumbing test has passed on the same install.
- C-8. **`AutoModel` initialization rule (NFR-16).** All model loading MUST use the generic `transformers.AutoModel*` factories — see PLAN ADR-009.

### 6.4 Out-of-scope
- O-1. Multi-GPU sharded inference (NCCL, tensor parallel).
- O-2. Production HTTP serving with autoscaling — single-process local execution only.
- O-3. Fine-grained custom CUDA kernels (the report *discusses* CUDA / PTX / SASS per L08 §2.2 but does not author kernels).
- O-4. A user-facing GUI — CLI + notebook + README only.
- O-5. Multi-tenant authentication.
- O-6. **Other providers in the runtime economic comparison** — Anthropic is the sole runtime provider (Q2). OpenAI / others may be discussed in prose but are not implemented in `backends/api_backend.py`.

---

## 7. Milestones & Timeline

> Time estimates derived from assignment §11.5 (the "Realistic Time Estimation" appendix).
> Calendar dates assume kick-off **2026-06-26** (today); each milestone has wall-clock and active-hours estimates.

| ID | Milestone | Active hrs | Wall hrs | Target date | Exit criterion |
|----|-----------|------------|----------|-------------|----------------|
| **M0** | **Planning approved** | — | — | 2026-06-26 | PRD v1.10 + PLAN v1.10 + TODO v1.10 reviewed and approved by user. |
| **M1** | **Infra + Setup ready** | 15 min | 1.5–3 h | 2026-06-27 | `uv` env, `pyproject.toml`, `ruff` + `pytest` configured, `.env-example`, `config/*.json` scaffolds (incl. `target_models` array + `plumbing_test_model`), `src/` and `tests/` skeletons, hardware profiler runs end-to-end. |
| **M2a** | **Plumbing test green** | 10 min | 0.5–1 h | 2026-06-27 | `run-plumbing-test` succeeds end-to-end on the small Q2 model; manifest captured. |
| **M2b** | **Baseline captured** | ≈ 30 min | ≈ 1–2 h | 2026-06-28 | Direct back-end run on each oversized target model; failure / extreme latency captured in `results/baseline_*`. |
| **M3** | **AirLLM + Quantization sweep done** | 30–45 min | 3–5 h | 2026-06-29 | For every model in `config.target_models`: ≥ 3 quantization levels measured; results CSVs + charts generated. |
| **M4** | **Economic analysis + 3-curve plot** | 20–30 min | 1–1.5 h | 2026-06-30 | `figures/break_even.png` contains On-Prem + Anthropic + Cloud GPU curves with marked crossovers; assumptions table in README. |
| **M5** | **§5.7 QLoRA fine-tune extension** | 20–30 min | 0.5–1 h | 2026-06-30 | QLoRA fine-tune completes; train/inference VRAM ratio reported; `docs/PRD_qlora.md` exists and is approved. |
| **M6** | **README report assembled** | ≈ 1 h | 1–1.5 h | 2026-07-01 | README contains every required section, every plot/table embedded, theory tied to data, QLoRA section present. |
| **M7** | **Quality gates green** | — | 0.5 h | 2026-07-01 | Ruff = 0, coverage ≥ 85 %, file-size check = 0 violations, no secrets, `uv.lock` committed. |
| **M8** | **Submission tag** | — | — | 2026-07-01 | Git tag `v1.00`, push, deliver. |

Total: 6.5–11 wall hours, ≈ 2–3 active hours (matches assignment §11.5).

---

## 8. KPIs Recap (for grading visibility)

| KPI | Source | Target |
|-----|--------|--------|
| K1 Plumbing run | FR-PT-1..3 | passes before any oversized run |
| K2 Oversized AirLLM runs | G2 / FR-5 / FR-2a | success for each target in `config.target_models` |
| K3 Quantization levels per model | G4 / FR-7 | ≥ 3 |
| K4 TTFT repetitions | FR-9 | ≥ 5 |
| K5 TPOT measurement | FR-9 | over ≥ 64 generated tokens |
| K6 Memory sampling rate | FR-9 | ≥ 2 Hz |
| K7 Break-even curves | G5 / FR-14 | 3 (On-Prem, Anthropic, Cloud GPU) with crossover labels |
| K8 QLoRA VRAM ratio | G6 / FR-21 | train vs inference reported |
| K9 Ruff violations | NFR-1 | 0 |
| K10 Test coverage | NFR-2 | ≥ 85 % |
| K11 Files > 150 LOC | NFR-3 | 0 |
| K12 Secrets in code | NFR-5 | 0 |
| K13 `pip` usages | NFR-6 | 0 |
| K14 Hard-coded model IDs | NFR-4 / FR-2a | 0 (all in `config.target_models`) |

---

## 9. Resolved Decisions (was: Open Questions)

| # | Decision | Where applied |
|---|----------|---------------|
| **D-1** | Target models = **Llama-3-8B** and **Qwen-7B-Q4**. **No 70B** (assignment §6.2 Don't). Defined as JSON array `config.target_models[]`; SDK iterates dynamically. | FR-2a, FR-2b, C-4, C-6; PLAN §6.4; ADR-013 |
| **D-2** | Sole third-party API provider for runtime economic comparison = **Anthropic**. | FR-6, FR-12, A-3, O-6; PLAN §6.5; ADR-011 |
| **D-3** | §5.7 extension = **QLoRA training**. Mandates `docs/PRD_qlora.md` per constitution §1.3. | FR-20, FR-21, FR-22, G6, K8, M5; TODO §7 (mandatory); PLAN ADR-014 |
| **D-4** | Break-even chart **MUST** include the **Cloud GPU curve** (3 curves total). | FR-13a, FR-14, K7, G5; PLAN §6.4 (`include_cloud_curve: true`); ADR-012 |

Additionally adopted as cross-cutting rules:
| Rule | Where applied |
|------|---------------|
| Mandatory pipeline plumbing test before oversized runs | FR-PT-1..3, G1, K1, SC-0, M2a; PLAN ADR-010; TODO M2a |
| Use generic `AutoModel` factories for initialization (Class-mismatch prevention) | NFR-16, C-8; PLAN ADR-009; PLAN §3.3 |

---

## 10. Glossary

| Term | Meaning (per L08) |
|------|-------------------|
| **TTFT** | Time To First Token — wall time from request submission to first output token. Dominated by Prefill stage. |
| **TPOT / ITL** | Time Per Output Token / Inter-Token Latency — average ms per generated token after the first. Dominated by Decode stage. |
| **Prefill** | Inference stage 1: processes the full prompt in parallel; compute-bound (GEMM, FLOPS). |
| **Decode** | Inference stage 2: generates one token at a time, sequential; memory-bound (GEMV, VRAM bandwidth). |
| **KV Cache** | Per-layer cache of Key/Value projections built during Prefill, reused during Decode. |
| **VRAM** | Video RAM on the GPU. Often the binding capacity constraint for LLMs. |
| **Roofline Model** | Visualization showing whether a workload is limited by compute or by memory bandwidth (assignment §3). |
| **AirLLM** | Library that runs giant LLMs layer-by-layer on CPU (or low-VRAM GPU) using `mmap` + virtual-memory paging analogy. |
| **Quantization** | Reducing weight precision (FP32 → FP16 → FP8 → Q4 → Q2 / NF4) to shrink memory and (sometimes) speed up compute. |
| **QLoRA** | Quantization + LoRA fine-tuning (Dettmers et al., NeurIPS 2023); uses 4-bit NormalFloat (NF4) + Double Quantization + Paged Optimizers. |
| **LoRA** | Low-Rank Adaptation: freeze base weights `W₀`, train tiny matrices `A, B` such that `W = W₀ + BA`. |
| **OLoRA** | LoRA with orthonormal initialization via QR decomposition (Büyükakyüz, 2024). |
| **SafeTensors** | Safe, zero-code, `mmap`-friendly tensor file format; preferred over pickle-based `pytorch_model.bin`. |
| **GGUF** | GGML-derived single-file format combining weights + metadata; Ollama's native format. |
| **API Gatekeeper** | Centralized component through which all external API calls flow; enforces rate limits + queueing (constitution §4). |
| **SDK** | Single entry point for all business logic; CLI / GUI / notebooks call only the SDK (constitution §3.1). |
| **Building Block** | Module defined by explicit Input / Output / Setup contracts (constitution §15). |
| **AutoModel** | `transformers.AutoModel` / `AutoModelForCausalLM` — generic factory that resolves the correct concrete model class from a model's `config.json`, preventing Class-mismatch errors. |
| **Plumbing test** | A small-model + Q2 end-to-end run executed before any oversized run, used to verify download / mmap / sampler / manifest plumbing. |

---

## 11. Approval

This document **MUST** be approved by the user before `docs/PLAN.md` and `docs/TODO.md` are considered binding, and before any code is written (constitution §1.5).

> Approval status: ☐ Pending user review of revision v1.10.
