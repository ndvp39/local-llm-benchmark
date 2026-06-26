# TODO — Task List

> **Document type:** Tasks Document (SDLC Phase 2 deliverable).
> **Project:** `on_prem_llm_lab`.
> **Companions:** `docs/PRD.md` v1.10, `docs/PLAN.md` v1.20.
> **Source authority:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00.
> **Document version:** 1.20 — 2026-06-26.
> **Status convention:** `[ ]` not started · `[~]` in progress · `[x]` done.
> **DoD:** Every task has a Definition of Done; no task is `[x]` until DoD is met.

> **Changelog (v1.10 → v1.20).** Renamed `HardwareProfiler` → `HardwareScanner` everywhere. Expanded **T-1.13 / T-1.14** to cover the three write side-effects of ADR-015 (inject `config/setup.json.hardware_constraints`, patch placeholder pair in `docs/PRD.md` + `README.md`). New tasks **T-1.16** (`init_env.py` bootstrap script — ADR-016) and **T-1.17** (placeholder markers added to PRD + README). Dedicated PRD list §7 gains **`PRD_hardware_scanner.md`** as RECOMMENDED. M1 §1.E section retitled. Existing tasks renumber-free (T-1.15 stays for AutoModel factory).
>
> **Changelog (v1.00 → v1.10).** New milestone **M2a — Plumbing test green** (was rolled into M2). New tasks for the **`AutoModel*`** initialization rule (`T-1.15`, `T-2.0`). New tasks for the **dynamic `target_models[]` JSON array** (`T-1.9`-revised, `T-3.5`-revised). API provider locked to **Anthropic** (`T-4.2`, `T-4.3` revised). §5.7 extension locked to **QLoRA training** (`T-5.3` mandatory, alternatives removed; **`T-5.0`** added to author `docs/PRD_qlora.md` BEFORE implementation). Break-even chart now **3 curves** including Cloud GPU (`T-4.4` revised). Dedicated PRD list (§7) updated: `PRD_qlora.md` is now MANDATORY (was conditional).

---

## 0. How to use this file

- Tasks are grouped by milestone (M0 → M8 — see PRD §7).
- Each task carries an **owner** placeholder (`@student` by default) and a **DoD**.
- Infra tasks (M1) are mandatory by the constitution and **MUST** be done before any business code.
- Tasks marked **(blocks code)** must complete before the next milestone may start.
- Tasks marked **🔒 constitution** are non-negotiable rules from `SOFTWARE_PROJECT_GUIDELINES.md`.
- Tasks marked **🆕 v1.10** are new in this revision.

---

## M0 — Planning approved · target 2026-06-26

- [x] **T-0.1** Write `docs/PRD.md`. — **owner:** @architect — **DoD:** present at `docs/PRD.md`, integrates assignment §§1–11 + L08 lecture; v1.10 published.
- [x] **T-0.2** Write `docs/PLAN.md`. — **owner:** @architect — **DoD:** SDK, Gatekeeper, Mixins, ADRs, C4 diagrams, folder layout; v1.10 published.
- [x] **T-0.3** Write `docs/TODO.md` (this file). — **owner:** @architect — **DoD:** all milestones decomposed; infra tasks listed; dedicated-PRD list present; v1.10 published.
- [ ] **T-0.4** User reviews and approves PRD v1.10 / PLAN v1.10 / TODO v1.10. **(blocks code)** — **owner:** @user — **DoD:** explicit written approval in chat.

---

## M1 — Infra + Setup ready · target 2026-06-27 · (blocks code)

> Mandatory infrastructure per constitution §§5, 6, 7. None of this is optional.
> **Execution order (revised 2026-06-26 by user):** T-1.1 → **T-1.10** → **T-1.2** → rest. Rationale: the hatchling wheel target in `pyproject.toml` needs `src/on_prem_llm_lab/` to exist, otherwise `uv sync` fails. IDs and DoDs are unchanged; only sequencing moves T-1.10 (skeleton) ahead of T-1.2 (lock/sync).

### 1.A — Package + environment 🔒 constitution §7.4
- [x] **T-1.1** Create `pyproject.toml` (project metadata, deps, ruff config, coverage config). — **DoD:** `uv sync` succeeds; `[project]`, `[tool.ruff]`, `[tool.coverage.run]`, `[tool.coverage.report]` sections present; `fail_under = 85`. Dependencies include `anthropic` (ADR-011), `peft`, `bitsandbytes`, `datasets` (ADR-014). **Done 2026-06-26:** file present at repo root; TOML syntax validated via `uv run` (17 runtime deps, 4 dev deps, ruff rule set & `fail_under=85` confirmed). Full `uv sync` will be exercised in T-1.2 once the `src/on_prem_llm_lab/` skeleton exists (T-1.10) — the hatch wheel target needs the package directory.
- [x] **T-1.2** Initialize `uv` and commit `uv.lock`. — **DoD:** `uv.lock` exists in the repo root; CI step `uv sync --frozen` passes. **Done 2026-06-26:** `uv lock` resolved 112 packages; `uv sync` installed 90 wheels into `.venv/` (first run ~1.5 GB download incl. torch / transformers / accelerate / bitsandbytes / scipy / pyarrow); `uv sync --frozen` re-verifies in 16 ms; `uv run python -c "import on_prem_llm_lab"` walks every one of the 27 modules cleanly. Two install issues hit and resolved — both documented in `docs/prompts_book.md` §2: (a) PEP 735 `[dependency-groups]` vs legacy `[project.optional-dependencies]` for `tool.uv.default-groups`; (b) Hatchling requires `README.md` to exist for `[project].readme` even on editable installs (placeholder README created).
- [x] **T-1.3** Ban forbidden tools. — **DoD:** repo grep finds zero occurrences of `pip install`, `python -m`, `virtualenv`, `venv` in source/scripts/docs; documented in `docs/prompts_book.md`. **Done 2026-06-26:** `tools/check_forbidden_tools.py` (96 LOC, ruff-clean) ships in two modes — default (CI gate over `src/`, `tests/`, `tools/`, `init_env.py`, `pyproject.toml`, `*.sh`, `scripts/**`, `.github/workflows/*.yml`) = **43 files scanned, 0 errors**; advisory `--include-docs` sweep surfaces 18 known constitution-discussion mentions in PRD/PLAN/TODO/prompts_book.md (not invocations). Per-line escape hatch `# ALLOW-FORBIDDEN: <reason>` supported. Full rationale in `docs/prompts_book.md` §4.

### 1.B — Linting + tests 🔒 constitution §§5, 6.1
- [x] **T-1.4** Configure `ruff` with rule set `["E","F","W","I","N","UP","B","C4","SIM"]`, `ignore = ["E501"]`, `target-version = "py310"` (or 3.11). — **DoD:** `uv run ruff check src/ tests/` returns 0 violations. **Done 2026-06-26:** Ruff config was already declared in `pyproject.toml` v1.00 (T-1.1) matching constitution §6.1 verbatim — rule set `["E","F","W","I","N","UP","B","C4","SIM"]`, `ignore = ["E501"]`, `target-version = "py311"`, `line-length = 100`. Verified with ruff 0.15.20: `uv run ruff check src/ tests/` → **All checks passed!** Also verified extended scope `uv run ruff check src/ tests/ tools/` → All checks passed (42 files); `uv run ruff format --check src/ tests/ tools/` → 42 files already formatted (one initial formatter nit on `tools/check_forbidden_tools.py` auto-fixed via `ruff format`).
- [ ] **T-1.5** Configure `pytest` + `pytest-cov` with `source = ["src"]`, `omit = ["src/main.py", "*/tests/*", "src/**/gui/*"]`, `fail_under = 85`. — **DoD:** `uv run pytest` runs (initially with placeholder tests) and prints coverage line.
- [ ] **T-1.6** Add `tools/check_file_size.py` (fails build if any `src/**/*.py` or `tests/**/*.py` exceeds 150 LOC, ignoring blank/comment-only lines). — **DoD:** script returns non-zero on a fixture file > 150 LOC; CI step wired.

### 1.C — Security + config scaffolding 🔒 constitution §§4.2, 6.2, 6.3, 6.4
- [x] **T-1.7** Create `.gitignore` (incl. `.env`, `*.pem`, `*.key`, `credentials.json`, `results/logs/`, AirLLM shard dir). — **DoD:** all entries present. **Done 2026-06-26 (advanced out of order to enable first commit + push):** comprehensive `.gitignore` at repo root — Python bytecode, `.venv/`, secret patterns (`.env`, `*.pem`, `*.key`, `credentials.json`, `*_secret*`), test/lint artifacts (`coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`), editor noise, `results/logs/*` (keeps `.gitkeep`), `airllm_shards/` + `D:/airllm_shards/` per ADR-005, `*.bak` per ADR-015 (HardwareScanner backups), HF cache spillover, Jupyter checkpoints.
- [ ] **T-1.8** Create `.env-example` with placeholders: `HF_TOKEN=`, `ANTHROPIC_API_KEY=`. **No `OPENAI_API_KEY`** entry (ADR-011). — **DoD:** committed; no real secrets present.
- [ ] **T-1.9** 🆕 v1.10 Create `config/setup.json`, `config/rate_limits.json`, `config/api_pricing.json`, `config/logging_config.json` per PLAN §6, each with `"version": "1.00"`. **`setup.json` MUST include the `target_models` JSON array (Llama-3-8B fp16, Qwen-7B-Q4) and the `plumbing_test_model` entry (small + Q2)** (ADR-013, ADR-010). **`api_pricing.json` MUST list only Anthropic** (ADR-011). **`setup.json.economic.cloud_gpu.include_in_chart` MUST default to `true`** (ADR-012). — **DoD:** files validate against `shared/config.py` reader (added in M3).

### 1.D — Source skeleton 🔒 constitution §§1.4, 13
- [x] **T-1.10** Create folder layout from PLAN §8 with empty `__init__.py` files. **Includes** `services/plumbing_test_runner.py`, `services/qlora_trainer.py`, `shared/automodel_factory.py`. — **DoD:** `python -c "import on_prem_llm_lab"` works under `uv run`. **Executes before T-1.2 per reorder note above.** **Done 2026-06-26:** 33 src files (6 subpackages: sdk, services, backends, mixins, shared, cli + root + constants) and 8 tests files (unit/, integration/, conftest.py) created. Top-level dirs added: `data/`, `results/` (+ `logs/`), `figures/`, `notebooks/`, `tools/` (each with `.gitkeep`). Structural import verified — every one of the 27 modules under `on_prem_llm_lab` imports cleanly. `uv run`-level verification will be re-confirmed at the end of T-1.2 (uv sync).
- [ ] **T-1.11** Add `src/on_prem_llm_lab/shared/version.py` with `__version__ = "1.00"`. — **DoD:** importable; unit-tested.
- [ ] **T-1.12** Add stub `OnPremLlmSDK` class (constructor only, no methods yet). — **DoD:** importable; unit-tested for construction.

### 1.E — HardwareScanner with config + doc injection 🆕 v1.20 · ADR-015
- [ ] **T-1.13** 🆕 v1.20 Implement `services/hardware_scanner.py` (TDD; renamed from `hardware_profiler.py`). — **DoD:** Test-first: returns typed `HardwareScanResult` with detected CPU/RAM/GPU/VRAM/disk/OS/Python; cross-platform Win/Linux; ≤ 150 LOC; ≥ 1 happy + 1 error test per side-effect. **Side-effects (ADR-015):** (1) injects scan payload into `config/setup.json.hardware_constraints` via atomic `write tmp → os.replace` and optional `.bak`; (2) replaces content between `<!-- HARDWARE_SPECS_PLACEHOLDER:START -->` / `<!-- HARDWARE_SPECS_PLACEHOLDER:END -->` markers in each path listed in `config.init.doc_targets` (default: `docs/PRD.md`, `README.md`) with a formatted Markdown table; (3) records per-file write status in `HardwareScanResult.write_receipts`. Idempotent (re-scan yields identical bytes modulo `captured_at`). Missing files / missing marker pair → `skipped`, never silent.
- [ ] **T-1.14** 🆕 v1.20 Wire `OnPremLlmSDK.scan_hardware()`, `OnPremLlmSDK.initialize_environment()` (composes scanner + future bootstrap steps), and CLI subcommand `initialize`. — **DoD:** `uv run on-prem-llm initialize` runs the full scan + inject + patch flow, prints a per-file summary, exits 0 on success / non-zero on any per-file failure.

### 1.F — `AutoModel*` factory helper 🆕 v1.10 · ADR-009
- [ ] **T-1.15** 🆕 v1.10 Implement `shared/automodel_factory.py` — a ≤ 50 LOC helper that resolves dtype / device_map per quantization level and returns `AutoModelForCausalLM` + `AutoTokenizer` instances. — **DoD:** ≤ 150 LOC (target ≤ 50); unit-tested with mocked `transformers`; happy + UnsupportedQuantization error paths.

### 1.G — `init_env.py` bootstrap + placeholder seeding 🆕 v1.20 · ADR-016
- [ ] **T-1.16** 🆕 v1.20 Create `init_env.py` at the repo root (`uv run init_env.py`). — **DoD:** ≤ 30 LOC; delegate-only — parses args, builds `OnPremLlmSDK`, calls `sdk.initialize_environment()`, prints summary, exits non-zero on failure. **Precondition guard everywhere:** every other SDK method (`run_plumbing_test`, `run_baseline`, `run_airllm`, `run_sweep`, `run_qlora_finetune`, `economic_analysis`, `assemble_readme`) raises `EnvironmentNotInitializedError` when `config.hardware_constraints` is `null` or older than `init.max_age_hours`, with a remediation hint pointing back to `uv run init_env.py`. Integration test `tests/integration/test_env_init_required.py` covers both "missing" and "stale" cases. Integration test `tests/integration/test_hardware_scanner_writes.py` covers the three write side-effects from T-1.13.
- [ ] **T-1.17** 🆕 v1.20 Add the paired placeholder markers `<!-- HARDWARE_SPECS_PLACEHOLDER:START -->` / `<!-- HARDWARE_SPECS_PLACEHOLDER:END -->` to **`docs/PRD.md`** (between §1.3 and §1.4) and **`README.md`** (under "Hardware profile" section once the README template lands in M6 — placeholder block seeded now so `init_env.py` has a target on first install). — **DoD:** both files contain the marker pair exactly once; markers preserved verbatim across scanner runs.

---

## M2a — Plumbing test green · target 2026-06-27 · 🆕 v1.10 · ADR-010 · (blocks M2b)

> Verifies the full pipeline (download → AirLLM mmap → metric collection → manifest) on a **small** model with **Q2** quantization before any oversized run. PRD FR-PT-1..3, G1, K1, SC-0.

- [ ] **T-2a.1** 🆕 v1.10 Implement `services/plumbing_test_runner.py`. — **DoD:** ≤ 150 LOC; runs the four stages defined in PRD §3.10; writes `results/plumbing_<timestamp>.json` with per-stage status; raises a structured `PlumbingStageError` on failure.
- [ ] **T-2a.2** 🆕 v1.10 Wire `OnPremLlmSDK.run_plumbing_test()` and CLI `run-plumbing-test`. — **DoD:** `uv run on-prem-llm run-plumbing-test` exits 0 on success, non-zero with a stage-specific message on failure.
- [ ] **T-2a.3** 🆕 v1.10 Integration test `tests/integration/test_plumbing_test_runner.py`. — **DoD:** runs against `plumbing_test_model` (TinyLlama-Q2 by default) and asserts all four stages reach `ok`.
- [ ] **T-2a.4** 🆕 v1.10 Add `SDK.run_sweep` precondition: requires a current plumbing manifest unless `skip_plumbing=True`. — **DoD:** unit test asserts `PlumbingNotRunError` raised when manifest absent; integration test `tests/integration/test_sweep_requires_plumbing.py` confirms.
- [ ] **T-2a.5** 🆕 v1.10 Execute the plumbing test on the student's actual machine and commit the resulting manifest under `results/`. — **DoD:** real manifest committed; status `overall: ok`.

---

## M2b — Baseline captured · target 2026-06-28

### 2b.A — Backends infra
- [ ] **T-2.0** 🆕 v1.10 Implement static AST guard `tests/unit/test_backends/test_init_uses_automodel.py` that blocks imports of concrete `*ForCausalLM` classes in `backends/` (ADR-009 enforcement). — **DoD:** test fails on a fixture that imports `LlamaForCausalLM`; passes on real code.
- [ ] **T-2.1** Define `backends/base.py` (`InferenceBackend` ABC, `BackendRunResult` dataclass, `Quant` enum). — **DoD:** ≤ 150 LOC; unit tests for the dataclass.
- [ ] **T-2.2** Implement `mixins/timing_mixin.py`. — **DoD:** isolated unit tests; computes TTFT and TPOT correctly on a faked generator.
- [ ] **T-2.3** Implement `mixins/memory_sampling_mixin.py`. — **DoD:** background thread sampling at configurable Hz, thread-safe stop, returns peak RAM/VRAM; unit-tested with a fake sampler.
- [ ] **T-2.4** Implement `mixins/manifest_logging_mixin.py`. — **DoD:** writes a JSON manifest with seed/config snapshot/git hash; includes `target_label` and `model_id`.
- [ ] **T-2.5** Implement `mixins/energy_accounting_mixin.py`. — **DoD:** computes Wh from configured wattage × wall_s.

### 2b.B — API Gatekeeper 🔒 constitution §4
- [ ] **T-2.6** Implement `shared/gatekeeper.py` (`ApiGatekeeper`, `RateLimitConfig`, `QueueStatus`, `GatekeeperError`). — **DoD:** FIFO queue, max depth from config, retry/backoff from config, structured JSON logging; ≤ 150 LOC across (split into helper file if needed).
- [ ] **T-2.7** Integration test `tests/integration/test_gatekeeper_queue.py` — burst beyond limit, assert FIFO, no crash, queue drains.
- [ ] **NOTE:** Author **`docs/PRD_api_gatekeeper.md`** alongside this work (see §7 below).

### 2b.C — Direct back-end + baseline run
- [ ] **T-2.8** Implement `backends/direct_backend.py` using **`AutoModelForCausalLM`** via `shared/automodel_factory.py` (ADR-009). Accepts `target_label` + `quantization`. — **DoD:** ≤ 150 LOC; unit tests with a tiny stub model; static AST test (T-2.0) passes.
- [ ] **T-2.9** Implement `services/benchmark_runner.py` (orchestrates load → sample-on → generate → sample-off → unload → manifest). — **DoD:** unit + integration tests.
- [ ] **T-2.10** Wire `SDK.run_baseline(target_label, ...)` and `cli run-baseline`. — **DoD:** executes against **each** oversized model in `config.target_models`; failure / extreme latency captured under `results/baseline_<label>_*`; logged, not silently swallowed (PRD SC-1).

---

## M3 — AirLLM + Quantization sweep · target 2026-06-29

### 3.A — AirLLM back-end (**central mechanism** — separate PRD required, see §7)
- [ ] **T-3.1** Implement `backends/airllm_backend.py`. Honors `airllm.layer_shards_saving_path` from config; validates free disk via `HardwareProfiler` precondition (ADR-005). Uses `AutoTokenizer` (ADR-009). — **DoD:** ≤ 150 LOC; unit-tested with mocked AirLLM; integration-tested on the plumbing-test model.
- [ ] **T-3.2** Author **`docs/PRD_airllm_integration.md`** (see §7).

### 3.B — Quantization (**central mechanism** — separate PRD required, see §7)
- [ ] **T-3.3** Add quantization adapter logic per back-end (FP16 / Q8 / Q4 mandatory; Q2 covered by plumbing + aggressive end; NF4 reserved for QLoRA M5). — **DoD:** SweepRunner can iterate the matrix; each backend either supports a level or raises `UnsupportedQuantization`.
- [ ] **T-3.4** Author **`docs/PRD_quantization.md`** (see §7).

### 3.C — Sweep runner & quality matrix
- [ ] **T-3.5** 🆕 v1.10 Implement `services/sweep_runner.py` — **iterates over `config.target_models` (ADR-013) × quantizations × backends**, with N repeats and warm-up, and asserts plumbing precondition (ADR-010). — **DoD:** ≤ 150 LOC; writes `results/sweep_<timestamp>.csv` + manifest; rows carry `target_label`.
- [ ] **T-3.6** Implement quality-matrix harness — same prompt, all quantizations, **per target model**, side-by-side outputs → `results/quality_matrix.md`. — **DoD:** referenced from README.

### 3.D — Plots
- [ ] **T-3.7** Add `services/plot_style.py` (style constants only) and chart helpers for: TTFT vs quantization, TPOT vs quantization, Throughput vs quantization, peak RAM/VRAM vs quantization, Roofline-style chart per assignment §3 — **all charts faceted by `target_label`**. — **DoD:** PNGs land in `figures/`; ≤ 150 LOC each helper file.

### 3.E — Config & shared
- [ ] **T-3.8** Implement `shared/config.py.load(config_dir)` and Pydantic/dataclass schemas — validates `version` on each JSON file; validates `target_models[]` schema (unique labels, valid quantizations). — **DoD:** unit tests for happy + version-mismatch + duplicate-label + invalid-quantization paths.
- [ ] **T-3.9** Implement `shared/logging_setup.py` (JSON-line file handler). — **DoD:** unit tests; honored by Gatekeeper.

---

## M4 — Economic analysis & 3-curve break-even · target 2026-06-30

- [ ] **T-4.1** Implement `services/economic_analyzer.py` — computes per-request Anthropic API cost, On-Prem amortized cost, **Cloud GPU per-request cost** (ADR-012), and the break-even crossover(s). — **DoD:** ≤ 150 LOC; unit-tested with synthetic inputs; honors PRD FR-12, FR-13, FR-13a, FR-14.
- [ ] **T-4.2** 🆕 v1.10 Implement `backends/api_backend.py` — **Anthropic Messages API only** (ADR-011), routed through Gatekeeper. — **DoD:** ≤ 150 LOC; unit-tested with mocked `anthropic` client.
- [ ] **T-4.3** 🆕 v1.10 Update `config/api_pricing.json` with current published **Anthropic** prices and `captured_at` date; cite source URL in `docs/prompts_book.md` and in the README assumption table. — **DoD:** numbers reflect the date of capture; placeholders removed.
- [ ] **T-4.4** 🆕 v1.10 Generate `figures/break_even.png` with **three curves** (On-Prem, Anthropic, Cloud GPU) and labeled crossover points, plus the assumption table in README. — **DoD:** all three series present; assumptions visible (electricity rate, capex, lifetime, wattage, cloud hourly rate, Anthropic prices); reproducible from CSVs.
- [ ] **T-4.5** Author **`docs/PRD_economic_analysis.md`** (see §7).

---

## M5 — §5.7 QLoRA training extension · target 2026-06-30 · 🆕 v1.10 LOCKED

> Q3 resolved: §5.7 extension = **QLoRA fine-tune** on one of the target models. The two alternative options from v1.00 (extra metric, cross-model comparison) are dropped.

- [ ] **T-5.0** 🆕 v1.10 **(blocks T-5.1..T-5.4)** Author **`docs/PRD_qlora.md`** BEFORE implementation begins (constitution §1.3, ADR-014). Must include: theoretical background (NF4 / Double Quantization / Paged Optimizers per L08 §5.1; LoRA per L08 §7.3), I/O contract (`target_label`, dataset, `LoraConfig` → `QloraReport`), constraints (tiny dataset, VRAM budget), alternatives considered (full FT, plain LoRA), success criteria (training completes, train-VRAM ≈ 3–5× inference per L08 §7.1), test scenarios. — **DoD:** file exists; user reviewed; status approved.
- [ ] **T-5.1** Implement `services/qlora_trainer.py` (Building Block per ADR-014). — **DoD:** ≤ 150 LOC; uses `peft.LoraConfig`, `bitsandbytes` NF4; loads base model via `AutoModelForCausalLM` (ADR-009); unit-tested with mocked PEFT.
- [ ] **T-5.2** Wire `SDK.run_qlora_finetune(...)` and CLI `run-qlora`. — **DoD:** end-to-end run on the smallest target model completes; emits `results/qlora_<label>_<timestamp>.json` with train + inference VRAM peaks.
- [ ] **T-5.3** Generate `figures/qlora_vram_ratio.png` — bar chart of training VRAM vs inference VRAM for the chosen target. — **DoD:** chart present in README §QLoRA.
- [ ] **T-5.4** Author the README §"QLoRA fine-tune extension" with motivation, NF4 explanation, method, dataset description, results, and connection to L08 §7. — **DoD:** section present; cites `docs/PRD_qlora.md`.

---

## M6 — README assembled · target 2026-07-01

- [ ] **T-6.1** Implement `services/report_assembler.py` — interpolates AUTOGEN regions in `README.md` from `results/` and `figures/` (ADR-007). — **DoD:** ≤ 150 LOC; idempotent; golden-file integration test.
- [ ] **T-6.2** Author the README template skeleton — sections: Project intro, Hardware profile, Methodology, **Plumbing test write-up (🆕 v1.10)**, Baseline (failure narrative, per target), AirLLM rescue (per target), Quantization sweep (per target), Quality matrix, **Economic analysis (3-curve break-even — 🆕 v1.10)**, Theory linkage (Prefill/Decode/Roofline/Paging), **QLoRA extension section (🆕 v1.10)**, Reproducibility (commands), Configuration, License & Credits. — **DoD:** every PRD K1..K14 visible in the rendered README; assignment §8 README requirements all present.
- [ ] **T-6.3** Author a top-level `README.md` user-manual section per constitution §1.1 (Installation, Usage, Examples, Configuration Guide, Contribution Guidelines, License & Credits). — **DoD:** all six subsections present.

---

## M7 — Quality gates green · target 2026-07-01

- [ ] **T-7.1** `uv run ruff check src/ tests/` returns 0 violations. — **DoD:** clean.
- [ ] **T-7.2** `uv run pytest --cov` returns coverage ≥ 85 %. — **DoD:** `fail_under` enforced.
- [ ] **T-7.3** `uv run python tools/check_file_size.py` returns 0 violations. — **DoD:** all files ≤ 150 LOC.
- [ ] **T-7.4** Secrets scan (grep for likely token patterns + manual review). — **DoD:** zero matches; `.env-example` present with `HF_TOKEN` + `ANTHROPIC_API_KEY` only.
- [ ] **T-7.5** 🆕 v1.10 Grep guard: **no model-ID literals in `src/`** outside `shared/config.py`'s reader (ADR-013). — **DoD:** CI step returns 0.
- [ ] **T-7.6** 🆕 v1.10 Grep guard: **no concrete `*ForCausalLM` imports in `src/backends/`** (ADR-009). — **DoD:** static AST test (T-2.0) passes.
- [ ] **T-7.7** Repository checklist per constitution §16 — every checkbox ticked or explicitly waived in `docs/prompts_book.md`. — **DoD:** §16 checklist annotated.

---

## M8 — Submission · target 2026-07-01

- [ ] **T-8.1** Tag `v1.00` on the main branch. — **DoD:** annotated git tag; CHANGELOG entry (optional).
- [ ] **T-8.2** Final smoke test: clone clean, `uv sync`, `uv run on-prem-llm profile-hardware`, `uv run on-prem-llm run-plumbing-test`, `uv run on-prem-llm run-sweep --dry-run`. — **DoD:** all four commands succeed on a fresh checkout.
- [ ] **T-8.3** Submission per assignment §7 (GitHub repo URL). — **DoD:** repo accessible; README renders correctly on GitHub.

---

## 7. Dedicated PRDs to author (constitution §1.3 mandatory) — REVISED v1.10

> Central mechanisms and complex algorithms **MUST** receive their own PRD: theory background, requirements, I/O, constraints, alternatives considered, success criteria, test scenarios (constitution §1.3). Create these documents as the corresponding milestones begin — they are themselves mandatory deliverables.

| ID | File | Status | When | Why this needs its own PRD |
|----|------|--------|------|----------------------------|
| DP-1 | `docs/PRD_api_gatekeeper.md` | **MANDATORY** | M2b (with T-2.6) | Centralized rate-limit + queue mechanism; constitutionally mandated (§4), and the project's core safety net for outbound calls. |
| DP-2 | `docs/PRD_airllm_integration.md` | **MANDATORY** | M3 (with T-3.1) | Layer-by-layer execution + `mmap` + OS-paging analogy is the headline algorithm (L08 §8); requires theory, IO contract, fault modes, success criteria. |
| DP-3 | `docs/PRD_quantization.md` | **MANDATORY** | M3 (with T-3.3) | Multi-level bit-width policy (FP16 / Q8 / Q4 / Q2 / NF4); requires per-level acceptance thresholds and quality-cliff documentation (L08 §5). |
| DP-4 | `docs/PRD_benchmarking_methodology.md` | **MANDATORY** | M2b (with T-2.9) | TTFT / TPOT / Throughput / peak-memory measurement methodology — non-trivial, must define exactly what we count, how we warm up, repetitions, statistics. |
| DP-5 | `docs/PRD_economic_analysis.md` | **MANDATORY** | M4 (with T-4.5) | Break-even calculation across **three curves** (Anthropic API, On-Prem, Cloud GPU); assumptions table, formula derivation, sensitivity to electricity rate, lifetime, cloud hourly rate. |
| DP-6 | `docs/PRD_roofline_analysis.md` | **MANDATORY** | M3 (with T-3.7) | Roofline model that proves compute-bound vs memory-bound (assignment §3) — theoretical derivation referencing Prefill (GEMM) and Decode (GEMV) regimes (L08 §3). |
| DP-7 | **`docs/PRD_qlora.md`** | **🆕 v1.10 MANDATORY** | M5 (with T-5.0 — **blocks T-5.1**) | QLoRA fine-tune is the §5.7 extension (Q3) and is a central mechanism (NF4, Double Quantization, Paged Optimizers, LoRA adapters) — constitution §1.3 mandates a dedicated PRD; ADR-014 confirms boundary. |
| DP-8 | `docs/PRD_plumbing_test.md` | **🆕 v1.10 RECOMMENDED** | M2a (with T-2a.1) | The plumbing-first execution rule (ADR-010) is itself a non-trivial mechanism: defines per-stage contracts, failure remediation hints, manifest schema, integration into `run_sweep` precondition. May be waived if `PlumbingTestRunner` stays ≤ 150 LOC and is fully covered by tests + ADR-010. |
| DP-9 | `docs/PRD_report_assembler.md` | OPTIONAL | M6 (with T-6.1) | Auto-interpolation of README sections from results is a non-trivial mechanism; PRD recommended but may be waived if implementation stays ≤ 150 LOC and is fully covered by tests. |
| DP-10 | `docs/PRD_hardware_scanner.md` | **🆕 v1.20 RECOMMENDED** | M1 (with T-1.13) | HardwareScanner has three write side-effects across two file types (JSON config + Markdown docs) with atomic semantics and an invariant about who else may edit the placeholder regions (nobody). May be waived if `services/hardware_scanner.py` stays ≤ 150 LOC AND the integration test `test_hardware_scanner_writes.py` covers all four outcomes (ok / skipped-missing-file / skipped-missing-marker / fail). |

> v1.20 changes vs v1.10: added `PRD_hardware_scanner.md` (recommended). v1.10 changes vs v1.00: `PRD_qlora.md` is now MANDATORY (was conditional on selecting QLoRA — Q3 resolved that selection). New entry `PRD_plumbing_test.md` (recommended). All other entries unchanged in mandatory/optional status.

---

## 8. Mandatory infrastructure tasks recap (constitution gates)

> These items are repeated here for visibility; each maps to a task above.

| Gate | Constitution ref. | Task |
|------|-------------------|------|
| `uv` only (no `pip`, no `venv`, no `python -m`, no `virtualenv`) | §7.4 | T-1.1..T-1.3 |
| `uv.lock` committed | §7.4 | T-1.2 |
| `ruff check` returns 0 | §6.1 | T-1.4, T-7.1 |
| TDD with coverage ≥ 85 % | §5.2 | T-1.5, every implementation task, T-7.2 |
| Files ≤ 150 LOC | §2.2 | T-1.6, T-7.3 |
| No hard-coded values (incl. **target model IDs as JSON array** — 🆕 v1.10) | §6.2 | T-1.9 + every config consumer + T-7.5 |
| No secrets in code; `.env-example` committed (HF + Anthropic only — 🆕 v1.10) | §6.4 | T-1.7, T-1.8, T-7.4 |
| Versioning `1.00` on code + each JSON config | §7.1 | T-1.9, T-1.11, T-3.8 |
| All API calls via Gatekeeper | §4 | T-2.6, T-2.7, DP-1 |
| All business logic via SDK; CLI delegate-only | §3.1 | T-1.12, every SDK method binding |
| Prompts Book maintained | §7.3 | T-1.3, T-4.3, T-7.7 (`docs/prompts_book.md`) |
| **`AutoModel*` init only** (🆕 v1.10) | NFR-16 / ADR-009 | T-1.15, T-2.0, T-2.8, T-3.1, T-7.6 |
| **Plumbing-first** (🆕 v1.10) | FR-PT-1..3 / ADR-010 | M2a all tasks, T-2a.4 |
| **3-curve break-even chart** (🆕 v1.10) | FR-14 / ADR-012 | T-4.1, T-4.4 |
| **Anthropic-only API** (🆕 v1.10) | FR-6 / ADR-011 | T-1.8, T-4.2, T-4.3 |
| **`docs/PRD_qlora.md` before code** (🆕 v1.10) | §1.3 / ADR-014 | T-5.0 (blocks T-5.1..T-5.4) |
| **HardwareScanner auto-injection + atomic writes** (🆕 v1.20) | ADR-015 | T-1.13, T-1.17 |
| **`init_env.py` env-init-first** (🆕 v1.20) | ADR-016 | T-1.16; precondition guard wired into every other SDK method |

---

## 9. Phase boundaries (DoD per milestone)

- **M1 done when:** `uv sync` works, ruff clean, pytest runs, **HardwareScanner writes config + patches PRD + README placeholders, `init_env.py` runs end-to-end (`uv run init_env.py`), `config.hardware_constraints` is populated**, `automodel_factory.py` is in place.
- **M2a done when:** plumbing test passes on the student's machine and the manifest is committed under `results/`.
- **M2b done when:** baseline run on each oversized target is captured (success or documented failure); Gatekeeper integration test passes; AST guard for `AutoModel*` is green.
- **M3 done when:** sweep CSV exists with rows for every `(target_label × quantization)` combination on the AirLLM back-end; all sweep charts in `figures/`; PRD_airllm_integration + PRD_quantization + PRD_benchmarking_methodology + PRD_roofline_analysis exist.
- **M4 done when:** `figures/break_even.png` exists with **three labeled curves** (On-Prem, Anthropic, Cloud GPU) and marked crossovers; `PRD_economic_analysis.md` exists.
- **M5 done when:** `docs/PRD_qlora.md` is approved; QLoRA training completes; train/inference VRAM ratio chart present in README.
- **M6 done when:** README is fully assembled and renders correctly; all required sections per assignment §8 present; QLoRA + plumbing + 3-curve break-even sections all included.
- **M7 done when:** all six quality gates green (ruff, coverage, file-size, secrets scan, no-model-ID literals, no concrete `*ForCausalLM` imports); constitution §16 checklist annotated.
- **M8 done when:** `v1.00` tag pushed; smoke test passes from a fresh clone, including `run-plumbing-test`.

---

## 10. Approval

This document **MUST** be approved by the user — alongside PRD v1.10 and PLAN v1.10 — before any code is written (constitution §1.5).

> Approval status: ☐ Pending user review of revision v1.10.
