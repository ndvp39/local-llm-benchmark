# Prompts Book

> **Mandate.** Constitution §7.3 requires a Prompt Engineering Log: significant prompts used to build the project, with context, goal, received outputs, iterative refinements, and lessons learned. This file also doubles as the project's **decision diary** — any non-obvious issue we hit during build is captured here so future-me / a reviewer can reconstruct the *why*.

## Contents
- [1. Planning prompts (PRD / PLAN / TODO authoring)](#1-planning-prompts-prd--plan--todo-authoring)
- [2. Installation & environment issues (T-1.2)](#2-installation--environment-issues-t-12)
- [3. Provider pricing capture (M4)](#3-provider-pricing-capture-m4)
- [4. T-1.3 — Forbidden-tools check: scope decision](#4-t-13--forbidden-tools-check-scope-decision)
- [5. T-2a.5 prep — storage drive + AirLLM compatibility](#5-t-2a5-prep--storage-drive--airllm-compatibility)

---

## 1. Planning prompts (PRD / PLAN / TODO authoring)
> Captured retroactively. The actual prompts live in the conversation transcript with the LLM architect. Key inflection points:
> - **v1.00 → v1.10**: four open questions resolved (target models, Anthropic-only API, QLoRA extension, Cloud GPU curve) + plumbing test + `AutoModel*` rule.
> - **v1.10 → v1.20 (PLAN/TODO)**: HardwareScanner gains write side-effects (config + PRD + README) + `init_env.py` bootstrap script + env-init-first precondition guard.

---

## 2. Installation & environment issues (T-1.2)

### 2.1 PEP 735 `dependency-groups` vs legacy `[project.optional-dependencies]`

**Symptom.** First `uv sync` printed:
```
error: Default group `dev` (from `tool.uv.default-groups`) is not defined
in the project's `dependency-groups` table
```
…even though `[project.optional-dependencies]` had a `dev` group.

**Root cause.** `tool.uv.default-groups` references the **PEP 735** `[dependency-groups]` table, not the older PEP 621 `[project.optional-dependencies]`. They are distinct surfaces in modern uv.

**Fix.** Moved `dev` deps from `[project.optional-dependencies]` to a top-level `[dependency-groups]` table:
```toml
[dependency-groups]
dev = [
    "ruff>=0.6.0",
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "hypothesis>=6.108",
]

[tool.uv]
default-groups = ["dev"]
```

**Lesson.** When using `tool.uv.default-groups`, always declare the group under PEP 735's `[dependency-groups]`, not under `[project.optional-dependencies]`. The two tables can coexist (legacy + modern), but `default-groups` only sees PEP 735.

### 2.2 Hatchling refused to build — missing `README.md`

**Symptom.** Second `uv sync` failed during the editable wheel build:
```
OSError: Readme file does not exist: README.md
hint: This usually indicates a problem with the package or the build environment.
```

**Root cause.** `pyproject.toml` declares `readme = "README.md"` (PEP 621 metadata), and Hatchling validates that the file exists at build time. Our README is the *report*, scheduled for assembly in M6 (T-6.1) — it didn't exist yet during T-1.2.

**Fix.** Created a placeholder `README.md` at the repo root that points to `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` and notes that it will be overwritten by `services/report_assembler.py` in M6. After v1.20 of PLAN/TODO, this same placeholder also seeds the `<!-- HARDWARE_SPECS_PLACEHOLDER:START/END -->` block for `init_env.py` (ADR-015).

**Lesson.** When `[project].readme` points to a file, Hatchling enforces its existence at build time — even for editable installs. Either ship a placeholder from day one, or omit the `readme` key until M6. We chose the placeholder route because the README is a mandatory deliverable anyway (constitution §1.1 + assignment §7 — "the report MUST be the README").

### 2.3 Outcome
- `uv lock` → 112 packages resolved in 516 ms.
- `uv sync` → 90 wheels installed into `.venv/` (full project + dev group); first run downloaded ~1.5 GB (torch, transformers, accelerate, bitsandbytes, scipy, pyarrow, etc.).
- `uv sync --frozen` → re-verifies 90 packages in 16 ms (cache hit); this is the CI-shape command.
- `uv run python -c "import on_prem_llm_lab"` → walks every one of the 27 modules under the package successfully (sdk/, services/, backends/, mixins/, shared/, cli/ + root + constants).
- `bitsandbytes` installed cleanly on Windows for this venv; the `platform_system != 'Darwin'` marker correctly excludes it on macOS.
- `torch` resolved to a GPU-capable wheel; if a future run on a non-CUDA machine fails to find a backend, pin `torch` to the CPU index (`https://download.pytorch.org/whl/cpu`) in `pyproject.toml` and re-lock.

---

## 3. Provider pricing capture (M4)
> Stub — to be filled when `config/api_pricing.json` is populated against Anthropic's published prices. Record: source URL, capture date, exact in/out per-million-token rates, model id at the time of capture.

---

## 4. T-1.3 — Forbidden-tools check: scope decision

### 4.1 What the constitution forbids
Constitution §7.4 / ADR-001: `uv` is the sole package manager + task runner. **`pip install`, `python -m`, `virtualenv`, and the bare `venv` command are forbidden** in invocations across the project.

### 4.2 Why the check is execution-scope by default
The DoD for T-1.3 reads literally: *"repo grep finds zero occurrences of `pip install`, `python -m`, `virtualenv`, `venv` in source/scripts/docs."* A literal grep across `docs/**/*.md` and `README.md` matches **18** lines today — all of which **discuss the prohibition** rather than invoke the tools. Examples:

- `docs/PLAN.md:526` — *"Context. Constitution §7.4 forbids `pip`/`venv`/`virtualenv`/`python -m`."*
- `docs/PRD.md:159` — *"NFR-6 … `pip`, `venv`, `virtualenv`, `python -m` strictly forbidden."*
- `docs/TODO.md:45` — the DoD for T-1.3 itself.
- `docs/prompts_book.md` — this file's discussion of the rule.

### 4.3 Design decision
`tools/check_forbidden_tools.py` ships in two modes:

| Mode | Files scanned | Failure behavior | When to run |
|------|---------------|------------------|-------------|
| **Default (CI gate)** | `src/**/*.py`, `tests/**/*.py`, `tools/**/*.py`, `init_env.py`, `pyproject.toml`, `*.sh`, `scripts/**`, `.github/workflows/*.yml` | Exit 1 on any unallow-listed match. | Pre-commit, CI, before every PR. |
| **`--include-docs` (audit)** | Default set + `docs/**/*.md` + `README.md` | Same — reports each line. | Manual audit when constitution discussion text changes; expected to surface the doc-mentions. |

Rationale: the goal of the rule is to prevent *invocations*, not discussion. The markdown lines that document the rule are the very mechanism that enforces it — neutralizing them would weaken the constitution's visibility. Adding per-line `<!-- ALLOW-FORBIDDEN: constitution discussion -->` markers inside markdown table cells is also impractical (markdown comment support inside table cells is unreliable across renderers).

### 4.4 Per-line escape hatch
For exceptional cases in code (e.g., a docstring that absolutely must show a `pip` command as a counter-example), append `# ALLOW-FORBIDDEN: <reason>` to the line. The match is then reported as a `WARNING` (visible to reviewers) and does **not** fail the build. Use sparingly.

### 4.5 Today's result
- Default scan: **43 files scanned · 0 errors · 0 warnings.** Gate passes.
- `--include-docs` audit: **48 files scanned · 18 expected mentions** in PRD / PLAN / TODO / prompts_book.md. Tracked here so the next reviewer sees that the audit-mode noise is *known and intentional*.

### 4.6 If the user wants strict literal enforcement
The alternative is to add `<!-- ALLOW-FORBIDDEN: discussion -->` markers to every constitution-discussion line in PRD / PLAN / TODO / prompts_book.md (≈18 lines). Quick to do but visually noisy in tables. Pivot to that interpretation on user request.

---

## 5. T-2a.5 prep — storage drive + AirLLM compatibility

> Captured 2026-06-30 during preparation for the real-machine plumbing test run. Two distinct problems surfaced before any model could be downloaded; both fixes were committed together in `1f76064`.

### 5.1 Symptom — disk on C: too small for even one target model

First scan run on 2026-06-26 reported:

| Component | Value |
|-----------|-------|
| Disk free | **2.8 GB** at `C:\AI_Agents_MSC_course\HW5` |
| RAM total / available | 7.8 GB / 0.9 GB |

The plumbing-test model alone (`TinyLlama-1.1B-Chat-v1.0`) is ~2 GB in safetensors form; the two real target models (Llama-3-8B fp16, Qwen2-7B) would each need ~14–16 GB of HF cache **plus** an equivalent footprint of AirLLM per-layer shards. C: was non-starter.

### 5.2 Root cause #1 — HF cache + AirLLM shard path both defaulted to small drives

* `huggingface_hub` writes to `~/.cache/huggingface` by default — on this user that resolves to `C:\Users\ndvp3\.cache\huggingface`. There was no env-var or config field guiding it elsewhere.
* `airllm.layer_shards_saving_path` had been set in `config/setup.json` v1.00 to a placeholder (`D:/airllm_shards`) — but the disk-free measurement in `HardwareScanner` reads the *same* path, so on a machine without a `D:` drive it reported the C: free instead and the placeholder went un-noticed. ADR-005 already says shards live on a dedicated drive; we just hadn't validated the path against a real machine yet.

### 5.3 Fix #1 — D: drive added; config points everything there

The student attached a 1 TB external drive (NADAV D:) with 392 GB free. We:

1. Created `D:/AI_agents_course/airllm_shards` (per-layer shard output) and `D:/AI_agents_course/hf_cache` (HF download cache).
2. Updated `config/setup.json`:
   * `hf.cache_dir` → `D:/AI_agents_course/hf_cache`
   * `airllm.layer_shards_saving_path` → `D:/AI_agents_course/airllm_shards`
3. Added `HF_HOME=` to `.env-example` as an OPTIONAL template line (with usage notes — see § "HF cache redirect" in the README). The user's own `.env` got the concrete value `HF_HOME=D:/AI_agents_course/hf_cache`.
4. Re-ran `uv run init_env.py`. `HardwareScanner` now reports:

   | Component | Value |
   |-----------|-------|
   | Disk free | **392.8 GB** at `D:\AI_agents_course\airllm_shards` |
   | RAM available | 2.9 GB |

   ` config/setup.json.hardware_constraints` + the `<!-- HARDWARE_SPECS_PLACEHOLDER -->` blocks in `README.md` and `docs/PRD.md` were all auto-patched in the same scan.

### 5.4 Why `HF_HOME` (env var) and not just `config.hf.cache_dir`

`huggingface_hub` reads `HF_HOME` directly — no `cache_dir=` argument is passed from our code today (no `services/model_acquirer.py` consumer exists yet; that's M2b+). Until then, the env var is the only mechanism that actually moves the cache. The config field is set anyway, so when `model_acquirer.py` lands it can read from one canonical place and pass it through.

### 5.5 Symptom #2 — AirLLM refused to import

Once disk was no longer the bottleneck, an early `uv run` hit a different wall: AirLLM imports `optimum.bettertransformer.BetterTransformer`, which was **removed in optimum 2.0** (released 2025). Fresh installs of optimum landed on 2.x by default, and AirLLM crashed at module load before any model touched it.

A second flavour of the same problem: `transformers>=4.49` ships breaking changes that AirLLM's `BetterTransformer` fallback path can't handle — calls into `transformers.models.llama.modeling_llama` no longer resolve the way AirLLM 2.11 expects.

### 5.6 Fix #2 — pin transformers and optimum below the breaking releases

`pyproject.toml` got tighter caps + two new tokenizer deps:

```toml
"transformers>=4.44,<4.49",
"optimum>=1.21,<2.0",
"sentencepiece>=0.2",
"protobuf>=4.25",
```

`sentencepiece` + `protobuf` are required by Llama-3 / TinyLlama / Qwen tokenizers (SentencePiece-tokenized models). They're listed transitively by `transformers` but installing them explicitly stops uv from yanking them when transformers upgrades.

`uv lock` re-resolved 112 → 113 packages; `uv sync --frozen` succeeded; `uv run python -c "import airllm"` walked cleanly.

### 5.7 What was tried and rolled back

Early in this troubleshooting we also patched `services/plumbing_default_stages.py` with two workarounds:

* Add `ignore_patterns=["original/*", "*.pth", "*.bin"]` to the HF `snapshot_download` call, to skip the duplicate Meta PyTorch checkpoints that Llama-3 ships under `original/` (~6 GB extra on disk).
* Monkeypatch `huggingface_hub.snapshot_download` and `BetterTransformer.transform` during the `mmap_allocation` stage, to inject the same `ignore_patterns` into AirLLM's *internal* downloader and to convert `NotImplementedError` → `ValueError` so AirLLM's fallback chain would engage on unsupported model types.

**Both were reverted before the T-2a.5 prep commit.** Once disk moved to D: with 390 GB free, the ~6 GB Meta-original overhead is irrelevant; once optimum was pinned below 2.0, the BetterTransformer path works without converting exceptions. The simpler `services/plumbing_default_stages.py` (no `ignore_patterns`, no monkeypatches) is what shipped. Keeping the workarounds would have masked the real root causes and left scary monkeypatch code in a Building Block.

### 5.8 Lessons

1. **Validate model-storage drive against `HardwareScanner` output BEFORE first model run.** A placeholder shard path that doesn't exist silently falls back to measuring whatever default drive Windows hands `psutil.disk_usage`. Make the scanner's `measured_at` field part of the eyeball check.
2. **AirLLM ↔ transformers ↔ optimum has tight version coupling.** AirLLM 2.11 expects pre-2.0 optimum (for `bettertransformer`) and pre-4.49 transformers. Pinning these explicitly in `pyproject.toml` is mandatory; do not assume `>=` is safe for ML-stack deps.
3. **Prefer environment / config changes over runtime monkeypatches.** Two early "fixes" patched library internals at runtime; both went away once the right env var (`HF_HOME`) and the right dependency pins were in place. Workarounds compound; root-cause fixes don't.
4. **`.env-example` is a template, not a snapshot.** The first version of `HF_HOME=` hard-coded the student's own `D:/AI_agents_course/hf_cache`; on review we made it an empty placeholder with usage notes so a fresh checkout on a different machine wouldn't break.
