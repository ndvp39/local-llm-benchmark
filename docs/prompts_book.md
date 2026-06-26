# Prompts Book

> **Mandate.** Constitution §7.3 requires a Prompt Engineering Log: significant prompts used to build the project, with context, goal, received outputs, iterative refinements, and lessons learned. This file also doubles as the project's **decision diary** — any non-obvious issue we hit during build is captured here so future-me / a reviewer can reconstruct the *why*.

## Contents
- [1. Planning prompts (PRD / PLAN / TODO authoring)](#1-planning-prompts-prd--plan--todo-authoring)
- [2. Installation & environment issues (T-1.2)](#2-installation--environment-issues-t-12)
- [3. Provider pricing capture (M4)](#3-provider-pricing-capture-m4)

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
