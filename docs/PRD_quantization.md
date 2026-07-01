# PRD — Quantization Policy (`shared/automodel_factory` + `backends/*_backend`)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-3 — **MANDATORY**.
> **Blocks:** **T-3.3** code (quantization adapter logic per backend). Approval required before implementation begins.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf.pdf` §5 (quantization overview) → `L08-summary-Lora-AirLLM.pdf` §5 (bit-width reduction, quality tradeoffs; §5.1 NF4 / Double Quantization / Paged Optimizers) → `docs/PRD.md` v1.10 FR-7 (support ≥ 3 levels) + FR-8 (backend refuses unsupported levels with `UnsupportedQuantizationError`) + G4 (quantify optimization payoff) + K3 (≥ 3 bit-widths per model).
> **Empirical anchor:** The M2b baseline manifests show fp16 weight footprint = 14.96 GiB (Llama-3-8B) and 14.19 GiB (Qwen2-7B) — both ~1.9× the 7.8 GiB system RAM. AirLLM's q4 compression option is the mechanism that makes the sweep interesting; without it, every measurable point on the memory/latency curve looks similar. Also the M2a AirLLM plumbing manifest confirms `compression="4bit"` maps correctly through `airllm.AutoModel.from_pretrained`.
> **Document version:** 1.00 — 2026-07-01.
> **Status:** DRAFT, awaiting user approval before code is written.

---

## 1. What and why

### 1.1 What
The project-wide **quantization policy** — which bit-widths are supported, by which back-end, with what tradeoffs, and how the sweep runner iterates the matrix. This PRD does NOT introduce a new component (there is no `services/quantizer.py`); it formalises how the existing surfaces cooperate:

* `on_prem_llm_lab.backends.base.Quant` (from T-2.1) — the canonical enum of quantization *labels* (`fp32`, `fp16`, `q8`, `q4`, `q2`, `nf4`).
* `on_prem_llm_lab.shared.automodel_factory.SUPPORTED_QUANTIZATIONS` + `_DTYPE_BY_QUANT` (from T-1.15) — the DirectBackend factory's dtype-resolution table.
* `on_prem_llm_lab.backends.airllm_backend._COMPRESSION` (from T-3.1) — AirLLMBackend's label → AirLLM `compression` kwarg map (`fp16 -> None`, `q4 -> "4bit"`, `q8 -> "8bit"`).
* `UnsupportedQuantizationError` (from T-1.15) — the canonical exception per FR-8.

T-3.3 lands (a) a **support matrix** helper that says which `(backend, quantization)` combinations are supported and which raise `UnsupportedQuantizationError`, (b) any per-backend adapter logic the sweep runner (T-3.5) needs to iterate the matrix cleanly, and (c) a truthful mapping in `_DTYPE_BY_QUANT` — because today it maps every `q*` label to `torch.float16`, which silently misleads callers into thinking Direct+q4 is real quantization when it is actually just fp16.

### 1.2 Why
Three motivations:

* **Assignment.** `ex05-AirLLM.pdf` §5 mandates quantization as one of the two techniques being combined with AirLLM. `docs/PRD.md` G4 + K3 + FR-7 require the sweep to cover ≥ 3 bit-widths per model.
* **L08 §5 theory.** Weight bit-width dominates the model's memory footprint (fp16 → 16 GB for 8B params; q4 → 4 GB). The report must show the memory-per-token and quality-per-token curves as bit-width varies, and explain the L08 §5 "quality cliff" — the empirical threshold below which output degrades unacceptably.
* **Silent-fp16 bug in the current factory.** `_DTYPE_BY_QUANT` maps `q4`/`q8`/`q2`/`nf4` all to `torch.float16`. That means "Direct + q4" currently means "load fp16", not "load 4-bit quantized". This PRD flags that as a real bug T-3.3 must fix (either by adding bitsandbytes-backed real quantization to Direct, or by having Direct explicitly refuse `q*` labels and defer to AirLLM).

### 1.3 Who consumes this policy
| Consumer | Where the policy applies | Landing task |
|---|---|---|
| `services/sweep_runner.py` | Iterates `(target, quantization, backend)`; needs the support-matrix helper | T-3.5 |
| `backends/direct_backend.py` | Refuses unsupported labels | T-3.3 (fix current silent-fp16 behaviour) |
| `backends/airllm_backend.py` | Already refuses unsupported labels via `_resolve_compression` | T-3.1 (already shipped) |
| Quality-matrix harness | Same prompt across all supported levels per target | T-3.6 |
| M6 report / README | Renders the sweep-CSV numbers as memory-vs-quality curves | T-6.1 |

---

## 2. Theoretical background

### 2.1 Quantization = per-weight bit-width reduction

An LLM's memory footprint at inference is dominated by the weight tensor stack. For an 8B-parameter model:

| Precision | Bytes per weight | 8B model weight footprint |
|---|---:|---:|
| fp32 | 4 | ~32 GB |
| **fp16 / bf16** | **2** | **~16 GB** ← M2b measured 14.96 GiB for Llama-3-8B |
| int8 (Q8) | 1 | ~8 GB |
| int4 / NF4 (Q4) | 0.5 | ~4 GB |
| int2 (Q2) | 0.25 | ~2 GB |

The scaling is (almost) linear because the activation + KV cache footprint is small relative to weights at inference time. AirLLM's per-layer streaming makes it possible to run any of these on a small box, but the quantized ones fit *more comfortably*: q4 Llama-3-8B is 4 GB weights, which fits entirely in the 7.8 GB reference RAM — the memory-bound Roofline regime opens up (L08 §3).

### 2.2 Quality cliff (L08 §5)

Bit-width and output quality are not linearly related. The literature reports:

* **fp16** — indistinguishable from fp32 for LLM inference; sometimes even beneficial (fewer numerical instabilities in attention softmax).
* **Q8** — near-lossless for chat / summarisation tasks; occasional token divergence on math / code.
* **Q4** — visible quality drop on complex reasoning; often acceptable for conversational tasks. NF4 (the QLoRA paper's NormalFloat 4-bit format) preserves quality better than naive int4 because it matches the empirical Gaussian distribution of pre-trained weights.
* **Q2** — quality typically collapses. Rarely usable for base models; may work post-fine-tuning.

The report's job under G4 is to *show* the quality cliff empirically on the two target models via the T-3.6 quality-matrix harness. This PRD does not decide *where* the cliff is; it defines the matrix that measures it.

### 2.3 Quantization implementations at play

| Level | Mechanism in this codebase | Backend that ships real support |
|---|---|---|
| fp32 | `torch_dtype=torch.float32` in `AutoModelForCausalLM.from_pretrained` | DirectBackend (via factory) |
| **fp16** | `torch_dtype=torch.float16` (DirectBackend) OR AirLLM default (no compression kwarg) | BOTH |
| **q4** | AirLLM's `compression="4bit"` — uses `bitsandbytes` 4-bit NF4 blocks under the hood via AirLLM's wrapper | AirLLMBackend |
| **q8** | AirLLM's `compression="8bit"` — uses `bitsandbytes` LLM.int8() | AirLLMBackend |
| q2 | No canonical implementation in this codebase (no library in the pinned deps supports 2-bit inference for the target architectures at production quality) | **none — placeholder label only** |
| nf4 | `bitsandbytes` `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` in principle; reserved for QLoRA training in M5 | **none in inference back-ends** (T-5.1) |

Reading this table left-to-right shows the honest matrix; T-3.3 must encode it in code and reject invalid combinations.

---

## 3. Functional requirements (FR-Q-*)

* **FR-Q-1.** A **single canonical support-matrix helper** `is_supported(backend: BackendId, quantization: Quant) -> bool` and `supported_matrix() -> dict[BackendId, frozenset[Quant]]` MUST live in `shared/automodel_factory.py` (or a new sibling module if size-cap pressure forces a split). Every consumer (SweepRunner, quality harness, CLI) queries this helper — no consumer re-encodes the matrix.
* **FR-Q-2.** The matrix (v1.00) is:
  ```
  BackendId.DIRECT: {Quant.FP32, Quant.FP16}
  BackendId.AIRLLM: {Quant.FP16, Quant.Q4, Quant.Q8}
  BackendId.API:    {Quant.FP16}    # opaque via the vendor; we tag it fp16 for accounting
  ```
* **FR-Q-3.** `DirectBackend.load()` MUST call the support helper and raise `UnsupportedQuantizationError` for any level not in its row (currently silently loads fp16 for `q4`/`q8`/`q2`/`nf4` — this bug ships to the sweep as false data unless fixed). Fix: `_DTYPE_BY_QUANT` shrinks to only supported levels + a pre-load guard.
* **FR-Q-4.** `AirLLMBackend._resolve_compression` already raises `UnsupportedQuantizationError` for unsupported labels — no change needed; T-3.3 verifies its behaviour matches FR-Q-2.
* **FR-Q-5.** `Quant.Q2` and `Quant.NF4` remain in the enum (they are referenced by PRD FR-7 and the T-5 QLoRA milestone) but the support helper reports them as unsupported by every inference backend. **The sweep runner treats them as "expected to skip"** and the quality-matrix harness omits them from its output — this matches the empirical reality that no library in our pinned deps runs Q2 inference on these architectures.
* **FR-Q-6.** The **quality-matrix harness** (T-3.6, downstream) runs the same prompt through EVERY supported `(target, quantization, backend)` combination and captures the output text side-by-side into `results/quality_matrix.md`. This PRD reserves the format shape but does not implement — T-3.6 does.
* **FR-Q-7.** Every `BackendRunResult` MUST carry `quantization` as the input label (already true from T-2.1). The label is what appears in CSV rows + charts; the *actual* dtype/compression the backend used is a diagnostic in the run's raw log, not the sweep row.

## 4. Non-functional requirements (NFR-Q-*)

* **NFR-Q-1.** File-size cap: the support helper stays inside `shared/automodel_factory.py` (currently 81 LOC after the T-2.11 kwargs). Total ≤ 150 LOC — if it grows past the cap, split the matrix + helper into `shared/quantization_matrix.py` (constitution §2.2 sanctioned extract).
* **NFR-Q-2.** No hard-coded label lists outside the matrix module. Consumers import `is_supported` + `supported_matrix()`.
* **NFR-Q-3.** Coverage ≥ 85 % on the touched surface (matrix helper + updated DirectBackend guard).
* **NFR-Q-4.** T-2.0 AST guard MUST remain green — no `*ForCausalLM` imports.
* **NFR-Q-5.** No new heavyweight dependency. `bitsandbytes` is already pinned (via `pyproject.toml` deps for `airllm`/`transformers` extras); nothing new to add.

---

## 5. I/O Contract

### 5.1 Support-matrix module (new symbols in `shared/automodel_factory.py`)

```python
def is_supported(backend: BackendId, quantization: Quant | str) -> bool:
    """True if this (backend, quantization) combination is exercisable."""

def supported_matrix() -> dict[BackendId, frozenset[Quant]]:
    """Canonical policy matrix (FR-Q-2)."""

def require_supported(
    backend: BackendId, quantization: Quant | str,
) -> None:
    """Raise UnsupportedQuantizationError if (backend, quantization) is not supported."""
```

### 5.2 `_DTYPE_BY_QUANT` shrinkage (fix FR-Q-3)

Before (silently coerces every q* to fp16 — the bug):
```python
_DTYPE_BY_QUANT = {
    "fp32": torch.float32, "fp16": torch.float16,
    "q8": torch.float16, "q4": torch.float16,     # <- silent fp16, misleading
    "q2": torch.float16, "nf4": torch.float16,     # <- silent fp16, misleading
}
```

After (honest — factory only knows about the labels the Direct path really supports):
```python
_DTYPE_BY_QUANT = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}
```

`load_causal_lm` now raises `UnsupportedQuantizationError` for anything else, and the Direct backend's `load()` calls `require_supported(BackendId.DIRECT, self._quantization)` before the factory — belt-and-suspenders.

### 5.3 `UnsupportedQuantizationError` message shape

Existing exception, message pattern preserved: `f"quantization {q!r} not in {SUPPORTED_QUANTIZATIONS}"`. T-3.3 may extend to include the backend in the message when raised via `require_supported`, e.g. `f"{backend.value} does not support quantization {q!r}; supported for this backend: {supported_matrix()[backend]}"`.

---

## 6. Constraints

* **C-Q-1.** No new heavy dep — `bitsandbytes` bounded to AirLLM's usage.
* **C-Q-2.** Matrix in FR-Q-2 is the v1.00 policy; changes require this PRD to be revised (not just the code).
* **C-Q-3.** T-2.0 AST guard stays green — no `*ForCausalLM` imports anywhere.
* **C-Q-4.** No hard-coded label sets outside the matrix module.
* **C-Q-5.** `Quant.Q2` + `Quant.NF4` cannot be silently dropped from the enum — they are referenced by FR-7 and the T-5 QLoRA milestone respectively.

---

## 7. Alternatives considered

| # | Option | Reason rejected |
|---|---|---|
| A-Q-1 | Add `bitsandbytes` 4-bit / 8-bit support to `DirectBackend` (via `BitsAndBytesConfig`) so Direct + q4 becomes real quantization | `bitsandbytes` inference on CPU is not production quality — the library's CPU codepath is a fallback intended for testing, not benchmarking. The reference box has zero GPUs; putting Direct+q4 in the matrix would produce numbers that don't reflect a real deployment. If a future revision moves to a GPU box, this alternative becomes attractive. |
| A-Q-2 | Delete `Quant.Q2` + `Quant.NF4` from the enum entirely since no backend supports them | Violates C-Q-5 — those labels are referenced by PRD FR-7 and by the M5 QLoRA milestone. Keep them; mark as "unsupported for inference" in the matrix. |
| A-Q-3 | Add `Quant.BF16` for TPU-style bfloat16 | Not called for by the assignment or planning docs. Out of scope for M3; can be added by revising this PRD if a use case appears. |
| A-Q-4 | Silent skip on unsupported combos in `SweepRunner` (log a WARNING, continue) instead of raising `UnsupportedQuantizationError` from the backend | Rejected — a silent-skip design lets callers write invalid `(backend, quantization)` combos and see zero output, hiding bugs. The backend raising a typed exception is the constitution's preferred pattern (§4 — no silent swallowing). SweepRunner catches the exception explicitly for skip-with-warning behaviour. |
| A-Q-5 | Read the support matrix from `config/setup.json` | Overkill for a policy that is code-shaped, not runtime-tunable. If we add a new backend the code has to change anyway. |

---

## 8. Success criteria

* **SC-Q-1.** `is_supported(BackendId.DIRECT, Quant.FP16)` is `True`; `is_supported(BackendId.DIRECT, Quant.Q4)` is `False` (unit test).
* **SC-Q-2.** `is_supported(BackendId.AIRLLM, Quant.Q4)` is `True`; `is_supported(BackendId.AIRLLM, Quant.FP32)` is `False` (unit test).
* **SC-Q-3.** `DirectBackend.load(quantization=Quant.Q4)` raises `UnsupportedQuantizationError` with the backend name in the message (unit test).
* **SC-Q-4.** `AirLLMBackend.__init__(quantization=Quant.FP32)` still raises `UnsupportedQuantizationError` (regression test — behaviour was shipped in T-3.1).
* **SC-Q-5.** `_DTYPE_BY_QUANT` contains ONLY `fp32` + `fp16` after T-3.3 (unit test asserts the map shape).
* **SC-Q-6.** `supported_matrix()[BackendId.DIRECT]` returns `frozenset({Quant.FP32, Quant.FP16})` and `supported_matrix()[BackendId.AIRLLM]` returns `frozenset({Quant.FP16, Quant.Q4, Quant.Q8})` (unit test).
* **SC-Q-7.** Every existing test on `DirectBackend` and `AirLLMBackend` still passes (regression check — 204 tests today, T-3.3 must land 204+ passing).
* **SC-Q-8.** `require_supported` includes the backend name and the allowed set in its error message so a caller reading the exception can immediately see what to try instead.

---

## 9. Test scenarios (informs T-3.3 unit suite)

### 9.1 Unit tests (in `tests/unit/test_shared/test_quantization_matrix.py` — new file)
| ID | Scenario |
|----|----------|
| U-Q-1 | `is_supported` returns True for every combo in the matrix |
| U-Q-2 | `is_supported` returns False for `(DIRECT, Q4)` |
| U-Q-3 | `is_supported` returns False for `(AIRLLM, FP32)` |
| U-Q-4 | `is_supported` returns False for every backend × `Q2` |
| U-Q-5 | `is_supported` returns False for every backend × `NF4` |
| U-Q-6 | `supported_matrix()` returns a dict with three backend keys (DIRECT, AIRLLM, API) |
| U-Q-7 | `require_supported` no-ops for supported combos |
| U-Q-8 | `require_supported` raises `UnsupportedQuantizationError` with the backend name AND the allowed set in the message for unsupported combos |
| U-Q-9 | `is_supported` accepts either `Quant` enum or `str` label as input (parametrized) |
| U-Q-10 | `_DTYPE_BY_QUANT` after the shrinkage contains exactly `{fp32, fp16}` |

### 9.2 Unit tests (extension of `test_direct_backend.py`)
| ID | Scenario |
|----|----------|
| U-Q-11 | `DirectBackend(quantization=Quant.Q4)` constructor OR `.load()` raises `UnsupportedQuantizationError` |
| U-Q-12 | `DirectBackend(quantization=Quant.FP32)` succeeds and passes `torch_dtype=torch.float32` to the factory |
| U-Q-13 | `test_calls_factory_with_model_id_and_quant_label` still passes with fp16 (regression) |

### 9.3 Unit tests (regression on `test_airllm_backend.py`)
The existing `TestConstruction.test_unsupported_quantization_raises` already covers `nf4` / `q2` / `fp32`. No new AirLLM test needed; T-3.3 does not touch AirLLMBackend.

### 9.4 No integration test
The support matrix is pure policy — no I/O, no external services, no wall-clock behaviour. Unit coverage is sufficient. The sweep runner (T-3.5) will exercise the matrix in its own integration tests.

---

## 10. Out of scope

* **NF4 quantization for QLoRA training.** Reserved for DP-7 (`docs/PRD_qlora.md`) + T-5.1.
* **Q2 inference support.** No library in the pinned deps supports it at production quality on the target architectures. Placeholder label only.
* **bitsandbytes CPU inference for DirectBackend.** See A-Q-1 rejection. Revisit if a GPU box appears.
* **Post-training quantization tooling.** We only *load* pre-quantized weights via AirLLM's `compression` kwarg; we do NOT re-quantize models ourselves.
* **AWQ / GPTQ / EXL2 back-ends.** Different quantization families with different toolchains; not called for by the assignment.
* **Per-target quantization defaults in `config.target_models[]`.** Each target entry already has its own `quantization` label. This PRD does not change that shape.

---

## 11. Decisions taken in this PRD

| ID | Decision | Why |
|----|----------|-----|
| D-Q-1 | Fix the silent-fp16 bug — shrink `_DTYPE_BY_QUANT` to `{fp32, fp16}` only; every other label rejected | Correctness. Sweep data must reflect what the backend actually did, not what the caller thought was requested. |
| D-Q-2 | Central support-matrix helper in `shared/automodel_factory.py` (no new module) | Keeps quantization concerns co-located with the factory that resolves dtypes. Split later if size-cap pressure forces it (constitution §2.2). |
| D-Q-3 | `Quant.Q2` + `Quant.NF4` stay in the enum, matrix says unsupported | Enum shape is a stable API surface referenced by other PRDs; deletion would ripple. Matrix reports the ground truth. |
| D-Q-4 | `require_supported` raises with backend name + allowed set in the message | UX: reading the exception should tell the user what to try instead. |
| D-Q-5 | No `bitsandbytes` CPU-backed DirectBackend q4 for now | A-Q-1 — the CPU codepath is not production quality; benchmark numbers would be misleading. |
| D-Q-6 | SweepRunner (T-3.5) is responsible for skip-with-warning on unsupported combos | The backend raises; the sweep chooses whether to abort or continue. Separation of concerns. |

---

## 12. Open questions for user

1. **Fix or defer the silent-fp16 bug?** The current `_DTYPE_BY_QUANT` maps `q*` to `torch.float16` silently. Options: (a) fix now in T-3.3 (this PRD's D-Q-1) — cleanest, matches SC-Q-5, but requires updating any test that assumed silent-fp16 behaviour; (b) defer to a later revision and just document the bug. My draft says (a).
2. **Should `require_supported` accept `str` in addition to `Quant`?** For CLI ergonomics (users pass `--quantization q4` as a string). My draft says yes — parametrized U-Q-9.
3. **Do we need a formal per-level acceptance threshold** (e.g., "Q4 must be within X % of fp16's tokens/sec")? My draft says no — the report's job is to *show* the tradeoff empirically via T-3.6's quality matrix; imposing a threshold would either be trivially true or misleadingly hard-coded.

---

## 13. Approval

This PRD MUST be approved by the user before any code is written for T-3.3. Approval flips DP-3 status in `docs/TODO.md` §7 to "MANDATORY · authored + approved".

> Approval status: ☑ **Approved 2026-07-01** by user — *"approved, fix now, accept str, no threshold"*. Answers to the 3 open questions in §12: (Q1) fix the silent-fp16 bug now — T-3.3 shrinks `_DTYPE_BY_QUANT` to `{fp32, fp16}` and adds `require_supported` at the DirectBackend load site; (Q2) `require_supported` accepts either `Quant` enum or `str` label; (Q3) no formal per-level acceptance threshold — T-3.6's quality matrix shows the tradeoff empirically.
> Implementation status: ☑ **Implemented 2026-07-01** by T-3.3 — `shared/automodel_factory.py` grew from 81 to 110 LOC with shrunk `SUPPORTED_QUANTIZATIONS` + shrunk `_DTYPE_BY_QUANT` (fixes silent-fp16 bug D-Q-1) + new `_SUPPORTED_MATRIX` + `is_supported` / `supported_matrix` / `require_supported` trio (accepts BackendId+Quant enums OR raw str, message includes backend + allowed set). `backends/direct_backend.py` gains `require_supported(BACKEND_ID, quantization)` at the top of `load()` (SC-Q-3). `tests/unit/test_shared/test_quantization_matrix.py` (170 LOC, 28 tests across 5 classes) covers all 10 §9.1 scenarios + 5 DirectBackend-DP-3-integration tests. Regressions on `test_automodel_factory.py` + `test_base.py` updated. All 236 tests pass; coverage 94.33 %; ruff clean; T-2.0 AST guard still green. Contract fulfilled verbatim — no deviations from v1.00.
