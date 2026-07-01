# PRD — AirLLM Integration (`backends/airllm_backend.py`)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-2 — **MANDATORY**.
> **Blocks:** **T-3.1** code (`backends/airllm_backend.py`). Approval required before implementation begins.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf.pdf` (assignment §3, §5.4, §6.1) → `L08-summary-Lora-AirLLM.pdf` §8 (the mmap / OS-paging analogy is the headline algorithm) → `docs/PRD.md` v1.10 FR-5 / SC-2 → `docs/PLAN.md` v1.20 §5.2 (Strategy pattern) + ADR-005 (`layer_shards_saving_path`) + ADR-009 (`AutoModel*` factories only).
> **Empirical anchor:** `results/plumbing_20260630T194154Z.json` (M2a T-2a.5 real-machine AirLLM plumbing manifest — Llama-3-8B fp16, 2 tokens, 18 min, TTFT/TPOT ≈ 6 min/tok, peak RAM 1.29 GB, 35 layers) and `results/witness_baseline_llama3-8b-fp16_20260701T125650Z.json` + `results/witness_baseline_qwen2-7b-q4_20260701T125740Z.json` (M2b T-2.11 witness manifests proving the Direct baseline cannot naive-load these same weights on this box).
> **Document version:** 1.00 — 2026-07-01.
> **Status:** DRAFT, awaiting user approval before code is written.

---

## 1. What and why

### 1.1 What
`AirLLMBackend` — the second concrete implementation of `InferenceBackend` (ABC in `backends/base.py` from T-2.1) that loads a Hugging Face causal-LM model via the [`AirLLM`](https://github.com/lyogavin/airllm) library. AirLLM streams model weights **layer-by-layer** from a shard cache on disk into RAM, executes the layer's forward pass, evicts it, then streams the next layer. The effect is that a model whose full weight footprint exceeds RAM can still run — at the cost of dramatically increased wall time per forward pass. The M2a plumbing test already proved the mechanism works on this specific box (Llama-3-8B fp16 ran 2 tokens end-to-end at 1.29 GB peak RSS against 7.8 GB total system RAM); this PRD frames the mechanism as a first-class back-end alongside `DirectBackend`.

### 1.2 Why
Three motivations stack:

* **Assignment.** `ex05-AirLLM.pdf` §5.4 explicitly mandates the AirLLM path as one of the deliverables, and §3 requires a Roofline analysis that only becomes interesting when the memory-bound regime (Decode) is *actually observable* — which it is under AirLLM's per-layer streaming.
* **L08 §8 headline algorithm.** The lecture presents AirLLM as the LLM-domain analog of OS virtual-memory paging: the OS pages code + data through a small physical-RAM window from a backing store; AirLLM pages transformer layers through a small RAM window from a safetensors shard cache. The report needs to make that analogy concrete with measured data.
* **M2b baseline confirmed direct load fails.** The witness manifests (`results/witness_baseline_*`) show Windows raises `OSError 1455 "The paging file is too small"` after ~10-28 s of naive `AutoModelForCausalLM.from_pretrained` attempts for both Llama-3-8B and Qwen2-7B on this 7.8 GB box. AirLLM is the mechanism that turns "cannot run" into "runs slowly" — this back-end is the rescue path SC-2 depends on.

### 1.3 Who calls it
| Caller | When | Landing task |
|---|---|---|
| `OnPremLlmSDK.run_airllm(target_label, prompt, ...)` | M3 baseline + full sweep | T-3.1 (this PRD) + wired to CLI in a follow-up |
| `services/sweep_runner.py` | M3 sweep (target × quantization matrix) | T-3.5 |
| `services/plumbing_default_stages.py` — the AirLLM branch of `_load_via_airllm` | M2a plumbing (already shipped) | already live |

The plumbing runner already loads AirLLM via a lightweight closure (M2a T-2a.2); T-3.1 lifts that logic into a proper `InferenceBackend` subclass with the full lifecycle contract, memory sampling composition, and `BackendRunResult` shape the M3 sweep expects.

---

## 2. Theoretical background

### 2.1 Why oversized LLMs don't fit
An 8B-parameter transformer at fp16 needs ~16 GB of contiguous RAM for the weight tensors alone, before activations, attention KV cache, or the tokenizer. On a 7.8 GB total RAM box (M2b measured 1.5 GB free during actual runs), the naive PyTorch load path never even completes the first shard — the OS `HeapAlloc`/`mmap` request for the shard's contiguous backing memory fails, producing `OSError 1455` on Windows or an OOM-kill on Linux. See `results/witness_baseline_*` for measured evidence on this specific box.

### 2.2 What AirLLM does
AirLLM splits the model on disk into **per-layer** safetensors shards, saved to `layer_shards_saving_path` on first load (~30 GB for Llama-3-8B — 1× fp16 weight cache in AirLLM's per-layer layout + tokenizer + config + AirLLM's index file). Then, at inference time:

1. Only the layer currently executing lives in RAM.
2. Each `forward()` pass iterates the layers sequentially: for each layer, memory-map the shard from disk, run the layer, evict.
3. Because the RAM working set is one layer plus activations (~1-2 GB for Llama-3-8B), the model fits.

### 2.3 The OS-paging analogy (L08 §8)
| OS virtual memory | AirLLM |
|---|---|
| Physical RAM | RAM working set (one layer at a time) |
| Backing store (disk / swap) | Per-layer safetensors shards under `layer_shards_saving_path` |
| Page table | AirLLM's `AutoModel` layer index |
| Page fault | `mmap` of the next shard when the current layer finishes |
| TLB / working set | Layer prefetch cache (bounded by AirLLM's `prefetching`) |

The report should present this table + the measured numbers side-by-side. The M2a manifest already provides the anchor: 35 layers streamed sequentially, TTFT ≈ TPOT ≈ 6 min/tok because every forward pass streams every layer, so the Prefill (usually compute-bound / GEMM) and Decode (usually memory-bound / GEMV) regimes both collapse to the disk-streaming rate.

### 2.4 Compatibility surface (learned from M2a T-2a.5, prompts_book §5–§10)
AirLLM 2.11 requires:

1. **Sharded multi-file safetensors layout** (`model-00001-of-000NN.safetensors` + `model.safetensors.index.json`). HF Hub auto-shards only above ~5 GB, so any model < ~7B parameters at fp16 is *usually* single-file and incompatible.
2. **Separate `lm_head`** — no tied input/output embeddings. Modern small LLMs (Llama-3.2, Qwen-2 small, Gemma, Phi-3) tie embeddings; AirLLM's layer-wiring code assumes a separate `lm_head` shard and crashes with `list index out of range` when it's missing.
3. **Architecture in AirLLM's supported list.** Llama-2, Llama-3, Mistral, Qwen-2 ≥ 7B, ChatGLM, Baichuan2 are supported. Random custom architectures are not.

The plumbing model choice (§10 of prompts_book) collapsed to "use the same target model at a tighter token budget" because there is no sub-7B model in mainstream namespaces that satisfies all three constraints. This PRD inherits that constraint: **AirLLMBackend only accepts models in `config.target_models`**, all of which are already vetted for AirLLM compatibility (Llama-3-8B-Instruct, Qwen2-7B-Instruct).

### 2.5 CPU-only device pinning (learned from M2a)
AirLLM defaults to `device="cuda:0"` and tries to allocate a CUDA stream at init even when CUDA is unavailable, crashing with `Torch not compiled with CUDA enabled` on CPU-only wheels. The M2a plumbing code fixed this by passing `device="cuda:0" if torch.cuda.is_available() else "cpu"` explicitly. This PRD mandates the same guard.

### 2.6 Compression mapping
AirLLM's `AutoModel.from_pretrained` accepts a `compression` kwarg with values `"4bit"` | `"8bit"` | (unset = fp16). Our config's `quantization` field uses labels `q4` | `q8` | `fp16`. The mapping is 1:1 (see plumbing_default_stages.py's `_AIRLLM_COMPRESSION`). This PRD mandates the mapping table live in `airllm_backend.py`, not re-derived per caller.

---

## 3. Functional requirements (FR-AL-*)

* **FR-AL-1.** `AirLLMBackend(InferenceBackend)` MUST provide the standard `InferenceBackend` triple `load()` / `generate(prompt, *, max_new_tokens, params)` / `unload()`, returning a `BackendRunResult` from `generate()` with `backend=BackendId.AIRLLM`.
* **FR-AL-2.** MUST load via `airllm.AutoModel.from_pretrained(model_id, layer_shards_saving_path=..., hf_token=..., compression=..., device=...)`. No concrete `*ForCausalLM` class import (ADR-009 + T-2.0 AST guard). No `transformers.AutoModelForCausalLM` — AirLLM has its own factory.
* **FR-AL-3.** MUST honour `config.airllm.layer_shards_saving_path` from setup.json (ADR-005). If the path is missing or empty, raise `AirLLMConfigError` — never fall back silently to a default temp dir.
* **FR-AL-4.** MUST validate free disk on the shard-cache drive **before** attempting the load: read `psutil.disk_usage(layer_shards_saving_path).free` and compare to a config-driven `airllm.min_free_disk_gb` (default 25 GB). If insufficient, raise `AirLLMDiskError` with the shortfall in GB. The M2a Llama-3-8B shard cache is ~30 GB — this pre-flight is analogous to the DirectBackend RAM pre-flight from T-2.11.
* **FR-AL-5.** MUST device-pin explicitly: `device="cuda:0" if torch.cuda.is_available() else "cpu"`. Never let AirLLM's default `cuda:0` fire on a CPU-only torch wheel.
* **FR-AL-6.** MUST map `quantization` (`fp16` / `q4` / `q8`) to AirLLM's `compression` kwarg via a module-level `_COMPRESSION` dict. `fp16` → no compression kwarg (default fp16). `q2` / `nf4` / `fp32` → raise `UnsupportedQuantizationError` (AirLLM doesn't support them).
* **FR-AL-7.** MUST resolve HF token: prefer explicit constructor kwarg `hf_token`, then `os.environ.get("HF_TOKEN")`, then `None` (public repos). Never log the token.
* **FR-AL-8.** MUST support the same two-generate TTFT/TPOT measurement pattern the Direct backend uses (T-2.8): one `generate(max_new_tokens=1)` for TTFT, one `generate(max_new_tokens=N)` for full TPOT. The M2a plumbing manifest showed this collapses TTFT ≈ TPOT under AirLLM streaming — that IS the L08 §3 Roofline observation the report needs.
* **FR-AL-9.** MUST report `peak_ram_mb=0.0` from `generate()` and leave real sampling to `BenchmarkRunner` (T-2.9) that composes the mixins around it — same discipline as DirectBackend. The runner writes the enriched `BackendRunResult`.
* **FR-AL-10.** MUST populate `n_layers` in a diagnostic field (`raw_log_path` metadata block) so the report's paging analogy can quote the shard count for each target — the M2a manifest recorded `layers: 35` for Llama-3-8B; the AirLLMBackend should preserve that.

## 4. Non-functional requirements (NFR-AL-*)

* **NFR-AL-1.** File size ≤ 150 LOC per `shared/gatekeeper.py` split precedent. If the module needs a state class, split into `backends/airllm_backend.py` (public) + `backends/airllm_backend_helpers.py` (pure helpers) — constitution §2.2 sanctioned.
* **NFR-AL-2.** Coverage ≥ 85 % (constitution §5.2). Target ≥ 95 % since the surface is small.
* **NFR-AL-3.** No hard-coded model IDs, paths, or quantization labels. Everything comes from config / constructor kwargs.
* **NFR-AL-4.** T-2.0 AST guard MUST remain green — no `*ForCausalLM` imports.
* **NFR-AL-5.** Unit tests MUST mock `airllm.AutoModel` at the import boundary (via `sys.modules["airllm"]` injection, following the M2a plumbing test pattern) — no real model download in unit tests.

---

## 5. I/O Contract

### 5.1 `AirLLMBackend` (constructor + methods)
```python
class AirLLMBackend(InferenceBackend):
    BACKEND_ID = BackendId.AIRLLM

    def __init__(
        self,
        *,
        target_label: str,
        model_id: str,
        quantization: Quant | str,
        layer_shards_saving_path: Path | str,
        hf_token: str | None = None,
        min_free_disk_gb: float = 25.0,
        factory: Callable[..., Any] | None = None,   # seams for tests
        clock: Callable[[], float] | None = None,
    ) -> None: ...

    def load(self) -> None: ...
    def generate(self, prompt: str, *, max_new_tokens: int,
                 params: dict[str, Any]) -> BackendRunResult: ...
    def unload(self) -> None: ...
```

### 5.2 Errors
```python
class AirLLMConfigError(ValueError):
    """Missing / empty layer_shards_saving_path, or unsupported quantization."""

class AirLLMDiskError(RuntimeError):
    """Insufficient free disk on the shard-cache drive (< min_free_disk_gb)."""
```

`UnsupportedQuantizationError` reused from `shared/automodel_factory.py`.

### 5.3 Module-level compression map
```python
_COMPRESSION: dict[str, str | None] = {
    "fp16": None,   # AirLLM default
    "q4": "4bit",
    "q8": "8bit",
}
```

Quantizations not in this map raise `UnsupportedQuantizationError`.

---

## 6. Constraints

* **C-AL-1.** File size ≤ 150 LOC.
* **C-AL-2.** No `transformers.*ForCausalLM` imports (ADR-009 + T-2.0 AST guard).
* **C-AL-3.** AirLLM is a mandatory dep already in `pyproject.toml` (M2a T-2a.5 pinned it alongside `optimum<2.0` + `transformers<4.49`). Do not add sub-dependencies without updating `pyproject.toml`.
* **C-AL-4.** No hard-coded values. `layer_shards_saving_path` + `min_free_disk_gb` + all quantization labels come from config / constructor.
* **C-AL-5.** CPU-only device pin (FR-AL-5) is mandatory; the M2a discovery is not optional.

---

## 7. Alternatives considered

| # | Option | Reason rejected |
|---|---|---|
| A-AL-1 | Use `accelerate` disk offload via `device_map="auto"` (same trick M2b Act 1 showed) | Already available implicitly through `DirectBackend`'s Act-1 path (`baseline_llama3-8b-fp16_20260630T221156Z.json`). The report needs BOTH data points, not one — different tradeoff shapes (see M2b comparison table). AirLLM's explicit per-layer streaming produces more informative memory-bound observations. |
| A-AL-2 | Roll our own layer-by-layer loader on top of `transformers` + `safetensors` | +500 LOC, and AirLLM already exists + is used in the reference literature. Reinventing the wheel violates DRY (constitution §5). |
| A-AL-3 | Use `llama.cpp` / GGUF quantization | Different quantization format (GGUF), different model coverage. The assignment specifically references AirLLM (§5.4). |
| A-AL-4 | Load into GPU only when CUDA is available; skip CPU path entirely | CPU-only is the reference machine's *actual configuration* (no GPU detected per `config.hardware_constraints.gpu.present = false`). Skipping CPU means skipping the whole project. |
| A-AL-5 | Read `layer_shards_saving_path` from an env var | Configuration goes through `config/setup.json` per the "no hard-coded values" rule and PLAN §6.5. Env vars are for secrets only. |
| A-AL-6 | Pre-flight only checks disk, not free RAM | Free-RAM check is not needed for AirLLM — one layer's working set is ~1-2 GB, well within any modern box. Disk is the actual scarce resource (the M2a shard cache is ~30 GB per model). |

---

## 8. Success criteria

* **SC-AL-1.** Unit test with mocked `airllm.AutoModel` produces a well-shaped `BackendRunResult` (backend=AIRLLM, target/model IDs pass through, timings from injected clock).
* **SC-AL-2.** `AirLLMConfigError` raised on missing `layer_shards_saving_path` (unit test).
* **SC-AL-3.** `AirLLMDiskError` raised when free disk < `min_free_disk_gb` (unit test with mocked `psutil.disk_usage`).
* **SC-AL-4.** `UnsupportedQuantizationError` raised for `nf4` / `q2` / `fp32` (unit test).
* **SC-AL-5.** Device pin fires — on CPU-only torch, `AutoModel.from_pretrained` receives `device="cpu"` (unit test asserts factory called with `device="cpu"` when `torch.cuda.is_available()` returns False).
* **SC-AL-6.** Compression mapping: `Quant.Q4` → `compression="4bit"`, `Quant.Q8` → `compression="8bit"`, `Quant.FP16` → no `compression` kwarg (unit test).
* **SC-AL-7.** Integration test (marked `@pytest.mark.integration`) runs the backend against the plumbing model config (Llama-3-8B fp16) with `airllm.AutoModel` faked via `sys.modules["airllm"]` injection — proves the backend + factory + config wiring works end-to-end without touching a real 30 GB shard cache.
* **SC-AL-8.** T-2.0 AST guard remains green — the AST parser finds zero concrete `*ForCausalLM` imports in `backends/airllm_backend.py`.

---

## 9. Test scenarios (informs T-3.1 unit + integration suite)

### 9.1 Unit (`tests/unit/test_backends/test_airllm_backend.py`)
| ID | Scenario |
|----|----------|
| U-AL-1 | Construction with all required kwargs succeeds; `_quantization` is `Quant` |
| U-AL-2 | Missing / empty `layer_shards_saving_path` → `AirLLMConfigError` |
| U-AL-3 | `psutil.disk_usage(...).free < min_free_disk_gb * 1024**3` → `AirLLMDiskError` (mock `psutil.disk_usage`) |
| U-AL-4 | Unsupported quantization (`nf4`, `q2`, `fp32`) → `UnsupportedQuantizationError` |
| U-AL-5 | `load()` calls factory with `device="cpu"` when `torch.cuda.is_available()` returns False (via monkeypatch) |
| U-AL-6 | `load()` calls factory with `compression="4bit"` when `quantization=Quant.Q4` |
| U-AL-7 | `load()` calls factory with `compression="8bit"` when `quantization=Quant.Q8` |
| U-AL-8 | `load()` calls factory with NO `compression` kwarg when `quantization=Quant.FP16` |
| U-AL-9 | `generate()` before `load()` raises `RuntimeError` |
| U-AL-10 | `generate()` returns `BackendRunResult` with `backend=BackendId.AIRLLM`, timing from injected clock |
| U-AL-11 | `unload()` releases the loaded model; idempotent |
| U-AL-12 | `hf_token` resolution: explicit kwarg wins > env var > None (parametrized) |
| U-AL-13 | Constructor rejects positional args (kw-only enforcement) |

### 9.2 Integration (`tests/integration/test_airllm_backend_integration.py`)
| ID | Scenario |
|----|----------|
| I-AL-1 | Full backend wired to a plumbing-model config with `airllm.AutoModel` faked via `sys.modules["airllm"]` injection — asserts `BackendRunResult` populated, factory called with the right kwargs, no real download triggered |
| I-AL-2 | Static AST guard (T-2.0) reruns against the real `backends/airllm_backend.py` file and finds zero concrete `*ForCausalLM` imports |

---

## 10. Out of scope

* **Real-machine 128-token AirLLM run for the M6 report.** That's a separate M3 task (analogous to T-2.11) — this PRD gates the *code*, not the run.
* **Multi-GPU sharding.** The reference machine has zero GPUs; irrelevant.
* **NF4 / QLoRA quantization.** Covered by DP-7 (`docs/PRD_qlora.md`) + T-5.1.
* **AirLLM's compressor pipeline internals.** Trust the library; test at the boundary via mocks.
* **Automatic shard-cache cleanup.** AirLLM's shard cache is a durable resource on disk (~30 GB per model). Cleanup policy is a separate concern (out of scope for M3 code, potentially a housekeeping tool in a later milestone).

---

## 11. Decisions taken in this PRD

| ID | Decision | Why |
|----|----------|-----|
| D-AL-1 | Two-generate TTFT/TPOT pattern (same as DirectBackend) | Consistent with T-2.8; the L08 §3 TTFT ≈ TPOT collapse observation is the report's centrepiece and needs both metrics reported (per FR-AL-8). |
| D-AL-2 | Explicit CPU device pin | M2a T-2a.5 lesson — AirLLM defaults to `cuda:0` and crashes on CPU-only torch. |
| D-AL-3 | Pre-flight disk check (analogous to DirectBackend's pre-flight RAM check) | Fail fast, produce a Python-level error that `baseline_service` / `sweep_runner` can catch and structure. |
| D-AL-4 | `AirLLMConfigError` + `AirLLMDiskError` as distinct exception types (vs. one generic `AirLLMError`) | Callers can distinguish config bugs from environmental limits — the sweep runner might retry on disk failures after cleanup but never on config errors. |
| D-AL-5 | Reuse `UnsupportedQuantizationError` from `shared/automodel_factory.py` | Single canonical exception for the whole project; PRD FR-8 references it. |
| D-AL-6 | Factory kwarg is `factory: Callable[..., Any] \| None` | Same seam pattern as DirectBackend (T-2.8) — test injection point for `airllm.AutoModel.from_pretrained`. |
| D-AL-7 | Compression map lives at module level, not on the class | Small immutable data; class-level constant is idiomatic Python and mirrors `shared/automodel_factory._DTYPE_BY_QUANT`. |

---

## 12. Open questions for user

Before T-3.1 begins, please confirm:

1. **`min_free_disk_gb` default = 25 GB — right size?** M2a Llama-3-8B produced a ~30 GB shard cache. 25 GB is conservative; 40 GB would guarantee a fresh Llama-3 fits even if concurrent Qwen shards are being written. My default in §5.1 is 25.
2. **Should `AirLLMBackend.load()` also emit a JSONL log line (analogous to `ApiGatekeeper._log`)?** Currently my draft doesn't — the memory sampling + manifest writing happen at the `BenchmarkRunner` layer. A load-event log line inside the backend is redundant. Confirming no.
3. **Anything missing from §9 test scenarios?** 13 unit + 2 integration. Adding scenarios now is cheap; retrofitting after T-3.1 is expensive.

Once these are answered, T-3.1 implementation proceeds against this contract verbatim. The PRD is the source of truth; deviations require updating this file first.

---

## 13. Approval

This PRD MUST be approved by the user before any code is written in `backends/airllm_backend.py`. Approval flips DP-2 status in `docs/TODO.md` §7 to "MANDATORY · authored + approved".

> Approval status: ☑ **Approved 2026-07-01** by user — *"approved, 25 GB, no log line, tests look complete, be aware that everything that related to download for models that dont need to be in the project, can download to D"*. Answers to the 3 open questions in §12: (1) `min_free_disk_gb = 25 GB` stays; (2) no in-backend log line — memory sampling + manifest writing stay at `BenchmarkRunner`; (3) 13 unit + 2 integration test scenarios in §9 stand. **Standing constraint honoured project-wide:** any transient model download (baseline, plumbing, sweep, or future) MUST land under `D:/AI_agents_course/hf_cache` via `HF_HOME`.
> Implementation status: ☑ **Implemented 2026-07-01** by T-3.1 — `src/on_prem_llm_lab/backends/airllm_backend.py` (149 LOC, 100 % stmt + branch coverage) + `tests/unit/test_backends/conftest.py` (51 LOC, shared stubs + `make_airllm` factory fixture) + `tests/unit/test_backends/test_airllm_backend.py` (139 LOC, 21 tests across 6 classes — covers U-AL-1..13) + `tests/integration/test_airllm_backend_integration.py` (82 LOC, 1 `@pytest.mark.integration` test — I-AL-1 via `sys.modules["airllm"]` injection). I-AL-2 (T-2.0 AST guard) reruns on every pytest invocation and stays green. Contract fulfilled verbatim — no deviations from v1.00.
