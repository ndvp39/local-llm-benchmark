# on-prem-llm-lab

> **Status:** placeholder. The real README is the **deep-dive technical report** mandated by Assignment 05 §7–§8 and assembled by `services/report_assembler.py` (task T-6.1).
> Until M6 is reached, see the planning documents under `docs/`:
>
> - [`docs/PRD.md`](docs/PRD.md) — Product Requirements (v1.10).
> - [`docs/PLAN.md`](docs/PLAN.md) — Architecture & Design (v1.20).
> - [`docs/TODO.md`](docs/TODO.md) — Task list with statuses.
> - [`docs/prompts_book.md`](docs/prompts_book.md) — Decision diary & troubleshooting log.

## Setup

```bash
uv sync                # install deps from pyproject.toml + uv.lock
cp .env-example .env   # then fill in HF_TOKEN (required for gated Llama-3)
uv run init_env.py     # scan hardware, patch config + this README's table below
```

### Plumbing model

`config.plumbing_test_model` mirrors `config.target_models[0]` byte-for-byte — **`meta-llama/Meta-Llama-3-8B-Instruct` (fp16) loaded via AirLLM** — and is capped at **`config.generation.plumbing_max_new_tokens = 2`** so a full plumbing run finishes in ~18-20 minutes on the under-resourced reference hardware. This means **plumbing exercises the exact production loader path** (AirLLM layer-split + shard write + per-layer mmap forward pass) on the exact production model — not a small smoke proxy. Six failed model-swap iterations (see [`docs/prompts_book.md` §6–§9](docs/prompts_book.md#6-t-2a5--plumbing-model-switched-from-tinyllama-to-llama-32-1b)) established that no AirLLM-compatible model exists below ~7B (AirLLM 2.11 needs sharded multi-file safetensors + separate `lm_head`), so "plumbing = first target at a tighter token budget" is the design that fits the constraint. Full rationale: [`docs/prompts_book.md` §10](docs/prompts_book.md#10-t-2a5-absolutely-final--plumbing-is-llama-3-8b-at-2-tokens-via-airllm-m2a-green).

**Targets:** `meta-llama/Meta-Llama-3-8B-Instruct` (fp16, the deliberately oversized run — 16 GB weights on 7.8 GB RAM, rescued by AirLLM's layer-by-layer mmap) + `Qwen/Qwen2-7B-Instruct` (Q4, second architecture for cross-family quantization tradeoff data). Llama-3 is gated (you click through Meta's form on HF); Qwen2 is Apache 2.0 / non-gated.

### M2a result — plumbing test green (2026-06-30)

Both manifests committed under `results/`:

| Manifest | Model | Loader | Tokens | TTFT | TPOT | Peak RAM | Wall | Status |
|---|---|---|---|---|---|---|---|---|
| `plumbing_20260630T184908Z.json` | TinyLlama-1.1B-Chat | transformers fp16 | 32 | **1.93 s** | **121 ms/tok** | 2.29 GB | 1 min 55 s | `ok` |
| `plumbing_20260630T194154Z.json` | **Meta-Llama-3-8B-Instruct** | **AirLLM fp16** | **2** | **367.28 s** (~6 min) | **368.35 s/tok** (~6 min) | **1.29 GB** | **18 min 25 s** | `ok` |

The Llama-3 run is the **decisive M2a result** — it proves the central thesis end-to-end on this machine: **16 GB of fp16 weights run successfully on a box with 7.8 GB total / 2.9 GB available RAM**, with peak process RSS holding at 1.29 GB throughout because AirLLM keeps only one transformer layer hot in RAM at a time. The cost of the rescue is visible: ~6 min/token sustained, ~9.8 tokens/hour throughput, TTFT collapsing to TPOT (no Prefill/Decode regime separation under extreme RAM pressure — see L08 §3 Roofline analysis and the discussion in [`docs/prompts_book.md` §10.5](docs/prompts_book.md#10-t-2a5-absolutely-final--plumbing-is-llama-3-8b-at-2-tokens-via-airllm-m2a-green)).

The TinyLlama run is preserved as cross-loader reference data — same hardware, same prompt, same code path but via `transformers.AutoModelForCausalLM` — illustrating what the box looks like when memory is not the bottleneck (~1.5 sec to first token, ~3000× faster TPOT than the AirLLM-rescued Llama-3 case).

### M2b result — baseline captured, two-act story (2026-07-01)

**Three complementary observations on the same 7.8 GB / CPU-only box, all captured under `results/`:**

| Manifest | Model | Loader path | Tokens | TTFT | TPOT | Peak RAM | Wall | Status |
|---|---|---|---|---|---|---|---|---|
| `plumbing_20260630T194154Z.json` (M2a) | Llama-3-8B-Instruct | **AirLLM per-layer mmap** fp16 | 2 | 367 s | 368 s/tok | **1.29 GB** | 18 min | `ok` |
| `baseline_llama3-8b-fp16_20260630T221156Z.json` (Act 1) | Llama-3-8B-Instruct | **Direct + accelerate silent offload** fp16 (`device_map="auto"`) | 128 | 64 s | 35 s/tok | 3.19 GB | 75 min | `ok` |
| `baseline_llama3-8b-fp16_20260701T120918Z_failure.json` (Act 2, SC-1) | Llama-3-8B-Instruct | **Direct naive-load** fp16 (`device_map=None`, pre-flight guard) | — | — | — | — | < 1 s | **`MemoryError`** |
| `baseline_qwen2-7b-q4_20260701T122111Z_failure.json` (Act 2, SC-1) | Qwen2-7B-Instruct | **Direct naive-load** fp16 (torch itself hits Windows paging limit) | — | — | — | — | download + partial load | **`OSError: The paging file is too small`** |
| `baseline_qwen2-7b-q4_20260701T123021Z_failure.json` (Act 2, SC-1) | Qwen2-7B-Instruct | **Direct naive-load** fp16 (`device_map=None`, pre-flight guard) | — | — | — | — | < 1 s | **`MemoryError`** |

**Act 1 — unexpected success via silent accelerate offload.** `shared/automodel_factory.load_causal_lm` was passing `device_map="auto"` + `low_cpu_mem_usage=True` — those two flags tell accelerate to figure out how to fit the model, using disk as an overflow device if RAM isn't enough. It did, transparently, without ever raising. The `Direct` back-end ran Llama-3-8B end-to-end for 128 tokens. The stdout carried one telltale line: `Some parameters are on the meta device because they were offloaded to the disk and cpu.` **This is essentially the same paging trick AirLLM makes explicit, done automatically by the HuggingFace stack.** Useful data point, but it defeats the whole point of the "Direct baseline" — the naive path was never attempted.

**Act 2 — forced naive load fails visibly (SC-1 anchor).** `DirectBackend.load` was corrected to override the factory defaults with `device_map=None` + `low_cpu_mem_usage=False` — the honest naive path. On Windows a raw oversized fp16 allocation is killed at OS level (`EXCEPTION_ACCESS_VIOLATION`, exit 139) *before* Python's `try/except` can catch, so a Python-level **pre-flight RAM check** was added: read local HF-cache safetensors sizes, compare to `psutil.virtual_memory().available`, raise `MemoryError` if weights > 90 % of avail RAM. The error message is engineered for the report: `Direct baseline pre-flight: meta-llama/Meta-Llama-3-8B-Instruct weights = 14.96 GiB (cached safetensors), available RAM = 1.52 GiB. Naive load would need ~14.96 GiB contiguous, exceeding the 90 % headroom. On Windows the OS kills such allocations at signal level (exit 139) before Python can catch. AirLLM (T-3.1) rescues via per-layer mmap streaming.`

### Head-to-head — three points on the memory ↔ speed frontier

| Metric | AirLLM (M2a) | Direct + accelerate offload (Act 1) | Direct naive load (Act 2, SC-1) |
|---|---|---|---|
| Runs successfully? | ✅ 2 tok in 18 min | ✅ 128 tok in 75 min | ❌ `MemoryError` at pre-flight |
| Peak RAM | **1.29 GB** (leanest) | 3.19 GB | n/a (never allocated) |
| TTFT | 367 s | **64 s** | n/a |
| TPOT | 368 s/tok | **35 s/tok** (fastest) | n/a |
| Prefill vs Decode regime | Collapsed (both memory-bound streaming) | **Distinct** (TTFT ≠ TPOT, 1.8× ratio) | n/a |

If your constraint is "≤ 4 GB free RAM, care about existence not speed", AirLLM wins. If it's "≥ 6-8 GB free RAM, need > 1 tok/min", Direct-with-accelerate-offload wins by ~10×. If your requirement is "attempt the naive load and fail visibly when it doesn't fit" (the honest baseline), the pre-flight guard is the right shape and its manifest is the SC-1 anchor. Full narrative pivot in [`docs/prompts_book.md` §11](docs/prompts_book.md#11-t-211-two-act-narrative--silent-accelerate-rescue--pre-flight-guard--honest-baseline-failure).

### Proof — weights vs RAM (byte-level, cross-checked)

The failure manifests report weight sizes and free RAM at run time. Those numbers are not opinions — they come from `pathlib.Path.stat().st_size` summed over the `.safetensors` files, and `psutil.virtual_memory().available` at the moment `load()` was called. Here is the same measurement replayed live against the current on-disk state so the manifest numbers can be audited.

**Llama-3-8B-Instruct — 4 safetensors shards under `D:/AI_agents_course/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/`:**

| Shard | Bytes | GiB |
|---|---:|---:|
| `model-00001-of-00004.safetensors` | 4,976,698,672 | 4.635 |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | 4.657 |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | 4.579 |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | 1.088 |
| **Total (fp16 weight footprint)** | **16,060,556,376** | **14.9576 GiB** |

Manifest `baseline_llama3-8b-fp16_20260701T120918Z_failure.json` recorded `weights = 14.96 GiB`. That matches the live measurement to the second decimal.

**Qwen2-7B-Instruct — 4 safetensors shards under `D:/AI_agents_course/hf_cache/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/`:**

| Shard | Bytes | GiB |
|---|---:|---:|
| `model-00001-of-00004.safetensors` | 3,945,426,872 | 3.674 |
| `model-00002-of-00004.safetensors` | 3,864,726,352 | 3.599 |
| `model-00003-of-00004.safetensors` | 3,864,726,408 | 3.599 |
| `model-00004-of-00004.safetensors` | 3,556,392,240 | 3.311 |
| **Total (all four shards, fully downloaded now)** | **15,231,271,872** | **14.1852 GiB** |

Manifest `baseline_qwen2-7b-q4_20260701T123021Z_failure.json` recorded `weights = 10.87 GiB` — that was the sum at run time (12:30 UTC), when the download was still resuming; the fourth shard finalised on disk after the pre-flight fired. The point is unchanged: at run time the pre-flight already saw more weight than free RAM.

**System RAM — `psutil.virtual_memory()` on this box:**

| Metric | Bytes | GiB | GB |
|---|---:|---:|---:|
| **Total** | 8,379,490,304 | **7.804 GiB** | 8.379 GB |
| **Available at Llama-3 pre-flight (2026-07-01T12:09Z)** | ≈ 1.633 × 10⁹ | **1.52 GiB** | 1.63 GB |
| **Available at Qwen2 pre-flight (2026-07-01T12:30Z)** | ≈ 1.653 × 10⁹ | **1.54 GiB** | 1.65 GB |
| Available now (verification, ~78 % used) | 1,780,338,688 | 1.658 GiB | 1.780 GB |

**The comparison the report rests on:**

| Target | Weights (GiB) | Free RAM at run (GiB) | Total RAM (GiB) | Over **free RAM** by | Over **total RAM** by |
|---|---:|---:|---:|---:|---:|
| Llama-3-8B fp16 | **14.958** | 1.52 | 7.804 | **+13.44 GiB (9.84×)** | **+7.15 GiB (1.92×)** |
| Qwen2-7B fp16 | **14.185** | 1.54 | 7.804 | **+12.65 GiB (9.21×)** | **+6.38 GiB (1.82×)** |

**Structural impossibility.** Both weight footprints exceed **the whole machine's RAM** (7.80 GiB total) by roughly **2×**, not just the fraction that happens to be free. Even if the operating system, browser, IDE, and every background service were shut down and 100 % of physical RAM were available for the model, the naive `AutoModelForCausalLM.from_pretrained(..., device_map=None, low_cpu_mem_usage=False)` allocation would still be short by 6.38 – 7.15 GiB per model. This is why AirLLM (per-layer mmap streaming — see M2a plumbing manifest at 1.29 GB peak RSS) and Direct + accelerate silent offload (see Act 1 manifest at 3.19 GB peak RSS) are not optional conveniences: they are the only paths that fit the constraint. The pre-flight guard's `MemoryError` is not a spurious safety-net — it is the honest expression of "this allocation cannot succeed on this box".

### Crash timeline — exact wall-clock times each failure was captured

Every failure manifest records TWO independent timestamps: the `captured_at` field inside the JSON body (set by `datetime.now(UTC)` at the moment `baseline_service.run_baseline` **begins the attempt** — line 66 of `services/baseline_service.py`), and the filesystem mtime (`Path.stat().st_mtime` — set by the OS when `write_text()` actually **flushes the failure manifest** to disk after the exception is caught). The gap between the two tells you how far into the load attempt the crash happened.

| # | Model | Attempt started (UTC) | Failure manifest written (fs mtime UTC) | Δ from start to write | error_type | Free RAM at crash | Weight footprint | Delta over free RAM |
|---:|---|---|---|---:|---|---:|---:|---:|
| 1 | Llama-3-8B fp16 | **2026-07-01 12:09:18** | 2026-07-01 12:09:18.342 | **~0.3 s** | `MemoryError` (pre-flight) | **1.52 GiB** | 14.96 GiB | **+13.44 GiB (9.84×)** |
| 2 | Qwen2-7B fp16 (bg task) | **2026-07-01 12:21:11** | 2026-07-01 12:31:09.170 | **~9 min 58 s** | `OSError` (Windows: *"The paging file is too small for this operation to complete. (os error 1455)"*) | — pre-flight passed, torch itself crashed later during shard allocation | ~10.87 GiB (partial DL at crash) | n/a — torch, not pre-flight |
| 3 | Qwen2-7B fp16 (fg retry) | **2026-07-01 12:30:21** | 2026-07-01 12:30:21.393 | **~0.4 s** | `MemoryError` (pre-flight) | **1.54 GiB** | 10.87 GiB | **+9.33 GiB (7.06×)** |

**Reading the timeline row by row:**
- **Row 1 (Llama-3, 12:09:18 UTC).** Pre-flight guard evaluated `weight_bytes = 16,060,556,376` (14.96 GiB) vs `psutil.virtual_memory().available = 1.52 GiB`. Since 14.96 GiB > 0.9 × 1.52 GiB, `MemoryError` raised. `baseline_service` caught it, wrote the manifest 0.3 s later. Manifest at `results/baseline_llama3-8b-fp16_20260701T120918Z_failure.json`.
- **Row 2 (Qwen2 background, 12:21:11 → 12:31:09 UTC).** Pre-flight *passed* at 12:21 (weight bytes on disk hadn't yet reached the crossover threshold given RAM state at that instant); the code proceeded into `AutoModelForCausalLM.from_pretrained` on the naive path. Torch spent ~10 minutes allocating shards until Windows' page-file backing store failed to grow, raising `OSError 1455 "The paging file is too small"`. `baseline_service` caught THAT exception and wrote the failure manifest at 12:31:09. Manifest at `results/baseline_qwen2-7b-q4_20260701T122111Z_failure.json`.
- **Row 3 (Qwen2 foreground retry, 12:30:21 UTC).** By 12:30, more shards had landed on D: (the pre-flight `for f in model_dir.rglob("*.safetensors")` sum now saw 10.87 GiB where it saw less at 12:21) AND `psutil.virtual_memory().available` had drifted to 1.54 GiB. `10.87 GiB > 0.9 × 1.54 GiB` → `MemoryError` raised, manifest written 0.4 s later. Manifest at `results/baseline_qwen2-7b-q4_20260701T123021Z_failure.json`.

**What the three rows together prove:** the same weights on the same machine hit **two distinct Windows failure modes** depending on which of `psutil.virtual_memory().available` or the safetensors-on-disk sum crosses the threshold first at run time. The pre-flight guard produces a clean Python-level `MemoryError` **within one second** of the attempt starting (rows 1 and 3). When the guard's own inputs are in a state where the condition doesn't fire (row 2), torch itself takes over the allocation attempt for ~10 minutes before Windows raises the OS-level `OSError 1455`. Either way, the failure is captured to disk with a byte-precise, timestamped audit trail — SC-1 is satisfied honestly on both paths.

### Witness runs — proof both models **actually attempted the load and crashed because of RAM**

The pre-flight guard is preemptive — it decides *not* to attempt the load when it can prove the load would fail. That is a protective safety net, but it does not by itself show that the load "would have crashed because of RAM". To produce that evidence, `tools/baseline_witness.py` spawns `uv run on-prem-llm run-baseline <target> --skip-preflight` as a subprocess, samples `psutil.virtual_memory()` every 2 s in the parent, walks the child process tree to sum RSS, and on child exit writes a `witness_baseline_<label>_<ts>.json` manifest carrying the full RAM timeline, exit code, wall time, and stderr tail. This is the reactive "the load tried and died" evidence — the pre-flight guard is disabled explicitly so torch actually reaches for the naive `AutoModelForCausalLM.from_pretrained(..., device_map=None, low_cpu_mem_usage=False)` allocation.

Both models were witnessed on 2026-07-01 T12:5x UTC on the same 7.804 GiB / CPU-only box. Both stderr streams contain `Loading checkpoint shards: 0%|          | 0/4` — hard proof that torch **began** the checkpoint load — followed by `OSError: The paging file is too small for this operation to complete. (os error 1455)`.

| Metric | Llama-3-8B fp16 (witness) | Qwen2-7B-Instruct fp16 (witness) |
|---|---|---|
| **Manifest** | `results/witness_baseline_llama3-8b-fp16_20260701T125650Z.json` | `results/witness_baseline_qwen2-7b-q4_20260701T125740Z.json` |
| Attempt started (UTC) | 2026-07-01 12:56:50.9 | 2026-07-01 12:57:40.5 |
| Wall time until crash | **10.15 s** | **28.36 s** |
| Exit code | 1 (caught at Python level by `baseline_service`) | 1 (caught at Python level by `baseline_service`) |
| **Peak child-tree RSS during load** | **0.286 GiB (293 MB)** | **0.287 GiB (294 MB)** |
| **Trough system available RAM during load** | **1.473 GiB** | **1.518 GiB** |
| torch actually started loading? | ✅ yes — stderr: `Loading checkpoint shards: 0%|          | 0/4` | ✅ yes — stderr: `Loading checkpoint shards: 0%|          | 0/4` |
| Terminal error | `OSError: The paging file is too small for this operation to complete. (os error 1455)` | `OSError: The paging file is too small for this operation to complete. (os error 1455)` |
| Companion Python-level manifest | `baseline_llama3-8b-fp16_20260701T125500Z_failure.json` (`error_type: "OSError"`) | `baseline_qwen2-7b-q4_20260701T125744Z_failure.json` (`error_type: "OSError"`) |

**Llama-3-8B RAM curve** (from `witness_baseline_llama3-8b-fp16_20260701T125650Z.json`, `samples[]`):

| Elapsed (s) | System available (GiB) | Child-tree RSS (GiB) | Notes |
|---:|---:|---:|---|
| 0.03 | 1.693 | 0.019 | subprocess just spawned, python still importing |
| 2.05 | 1.508 | 0.191 | transformers imported + torch initialised |
| 4.08 | 1.490 | 0.229 | tokenizer + config loaded, first shard opened |
| 6.10 | 1.510 | 0.247 | mmap of first shard in progress |
| 8.13 | 1.473 | **0.286** ← peak | Windows raises `OSError 1455` when expanding the page file to accept more |
| 10.15 | 1.728 | 0.000 | process dead, RAM reclaimed |

**Qwen2-7B RAM curve** (from `witness_baseline_qwen2-7b-q4_20260701T125740Z.json`, `samples[]`):

| Elapsed (s) | System available (GiB) | Child-tree RSS (GiB) | Notes |
|---:|---:|---:|---|
| 0.03 | 1.764 | 0.019 | subprocess just spawned |
| 2.04 | 1.597 | 0.193 | transformers imported + torch initialised |
| 4.06 | 1.541 | 0.242 | first shard mmap started |
| 6.08 – 24.29 | 1.518 – 1.554 | **0.258** (held for ~20 s) | Windows attempting to expand the page file — repeated OS-level retries visible as a memory plateau |
| 26.32 | 1.531 | **0.287** ← peak | Windows finally gives up expanding, raises `OSError 1455` |
| 28.36 | 1.793 | 0.000 | process dead |

**Interpretation.** Peak child-tree RSS ≈ **287 MB** for both models — that is torch + transformers + Python + partial shard mmap. It is **~50× smaller than the model weights** (14.96 GiB Llama, 14.19 GiB Qwen) because Windows kills the load *before* torch can commit even one full shard: the OS page file cannot be expanded fast enough to back the ~15 GiB contiguous allocation torch requests. The load did not fail because our code was wrong; the load failed because the *operating system refused the memory request* the naive load path issues. This is the honest baseline failure the M3 AirLLM back-end has to rescue — and every column in the tables above traces to a specific JSON field in a manifest committed under `results/`.

### Model storage drive (important on small system disks)

The target models (Llama-3-8B fp16 ≈ 16 GB, Qwen2-7B ≈ 14 GB) plus their AirLLM per-layer shards can easily exceed 60 GB of disk. If your home drive (`C:` on Windows, `~` on Linux/macOS) does not have ~50 GB free, redirect both the HF cache and the AirLLM shard path to a larger drive **before** the first model download:

1. In `config/setup.json`, point `airllm.layer_shards_saving_path` and `hf.cache_dir` at directories on the larger drive (e.g. `D:/AI_agents_course/airllm_shards` + `D:/AI_agents_course/hf_cache`).
2. In your local `.env`, set `HF_HOME=` to the same `hf_cache` path — `huggingface_hub` reads this env var directly.
3. Re-run `uv run init_env.py` so `HardwareScanner` measures the *new* path and refreshes the disk reading in this file + `docs/PRD.md`.

Full root-cause writeup of why both knobs are needed (env var + config) lives in [`docs/prompts_book.md` §5](docs/prompts_book.md#5-t-2a5-prep--storage-drive--airllm-compatibility).

## Hardware profile (auto-populated by `init_env.py` — do not hand-edit)

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
