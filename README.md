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
