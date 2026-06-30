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
