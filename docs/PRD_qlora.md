# PRD — QLoRA Fine-Tune Extension (Assignment §5.7 · Milestone M5)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory). **Explicitly required** by `docs/PRD.md` FR-22: "*The QLoRA mechanism MUST have its own dedicated requirements document at `docs/PRD_qlora.md` before implementation begins*."
> **Tracked by:** `docs/TODO.md` §7 DP-7 — **MANDATORY**.
> **Blocks:** **M5 milestone** entirely — T-5.1 (`services/qlora_trainer.py`), T-5.2 (`SDK.run_qlora_finetune`), T-5.3 (CLI `run-qlora`), T-5.4 (VRAM measurement mixin), T-5.5 (loss-curve chart), T-5.6 (before/after quality comparison), T-5.7 (README §QLoRA Extension). Approval required before any M5 code is written.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 → `ex05-AirLLM.pdf` §5.7 (§5 extensions — QLoRA / SmoothQuant / GPTQ options, D-3 locks in QLoRA) → `L08-summary-Lora-AirLLM.pdf` §5.1 (NF4 + Paged Optimizers), §7.1 (train VRAM = 3-5× inference VRAM), §7.3 (LoRA algebra), §7.5 (QLoRA = NF4 base + LoRA adapters) → `docs/PRD.md` v1.10 D-3 (§5.7 = QLoRA locked) + G6 (fine-tune demonstration) + K8 (train-vs-inference VRAM ratio) + FR-20/21/22 + ADR-014.
> **Empirical anchor:** none yet — M5 is not started. Once M5 lands, the anchor becomes `results/qlora_<label>_<ts>.json` (loss curve + train/inference VRAM peaks).
> **Document version:** 1.00 — 2026-07-01.
> **Status:** **APPROVED 2026-07-01** — user (ndvp39@gmail.com) confirmed all 5 open-question defaults: Q-QL-1 = TinyLlama-1.1B base, Q-QL-2 = alpaca-cleaned 500 samples, Q-QL-3 = max_steps=200, Q-QL-4 = rank=8/alpha=16, Q-QL-5 = `torch.cuda.max_memory_allocated()`. M5 code (T-5.1..T-5.7) cleared to start.

---

## 1. What and why

### 1.1 What
Formalise the **QLoRA fine-tune extension** — how the SDK trains LoRA adapters on top of an NF4-quantized base model, measures the train/inference VRAM ratio the L08 §7.1 curriculum highlights, and produces a before/after qualitative demonstration for the report's §5.7 extension section. Covers:

* **Training venue** — where the QLoRA training runs (spoiler: Colab / cloud GPU, not the on-prem CPU laptop — see §2.4 for the hard constraint).
* **Base model choice** — which of the sweep's target models is used, and why.
* **Dataset choice** — a tiny public SFT dataset the training loop consumes.
* **LoRA hyperparameters** — rank, alpha, target modules, learning rate, batch size, gradient accumulation, epochs / steps.
* **Measurement** — train VRAM peak, inference VRAM peak, wall time, loss curve, output samples on a fixed prompt before and after fine-tuning.
* **Artifact** — `results/qlora_<label>_<ts>.json` manifest + `results/qlora_loss_<ts>.csv` + `figures/qlora_loss.png` + before/after text samples inline in the report.

This PRD does introduce new code (`services/qlora_trainer.py`) — the module doesn't exist yet.

### 1.2 Why
* **Assignment §5.7 = QLoRA (D-3 lock).** The extension is committed at the PRD level; skipping it means failing G6 and K8. Course guidelines require ONE §5.7 extension; D-3 picked QLoRA over the alternatives (SmoothQuant, GPTQ).
* **L08 §7.1 curriculum linkage.** The lecture's headline observation — "training memory is 3-5× inference memory because of optimizer state + gradients + activations, and Paged Optimizers + NF4 make it possible on a single consumer GPU" — is the pedagogical point of §5.7. The report MUST demonstrate this empirically.
* **Report narrative closure.** Without QLoRA, the report's economic argument ("3-way deployment options") leaves the "on-prem is only for inference" gap open — LoRA/QLoRA is the counter that says "on-prem can even train the model if you have modest GPU memory."
* **Assignment K8.** "LoRA/QLoRA fine-tune completes; train-vs-inference VRAM ratio reported." This PRD encodes both parts into concrete measurable success criteria.

### 1.3 Who consumes this policy
| Consumer | Where the policy applies | Landing task |
|---|---|---|
| `services/qlora_trainer.py` | New module — orchestrates NF4 base load + LoRA attach + Trainer loop + VRAM sampling | T-5.1 |
| `mixins/vram_sampling_mixin.py` | Extends `memory_sampling_mixin.py` with GPU-VRAM sampling via `torch.cuda.max_memory_allocated()` | T-5.4 |
| SDK method `run_qlora_finetune(target_label, dataset_path, lora_config)` | Replaces the current `_future_stubs.py` stub | T-5.2 |
| CLI `run-qlora` | Delegates to SDK; prints train VRAM / inference VRAM / final loss / where the artifacts land | T-5.3 |
| Report assembler (T-6.1) | Embeds `figures/qlora_loss.png` + train/inference VRAM ratio + before/after text into README §QLoRA Extension | T-6.1 |

---

## 2. Theoretical background & definitions

### 2.1 LoRA (Low-Rank Adaptation) — L08 §7.3
For a pretrained weight matrix `W₀ ∈ R^(d×k)`, LoRA freezes `W₀` and adds a low-rank update `ΔW = B · A`, where `A ∈ R^(r×k)` and `B ∈ R^(d×r)` with rank `r ≪ min(d, k)`. Forward pass becomes `Wx = W₀x + B · A · x`. Only `A` and `B` are trainable, reducing trainable-parameter count from `d·k` to `r·(d+k)` — typically 100-10 000× less.

### 2.2 QLoRA — L08 §5.1 + §7.5
QLoRA = LoRA where **the base model `W₀` is stored in NF4** (4-bit normal float, a data type Anthropic Dettmers et al. developed for weight quantization with information-preserving guarantees). Training math still happens in bf16/fp16 via **on-the-fly dequantization**; the frozen base stays in NF4 in VRAM. Combined with:
* **Paged Optimizers** — CPU-RAM-backed offload of optimizer state (Adam's momentum + variance buffers) that gets paged into VRAM only on the step where the optimizer runs. Cuts peak optimizer VRAM by ~50-75%.
* **Gradient checkpointing** — trades compute for memory by re-computing forward activations during backward pass. Cuts activation VRAM by ~40-60%.
* **Double quantization** — quantize the quantization constants themselves, saving ~0.4 bits per parameter.

Result: **7 B parameter models can be QLoRA-fine-tuned on a single 24 GB consumer GPU (RTX 3090 / 4090)** or a **13 B model on a 40 GB A100**, or a **1-3 B model on a free-tier T4 (16 GB VRAM)**.

### 2.3 Train VRAM vs Inference VRAM ratio — L08 §7.1
For a `P`-parameter model at bit-width `b`:
* **Inference VRAM ≈ P × b / 8** (weights only; KV cache is per-request and comparable to activations).
* **Training VRAM ≈ P × b / 8 + P × (2 gradient-bytes + 8 Adam-state-bytes + activation-bytes) = 3-5× inference.**

QLoRA's contribution: replace `P × b / 8` for the frozen base with `P × 4 / 8` (NF4) and only carry gradients/optimizer state for the ~1% LoRA parameters. Net: training VRAM approaches inference VRAM (within ~1.5×) instead of 3-5×.

**The report's K8 claim:** measure both peaks empirically and confirm the QLoRA ratio is dramatically flatter than the naive `3-5×` L08 §7.1 formula predicts.

### 2.4 Why QLoRA CANNOT run on the reference CPU-only laptop

`bitsandbytes` NF4 kernels are **CUDA-only** — there is no CPU implementation as of Jan 2026. Without NF4, the "QLoRA" name is a misnomer; you'd be doing plain LoRA on fp16 (which fits on CPU but is 4× slower per step and doesn't demonstrate the K8 VRAM story).

**Hard constraint:** M5 training runs on a **rented cloud GPU** (Colab free-tier T4 is sufficient — 16 GB VRAM). Inference of the fine-tuned model then also runs on GPU (adapter merge → AirLLM path on-prem is optional).

**This is CONSISTENT with the assignment.** §5.7 extension does not require training on the on-prem laptop; the on-prem constraint is for INFERENCE benchmarks (§1-§4). Cloud GPU for training is the natural application of the "3 deployment options" thesis the report already argues.

---

## 3. What this PRD is NOT

* **Not the inference benchmark.** M2b/M3's baseline + AirLLM sweep already covers the inference side (K3, K4, K5).
* **Not a re-pretraining exercise.** LoRA adapts a pretrained model to a downstream task on a small dataset — measured in hundreds of examples, not billions.
* **Not RLHF / DPO.** Preference-based fine-tuning is out of scope; this is supervised fine-tuning (SFT).
* **Not a serving system.** Post-training, the report shows generation quality on a fixed prompt. Deploying the fine-tuned adapter for real users is out of scope.
* **Not multi-GPU / distributed training.** Single-GPU only, Colab-scale.

---

## 4. Building block: `services/qlora_trainer.py`

### 4.1 Input
* `config/setup.json.qlora` — new config subtree:
  ```json
  "qlora": {
      "base_model_label": "llama3-8b-fp16",
      "dataset": {
          "hf_id": "yahma/alpaca-cleaned",
          "subset_size": 500,
          "split": "train"
      },
      "lora": {
          "rank": 8,
          "alpha": 16,
          "dropout": 0.05,
          "target_modules": ["q_proj", "v_proj"]
      },
      "training": {
          "epochs": 1,
          "max_steps": 200,
          "batch_size": 1,
          "gradient_accumulation_steps": 8,
          "learning_rate": 2e-4,
          "warmup_steps": 20,
          "gradient_checkpointing": true,
          "paged_optimizer": true
      },
      "measurement": {
          "eval_prompt": "Explain quantum entanglement in one paragraph.",
          "max_new_tokens": 128
      }
  }
  ```
* `.env` — `HF_TOKEN` (existing) + optional `WANDB_API_KEY` (opt-in for logging).

### 4.2 Output
* `results/qlora_<label>_<ts>.json` — manifest with: `train_vram_peak_mb`, `inference_vram_peak_mb`, `vram_ratio` (train/inference), `final_train_loss`, `total_steps`, `wall_s_train`, `wall_s_inference_before`, `wall_s_inference_after`, `output_before`, `output_after`, `config_snapshot`, `git_hash`, `python`, `package_version`, `trainable_params`, `total_params`, `trainable_pct`, `venue` (e.g. `"colab_t4_16gb"`).
* `results/qlora_loss_<ts>.csv` — per-step training loss: `step`, `loss`, `learning_rate`, `elapsed_s`.
* `figures/qlora_loss.png` — matplotlib plot of loss vs step.
* `results/qlora_adapter_<label>_<ts>/` — saved LoRA adapter weights (small — usually 10-50 MB).

### 4.3 Setup
* **`peft`** — LoRA / QLoRA APIs (`LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained`).
* **`bitsandbytes`** — NF4 quantization at load time (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`).
* **`accelerate`** — required by peft/trl for the training loop.
* **`transformers.Trainer`** — the SFT loop itself. Trainer handles gradient accumulation, learning-rate schedule, checkpointing.
* **`datasets`** — Hugging Face datasets library for loading the SFT dataset.
* **CUDA** — CUDA 11.8+ available (Colab T4 provides this by default).

### 4.4 Venue = Colab notebook
* Training runs in a Colab notebook `notebooks/qlora_training.ipynb`.
* Notebook is checked in to the repo; the report's §QLoRA Extension embeds screenshots of the loss curve + terminal output + before/after generation.
* Notebook parameters come from `config/setup.json.qlora` (copied into the notebook via cell 1) — so re-running with different LoRA rank / dataset size doesn't need notebook edits.
* Notebook outputs are downloaded and committed to `results/qlora_*` + `figures/qlora_*` in the repo.

---

## 5. The T-5.x API surface

### 5.1 Data types (`sdk/qlora_types.py`)

```python
@dataclass(frozen=True, kw_only=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

@dataclass(frozen=True, kw_only=True)
class QLoraTrainingConfig:
    epochs: int
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    paged_optimizer: bool

@dataclass(frozen=True, kw_only=True)
class QLoraRunResult:
    run_id: str
    started_at: str
    venue: str
    base_model_label: str
    base_model_id: str
    dataset_hf_id: str
    dataset_subset_size: int
    lora: LoraConfig
    training: QLoraTrainingConfig
    trainable_params: int
    total_params: int
    trainable_pct: float
    train_vram_peak_mb: float
    inference_vram_peak_mb: float
    vram_ratio: float
    final_train_loss: float
    total_steps: int
    wall_s_train: float
    wall_s_inference_before: float
    wall_s_inference_after: float
    output_before: str
    output_after: str
    adapter_path: str
    loss_curve_csv: str
```

### 5.2 SDK method

```python
def run_qlora_finetune(
    self,
    target_label: str,
    *,
    dataset_path: Path | None = None,       # override config.qlora.dataset if given
    lora_config: LoraConfig | None = None,   # override config.qlora.lora if given
    training_config: QLoraTrainingConfig | None = None,
    venue: str = "colab_t4_16gb",
) -> QLoraRunResult: ...
```

### 5.3 CLI

```bash
uv run on-prem-llm run-qlora llama3-8b-fp16 \
    --dataset yahma/alpaca-cleaned \
    --rank 8 --alpha 16 \
    --max-steps 200 \
    --venue colab_t4_16gb
```

### 5.4 Notebook contract
`notebooks/qlora_training.ipynb` cells:
1. **Setup** — install `peft bitsandbytes accelerate transformers datasets`; check CUDA.
2. **Config** — paste `config/setup.json.qlora` verbatim.
3. **Load base + tokenizer** — HF snapshot → `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(...))`.
4. **Attach LoRA** — `peft.get_peft_model(base, peft.LoraConfig(...))`.
5. **Load dataset** — `datasets.load_dataset(...)`, slice `[:subset_size]`, tokenize.
6. **Baseline inference** — record `output_before` at fixed prompt + inference VRAM peak.
7. **Train** — `transformers.Trainer(model, ...).train()` with `on_step_end` callback logging step + loss + wall time to CSV.
8. **Post-train inference** — record `output_after` at fixed prompt + inference VRAM peak.
9. **Save adapter + manifest** — `model.save_pretrained(...)`; write `results/qlora_<label>_<ts>.json` with all measurements.
10. **Plot loss** — matplotlib CSV → PNG.

---

## 6. Functional Requirements (FR-QL)

* **FR-QL-1.** `services/qlora_trainer.py` MUST expose a pure orchestration function that takes `LoraConfig` + `QLoraTrainingConfig` + resolved dataset + base model reference and returns `QLoraRunResult`. No file I/O side effects at the orchestration layer — the notebook handles I/O.
* **FR-QL-2.** VRAM peaks MUST be sampled via `torch.cuda.max_memory_allocated(device=0)` reset before each measurement window (train / inference_before / inference_after).
* **FR-QL-3.** Trainable-parameter count MUST be reported alongside total-parameter count so the report can quote the "~1% of total params" LoRA efficiency claim.
* **FR-QL-4.** Training MUST use `paged_adamw_8bit` optimizer when `paged_optimizer=true` (default), else `adamw_torch` — the config toggle allows measuring the paged-optimizer VRAM savings if desired.
* **FR-QL-5.** Gradient checkpointing MUST be enabled by default (`gradient_checkpointing=true`).
* **FR-QL-6.** The dataset loader MUST honour `subset_size` (slice from the beginning of `split`); MUST NOT shuffle by default so the run is deterministic.
* **FR-QL-7.** Training MUST use a fixed random seed from `config.generation.seed` (currently 42) so re-runs are reproducible.
* **FR-QL-8.** Final training loss + step curve MUST be persisted as CSV `results/qlora_loss_<ts>.csv` for T-5.5 plotting.
* **FR-QL-9.** Adapter weights MUST be saved to `results/qlora_adapter_<label>_<ts>/` via `PeftModel.save_pretrained()` (10-50 MB — small enough to git-commit).
* **FR-QL-10.** The fixed evaluation prompt is `config.qlora.measurement.eval_prompt` (default `"Explain quantum entanglement in one paragraph."`). Both `output_before` and `output_after` are generated at `max_new_tokens=128` with `temperature=0.0` (matches inference config).
* **FR-QL-11.** SDK method `run_qlora_finetune()` MUST replace the current `_future_stubs.py` stub — env guard first, then delegate to the trainer.
* **FR-QL-12.** CLI `run-qlora` MUST invoke the SDK method + print `OK: qlora_<label>_<ts>.json (train/inference VRAM ratio = <ratio>)` on success.
* **FR-QL-13.** The manifest MUST record `venue` (e.g., `"colab_t4_16gb"`) so the report can be honest about where training ran (versus the on-prem CPU inference venue).

## 7. Non-Functional Requirements (NFR-QL)

* **NFR-QL-1.** Zero hard-coded numeric constants — LoRA rank/alpha, epochs, max_steps, batch_size, learning_rate ALL come from `config.qlora`.
* **NFR-QL-2.** No new heavy dependency added to `pyproject.toml` main group — `peft`, `bitsandbytes`, `accelerate`, `datasets` land in an OPTIONAL `[qlora]` extra so the CPU-only inference stack doesn't get bloated. Colab installs the extra explicitly.
* **NFR-QL-3.** File-size cap: `qlora_trainer.py` ≤ 150 LOC per constitution §2.2. May need split into `qlora_orchestrator.py` + `qlora_measurement.py` mirroring the T-3.5 sweep_runner + sweep_stats split.
* **NFR-QL-4.** Notebook `notebooks/qlora_training.ipynb` MUST be committed CLEAN (outputs stripped except the loss-plot cell which stays for the report screenshot). Use `nbstripout` pre-commit hook.
* **NFR-QL-5.** Notebook MUST use ONLY `uv run`-compatible commands — no `pip install`, no `conda`. On Colab, this reduces to `!pip install .[qlora]` at cell 1 (Colab pre-installs pip and doesn't ship uv).

---

## 8. Success Criteria (SC-QL)

* **SC-QL-1.** Training completes without OOM on Colab T4 (16 GB VRAM) for `base_model_label = tinyllama-1.1b` OR `base_model_label = llama3-8b-fp16` with `max_steps ≤ 200`.
* **SC-QL-2.** Loss curve shows monotonic-ish decrease over first 100 steps (allow noise; require `final_loss < 0.8 × initial_loss`).
* **SC-QL-3.** `output_after` differs from `output_before` textually — measured by simple token-diff (`sacrebleu` BLEU < 0.7 between them is fine).
* **SC-QL-4.** `train_vram_peak_mb` and `inference_vram_peak_mb` are both non-zero (proves CUDA VRAM sampling wired correctly).
* **SC-QL-5.** `vram_ratio = train_vram_peak_mb / inference_vram_peak_mb` is reported. **Expected: 1.5-2.5×** (QLoRA's flatter ratio due to NF4 base + paged optimizer). The report contrasts this against L08 §7.1's 3-5× naïve ratio. Both directions are publishable.
* **SC-QL-6.** `trainable_pct` is reported (expected: 0.1-1% depending on rank). This IS the LoRA efficiency claim.
* **SC-QL-7.** Adapter file `results/qlora_adapter_<label>_<ts>/adapter_model.safetensors` exists + is loadable via `PeftModel.from_pretrained()` (round-trip test).
* **SC-QL-8.** Ruff clean, ≤ 150 LOC per file, unit tests for `qlora_trainer.py` orchestration cover the pure functions (mock `Trainer` + mock CUDA calls). Notebook itself is not unit-tested.

---

## 9. Test Scenarios

* **U-QL-1** — `LoraConfig` dataclass is frozen + immutable (cannot mutate rank at runtime).
* **U-QL-2** — `QLoraTrainingConfig` accepts all documented fields; extra fields raise `TypeError`.
* **U-QL-3** — Orchestrator function accepts a mock `Trainer` (via dependency injection) and produces a `QLoraRunResult` with all expected fields populated.
* **U-QL-4** — `subset_size` slicing is deterministic — same seed → same subset.
* **U-QL-5** — `vram_ratio` computation: `train=6144, inference=3072` → ratio=2.0.
* **U-QL-6** — Manifest JSON round-trip: write → read → dataclass reconstruction identity.
* **U-QL-7** — CSV loss round-trip: read via `csv.DictReader` produces one row per training step.
* **U-QL-8** — `paged_optimizer=false` config → orchestrator selects `adamw_torch`, not `paged_adamw_8bit`.
* **U-QL-9** — CLI `run-qlora --help` lists all documented flags.
* **U-QL-10** — CLI `run-qlora <invalid_label>` exits 1 with `FAIL: unknown target_label` message.
* **I-QL-1** — Golden-file integration: run a tiny 5-step training on a mock model (2-layer transformer, 32-dim), verify manifest + CSV + adapter directory land correctly. No CUDA required (uses CPU trainer for the test).
* **I-QL-2** — Notebook-execution smoke: `jupyter nbconvert --execute notebooks/qlora_training.ipynb --to notebook --output /tmp/executed.ipynb --ExecutePreprocessor.timeout=60` succeeds against a stubbed model. Notebook is treated as executable spec.

---

## 10. Alternatives Considered

| # | Alternative | Rejected because |
|---|---|---|
| A-QL-1 | Run QLoRA locally on CPU using regular LoRA (no NF4) | Violates "QLoRA" name; L08 §5.1 pedagogy depends on NF4; wall time impractical (days per epoch) |
| A-QL-2 | Skip QLoRA, do plain LoRA on fp16 CPU as the extension | Assignment D-3 explicitly picks QLoRA; downgrading defeats §5.7 lock |
| A-QL-3 | Use `unsloth` (2-5× QLoRA speedup) instead of vanilla peft/bitsandbytes | Adds a heavy dependency; peft/bitsandbytes are already the L08 §5.1 canonical stack; unsloth is optional NFR-QL-5 future extension |
| A-QL-4 | Fine-tune Llama-3-8B (matches sweep) instead of TinyLlama-1.1B | Llama-3-8B QLoRA fits on Colab T4 but leaves ~2 GB headroom; if training OOMs on the assignment day, we lose data. **Config-controllable — user picks** |
| A-QL-5 | Use a larger dataset (e.g., 10 000 samples) | Wall time explodes on free-tier Colab (12 hr session limit); tiny SFT demo is what the assignment asks for |
| A-QL-6 | Full DPO / RLHF instead of SFT | Out of scope — SFT is the L08 §7.5 QLoRA canonical example |
| A-QL-7 | Rent an A100 hourly for QLoRA training | Adds cost + billing complexity; Colab free-tier T4 is sufficient for demonstration and matches "under-resourced" narrative |
| A-QL-8 | Skip the notebook, do everything in the CLI | Notebook is the natural venue for a cloud-GPU exploration workflow; report can embed the notebook's loss plot directly |

---

## 11. Open Questions (blocking approval)

* **Q-QL-1 (Base model for the QLoRA demo).**
  * Option A: **`llama3-8b-fp16`** (matches sweep, story-consistent, tight on T4 16 GB — max_steps limited to ~50-100)
  * Option B: **`tinyllama-1.1b`** (spare capacity, safer, but doesn't match the sweep's models — report has to explain why the fine-tune subject is a different model than the inference-benchmark subject)
  * Author's suggested default: **Option B (tinyllama-1.1b) for reliability.** The K8 VRAM-ratio story is model-agnostic. Report explicitly notes the choice.
* **Q-QL-2 (Dataset).**
  * Option A: **`yahma/alpaca-cleaned`** (52k instructions, well-known Alpaca-style SFT)
  * Option B: **`databricks/databricks-dolly-15k`** (15k open-licence instruction-response)
  * Option C: a custom 500-example synthetic dataset committed to the repo (100% reproducible, no HF Hub dependency)
  * Author's suggested default: **Option A (Alpaca-cleaned)** with `subset_size=500`. Standard practice, canonical benchmark, HF Hub already reachable per our `HF_TOKEN` + gatekeeper setup.
* **Q-QL-3 (max_steps).**
  * Option A: **`max_steps=200`** (train VRAM measurable, ~10-15 min on T4; loss trajectory clear)
  * Option B: **`max_steps=500`** (fuller loss curve; ~25-40 min on T4)
  * Option C: **`epochs=1`** (dataset-size-driven)
  * Author's suggested default: **Option A (max_steps=200).** Demonstrates the K8 story without hoarding Colab hours; enough steps for a meaningful loss curve.
* **Q-QL-4 (LoRA rank).**
  * Option A: **`rank=8, alpha=16`** (parameter-efficient, ~0.1% trainable — canonical QLoRA paper setting)
  * Option B: **`rank=16, alpha=32`** (more capacity; ~0.2% trainable)
  * Author's suggested default: **Option A (rank=8, alpha=16).** L08 §7.5's canonical example; report can quote it verbatim.
* **Q-QL-5 (VRAM measurement approach).**
  * Option A: `torch.cuda.max_memory_allocated()` (per-process peak — simple, exact-for-what-we-need)
  * Option B: `nvidia-smi` polling in a background thread (external, catches other GPU processes too — noisy in shared Colab)
  * Author's suggested default: **Option A (torch.cuda.max_memory_allocated()).** Cleaner isolation; Colab's shared GPU won't confound the measurement.

Author's suggested defaults: **Q-QL-1 = tinyllama-1.1b; Q-QL-2 = alpaca-cleaned 500 samples; Q-QL-3 = max_steps=200; Q-QL-4 = rank=8/alpha=16; Q-QL-5 = torch.cuda API.**

**RESOLVED 2026-07-01:** all 5 defaults approved by user (ndvp39@gmail.com) inline in chat. M5 (T-5.1..T-5.7) cleared to begin implementation once DP-6 (Roofline) is also approved and the T-3.6b sweep is complete.

---

## 12. Landing plan (informative, not gated by this PRD)

* **T-5.0** — This PRD authored + approved (2026-07-01).
* **T-5.1** — `services/qlora_trainer.py` (pure orchestration — mockable Trainer) + `services/qlora_measurement.py` (VRAM sampling helpers) + unit tests.
* **T-5.2** — SDK method `run_qlora_finetune()` (replaces `_future_stubs` stub) + integration test (golden-file with mocked Trainer).
* **T-5.3** — CLI `run-qlora` command + `--help` documentation.
* **T-5.4** — `mixins/vram_sampling_mixin.py` (extends memory_sampling_mixin with `torch.cuda.max_memory_allocated()` peaks).
* **T-5.5** — `services/qlora_loss_chart.py` — CSV → PNG matplotlib plot.
* **T-5.6** — `notebooks/qlora_training.ipynb` authored, executed on Colab T4, outputs downloaded to `results/qlora_*` + `figures/qlora_*`.
* **T-5.7** — README §QLoRA Extension section — embeds screenshots, quotes VRAM ratio, before/after text samples, adapter path.

Every step maps to constitution §1.3's SDLC phases: **PRD → SDK method + tests → CLI wrap → notebook run → integration + docs**.

**Rough M5 wall clock (informative):** T-5.1..T-5.5 code = ~4-6 hours. T-5.6 Colab notebook = ~1-3 hours setup + training. T-5.7 README section = ~1 hour. **Total M5 = ~1-2 days of focused work** assuming Colab quota allows.

---

## 13. Ready to review?

Every numeric target in this PRD's §8 references `config.qlora` fields (not magic constants). The 5 open questions in §11 each have a proposed default. Approval unblocks M5's T-5.1 through T-5.7.

**Approver:** user (ndvp39@gmail.com) — signs off inline in a chat reply.
