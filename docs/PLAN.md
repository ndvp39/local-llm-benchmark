# PLAN — Architecture & Design

> **Document type:** Architecture & Design Document (SDLC Phase 2).
> **Project:** `on_prem_llm_lab`.
> **Companion to:** `docs/PRD.md` v1.10.
> **Source authority:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00.
> **Document version:** 1.20 — 2026-06-26.
> **Status:** REVISED DRAFT — incorporates the HardwareScanner auto-injection workflow and the `init_env.py` bootstrap script. Awaiting approval before remaining M1 tasks proceed.

> **Changelog (v1.10 → v1.20).** Renamed `HardwareProfiler` → `HardwareScanner` and gave it two write side-effects: (i) it injects detected CPU / RAM / GPU / VRAM / disk specs into `config/setup.json.hardware_constraints`; (ii) it replaces the paired HTML-comment markers `<!-- HARDWARE_SPECS_PLACEHOLDER:START -->` / `<!-- HARDWARE_SPECS_PLACEHOLDER:END -->` in `docs/PRD.md` and `README.md` with a formatted Markdown table. New top-level script **`init_env.py`** wires this scan + inject + patch flow as a one-shot bootstrap (`uv run init_env.py`) that runs before any other pipeline action. Added **ADR-015** (HardwareScanner side-effects + atomic writes) and **ADR-016** (`init_env.py` bootstrap rule). Added §6.7 (HardwareScanner write contracts). Updated SDK surface (`initialize_environment()`), Building Blocks table, folder layout (file rename + new root script), and risks table.

> **Changelog (v1.00 → v1.10).** Anthropic is now the sole runtime API provider (Q2). Cloud GPU curve is required in the break-even chart (Q4). §5.7 extension = QLoRA training (Q3). Target models are a JSON **array** under `config.target_models[]` and the SDK iterates them (Q1). Added a mandatory **`PlumbingTestRunner`** Building Block + `SDK.run_plumbing_test()`. Added four new ADRs: **ADR-009** (use `AutoModel*` factories for robust init), **ADR-010** (plumbing-first execution rule), **ADR-011** (Anthropic as sole API provider), **ADR-012** (3-curve break-even chart). Added **ADR-013** (target_models as JSON array, dynamically iterated) and **ADR-014** (QLoRA mechanism boundary). Updated `setup.json` schema to include `target_models[]`, `plumbing_test_model`, and `cloud_gpu.include_in_chart: true`.

---

## 1. Architectural Principles (binding)

Every design decision below derives from the constitution. The five load-bearing rules:

1. **SDK-first** (§3.1) — every business operation is reachable through `OnPremLlmSDK`. CLI / notebooks / future GUI **only** delegate. Zero business logic above the SDK line.
2. **Centralized API Gatekeeper** (§4) — every outbound API call (HF Hub download, Anthropic API, any future telemetry) goes through `ApiGatekeeper`. Rate limits, queues, retries come from `config/rate_limits.json`, never from source code.
3. **No hard-coded values** (§6.2) — model IDs (now an iterated JSON array — ADR-013), paths, URLs, prices, limits, timeouts, energy assumptions, cloud GPU rates: all in `config/*.json` or environment variables.
4. **Building Blocks with explicit Input / Output / Setup** (§15) — each component states its contract in its docstring.
5. **DRY via inheritance + Mixins** (§3.2) — when two back-ends share measurement or logging behavior, that behavior becomes a Mixin; never a copy-paste.

Two additional cross-cutting rules ratified in v1.10:

6. **Plumbing-first execution** (ADR-010) — no oversized run may execute until a plumbing test on the small/Q2 model has passed in the current install.
7. **Robust model init via `AutoModel*`** (ADR-009) — all `transformers`-based model loading goes through the generic `AutoModel` / `AutoModelForCausalLM` factories so the correct concrete class is resolved from the model's `config.json` and Class-mismatch errors are avoided (assignment §6.1 Do).

One additional cross-cutting rule ratified in v1.20:

8. **Environment-init-first execution** (ADR-016) — the bootstrap script `uv run init_env.py` MUST run once after install. It invokes the `HardwareScanner` (ADR-015), which scans hardware AND injects results into `config/setup.json` AND patches the paired placeholder markers in `docs/PRD.md` and `README.md`. Subsequent pipeline actions (plumbing test, baseline, sweep, qlora) MUST refuse to run if `config.hardware_constraints` is absent or stale (older than `init.max_age_hours` in config) — with a clear remediation message pointing back to `init_env.py`.

---

## 2. C4-Model Diagrams (ASCII; PNGs will be rendered in `assets/` during implementation)

### 2.1 C4 Level 1 — System Context

```text
                       +-----------------------------+
                       |   Reviewer / Student CLI    |
                       |    (Sarah, Dr. Yoram)       |
                       +--------------+--------------+
                                      |
                                      v
+--------------+      +-----------------------------+      +-------------------+
|  Local FS    |<---->|   on_prem_llm_lab (system)  |<---->|  Hugging Face Hub |
| (models,     |      |                             |      |  (model weights)  |
|  results,    |      |   SDK + Backends + GK       |      +-------------------+
|  figures)    |      |                             |      +-------------------+
+--------------+      |                             |<---->|  Anthropic API    |
                      +-----------------------------+      |  (sole 3rd-party) |
                                      |                    +-------------------+
                                      v
                              +---------------+
                              |  Ollama       |
                              |  (local       |
                              |  daemon, opt.)|
                              +---------------+
```

### 2.2 C4 Level 2 — Containers (process boundaries)

```text
+----------------------------------------------------------+
|  Python process: `uv run on-prem-llm <cmd>`              |
|                                                          |
|  +-----------+   +-----------+   +--------------------+  |
|  |  CLI      |-->|  SDK      |-->|  Domain Services   |  |
|  |  (typer)  |   |  facade   |   |  (orchestration)   |  |
|  +-----------+   +-----------+   +---------+----------+  |
|                                            |             |
|                                            v             |
|             +------------------------------+-----------+ |
|             |  Inference Backends (strategy)           | |
|             |  +-----------+ +-----------+ +---------+ | |
|             |  | Direct    | | AirLLM    | | Api     | | |
|             |  | (AutoMd*) | | (AutoMd*) | | (Anthr) | | |
|             |  +-----------+ +-----------+ +---------+ | |
|             +-----+--------------+--------------+------+ |
|                   |              |              |        |
|                   v              v              v        |
|             +---------------------------------------+    |
|             |  Infrastructure                       |    |
|             |  +----------+ +----------+ +--------+ |    |
|             |  | HF Hub   | | FS / SS  | | Sampler| |    |
|             |  | client   | | (mmap)   | | (psutil| |    |
|             |  +----------+ +----------+ |  pynvml)| |   |
|             |                            +--------+ |    |
|             |   +-----------------------------+    |    |
|             |   |       ApiGatekeeper         |    |    |
|             |   |  (rate, queue, retry, log)  |    |    |
|             |   +-----------------------------+    |    |
|             +---------------------------------------+    |
+----------------------------------------------------------+
```

### 2.3 C4 Level 3 — Components inside the SDK

```text
src/on_prem_llm_lab/
└── sdk/
    └── sdk.py                 # OnPremLlmSDK — single entry point (facade)
└── services/
    ├── hardware_scanner.py    # scan + inject config + patch docs (ADR-015)
    ├── model_acquirer.py      # HF download via gatekeeper
    ├── plumbing_test_runner.py# small-model + Q2 pre-flight (ADR-010)
    ├── benchmark_runner.py    # orchestrates a run + sampling
    ├── sweep_runner.py        # iterates config.target_models × quant
    ├── qlora_trainer.py       # §5.7 extension (ADR-014; PRD_qlora.md)
    ├── economic_analyzer.py   # On-Prem + Anthropic + Cloud GPU curves
    └── report_assembler.py    # interpolates README from results/
└── backends/
    ├── base.py                # InferenceBackend (abstract)
    ├── direct_backend.py      # uses AutoModelForCausalLM (ADR-009)
    ├── airllm_backend.py      # uses AirLLM with AutoModel-style init
    └── api_backend.py         # Anthropic (sole provider — ADR-011)
└── mixins/
    ├── timing_mixin.py        # TTFT / TPOT capture
    ├── memory_sampling_mixin.py
    └── manifest_logging_mixin.py
└── shared/
    ├── gatekeeper.py          # ApiGatekeeper
    ├── rate_limit_config.py   # typed reader for rate_limits.json
    ├── config.py              # central config loader; reads target_models[]
    ├── version.py             # __version__ = "1.00"
    ├── logging_setup.py
    └── paths.py               # pathlib helpers, relative paths only
└── constants.py               # immutable physical/mathematical constants
└── cli/
    └── main.py                # typer / argparse — delegate-only
```

### 2.4 C4 Level 4 — Code (key abstraction)

```python
# backends/base.py (illustrative — actual file will obey 150-LOC rule)
class InferenceBackend(ABC):
    """
    Input:  prompt (str), max_new_tokens (int), generation_params (dict).
    Output: BackendRunResult (text, prompt_tokens, completion_tokens,
            ttft_ms, tpot_ms, throughput_tps, peak_ram_mb, peak_vram_mb,
            wall_s, energy_wh, raw_logs).
    Setup:  model_id (str), quantization (Enum), backend_config (dict),
            gatekeeper (ApiGatekeeper), sampler (MemorySampler).
    Init:   subclasses MUST load weights via transformers.AutoModel*
            factories (ADR-009) — never via a concrete model class.
    """

    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int,
                 params: dict) -> BackendRunResult: ...
    @abstractmethod
    def unload(self) -> None: ...
```

### 2.5 Deployment diagram (single host)

```text
+---------------- Workstation (Windows 10) ----------------+
|                                                          |
|  Disk (NVMe/SSD)                                         |
|  ├─ ~/.cache/huggingface/  (SafeTensors / GGUF shards)   |
|  ├─ <repo>/results/        (CSV, JSON manifests)         |
|  └─ <repo>/figures/        (PNG plots)                   |
|                                                          |
|  RAM (sampled @ ≥ 2 Hz)                                  |
|  └─ AirLLM hot layer (one transformer block at a time)   |
|                                                          |
|  GPU VRAM (if present, sampled @ ≥ 2 Hz via pynvml)      |
|                                                          |
|  Python interpreter (uv-managed venv, pinned in uv.lock) |
|                                                          |
|  Optional: Ollama daemon  http://localhost:11434/v1       |
|                                                          |
+----------------------------------------------------------+
```

---

## 3. SDK Layer — Single Entry Point (constitution §3.1)

### 3.1 Public surface (illustrative)

```python
# sdk/sdk.py — public facade
class OnPremLlmSDK:
    """
    Single entry point for ALL business logic. CLI, notebooks, and any
    future GUI MUST go through this facade — they are forbidden to import
    services/, backends/, or shared/ directly.

    Input  (constructor): config_path (Path), env (Mapping[str, str]).
    Output (methods):     typed result objects (HardwareProfile,
                          BackendRunResult, SweepReport, EconomicReport,
                          PlumbingResult, QloraReport).
    Setup:                ApiGatekeeper assembled from config; logger.
    """

    def scan_hardware(self) -> HardwareScanResult: ...             # ADR-015
    def initialize_environment(self) -> InitEnvResult: ...         # ADR-016
    def download_model(self, target_label: str) -> Path: ...
    def run_plumbing_test(self) -> PlumbingResult: ...             # ADR-010
    def run_baseline(self, target_label: str, prompt: str, *,
                     max_new_tokens: int) -> BackendRunResult: ...
    def run_airllm(self, target_label: str, prompt: str, *,
                   quantization: Quant, max_new_tokens: int
                   ) -> BackendRunResult: ...
    def run_sweep(self, prompts: list[str], *,
                  quantizations: list[Quant],
                  backends: list[BackendId],
                  skip_plumbing: bool = False,         # ADR-010
                  ) -> SweepReport: ...
    def run_qlora_finetune(self, target_label: str,
                           dataset_path: Path,
                           lora_config: LoraConfig) -> QloraReport: ...
    def economic_analysis(self, sweep: SweepReport) -> EconomicReport: ...
    def assemble_readme(self, *, dry_run: bool = False) -> Path: ...
```

> `target_label` resolves to an entry inside `config.target_models[]` (ADR-013). All target-model iteration in `run_sweep` is over that array.

### 3.2 Hard rules
- The CLI module (`cli/main.py`) **MUST** contain only argument parsing + calls to `OnPremLlmSDK`. Any branch that computes a number, formats a metric, or decides a strategy is a bug.
- The notebook in `notebooks/analysis.ipynb` **MUST** use only `OnPremLlmSDK` and `pandas` for plotting — no direct imports from `backends/` or `services/`.
- `run_sweep` **MUST** assert that a successful plumbing manifest exists in `results/` (or call `run_plumbing_test` itself) before launching any oversized run (ADR-010, FR-PT-1..3).
- **Every** SDK method except `scan_hardware()` and `initialize_environment()` **MUST** assert that `config.hardware_constraints` exists and is fresh (within `init.max_age_hours`) before doing real work, raising `EnvironmentNotInitializedError` otherwise (ADR-016). The error MUST point the user back to `uv run init_env.py`.

### 3.3 Backend loading rule (ADR-009)
Every back-end implementation that uses the `transformers` library MUST instantiate models via the generic factories:

```python
# direct_backend.py — illustrative
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=resolved_torch_dtype,   # from quantization config
    device_map=resolved_device_map,
    low_cpu_mem_usage=True,
)
```

- Direct passing of a concrete class (e.g., `LlamaForCausalLM`) is **forbidden** — `AutoModel*` resolves the right class from `config.json` and avoids the Class-mismatch failure mode flagged in assignment §6.1 Do.
- The AirLLM back-end uses AirLLM's own loader, but the *tokenizer* and any auxiliary `transformers` calls within the back-end **MUST** also use `AutoTokenizer.from_pretrained`.
- A unit test (`tests/unit/test_backends/test_init_uses_automodel.py`) statically inspects each back-end module to ensure no concrete `*ForCausalLM` class is imported.

---

## 4. API Gatekeeper — Centralized Rate Control (constitution §4)

### 4.1 Interface (Building Block contract)

```python
# shared/gatekeeper.py
class ApiGatekeeper:
    """
    Input:  callable + args/kwargs (any outbound network call).
    Output: callable's return value, OR enqueued + later returned via future.
    Setup:  RateLimitConfig (loaded from config/rate_limits.json),
            QueuePolicy (FIFO, max depth from config),
            RetryPolicy (max_retries, retry_after_seconds from config).
    """

    def execute(self, service: str, callable_: Callable[..., T],
                *args, **kwargs) -> T: ...

    def get_queue_status(self, service: str) -> QueueStatus: ...
```

### 4.2 Required behavior (matches constitution §4.1, §4.3)
- Before each call: check the per-service window (`requests_per_minute`, `requests_per_hour`, `concurrent_max`).
- If allowed: execute and log (structured JSON line: timestamp, service, latency_ms, status).
- If denied due to rate: enqueue (FIFO), apply backpressure flag when `queue_max_depth` is reached, drain as windows reset.
- On transient failure: retry up to `max_retries` with `retry_after_seconds` backoff.
- On hard failure: raise a typed `GatekeeperError` with structured context — never silent.

### 4.3 Rate-limit configuration source (no hard-coding)

```json
// config/rate_limits.json
{
  "version": "1.00",
  "services": {
    "default":            { "requests_per_minute": 30,  "requests_per_hour": 500,  "concurrent_max": 5, "retry_after_seconds": 30, "max_retries": 3, "queue_max_depth": 100 },
    "huggingface_hub":    { "requests_per_minute": 60,  "requests_per_hour": 1000, "concurrent_max": 4, "retry_after_seconds": 15, "max_retries": 5, "queue_max_depth": 50  },
    "anthropic_messages": { "requests_per_minute": 20,  "requests_per_hour": 400,  "concurrent_max": 2, "retry_after_seconds": 30, "max_retries": 3, "queue_max_depth": 50  }
  }
}
```

> Anthropic is the only third-party LLM provider configured at runtime (ADR-011). Numeric values are defaults to be tuned against Anthropic's published limits — they live in JSON precisely so they can be tuned without touching code.

### 4.4 Cross-cutting wiring
Every `InferenceBackend.api_call` and every `model_acquirer.download` accepts the `gatekeeper` via constructor injection. No service may construct its own gatekeeper — the SDK builds one and shares it.

---

## 5. OOP, DRY, Mixins, Building Blocks (constitution §§3.2, 15)

### 5.1 Mixin catalog (each = exactly ONE concern; independently testable)

| Mixin | Concern | Composed into |
|-------|---------|---------------|
| `TimingMixin` | Wall-clock + per-token timing (`ttft_ms`, `tpot_ms`) | All three backends |
| `MemorySamplingMixin` | Background sampler thread for RAM (psutil) + VRAM (pynvml) | All three backends |
| `ManifestLoggingMixin` | Write `results/run_<timestamp>.json` with seed/config snapshot | All three backends |
| `EnergyAccountingMixin` | Multiply wall-clock × configured wattage → Wh + $ | Direct, AirLLM only (API path uses provider billing) |

Rules (per constitution §3.2 table):
- Each Mixin provides one concern.
- No Mixin overrides another's methods.
- Each Mixin has its own `tests/unit/test_mixins/test_<mixin>.py`.

### 5.2 Strategy pattern for back-ends

```text
InferenceBackend (ABC, declares load/generate/unload + Building-Block contract)
   ├── DirectBackend          (uses transformers.AutoModelForCausalLM)  -- ADR-009
   ├── AirLLMBackend          (uses airllm.AirLLM*; AutoTokenizer init) -- ADR-009
   └── ApiBackend             (uses anthropic SDK via gatekeeper)        -- ADR-011
```

### 5.3 Building Blocks (Input / Output / Setup declared in every block)

| Block | Input | Output | Setup |
|-------|-------|--------|-------|
| `HardwareScanner` | paths to `setup.json`, `docs/PRD.md`, `README.md` | `HardwareScanResult` (detected specs + write-receipts) **plus side-effects:** injects `config.hardware_constraints`; replaces placeholder block in PRD + README | OS detection scope flags; atomic-write tempfile root; `.bak` keep policy |
| `EnvInitializer` (`init_env.py` thin entry) | — | `InitEnvResult` (overall ok, per-step receipts) | reads `config.init` block (max_age_hours, paths) |
| `ModelAcquirer` | `target_label` | local `Path` | cache root, HF token (env) |
| `PlumbingTestRunner` | — | `PlumbingResult` (per-stage status) | `config.plumbing_test_model` |
| `BenchmarkRunner` | prompt, params, backend, target | `BackendRunResult` | sampler hz, repeat count |
| `SweepRunner` | matrix(target × quantization × backend) | `SweepReport` | warm-up policy, seed list, plumbing precondition, env-init precondition |
| `QloraTrainer` | `target_label`, dataset, LoraConfig | `QloraReport` (train + inference VRAM) | NF4 / Paged Optimizers config |
| `EconomicAnalyzer` | `SweepReport`, `PricingConfig` | `EconomicReport` + PNG (3 curves) | wattage, capex, lifetime, cloud hourly |
| `ReportAssembler` | `SweepReport`, `EconomicReport`, `QloraReport`, template | rendered `README.md` | template path |

### 5.4 DRY triggers (constitution §3.2 table) — pre-declared resolutions
| Trigger we expect to hit | Resolution we will apply |
|--------------------------|--------------------------|
| Same `try/except` around HF and Anthropic calls | `ApiGatekeeper.execute` is the wrapper. |
| Same timing & sampling logic in 3 backends | Mixins (TimingMixin, MemorySamplingMixin). |
| Same manifest schema across runs | Single Pydantic / dataclass model in `shared/`. |
| Same plot styling in multiple charts | One `plot_style.py` helper in `services/` (style constants only). |
| Same model-loading boilerplate across backends | Shared helper `shared/automodel_factory.py` (≤ 50 LOC) wraps `AutoModel*` resolution per quantization (ADR-009). |

---

## 6. Data Schemas & Contracts

### 6.1 `HardwareScanResult` (was `HardwareProfile` in v1.10)
```jsonc
{
  "captured_at": "2026-06-26T10:00:00Z",
  "os": "Windows-10-10.0.19045",
  "python": "3.11.x",
  "cpu":  { "model": "...", "cores_physical": 8, "cores_logical": 16 },
  "ram":  { "total_gb": 32, "available_gb": 24 },
  "gpu":  { "present": true, "model": "RTX 3060", "vram_gb": 12 },
  "disk": { "free_gb": 250, "fs": "NTFS", "kind": "NVMe|SSD|HDD" },
  "write_receipts": {
    "config_setup_json":  { "status": "ok|skipped|fail", "path": "config/setup.json", "bak": "config/setup.json.bak" },
    "docs_prd_md":        { "status": "ok|skipped|fail", "path": "docs/PRD.md",       "bak": "docs/PRD.md.bak"       },
    "readme_md":          { "status": "ok|skipped|fail", "path": "README.md",         "bak": "README.md.bak"         }
  }
}
```

### 6.2 `BackendRunResult`
```jsonc
{
  "run_id": "uuid4",
  "started_at": "...",
  "backend": "airllm|direct|api",
  "target_label": "llama3-8b-fp16|qwen2-7b-q4|...",
  "model_id": "...",
  "quantization": "fp16|q8|q4|q2|nf4|none",
  "prompt_tokens": 128,
  "completion_tokens": 256,
  "ttft_ms": 450.2,
  "tpot_ms": 87.3,
  "throughput_tps": 11.4,
  "peak_ram_mb": 14820,
  "peak_vram_mb": 7610,
  "wall_s": 22.7,
  "energy_wh": 0.85,
  "completion_text": "...",
  "raw_log_path": "results/logs/run_<id>.jsonl"
}
```

### 6.3 `PlumbingResult` (new in v1.10; shape updated in v1.50 per prompts_book §10)
```jsonc
{
  "captured_at": "2026-06-30T19:41:54Z",
  "plumbing_test_model": {
    "id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "quantization": "fp16",
    "label": "llama3-8b-fp16-airllm-plumbing",
    "loader": "airllm"
  },
  "stages": {
    "download":          { "status": "ok", "duration_s": 0.97, "path": "..." },
    "mmap_allocation":   { "status": "ok", "duration_s": 9.53, "layers": 35, "loader": "airllm" },
    "metric_collection": { "status": "ok", "duration_s": 1103.99,
                            "ttft_ms": 367282.43, "tpot_ms": 368350.11,
                            "peak_ram_mb": 1286.99, "tokens_generated": 2 },
    "manifest_write":    { "status": "ok", "path": "results/plumbing_<id>.json" }
  },
  "overall": "ok",
  "remediation_hint": null
}
```
> Shape carries `loader` per stage where relevant; under M2a's "plumbing = first target at 2 tokens" final shape, `mmap_allocation.layers` reflects the production model's actual transformer-block count (Llama-3-8B = 35).

### 6.4 `rate_limits.json` — see §4.3 above.

### 6.5 `setup.json` (main app config — REVISED v1.20)
```jsonc
{
  "version": "1.00",
  "init": {
    "max_age_hours": 168,                 // env-init freshness window (ADR-016)
    "doc_targets": [                       // files patched by HardwareScanner (ADR-015)
      "docs/PRD.md",
      "README.md"
    ],
    "placeholder_start": "<!-- HARDWARE_SPECS_PLACEHOLDER:START -->",
    "placeholder_end":   "<!-- HARDWARE_SPECS_PLACEHOLDER:END -->",
    "keep_bak": true                       // write a .bak per file per scan
  },
  "hardware_constraints": null,            // FILLED BY HardwareScanner (ADR-015)
  // After `uv run init_env.py`, hardware_constraints becomes the HardwareScanResult
  // payload (less the write_receipts) so the rest of the pipeline can read it.
  "hf": {
    "cache_dir": "~/.cache/huggingface"
  },
  "airllm": {
    "layer_shards_saving_path": "D:/airllm_shards"
  },
  "plumbing_test_model": {
    "id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "quantization": "fp16",
    "label": "llama3-8b-fp16-airllm-plumbing",
    "loader": "airllm"
  },
  "target_models": [
    { "id": "meta-llama/Meta-Llama-3-8B-Instruct", "quantization": "fp16", "label": "llama3-8b-fp16", "loader": "airllm" },
    { "id": "Qwen/Qwen2-7B-Instruct",              "quantization": "q4",   "label": "qwen2-7b-q4",   "loader": "airllm" }
  ],
  // `plumbing_test_model` mirrors `target_models[0]` byte-for-byte in v1.50+
  // (prompts_book §10): plumbing runs the production AirLLM loader on the
  // production model at a 2-token budget (`plumbing_max_new_tokens` below).
  "generation": { "max_new_tokens": 128, "temperature": 0.0, "seed": 42,
                  "plumbing_max_new_tokens": 2 },
  "sampling":   { "memory_hz": 5, "repeat": 5 },
  "energy":     { "assumed_watts_idle": 30, "assumed_watts_active": 180,
                  "electricity_price_per_kwh_usd": 0.16 },
  "economic":   { "hardware_capex_usd": 2500,
                  "lifetime_hours": 5000,
                  "cloud_gpu": { "hourly_usd": 0.60, "include_in_chart": true } }
}
```

> `target_models` is a JSON array (ADR-013). The SDK loops over it; no code change is required to add or remove a target.
> `cloud_gpu.include_in_chart` is `true` per Q4; the break-even chart MUST contain three curves.
> `plumbing_test_model` is required by FR-PT-1..3 / ADR-010.
> `init` and `hardware_constraints` are new in v1.20 (ADR-015 / ADR-016). `hardware_constraints` is `null` at rest and populated by `init_env.py`. Other modules raise `EnvironmentNotInitializedError` if it is still `null` when they run.

### 6.6 `api_pricing.json` (REVISED v1.10 — Anthropic only)
```jsonc
{
  "version": "1.00",
  "providers": {
    "anthropic": {
      "model": "claude-haiku-4-5-20251001",
      "in_per_million_usd":  1.00,
      "out_per_million_usd": 5.00,
      "captured_at": "2026-06-26",
      "source_url": "<to be filled in M4 from Anthropic's pricing page>"
    }
  }
}
```

> Numeric values are placeholders to be replaced from Anthropic's *current* published pricing during M4. OpenAI / other providers are out of scope at runtime (ADR-011) — they MAY be mentioned in prose only.

### 6.7 `HardwareScanner` write contracts (NEW in v1.20 — ADR-015)

`HardwareScanner` is the only component in the system that mutates `config/setup.json`, `docs/PRD.md`, and `README.md`. Its contract:

1. **Detect.** Read hardware via `psutil` (CPU, RAM, disk) + `pynvml` (GPU/VRAM, optional) + `platform` / `sys` (OS, Python). Build a `HardwareScanResult` dataclass.
2. **Inject into `config/setup.json`.**
   - Read existing JSON, set `hardware_constraints` to the scan payload (without `write_receipts`), bump nothing else.
   - Write to `config/setup.json.tmp`, then `os.replace(tmp, config/setup.json)` for atomicity.
   - If `init.keep_bak`, snapshot the prior file to `config/setup.json.bak` first.
3. **Patch placeholder markers in `docs/PRD.md` and `README.md`.**
   - For each path in `init.doc_targets`, locate the **pair**:
     ```
     <!-- HARDWARE_SPECS_PLACEHOLDER:START -->
     ...anything in between (will be replaced)...
     <!-- HARDWARE_SPECS_PLACEHOLDER:END -->
     ```
   - Replace the content **between** the markers with a fresh Markdown table rendered from the scan payload. The markers themselves are preserved verbatim so the next scan can find them again.
   - Atomic write + optional `.bak` snapshot, same as step 2.
   - If a file is missing or has no marker pair: status = `skipped`, recorded in `write_receipts`. **Never** silently fail.
4. **Return.** A `HardwareScanResult` with `write_receipts` describing each side-effect.

Rendered table example (this is what gets injected between the markers):

```markdown
| Component | Value |
|-----------|-------|
| Captured at | 2026-06-26T10:00:00Z |
| OS / Python | Windows-10-10.0.19045 / 3.12.13 |
| CPU | Intel Core i7-... · 8 physical / 16 logical |
| RAM | 32 GB total · 24 GB available |
| GPU | NVIDIA RTX 3060 · 12 GB VRAM |
| Disk (project drive) | 250 GB free · NTFS · NVMe |
```

**Invariants.**
- No other component writes to `config/setup.json` after the version field. Mutations are HardwareScanner-only.
- No other component edits `docs/PRD.md` or `README.md` between the placeholder markers.
- A successful run is idempotent: scanning twice in a row writes the same content (modulo `captured_at`).

---

## 7. Architectural Decision Records (ADRs)

> Each ADR follows: Context → Decision → Alternatives → Consequences. (constitution §1.2 PLAN.md requirement.)

### ADR-001 — Use `uv` as the sole package manager and task runner
- **Context.** Constitution §7.4 forbids `pip`/`venv`/`virtualenv`/`python -m`.
- **Decision.** All dependency install / scripts / tests run via `uv`.
- **Alternatives.** Poetry, Hatch, raw `pip` — all rejected by constitution.
- **Consequences.** `uv.lock` committed; CI checks for forbidden commands.

### ADR-002 — Three back-ends behind one Strategy interface (Direct / AirLLM / API)
- **Context.** Three measurement paths are required (assignment §5.2 / 5.3 / 5.5).
- **Decision.** Single `InferenceBackend` ABC + three implementations, swapped from config.
- **Alternatives.** Branching `if backend == ...` in services. Rejected (violates DRY + Open/Closed).
- **Consequences.** New back-end (e.g., Ollama local API) added later costs one file.

### ADR-003 — Mixins for cross-cutting timing / sampling / manifest concerns
- **Context.** All three back-ends need the same TTFT/TPOT capture, the same memory sampler and the same manifest writer.
- **Decision.** Mixins (`TimingMixin`, `MemorySamplingMixin`, `ManifestLoggingMixin`).
- **Alternatives.** Decorators, helper functions, base class with concrete methods. Mixins win because each concern is independently testable (constitution §3.2) and back-ends remain a thin Strategy hierarchy without diamond inheritance pain.
- **Consequences.** Each Mixin gets its own unit-test file.

### ADR-004 — Configuration is hierarchical JSON + `.env` + `constants.py`
- **Context.** Constitution §6.2 forbids hard-coded values.
- **Decision.** `setup.json` (app, incl. `target_models[]`), `rate_limits.json` (gatekeeper), `api_pricing.json` (economics), `.env` (secrets), `constants.py` (physical/mathematical only).
- **Alternatives.** YAML, TOML in `pyproject.toml`. JSON chosen because it's already in the constitution's examples and trivially diffable.
- **Consequences.** Each JSON file carries a `"version"` field validated at startup (constitution §7.1).

### ADR-005 — AirLLM layer shard cache on a fast disk, path from config
- **Context.** Assignment §6.1 Do warns explicitly: re-route `layer_shards_saving_path` away from a near-full system drive (often `C:`), onto a fast SSD.
- **Decision.** `setup.json.airllm.layer_shards_saving_path` is mandatory; SDK validates free space and disk class (NVMe/SSD/HDD) at runtime via `HardwareProfiler`.
- **Alternatives.** Default to `~/.cache/airllm`. Rejected because the failure mode (disk full mid-run) is exactly the trap the assignment warns about.
- **Consequences.** Hardware profile is read both for reporting and for guard-rail validation.

### ADR-006 — Measurement separates Prefill from Decode metrics
- **Context.** Assignment §3 and §5.6 explicitly demand showing where the bottleneck is, which is exactly the Prefill (compute-bound) vs Decode (memory-bound) distinction from L08 §3.
- **Decision.** TTFT (Prefill proxy) and TPOT (Decode proxy) are reported separately, with peak RAM/VRAM time-series so the reader can correlate them.
- **Alternatives.** Report only end-to-end latency. Rejected — collapses the two regimes the analysis must distinguish.
- **Consequences.** `BackendRunResult` carries both metrics; the report contains a Roofline-style chart.

### ADR-007 — README is the technical report (assembled, not hand-edited)
- **Context.** Assignment §7 / §8 — "the report MUST be the README; tables/charts/screenshots inside the README itself".
- **Decision.** `ReportAssembler` interpolates numeric sections into `README.md` from `results/` so re-running experiments updates the report.
- **Alternatives.** Hand-maintain README. Rejected — guarantees stale numbers.
- **Consequences.** README has clearly-delimited regions (`<!-- AUTOGEN:... -->`); a CI check guards against hand-edits to those regions.

### ADR-008 — Defer GUI; CLI + notebook are sufficient
- **Context.** Constitution §3.1 allows multiple consumers but the project's audience is one reviewer + one student. Time budget is ~3 active hours.
- **Decision.** Ship CLI + Jupyter notebook only; SDK leaves the door open for a GUI later.
- **Alternatives.** Streamlit dashboard. Rejected — out of scope (PRD §6.4 O-4).
- **Consequences.** Reduces test surface; no `gui/` package this iteration.

### ADR-009 — Initialize models exclusively through `AutoModel*` factories (NEW in v1.10)
- **Context.** Assignment §6.1 Do explicitly warns: when loading HF models — especially from new families — instantiating a concrete class like `LlamaForCausalLM` against a checkpoint that registers a slightly different architecture raises a *Class-mismatch* error at init time. The generic `AutoModel` / `AutoModelForCausalLM` factories read the model's `config.json` and pick the right concrete class.
- **Decision.** Every back-end that loads `transformers` weights does so via `AutoModelForCausalLM.from_pretrained(...)` and tokenizers via `AutoTokenizer.from_pretrained(...)`. A shared helper `shared/automodel_factory.py` (≤ 50 LOC) centralizes dtype/device-map resolution per quantization level.
- **Alternatives.** Per-family concrete classes — rejected (brittle, exactly the failure mode the assignment flags). Custom loader — rejected (re-implements what HF already provides).
- **Consequences.** Adding a new model family (e.g., Mistral) requires no code change. A static test (`tests/unit/test_backends/test_init_uses_automodel.py`) blocks imports of concrete `*ForCausalLM` classes.

### ADR-010 — Plumbing-first execution rule (NEW in v1.10)
- **Context.** Oversized runs are expensive in time and may hang the machine. We need fast feedback that the pipeline (HF download → AirLLM `mmap` → sampler → manifest) is intact before committing to a long run.
- **Decision.** Introduce a `PlumbingTestRunner` Building Block driven by `config.plumbing_test_model` (small model + Q2). `SDK.run_plumbing_test()` exposes it. `SDK.run_sweep()` requires a successful plumbing manifest from the current install, or runs the plumbing test itself, unless explicitly bypassed (`skip_plumbing=True`, which logs a warning).
- **Alternatives.** Trust the implementation. Rejected — assignment §6.1 Do explicitly recommends warming up plumbing on a small/Q2 model first.
- **Consequences.** Added Building Block (`PlumbingTestRunner`), added SDK method, added CLI subcommand `run-plumbing-test`, added integration test that asserts `run_sweep` aborts when no plumbing manifest exists.

### ADR-011 — Anthropic is the sole runtime third-party API provider (NEW in v1.10)
- **Context.** PRD Q2 resolved: keep the economic comparison clean by anchoring it to a single provider.
- **Decision.** `backends/api_backend.py` calls the Anthropic Messages API only. `config/api_pricing.json` lists Anthropic only. Other providers MAY be cited in prose; they MUST NOT be wired into the runtime.
- **Alternatives.** Multi-provider matrix (OpenAI + Anthropic + …). Rejected — increases scope and dilutes the break-even analysis.
- **Consequences.** Single API key (`ANTHROPIC_API_KEY`) in `.env-example`; single gatekeeper service `anthropic_messages`. Cost math has a single source of truth.

### ADR-012 — Break-even chart MUST show three curves (NEW in v1.10)
- **Context.** PRD Q4 resolved: include Cloud GPU in the economic comparison.
- **Decision.** `EconomicAnalyzer` produces a single chart with three curves — On-Prem, Anthropic API, Cloud GPU — plus marked crossover points. `cloud_gpu.include_in_chart` defaults to `true`; setting it to `false` is supported but discouraged (and logged as a warning).
- **Alternatives.** Two-curve chart. Rejected — Q4 says "absolutely include the Cloud GPU curve".
- **Consequences.** `EconomicReport` carries `cloud_gpu_curve` array; the chart helper draws three series; the assumption table includes `cloud_gpu.hourly_usd`.

### ADR-013 — Target models are a JSON array, iterated by the SDK (NEW in v1.10)
- **Context.** PRD Q1 resolved + constitution §6.2 (no hard-coded values). The initial targets are Llama-3-8B and Qwen-7B-Q4, but the system MUST allow appending more without code changes.
- **Decision.** `config.target_models[]` is the single source of truth for which models to evaluate. Each entry: `{ id, quantization, label }`. `SweepRunner` iterates the array. CLI subcommands accept `target_label` (resolves to one entry) or default to "all".
- **Alternatives.** Hard-coded model list in `constants.py`. Rejected — violates §6.2.
- **Consequences.** Adding a third model family (e.g., Mistral-7B) is a one-line JSON edit. A CI grep ensures no model ID literal appears in `src/` outside `config.py`'s reader.

### ADR-014 — QLoRA mechanism is encapsulated in `services/qlora_trainer.py` with its own PRD (NEW in v1.10)
- **Context.** PRD Q3 resolved: §5.7 extension = QLoRA training. Constitution §1.3 mandates a dedicated PRD for "central mechanisms or specific complex algorithms" — QLoRA qualifies.
- **Decision.** A separate Building Block `QloraTrainer` (Input: target_label + dataset + LoraConfig; Output: QloraReport with train & inference VRAM peaks; Setup: NF4 + Paged Optimizers config). Mechanism is fully documented in **`docs/PRD_qlora.md`** before implementation begins (TODO §7, DP-7).
- **Alternatives.** Inline QLoRA into `SweepRunner`. Rejected — QLoRA is a distinct mechanism and merits its own contract + tests.
- **Consequences.** New SDK method `run_qlora_finetune`; new CLI subcommand `run-qlora`; new dependencies (`peft`, `bitsandbytes`, `datasets`).

### ADR-015 — HardwareScanner has two write side-effects with atomic file ops (NEW in v1.20)
- **Context.** Assignment §5.1 requires documenting exact hardware (CPU/RAM/GPU/VRAM) in the report and using it to justify model choice. Hand-copying these specs into `config/setup.json`, `docs/PRD.md`, and `README.md` is exactly the "manual transcription" failure the constitution warns against in §0.2.
- **Decision.** `HardwareScanner` (in `services/hardware_scanner.py`, renamed from `hardware_profiler.py`) detects hardware AND mutates three files in one pass:
  1. Inject scan payload into `config/setup.json.hardware_constraints`.
  2. Replace the content between `<!-- HARDWARE_SPECS_PLACEHOLDER:START -->` and `<!-- HARDWARE_SPECS_PLACEHOLDER:END -->` markers in `docs/PRD.md` and `README.md` with a formatted Markdown table.
  All writes are atomic (`write tmp → os.replace`) with optional `.bak` snapshots (`init.keep_bak`). Missing files or absent marker pairs are reported in `write_receipts` as `skipped` — never silently swallowed.
- **Alternatives.** (a) Print-only — rejected, the manual transcription is the bug. (b) Separate writers per file — rejected, three writers means three places to forget; one Scanner owns all three side-effects.
- **Consequences.** New `init` block + `hardware_constraints` field in `setup.json`. New placeholder convention in PRD + README. New integration test `tests/integration/test_hardware_scanner_writes.py`. New invariant: no other module writes those three files between the markers.

### ADR-016 — `init_env.py` bootstrap script enforces env-init-first execution (NEW in v1.20)
- **Context.** ADR-015 only matters if it actually runs before the rest of the pipeline. We need a single, obvious entry point AND a runtime precondition.
- **Decision.** Add a top-level `init_env.py` script invoked via `uv run init_env.py`. It is a ≤ 30 LOC delegate-only wrapper: parses args, builds the SDK, calls `SDK.initialize_environment()` (which composes `HardwareScanner` + any future bootstrap steps), prints a summary, exits non-zero on any per-file failure. All other SDK methods (`run_plumbing_test`, `run_baseline`, `run_airllm`, `run_sweep`, `run_qlora_finetune`, `economic_analysis`, `assemble_readme`) assert `config.hardware_constraints` is present and fresh (within `init.max_age_hours`), raising `EnvironmentNotInitializedError` with a remediation hint pointing back to `uv run init_env.py`.
- **Alternatives.** (a) Hide it inside every entry point — rejected, hidden side-effects on every CLI call are surprising. (b) Document the manual step in README and trust the user — rejected, exactly the workflow gap the constitution §0.2 calls out. (c) A `pre-commit`-style hook — rejected, too tied to git workflow.
- **Consequences.** New top-level file `init_env.py`; new SDK method `initialize_environment()`; new CLI subcommand `initialize`; new precondition assertion in every other SDK method; new test `tests/integration/test_env_init_required.py` asserting `EnvironmentNotInitializedError` is raised when `hardware_constraints` is null or stale.

---

## 8. Folder Layout (binding — REVISED v1.10)

```text
on_prem_llm_lab/
├── init_env.py                    # Bootstrap (ADR-016) — uv run init_env.py
├── src/
│   └── on_prem_llm_lab/
│       ├── __init__.py            # exports OnPremLlmSDK, __version__
│       ├── sdk/
│       │   ├── __init__.py
│       │   └── sdk.py             # OnPremLlmSDK facade
│       ├── services/
│       │   ├── __init__.py
│       │   ├── hardware_scanner.py        # ADR-015 (was hardware_profiler.py)
│       │   ├── model_acquirer.py
│       │   ├── plumbing_test_runner.py    # ADR-010
│       │   ├── benchmark_runner.py
│       │   ├── sweep_runner.py
│       │   ├── qlora_trainer.py           # ADR-014
│       │   ├── economic_analyzer.py
│       │   └── report_assembler.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── direct_backend.py          # AutoModelForCausalLM (ADR-009)
│       │   ├── airllm_backend.py          # AirLLM + AutoTokenizer
│       │   └── api_backend.py             # Anthropic only (ADR-011)
│       ├── mixins/
│       │   ├── __init__.py
│       │   ├── timing_mixin.py
│       │   ├── memory_sampling_mixin.py
│       │   ├── manifest_logging_mixin.py
│       │   └── energy_accounting_mixin.py
│       ├── shared/
│       │   ├── __init__.py
│       │   ├── gatekeeper.py
│       │   ├── rate_limit_config.py
│       │   ├── config.py                  # reads target_models[] (ADR-013)
│       │   ├── automodel_factory.py       # AutoModel* helper (ADR-009)
│       │   ├── logging_setup.py
│       │   ├── paths.py
│       │   └── version.py                 # __version__ = "1.00"
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py                    # delegate-only
│       └── constants.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_sdk/
│   │   ├── test_services/
│   │   ├── test_backends/
│   │   │   └── test_init_uses_automodel.py    # ADR-009 enforcement
│   │   ├── test_mixins/
│   │   └── test_shared/
│   └── integration/
│       ├── test_gatekeeper_queue.py
│       ├── test_plumbing_test_runner.py       # ADR-010
│       ├── test_sweep_requires_plumbing.py    # ADR-010
│       ├── test_hardware_scanner_writes.py    # ADR-015
│       ├── test_env_init_required.py          # ADR-016
│       ├── test_sweep_runner_end_to_end.py
│       └── test_report_assembler.py
├── config/
│   ├── setup.json                 # target_models[], plumbing_test_model, cloud_gpu
│   ├── rate_limits.json           # anthropic_messages only
│   ├── api_pricing.json           # Anthropic only
│   └── logging_config.json
├── data/
├── results/
├── figures/
├── notebooks/
│   └── analysis.ipynb
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   ├── prompts_book.md
│   ├── PRD_qlora.md               # MANDATORY per ADR-014
│   └── PRD_*.md                   # other central mechanisms — see TODO §7
├── README.md                      # the technical report (assembled)
├── pyproject.toml
├── uv.lock
├── .env-example                   # HF_TOKEN=, ANTHROPIC_API_KEY=
└── .gitignore
```

---

## 9. Cross-Cutting Concerns

### 9.1 Logging
- One JSON-line file per run under `results/logs/run_<id>.jsonl`.
- Levels: `INFO` for lifecycle events, `DEBUG` for sampler ticks (off by default), `ERROR` for gatekeeper failures.

### 9.2 Configuration loading & validation
- `shared/config.py.load(config_dir: Path) -> AppConfig` reads all four JSON files and returns a typed object.
- Each JSON file's `version` is validated against the constants in `version.py` — startup fails fast on mismatch (constitution §7.1).
- `target_models[]` is validated: each entry has a non-empty `id`, a `quantization` from the supported enum, and a unique `label`.

### 9.3 Concurrency / thread safety (constitution §14)
- Memory sampler runs in a background thread, communicates via `queue.Queue` (no shared mutable state).
- Gatekeeper uses `threading.Lock` around the rate-limit token bucket; queue is `collections.deque` guarded by lock or `queue.Queue`.

### 9.4 Security (constitution §6.4)
- `HF_TOKEN`, `ANTHROPIC_API_KEY` from env only.
- `.env-example` lists every expected variable with placeholder; **no `OPENAI_API_KEY`** entry (ADR-011).
- `.gitignore` excludes `.env`, `*.pem`, `*.key`, `credentials.json`, plus AirLLM shard dir and `results/logs/`.

### 9.5 Versioning
- `src/on_prem_llm_lab/shared/version.py` → `__version__ = "1.00"`.
- Each JSON config carries `"version": "1.00"`; startup asserts compatibility.

### 9.6 Linting & coverage gates
- `ruff` rule set per constitution §6.1 (E, F, W, I, N, UP, B, C4, SIM).
- `pytest --cov` with `fail_under = 85` in `pyproject.toml`.
- Pre-commit (optional) wires both.

---

## 10. Test Strategy (constitution §5)

- **Unit tests** mirror `src/` structure under `tests/unit/`. Every public method has ≥ 1 happy-path and ≥ 1 error-case test. External I/O (HF Hub, file system writes to model cache, Anthropic API) is **mocked**.
- **Backend init enforcement (ADR-009).** `tests/unit/test_backends/test_init_uses_automodel.py` parses each back-end module's AST and asserts no `*ForCausalLM` concrete class is imported; only `AutoModel*` / `AutoTokenizer` are allowed.
- **Plumbing precondition (ADR-010).** `tests/integration/test_sweep_requires_plumbing.py` asserts `SDK.run_sweep` raises `PlumbingNotRunError` if no plumbing manifest exists and `skip_plumbing=False`.
- **Integration tests** under `tests/integration/`:
  - `test_gatekeeper_queue.py` — fires a burst beyond `requests_per_minute`, asserts FIFO ordering, asserts no exception escapes, asserts queue drains.
  - `test_plumbing_test_runner.py` — runs the small/Q2 model end-to-end and asserts the four stages all reach `ok`.
  - `test_sweep_runner_end_to_end.py` — uses a tiny stub model (≤ 100 MB) to exercise the full SweepRunner → ReportAssembler chain over a 2-entry `target_models` fixture.
  - `test_report_assembler.py` — golden-file comparison against a fixture README template.
- **Coverage** ≥ 85 % global; `src/main.py` and any future `gui/` excluded per constitution §5.2.

---

## 11. Risks & Mitigations (REVISED v1.10)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AirLLM cannot run on student's exact Python/Torch combo | Medium | High | M1 includes a smoke test on a tiny model; pin AirLLM-compatible Torch in `pyproject.toml`. |
| Chosen target is *too* big — AirLLM also fails (assignment §6.2 Don't) | Low | High | Capped at 8B-class (Q1 resolved); plumbing-first rule (ADR-010) gives fast-fail signal before committing to a long run. |
| Layer shard cache fills `C:` and crashes the machine | Medium | High | `layer_shards_saving_path` is config-driven; HardwareProfiler enforces free-space precondition. |
| HF or Anthropic token committed by accident | Low | Critical | `.env-example` only; `.gitignore`; pre-flight scan in M7. |
| File size creep | Medium | Medium | CI script `tools/check_file_size.py` fails build if any source file > 150 LOC. |
| Test coverage drift | Medium | Medium | `fail_under = 85` in `pyproject.toml`. |
| Measurement overhead inflates TTFT/TPOT | Medium | Medium | Null-op benchmark (NFR-10) gates the sampler frequency. |
| **Class-mismatch error on model init** (new family checkpoint vs concrete class) | Medium | High | **ADR-009**: enforce `AutoModel*` everywhere; static AST test blocks concrete-class imports. |
| **Sweep launched without plumbing test** | Medium | High | **ADR-010**: `run_sweep` requires a current plumbing manifest or runs the plumbing test itself; `--skip-plumbing` only logs a warning. |
| **QLoRA fine-tune exceeds VRAM budget** | Medium | Medium | NF4 + Paged Optimizers from QLoRA paper; tiny dataset; document in `PRD_qlora.md`. |
| **Anthropic pricing changes mid-project** | Low | Low | `api_pricing.json` carries `captured_at`; assumption table in README cites the date. |
| **HardwareScanner corrupts `PRD.md` / `README.md`** (e.g., process killed mid-write, replaces wrong region) | Low | High | **ADR-015**: atomic `write tmp → os.replace`; `.bak` per file per scan; placeholder pair (`:START` / `:END`) limits replacement to one well-defined region; integration test asserts idempotency. |
| **User runs pipeline without `init_env.py`** | Medium | Medium | **ADR-016**: every other SDK method asserts `config.hardware_constraints` is present and fresh; raises `EnvironmentNotInitializedError` with remediation hint. |
| **Hardware changes between scans** (e.g., GPU added) cause stale `hardware_constraints` | Low | Medium | `init.max_age_hours` window forces a re-scan; manual `uv run init_env.py` flushes the receipts. |

---

## 12. Approval

This document **MUST** be approved by the user (alongside PRD v1.10 and TODO v1.10) before development begins (constitution §1.5).

> Approval status: ☐ Pending user review of revision v1.10.
