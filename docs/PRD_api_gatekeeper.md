# PRD — API Gatekeeper (`shared/gatekeeper.py`)

> **Document type:** Dedicated PRD for a central mechanism (SDLC Phase 1 deliverable, constitution §1.3 mandatory).
> **Tracked by:** `docs/TODO.md` §7 DP-1 — **MANDATORY**.
> **Blocks:** **T-2.6** implementation. **T-2.7** integration test depends on the I/O contract authored here.
> **Source authority chain:** `SOFTWARE_PROJECT_GUIDELINES.md` v3.00 §4 (binding constitution rule — every API call MUST be gated) → `docs/PRD.md` v1.10 FR-15 / FR-16 / SC-5 → `docs/PLAN.md` v1.20 §4 (interface sketch + ADR-002 cross-cutting wiring).
> **Document version:** 1.00 — 2026-06-30.
> **Status:** DRAFT, awaiting user approval before any code is written.

---

## 1. What and why

### 1.1 What
A **single, centralized component** that gates every outbound API call made by the project. It owns three responsibilities:

1. **Rate limiting** — per-service quotas (requests/min, requests/hour, concurrent_max) loaded from `config/rate_limits.json`.
2. **FIFO overflow queueing with bounded depth** — when a service is at its rate limit, callers wait in arrival order; when the queue is full, calls fail loudly rather than silently buffer.
3. **Structured JSON logging + typed errors** — every event (acquire, queue, wait, retry, error) emits one JSON-line; hard failures raise `GatekeeperError` with `reason`/`service`/`attempt` fields.

### 1.2 Why
Three motivations stack:

* **Constitutional.** `SOFTWARE_PROJECT_GUIDELINES.md` §4 makes this non-negotiable: "every outbound API call MUST go through `ApiGatekeeper`. No bypass is allowed." PRD §3.7 FR-15 + FR-16 reflect this verbatim, including "no numeric limit may be hard-coded in source."
* **Operational.** This lab calls **two** external vendors — Hugging Face Hub (for model downloads) and Anthropic (for the M4 break-even API curve). Each has its own quota shape; the wrong-shape burst can cause a 429-storm that delays or breaks the M3 sweep. Centralizing the policy means **one** place to tune.
* **Reproducibility.** Rate-limit configuration in `config/rate_limits.json` (not in source) means a reviewer can dial down limits to match their own keys without code changes, and the project's run manifests can record the exact policy that was in force at run time.

### 1.3 Who calls it
| Caller | Service name | Lands in |
|---|---|---|
| `services/model_acquirer.py` (T-2.x) | `huggingface_hub` | M2b+ |
| `backends/api_backend.py` (T-4.2) | `anthropic_messages` | M4 |
| Future telemetry / metrics emitter | `default` | post-v1.00 |

The list is closed by ADR-011 (Anthropic only) + ADR-013 (target_models as JSON array — no per-target API). If a fourth service is added later, only `config/rate_limits.json` needs a new section + the new caller passes a new `service` string; the gatekeeper code is untouched.

---

## 2. Theoretical background

### 2.1 Rate limits in vendor APIs
Public LLM APIs publish quotas in three orthogonal dimensions:

* **Time-window counters** — requests per minute (RPM), per hour (RPH), per day. Crossing the window → HTTP **429 Too Many Requests** with a `Retry-After` header.
* **Concurrency caps** — at most N in-flight requests at any instant. Crossing → 429 or 503.
* **Token-bandwidth caps** (for LLM APIs) — input + output tokens per minute (TPM, OTPM). This project does *not* meter TPM/OTPM at the gatekeeper layer — the token counts live on `BackendRunResult` (PLAN §6.2) and the cost analyzer (T-4.1) consumes them downstream. The break-even chart's API curve uses request count × per-request cost, so RPM gating is sufficient for the lab's scope.

### 2.2 Algorithm choice
Three textbook strategies are in scope:

| Strategy | Pros | Cons | Verdict |
|---|---|---|---|
| **Fixed-window counter** | trivially simple; matches "60 req/min" vendor wording verbatim | allows 2× burst at window boundary (59th sec + 1st sec of next window) | **chosen** |
| **Sliding-window counter** | smooths boundary burst | needs O(window) memory + timestamp tracking | rejected — overkill at this scale (≤ 50 outbound calls per sweep) |
| **Token-bucket** | continuous refill, controlled burst | +50 LOC, harder to reason about; not how vendors document quotas | rejected |

**Decision: fixed-window** for RPM and RPH counters, paired with a `threading.Semaphore(concurrent_max)` for the concurrency cap. Boundary-burst risk is theoretical for our load (sweep ≪ window size).

### 2.3 Queue semantics
A bounded `collections.deque` per service holds waiting callers. Arrival order is preserved (FIFO — `popleft` discipline). On `queue_max_depth` overflow the gatekeeper raises `GatekeeperError(reason="queue_full")` — **never** silently drops or unblockingly buffers. Callers up the stack (sweep runner) choose how to react (sleep + retry, abort, etc.).

### 2.4 Retry / backoff
On a transient inner-callable failure (network error, 5xx, 429), the gatekeeper retries up to `max_retries` times with **exponential backoff + full jitter**:

```
sleep_s = uniform(0, retry_after_seconds * 2^attempt)
```

Exponential because: a real 429 means the vendor is pushing us off; a linear retry guarantees we hit them again at the same cadence after the window resets. Jitter because: when multiple workers retry simultaneously, deterministic backoff creates synchronized hammering.

Persistent failure (all retries exhausted) raises `GatekeeperError(reason="max_retries_exceeded")` carrying the last underlying exception.

### 2.5 Structured logging
Each gatekeeper event emits **exactly one JSON-line** with: `ts` (ISO UTC), `event` (one of `acquire | queue_wait | release | retry | queue_full | error`), `service`, `queue_depth`, `in_flight`, `attempt?`, `error?`. The JSON-line format is constitution §6.3 + PLAN §6.5 (`logging_config.json` → JSONL formatter for `lab.log`). At construction, the gatekeeper accepts an injectable `logger` (default = `logging.getLogger("on_prem_llm_lab.gatekeeper")`).

---

## 3. Functional requirements (FR-GK-*)

* **FR-GK-1** — Single public class `ApiGatekeeper` exposes `acquire(service)` as a **context manager** and a higher-level `call(service, fn, *args, **kwargs)` that composes `acquire()` + retry loop around `fn`.
* **FR-GK-2** — Per-service limits MUST be loaded from `config/rate_limits.json` via `RateLimitConfig.from_dict(payload)`. Missing service name → unknown-service error at lookup time (NOT at construction — registered services may be added by reviewers without forcing a reload).
* **FR-GK-3** — Bounded FIFO queue per service. Overflow raises `GatekeeperError(reason="queue_full")` with the current depth in `.context`. No silent drop. No unbounded buffer.
* **FR-GK-4** — Retry loop applies exponential backoff with jitter, bounded by `max_retries`. Exhaustion raises `GatekeeperError(reason="max_retries_exceeded", cause=<last_exc>)`.
* **FR-GK-5** — One JSON-line log per event. Required fields: `ts`, `event`, `service`. Optional fields per event type. Logger is injectable.
* **FR-GK-6** — Thread-safe. The sweep runner (T-3.5) may parallelise across targets; the gatekeeper holds a single `threading.Lock` per service to guard the counter + queue + semaphore state.
* **FR-GK-7** — Per-service isolation. A burst against `huggingface_hub` MUST NOT consume capacity from `anthropic_messages`. Each service has its own counters, queue, semaphore, and lock.
* **FR-GK-8** — Observable state via `get_queue_status(service) -> QueueStatus`. Used by tests and (future) the M6 report.

## 4. Non-functional requirements (NFR-GK-*)

* **NFR-GK-1** — File-size cap: `shared/gatekeeper.py` ≤ 150 LOC (constitution §2.2). If the FR-GK-1..8 set exceeds 150 LOC, split into `shared/gatekeeper.py` (public API) + `shared/gatekeeper_state.py` (per-service state machine — sanctioned by constitution §2.2 "extract logic" exception). `shared/rate_limit_config.py` (already a stub) hosts the typed config readers separately.
* **NFR-GK-2** — Coverage ≥ 85 % (constitution §5.2). Target ≥ 95 % since the surface is small and pure logic-heavy.
* **NFR-GK-3** — No hard-coded numerics. No literal `time.sleep(30)` anywhere; every duration comes from `RateLimitConfig`.
* **NFR-GK-4** — `clock` (current time) and `sleeper` (sleep function) are injectable so tests are deterministic and never wall-sleep.
* **NFR-GK-5** — Synchronous only. Asyncio is not used elsewhere in the project; introducing it for the gatekeeper would violate the constitution's KISS + DRY principles.

---

## 5. I/O Contract

### 5.1 `ServiceLimits` (frozen view of one service entry)
```python
@dataclass(frozen=True)
class ServiceLimits:
    requests_per_minute: int
    requests_per_hour: int
    concurrent_max: int
    retry_after_seconds: float
    max_retries: int
    queue_max_depth: int
```

### 5.2 `RateLimitConfig` (frozen, loaded from `rate_limits.json`)
```python
@dataclass(frozen=True)
class RateLimitConfig:
    version: str
    services: dict[str, ServiceLimits]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RateLimitConfig": ...
    def for_service(self, name: str) -> ServiceLimits: ...   # raises KeyError
```

### 5.3 `QueueStatus`
```python
@dataclass(frozen=True)
class QueueStatus:
    service: str
    queue_depth: int        # callers currently waiting
    in_flight: int          # callers currently inside acquire/release
    requests_in_last_minute: int
    requests_in_last_hour: int
```

### 5.4 `GatekeeperError`
```python
class GatekeeperError(RuntimeError):
    """Hard failure. Reasons: queue_full | max_retries_exceeded | unknown_service."""

    def __init__(
        self,
        *,
        reason: str,
        service: str,
        attempt: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        ...
```

### 5.5 `ApiGatekeeper`
```python
class ApiGatekeeper:
    def __init__(
        self,
        config: RateLimitConfig,
        *,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,      # monotonic seconds
        sleeper: Callable[[float], None] | None = None,  # accepts seconds
    ) -> None: ...

    @contextmanager
    def acquire(self, service: str) -> Iterator[None]: ...

    def call(
        self,
        service: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    def get_queue_status(self, service: str) -> QueueStatus: ...
```

`acquire()` is the low-level primitive (use it when the caller needs custom retry behaviour — e.g., HF Hub's resumable download already does its own retries). `call()` is the convenience wrapper that composes `acquire` + the retry loop and is the recommended path for Anthropic + general telemetry.

---

## 6. Constraints

* **C-GK-1** — File size ≤ 150 LOC per NFR-GK-1.
* **C-GK-2** — No hard-coded values per NFR-GK-3 / FR-16.
* **C-GK-3** — `from anthropic import ...` and `from huggingface_hub import ...` MUST NOT appear in `shared/gatekeeper.py`. The gatekeeper is vendor-agnostic — it accepts opaque callables. The vendor-specific code lives in the caller (`api_backend.py`, `model_acquirer.py`).
* **C-GK-4** — Synchronous, single-process. Distributed coordination is out of scope.
* **C-GK-5** — The gatekeeper does NOT meter input/output tokens. Token accounting lives on `BackendRunResult`.

---

## 7. Alternatives considered

| # | Option | Reason rejected |
|---|---|---|
| A-GK-1 | External library (`tenacity` for retry, `ratelimit` for RPM) | Doesn't centralize — per-call decorator is bypassable. Adds two deps the constitution would flag as unjustified. Custom logging-format integration would be more code than the from-scratch impl. |
| A-GK-2 | Per-call decorator (`@rate_limited`) | Centralization is the rule (FR-15). Decorator-based design relies on developer discipline to apply it everywhere. Constitutionally noncompliant. |
| A-GK-3 | Token-bucket algorithm | +50 LOC, harder to reason about, doesn't match vendor documentation phrasing. Boundary burst risk is theoretical at our load (a sweep is ≤ 50 outbound calls; the 2× burst at window edge is harmless). |
| A-GK-4 | Asyncio + `aiohttp` | The rest of the project is synchronous. Mixing sync + async creates the worst of both worlds (the "what colour is your function" problem). KISS. |
| A-GK-5 | Worker-thread pool with explicit queue dequeuing | Simpler in-line wait + semaphore covers our case. A worker pool would need its own lifecycle management (start/stop, thread leaks), which doubles the gatekeeper's surface area. |
| A-GK-6 | Linear (non-exponential) backoff | Linear backoff against a real 429 means we hit the vendor at the same cadence after the window resets — guaranteed re-throttle. Exponential + jitter is industry standard (AWS SDK docs reference it directly). |

---

## 8. Success criteria

* **SC-GK-1 — Burst FIFO.** With `anthropic_messages` rpm=20, submit 60 calls in rapid succession. All 60 eventually complete; observed completion order matches submission order (FIFO). No exceptions. (T-2.7 integration test.)
* **SC-GK-2 — Queue-full bites.** Configure `queue_max_depth=10`. Submit 12 calls with an inner callable that blocks on a barrier. Calls 1–10 enter the queue, calls 11–12 raise `GatekeeperError(reason="queue_full")`. (Unit test.)
* **SC-GK-3 — Transient retry succeeds.** Inner callable raises a `TransientError` on the first 2 attempts, returns 42 on the third. Gatekeeper returns 42; the retry attempts are logged. (Unit test.)
* **SC-GK-4 — Max-retries surfaces.** Inner callable always raises `TransientError`. After `max_retries` attempts, `GatekeeperError(reason="max_retries_exceeded", cause=<TransientError>)` is raised. (Unit test.)
* **SC-GK-5 — Per-service isolation.** Burst against `huggingface_hub` does NOT affect `anthropic_messages`' `get_queue_status` counters. (Unit + integration.)
* **SC-GK-6 — Structured logging.** Every event produces exactly one JSON-line on the injected logger. Each line is valid JSON and contains the required fields. (Unit test parses the captured log records.)
* **SC-GK-7 — Determinism in tests.** All tests run with injected `clock` + `sleeper`. No test wall-sleeps beyond ~50 ms (for the deliberate burst-test in SC-GK-1).

---

## 9. Test scenarios (informs T-2.6 unit suite and T-2.7 integration suite)

### 9.1 Unit (T-2.6, in `tests/unit/test_shared/`)
| ID | Module | Scenario |
|----|--------|----------|
| U-GK-1 | `test_rate_limit_config.py` | `from_dict` happy path with all three services |
| U-GK-2 | `test_rate_limit_config.py` | Version mismatch → ValueError |
| U-GK-3 | `test_rate_limit_config.py` | `for_service("missing")` → KeyError |
| U-GK-4 | `test_rate_limit_config.py` | `ServiceLimits` is frozen |
| U-GK-5 | `test_gatekeeper.py` | `acquire` under limit returns immediately (no clock advance, no sleeper call) |
| U-GK-6 | `test_gatekeeper.py` | `acquire` over rpm enters queue and waits via injected sleeper; release advances clock past window |
| U-GK-7 | `test_gatekeeper.py` | `acquire` when queue full raises `GatekeeperError(reason="queue_full")` |
| U-GK-8 | `test_gatekeeper.py` | `call` retries on transient error and eventually returns success (SC-GK-3) |
| U-GK-9 | `test_gatekeeper.py` | `call` raises `GatekeeperError(reason="max_retries_exceeded")` with `cause` set (SC-GK-4) |
| U-GK-10 | `test_gatekeeper.py` | `get_queue_status` returns current counters per service (SC-GK-5 lite) |
| U-GK-11 | `test_gatekeeper.py` | Every event emits a structured log line; parsed JSON contains required fields (SC-GK-6) |
| U-GK-12 | `test_gatekeeper.py` | Concurrent caps respected — `concurrent_max=2` blocks a third caller until one releases (uses threading) |
| U-GK-13 | `test_gatekeeper.py` | Unknown service raises `GatekeeperError(reason="unknown_service")` |

### 9.2 Integration (T-2.7, in `tests/integration/test_gatekeeper_queue.py`)
| ID | Scenario |
|----|----------|
| I-GK-1 | 60-call burst against `anthropic_messages` (rpm=20): all complete in FIFO order; total wall time ≥ ~150 s (3 windows × 50 ms simulated per second via injected clock). Test uses injected clock + sleeper so the *real* test wall time is < 100 ms. |
| I-GK-2 | Interleaved HF + Anthropic burst — per-service counters update independently; HF queue depth does not impact Anthropic counters. |

---

## 10. Out of scope

* TPM / OTPM (token-bandwidth) accounting at the gatekeeper layer. (Lives on `BackendRunResult`.)
* Distributed / multi-process coordination. (Single-process project.)
* Adaptive rate-limit tuning (auto-detecting vendor limits from 429 headers).
* Async/await primitives.
* GUI / progress-bar integration.

---

## 11. Decisions taken in this PRD

| ID | Decision | Why |
|----|----------|-----|
| D-GK-1 | Fixed-window counters (not token-bucket) | Matches vendor wording; simpler; A-GK-3. |
| D-GK-2 | Exponential backoff with full jitter | Industry standard; avoids synchronised retry hammering; A-GK-6. |
| D-GK-3 | Synchronous in-line wait (not worker-pool) | KISS; A-GK-5. |
| D-GK-4 | `clock` + `sleeper` injectable | NFR-GK-4; tests must be deterministic + fast. |
| D-GK-5 | `acquire()` AND `call()` exposed | HF Hub does its own retries → use `acquire()`; Anthropic + telemetry → use `call()`. |
| D-GK-6 | Default logger is `logging.getLogger("on_prem_llm_lab.gatekeeper")` | Composes with `config/logging_config.json` JSONL formatter (PLAN §6.5). |
| D-GK-7 | Split file allowed: `gatekeeper.py` + `gatekeeper_state.py` if needed | NFR-GK-1; constitution §2.2 sanctioned exception. |
| D-GK-8 | Per-service `threading.Lock` (not a global lock) | FR-GK-6 thread-safety + FR-GK-7 per-service isolation. |

---

## 12. Open questions for user

Before T-2.6 begins, please confirm:

1. **`call()` vs `acquire()` API surface.** This PRD exposes both. Do you want me to ship both, or `acquire()` only (smaller surface, callers wrap their own retry)? Default in §5.5: both.
2. **Logger default.** Stdlib `logging` named `on_prem_llm_lab.gatekeeper` (composes with M3's `logging_config.json`) vs `print()` to stderr (simpler, but less integration). Default in §11 D-GK-6: stdlib `logging`.
3. **Anything you want measured that isn't in §9?** Open the test list now while the contract is still in flight.

Once these are answered, T-2.6 implementation proceeds against this contract verbatim. The PRD is the source of truth; deviations require updating this file first.

---

## 13. Approval

This PRD MUST be approved by the user before any code is written in `shared/gatekeeper.py`. Approval flips DP-1 status in `docs/TODO.md` §7 from "MANDATORY" to "MANDATORY · authored · approved".

> Approval status: ☑ **Approved 2026-07-01** by user — *"approved, both methods, stdlib logging, no missing tests"*.
> Implementation status: ☑ **Implemented 2026-07-01** by T-2.6 (`shared/rate_limit_config.py` 46 LOC + `shared/gatekeeper_state.py` 93 LOC + `shared/gatekeeper.py` 138 LOC — 3-file split per D-GK-7) and T-2.7 (`tests/integration/test_gatekeeper_queue.py` 90 LOC).
>
> All 13 unit + 2 integration scenarios from §9 are green: U-GK-1..4 in `test_rate_limit_config.py` (8 tests); U-GK-5/6/10/11/13 in `test_gatekeeper.py` (6 tests); U-GK-7/8/9/12 in `test_gatekeeper_retry.py` (4 tests, including the real-threading `TestConcurrentMax` for U-GK-12); I-GK-1 + I-GK-2 in `test_gatekeeper_queue.py` (3 tests). Coverage: `rate_limit_config.py` 100 % stmt, `gatekeeper.py` 99 % stmt, `gatekeeper_state.py` 93 % stmt. Contract fulfilled verbatim — no deviations from v1.00.
