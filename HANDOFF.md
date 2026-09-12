# HANDOFF: LLM Inference Cost & Latency Optimization Router

## PROJECT DESCRIPTION
- **What this project is:** An LLM Inference Cost & Latency Optimization Router. It routes incoming queries to one of several LLMs using a continuous contextual bandit that learns online which model to use for which query, balancing cost, latency, and response quality, instead of a traditionally trained classifier.
- **Why it exists:** Built as a portfolio project for ML/AI placement interviews. It is designed to demonstrate systems thinking and decision-making under uncertainty (not just standard model training), and to mirror a real problem every company running LLM products faces (routing between cheap and expensive models).
- **Scope:** This is the full, universal-best version of the project, not a scoped-down one. Correctness and depth are prioritized over speed.
- **Zero-cost constraint:** Uses Ollama-served local open-source models (`llama3.2:1b`, `llama3.2:3b`, `mistral`, `phi3`) instead of paid hosted APIs. It uses simulated per-call cost based on public pricing of comparable hosted models, plus a mock mode for fast, free iteration.

## WHAT'S BUILT SO FAR

### Phase 1 - Foundation
- **Project scaffold:** `router/`, `judge/`, `dashboard/`, `experiments/`, `observability/`, `api/`, `tests/`
- **Unified client wrapper (`router/client.py`):** Calls local Ollama models, tracks simulated cost and real latency, and includes a mock mode toggle.
- **Query embeddings (`router/embeddings.py`):** Continuous context vectors via `sentence-transformers`, with multi-turn conversation support.

### Phase 2 - Core Routing Intelligence
- **Continuous Contextual Bandit (`router/bandit.py`):** LinUCB implementation that utilizes embedding vectors as context to balance exploration vs. exploitation mathematically.
- **Non-Stationary Handling & Shocks:** Exponential decay (sliding window) on the bandit matrices allows it to gracefully forget stale data. A deterministic `set_shock` mechanism allows us to simulate sudden model quality drops.
- **LLM-as-a-Judge & Escalation (`judge/judge.py`, `router/router_core.py`):** Uses an impartial LLM prompt to grade answers on a 0-1 scale with built-in heuristic length-bias mitigation. The `OptimizationRouter` ties the system together, intercepting low-confidence decisions or low-quality initial answers to safely escalate to a fallback model.

### Phase 3 - Realistic Evaluation
- **Streaming Traffic Simulator (`experiments/simulator.py`):** Simulates live production traffic with distinct topical distributions (chat, math, code) and executes mid-run distribution shifts to explicitly test the bandit's adaptability. Now has a runnable `if __name__ == "__main__":` entry point (added when we discovered `python -m experiments.simulator` previously did nothing - it only defined the class) that wires up a full mock-mode stack with a `StructuredLogger` and runs 300 queries with a distribution shift and a model shock, so it actually populates `observability/router_logs.jsonl` when run directly.
- **Validation Harness (`experiments/validation.py`):**
  - *Sequential A/B Testing* to track cumulative metrics over time (avoiding the non-IID pitfalls of naive t-tests on shifting streaming data).
  - *Doubly Robust Off-Policy Evaluation* to robustly estimate counterfactual routing policies using logged data, avoiding the massive variance of pure IPS and the bias of pure Direct Method imputation.

### Phase 4 - Production Layer (Complete)
- **Structured Telemetry Logging (`observability/logger.py`):** Extracted routing telemetry (latency, cost, confidence, scores) into robust JSONL logs.
- **Per-Model SLA Tracking (`observability/metrics.py`):** Computes rolling p50, p95, and p99 latency percentiles alongside model escalation rates using memory-efficient rolling windows (`collections.deque`).
- **SLA Alerting (`observability/alerts.py`):** Evaluates SLA metrics against thresholds and fires alerts (and configurable webhooks) upon detecting silent model degradation.
- **Dashboard Data Layer (`dashboard/data_layer.py`):** Parses JSONL logs and computes aggregated metrics, timeseries data, and per-model statistics (zero external dependencies like Pandas to keep it robust).
- **Streamlit Dashboard UI (`dashboard/app.py`, `dashboard/chart_utils.py`):** Live dashboard built on top of the data layer - router cost/quality/latency projected against static "always use model X" baselines (derived from that model's own observed per-query averages, no extra simulation runs needed), routing distribution over time (rolling window, so it stays responsive to a live shift), and escalation frequency (rolling rate + reason breakdown). Includes a manual/auto-refresh control. Pure aggregation helpers (`chart_utils.py`) are kept separate from the Streamlit rendering script (`app.py`) so they're unit-testable without a script-run context; `app.py` itself is smoke-tested via Streamlit's `AppTest`.

### Deployment Layer (Complete)
- **FastAPI Service (`api/`):** Exposes the router as `POST /v1/route` (plus an unauthenticated `GET /health` liveness probe). `api/dependencies.py` builds a lazy singleton `OptimizationRouter` (mock mode by default, matching the zero-cost constraint; `ROUTER_MOCK_MODE=false` switches to real Ollama inference).
- **Authentication (`api/auth.py`):** Static API-key check via the `X-API-Key` header, keys sourced from the `ROUTER_API_KEYS` env var (comma-separated).
- **Rate Limiting (`api/rate_limiter.py`):** In-memory sliding-window limiter, keyed per API key (`ROUTER_RATE_LIMIT_MAX_REQUESTS` / `ROUTER_RATE_LIMIT_WINDOW_SECONDS` env vars). Auth is checked before rate limiting so invalid keys can't be used to burn a real client's quota. Documented as single-process only - a multi-worker deployment would need a shared store (Redis) instead.
- **Integration Tests (`tests/test_integration_pipeline.py`):** Wires the real (unmocked) embedder, bandit, and judge behind the actual FastAPI endpoint and asserts on real emergent behavior end-to-end - e.g. a cold-start bandit deterministically escalating its first query, a primed low-quality arm triggering the judge-score fallback, and API-produced logs being correctly consumable by `SLATracker`, `AlertManager`, and `DashboardDataLayer` with no transformation needed.

### Documentation
- **`README.md`:** Comprehensive, interview-prep-oriented writeup covering the full architecture (with a Mermaid pipeline diagram), why a contextual bandit instead of a classifier, why LinUCB specifically (vs. epsilon-greedy/UCB1/Thompson/deep bandits), why sequential A/B testing and doubly robust OPE instead of naive t-tests/IPS/DM, an explicit zero-cost design section (Ollama + simulated pricing tiers), a documented known limitation (see below), and an anticipated interview Q&A section.

## KEY DESIGN DECISIONS AND WHY
- **Continuous embeddings instead of discrete query buckets:** Needed for the contextual bandit to learn non-linear decision boundaries as query diversity grows.
- **Mock mode with deterministic hashing:** The same query always produces the same mock embedding/response *within a process*, so bandit convergence can be tested without waiting on real inference every time.
- **Simulated pricing mapped to real hosted-model tiers:** Maps local models (e.g., `llama3.2:1b` as a cheap-tier proxy, `mistral` as an expensive-tier proxy). This keeps the project free to run while preserving a realistic cost/quality/latency tradeoff. Latency is real wall-clock time; only cost is simulated.
- **Cumulative tracking and Doubly Robust OPE:** Chosen specifically to demonstrate advanced ML evaluation understanding that goes beyond simple static train/test splits.
- **Baseline projection from observed per-model averages (dashboard):** Rather than re-running the simulator once per static baseline to compare against, the dashboard projects "always use model X" from that model's own logged per-query averages - correct because the bandit already explores every arm (and always falls back through the fallback model on escalation).
- **Sliding-window rate limiting, in-memory, auth-before-limit:** A fixed window lets a client burst 2x at the boundary; sliding closes that. In-memory is a deliberate single-process scope decision (documented as the first thing to swap for Redis in a multi-worker deployment). Rate-limiting by the *authenticated* key (checked first) means unauthenticated spam can't drain a real client's quota.
- **Integration tests use real components, unit tests mock them:** `tests/test_api.py` mocks embedding generation for a fast, isolated API-contract check; `tests/test_integration_pipeline.py` deliberately leaves the embedder/bandit/judge real so it's actually testing the emergent behavior of the wired-together system, not a scripted mock response.

## BANDIT-COLLAPSE BUG: FIXED, PLUS A NEW FINDING
The original limitation (routing collapsing to 100% `mistral`, documented here previously) turned out to have a more fundamental cause than first written up: `router/router_core.py` only ever called `bandit.update()` with the *final* `selected_model` variable, which after any escalation had already been reassigned to the fallback. That meant a non-fallback arm could **never** earn the update needed to clear the confidence gate that was blocking it in the first place - a structural deadlock, not a slow convergence, independent of decay/embedding effects.

**Fixed** (`router/bandit.py`, `router/router_core.py`):
- `LinUCBRouter.select_model` now forces every arm through one unconditional warm-start trial (round-robin, in `models` order) before UCB comparisons apply, reported via a new `is_forced_exploration` flag (return signature is now a 4-tuple).
- The router's low-confidence gate is exempted for forced-exploration picks.
- Every model actually executed in a round (not just whichever response is served) now gets its own `bandit.update()` call with its own observed reward - closes the same blind spot for the low-judge-score escalation path.

**Verified:** re-ran the 300-query simulation - `{'mistral': 295, 'llama3.2:3b': 2, 'phi3': 3}` vs. `{'mistral': 300}` before. Every arm now gets real trials; convergence toward `mistral` is a genuine (and here, objectively correct, since the mock judge scores it highest deterministically) learned preference, not a bypass. New/updated tests in `test_bandit.py`, `test_router_core.py`, `test_integration_pipeline.py`. `README.md` Sections 4/5/11/12 updated to match.

**Second finding: also fixed.** Shocking the *fallback* model itself (as opposed to a non-fallback arm) did not cause migration away from it - verified interactively (75 queries in, `judge.set_shock("mistral", 0.6)`, next 75 queries still 100% `mistral`). Root cause: the low-confidence gate always routed to `self.fallback_model` unconditionally, with no check on whether the fallback itself was currently trustworthy - a circular trust assumption.

**Fix** (`router/bandit.py`, `router/router_core.py`): added `LinUCBRouter.best_known_model(context, exclude, prefer)` - returns the arm with the highest pure-exploitation estimate (no exploration bonus), excluding the arm that just triggered escalation. Both escalation paths (low-confidence and low-judge-score) now target this instead of a hardcoded model name; `fallback_model` is now only a *tie-break preference* used when candidates are genuinely tied (common right after warm-start), never an unconditional override once real evidence differs.

**Verified:** repeated a single query so the bandit builds real convergent evidence (`{'mistral': 38, 'llama3.2:3b': 1, 'phi3': 1}`), shocked the dominant model, and traffic migrated (`{'llama3.2:3b': 21, 'phi3': 5, 'mistral': 14}` over the next 40 queries) - the system now demonstrates the exact adaptive behavior `judge.set_shock()` exists to prove. New tests: `test_bandit.py` (`test_best_known_model_*`), `test_router_core.py::test_low_confidence_escalation_avoids_a_degraded_fallback`, `test_integration_pipeline.py::test_shocked_dominant_model_traffic_migrates_away` (this last one needed deterministic judge scoring to avoid flakiness from Python's per-process string-hash randomization affecting the mock judge's jitter term - the underlying fix itself was never flaky, only that one test's convergence-speed assertion was, before the judge mock was made deterministic). `README.md` Sections 5/11/12 updated to match.

## REAL-OLLAMA SMOKE TEST: DONE
Verified real (non-mock) inference against a locally running Ollama server (all four models already pulled: `llama3.2:1b`, `llama3.2:3b`, `phi3`, `mistral`). Real embeddings (`sentence-transformers`/`torch`) were deliberately left mocked per user's choice - a separate, heavier dependency the router doesn't otherwise need installed to exercise real LLM calls.

- `UnifiedLLMClient(mock_mode=False).generate(...)` - real answer in ~2s, correct token counts/cost.
- `LLMJudge.evaluate(...)` (using `mistral` as judge) - correctly parsed a real `SCORE: 10` response into `1.0` in ~17s.
- Full `OptimizationRouter` pipeline (real client+judge, mocked embedder) - warm-started through `llama3.2:1b` then `llama3.2:3b` with real generations and real judge scores; correct JSONL log entries.
- `POST /v1/route` with `ROUTER_MOCK_MODE=false` - served a real answer end-to-end through the deployed API; auth (401) and validation (422) still correctly enforced.

**Bug found and fixed along the way:** `api/dependencies.py`'s `get_router()` tied a single `ROUTER_MOCK_MODE` env var to *both* the LLM client and the embedder - so there was no way to run real Ollama inference via the API without also requiring `sentence-transformers` installed (it raised `ImportError` otherwise, confirmed via a live 500 error). Fixed by adding an independent `ROUTER_MOCK_EMBEDDINGS` env var (defaults to following `ROUTER_MOCK_MODE`, so the common case is unaffected). New tests in `tests/test_api_dependencies.py`. `README.md` Sections 10/14/15 updated to match.

## FAULT-TOLERANCE, FAILOVER & ALERTING LAYER (Complete)

### Fault injection on the client (`router/client.py`)
- `UnifiedLLMClient.inject_fault(model, probability=1.0, hard_down=False, error_types=None)` / `.clear_fault(model=None)` / `.is_faulted(model)` arm, disarm, and query a synthetic fault on a chosen backend - either probabilistic per-call failure or a full `hard_down` outage for the run.
- Three realistic failure shapes, each subclassing the real exception a live client would throw so `except ConnectionError`/`except TimeoutError` code behaves identically against a simulated fault: `SimulatedConnectionRefusedError`, `SimulatedTimeoutError`, `SimulatedServerError` (carries a real 5xx `status_code`).
- `classify_backend_error(exc)` / `backend_error_code(exc)` turn any of the above (plus `OllamaUnavailableError` for the real-Ollama path) into an explicit reason string/stable bucket key; anything unrecognized is tagged `unclassified_error[ClassName]` rather than folded into a generic bucket.

### Router-level failover (`router/router_core.py`)
- `route_and_execute()`'s primary `client.generate()` call is now wrapped: on a raised backend exception it retries exactly once against `bandit.best_known_model(exclude=selected_model)` (the same call already used for quality-based escalation, not a hardcoded fallback name). Success sets `backend_failed_over=True`/`backend_failure_reason`/`backend_failure_code` on the result dict and continues normally; if the failover target also raises (or no alternate exists), it raises `RoutingFailoverError` with **both** the primary and failover failure reasons preserved as attributes (`.primary_model/.primary_reason/.primary_code`, `.failover_model/.failover_reason/.failover_code` - `"no_alternate_backend_available"` when there was nothing to fail over to).
- Result dict also now carries `bandit_selected_model` (the raw bandit pick before any escalation) and `attempted_backend` (whichever model was actually dialed first, post confidence-escalation) so callers can tell "what the bandit wanted" apart from "what actually got called" apart from "what ultimately served the response" (`model_used`).
- **Coexistence bug found and fixed while testing the above:** the pre-existing low-judge-score retry (step 6) picks `best_known_model(exclude=selected_model)` too, which can point at a backend that's down for reasons unrelated to quality - including the very one step 4 just failed over away from. That retry's `client.generate()` call was unguarded and could crash an already-successful request. Now wrapped: on failure it logs via `classify_backend_error` and keeps the response already in hand instead of raising. Regression test: `test_router_core.py::test_quality_retry_hitting_a_faulted_backend_does_not_crash_the_request`.
- New tests: `test_router_core.py` gained 6 cases (failover succeeds, failover exhausted with both reasons preserved and distinct, no-alternate-backend, defaults on the happy path unaffected, failed arm never rewarded, the step-6 regression above).

### Fault-injection experiment (`experiments/fault_injection.py`)
- `FaultInjectionExperiment` fires N requests through the **real** `route_and_execute()` (not a reimplementation) with one backend faulted, and normalizes every outcome into an explicit record: `success_direct`, `success_failover`, `failed_no_alternate_backend`, `failed_failover_target_error` - catching only `RoutingFailoverError` for the total-failure case, reading everything else off the router's own result dict.
- Output is a single JSON results file (`experiments/results/fault_injection_results.json`, deliberately not `.jsonl`/`.log` so it isn't `.gitignore`d) with per-request records, totals, and a failure-cause breakdown grouped by stable reason codes - never a generic "error" bucket.
- The bandit is warmed up (ordinary fault-free `route_and_execute` calls) before the fault is armed so `select_model()` makes a real UCB-driven pick from the start, then runs unfrozen for the rest of the experiment (real `bandit.update()` calls and all) - this is the real router's behavior, not an isolated slice of it.
- CLI: `python -m experiments.fault_injection --backend <model> [--hard-down | --probability P] [--n-requests N] [--error-types ...] [--seed S] [--output path]`.

**Bug found and fixed: failover counts silently plateaued regardless of load.** With `mistral` hard-down, successful failovers capped near ~30-40 whether 200 or 1000 requests were fired. Root cause: `mistral`'s bandit arm (`A`/`b`/`theta`) freezes the moment it starts failing, since `bandit.update()` is never called for an arm that raised - there's no judge score to reward it with. Meanwhile the experiment only ever sent `DEFAULT_QUERIES[i % 10]` - 10 literal, repeating strings - and `ContextEmbedder`'s mock embedding is deterministic per *exact* string, so the bandit was really only ever comparing 10 fixed context vectors. Every other arm's shared linear model keeps drifting (real judge-driven updates elsewhere); once a competitor's drift happens to overtake mistral's now-immovable score at one of those 10 fixed vectors, that slot can never route back to mistral again - a one-way ratchet, once per string, that exhausts itself early and then plateaus forever. Confirmed via a diagnostic run: last-ever mistral selection at iteration 185 of 1000.
- **Fix:** `_tag_query(base_query, tag)` appends a unique, monotonically-distinct tag to every request's query text (both warm-up and main loop), so no two requests in a run ever share an identical embedding vector.
- **Verified:** `requests_routed_to_faulted_backend` / `successful_failovers` went from a fixed ~30-42 regardless of N to scaling with load: **200 requests -> 99/99 recovered, 1000 requests -> 336/336 recovered, 0 failed** in both cases.
- New regression tests: `test_no_two_requests_in_a_run_share_identical_query_text`, `test_failover_count_scales_with_n_requests_instead_of_plateauing` (asserts a >2x growth between a small and large run - a real plateau would make them nearly equal).

### HTTP webhook alerting (`observability/alerts.py`)
- `HTTPAlertHook(url, timeout=5.0)` is a real, callable `alert_hook` (POSTs `{"model","metric","message"}` as JSON via `httpx.post`) - replaces the previous docstring-only claim that the hook "would fire webhooks to PagerDuty or Slack" with actual HTTP code.
- Failures are caught and logged with the *specific* cause, never collapsed: `httpx.TimeoutException`, `httpx.HTTPStatusError` (non-2xx, logs the real status code + body), `httpx.RequestError` (connection refused, DNS failure, TLS error, etc.), plus a final bare `Exception` catch as defense-in-depth. None of these ever propagate - `AlertManager.check_metrics()` cannot be crashed by a dead or misconfigured webhook endpoint.
- New tests: `test_alerts.py` gained 7 cases (correct payload, timeout, connection error, non-2xx, all three reasons logged as distinguishably different messages, wired into `AlertManager` without crashing `check_metrics`, a successful call logs nothing at ERROR level) - HTTP layer fully mocked via `unittest.mock.patch("observability.alerts.httpx.post", ...)`, no real network calls.

### Known follow-up (not yet done)
- `observability/logger.py`'s `StructuredLogger.log_decision()` does not yet record `backend_failed_over`/`backend_failure_reason`/`backend_failure_code` - production telemetry (`router_logs.jsonl`) won't show a backend failover happened even though `route_and_execute()`'s return value now carries it. Flagged, not implemented, since it wasn't asked for.

## CURRENT STATUS
- **All phases (1 through 4), the deployment layer, documentation, and the fault-tolerance/failover/alerting layer are complete. Both bandit-collapse findings are fixed and verified. Real-Ollama inference is verified end-to-end (client, judge, full router pipeline, and the API). Router-level backend failover is implemented and verified (336/336 recovered across 1,000 injected-fault requests), and a plateau bug in that same verification harness was found and fixed.**
- 111 tests passing across the project (`pytest tests/`, ~25s, all mock mode, no external services required), confirmed stable across repeated independent runs and across randomized `PYTHONHASHSEED` (relevant here since `ContextEmbedder`'s mock embeddings hash query text).
- `requirements.txt` now includes `streamlit`, `pandas` (dashboard) and `fastapi`, `uvicorn`, `httpx` (API + HTTP alert webhook), in addition to the original `ollama`, `sentence-transformers`, `numpy`, `pytest`.
- `.gitignore` excludes `*.jsonl`/`*.log` (generated telemetry logs shouldn't be committed) but deliberately **not** `*.json` - `experiments/results/fault_injection_results.json` is a results file meant to be visible in the repo, not a rotating log.
- No open bandit-correctness or verification findings remain. Real embeddings (`sentence-transformers`/`torch`) still haven't been installed/exercised in this environment - the resolution logic for it is tested (`test_api_dependencies.py`), but not a live run - a candidate next item if full end-to-end realism (including embeddings) is ever wanted. See "Known follow-up" above for the one open item in the new layer (structured-log fields for backend failover).

## CONVENTIONS TO KEEP CONSISTENT
- Folder/file naming as established above.
- Clean, well-commented, idiomatic Python, with comments explaining "why" not just "what" - this needs to be defensible in interviews.
- Build one stage at a time (stages are now the working unit, e.g. "Stage 10a/10b/10c"), pause after each for review before continuing.
- Actually run what was built (tests, simulator, dashboard, API smoke checks) and show real output before declaring a stage done, rather than only reasoning about correctness.
