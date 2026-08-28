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

## KNOWN LIMITATION (documented, not yet fixed)
Running the simulator in mock mode causes routing to collapse onto a single arm (`mistral`) after the first couple of queries, even with a configured distribution shift and shock. Root cause (fully traced, written up in `README.md` Section 11): a cold bandit's first query always escalates to the fallback via the low-confidence path (correct behavior); but `bandit.update()` only touches the arm that actually ran, and per-update decay (`gamma=0.99`) inflates that arm's uncertainty estimate in *all* directions (not just the observed one), which keeps pushing its UCB above every untouched arm's static baseline - so the bandit keeps picking it without ever tripping escalation again, and no other arm gets a second data point. Mock embeddings (independent random vectors per query, no real semantic structure) compound this. Candidate fixes, not yet implemented: forced round-robin warm-start exploration, Thompson Sampling instead of LinUCB, decaying every arm every timestep (not only the updated one), an epsilon-exploration floor, or switching to real `sentence-transformers` embeddings. User has said to revisit this later.

## CURRENT STATUS
- **All phases (1 through 4), the deployment layer, and documentation are complete.**
- 57 tests passing across the project (`pytest tests/`, ~2-3 seconds, all mock mode, no external services required).
- `requirements.txt` now includes `streamlit`, `pandas` (dashboard) and `fastapi`, `uvicorn`, `httpx` (API), in addition to the original `ollama`, `sentence-transformers`, `numpy`, `pytest`.
- `.gitignore` now excludes `*.jsonl` (generated telemetry logs shouldn't be committed).
- The one open item is the known bandit-collapse limitation above - explicitly deferred by the user ("tell me this issue later"), not currently scheduled.

## CONVENTIONS TO KEEP CONSISTENT
- Folder/file naming as established above.
- Clean, well-commented, idiomatic Python, with comments explaining "why" not just "what" - this needs to be defensible in interviews.
- Build one stage at a time (stages are now the working unit, e.g. "Stage 10a/10b/10c"), pause after each for review before continuing.
- Actually run what was built (tests, simulator, dashboard, API smoke checks) and show real output before declaring a stage done, rather than only reasoning about correctness.
