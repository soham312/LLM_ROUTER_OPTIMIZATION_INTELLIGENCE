# HANDOFF: LLM Inference Cost & Latency Optimization Router

## PROJECT DESCRIPTION
- **What this project is:** An LLM Inference Cost & Latency Optimization Router. It routes incoming queries to one of several LLMs using a continuous contextual bandit that learns online which model to use for which query, balancing cost, latency, and response quality, instead of a traditionally trained classifier.
- **Why it exists:** Built as a portfolio project for ML/AI placement interviews. It is designed to demonstrate systems thinking and decision-making under uncertainty (not just standard model training), and to mirror a real problem every company running LLM products faces (routing between cheap and expensive models).
- **Scope:** This is the full, universal-best version of the project, not a scoped-down one. Correctness and depth are prioritized over speed.
- **Zero-cost constraint:** Uses Ollama-served local open-source models (`llama3.2:1b`, `llama3.2:3b`, `mistral`, `phi3`) instead of paid hosted APIs. It uses simulated per-call cost based on public pricing of comparable hosted models, plus a mock mode for fast, free iteration.

## WHAT'S BUILT SO FAR

### Phase 1 - Foundation
- **Project scaffold:** `router/`, `judge/`, `dashboard/`, `experiments/`, `observability/`, `tests/`
- **Unified client wrapper (`router/client.py`):** Calls local Ollama models, tracks simulated cost and real latency, and includes a mock mode toggle.
- **Query embeddings (`router/embeddings.py`):** Continuous context vectors via `sentence-transformers`, with multi-turn conversation support.

### Phase 2 - Core Routing Intelligence
- **Continuous Contextual Bandit (`router/bandit.py`):** LinUCB implementation that utilizes embedding vectors as context to balance exploration vs. exploitation mathematically.
- **Non-Stationary Handling & Shocks:** Exponential decay (sliding window) on the bandit matrices allows it to gracefully forget stale data. A deterministic `set_shock` mechanism allows us to simulate sudden model quality drops.
- **LLM-as-a-Judge & Escalation (`judge/judge.py`, `router/router_core.py`):** Uses an impartial LLM prompt to grade answers on a 0-1 scale with built-in heuristic length-bias mitigation. The `OptimizationRouter` ties the system together, intercepting low-confidence decisions or low-quality initial answers to safely escalate to a fallback model.

### Phase 3 - Realistic Evaluation
- **Streaming Traffic Simulator (`experiments/simulator.py`):** Simulates live production traffic with distinct topical distributions (chat, math, code) and executes mid-run distribution shifts to explicitly test the bandit's adaptability.
- **Validation Harness (`experiments/validation.py`):** 
  - *Sequential A/B Testing* to track cumulative metrics over time (avoiding the non-IID pitfalls of naive t-tests on shifting streaming data).
  - *Doubly Robust Off-Policy Evaluation* to robustly estimate counterfactual routing policies using logged data, avoiding the massive variance of pure IPS and the bias of pure Direct Method imputation.

### Phase 4 - Production Layer (In Progress)
- **Structured Telemetry Logging (`observability/logger.py`):** Extracted routing telemetry (latency, cost, confidence, scores) into robust JSONL logs.
- **Per-Model SLA Tracking (`observability/metrics.py`):** Computes rolling p50, p95, and p99 latency percentiles alongside model escalation rates using memory-efficient rolling windows (`collections.deque`).
- **SLA Alerting (`observability/alerts.py`):** Evaluates SLA metrics against thresholds and fires alerts (and configurable webhooks) upon detecting silent model degradation.
- **Dashboard Data Layer (`dashboard/data_layer.py`):** Built the underlying Python backend to parse JSONL logs and compute aggregated metrics, timeseries data, and per-model statistics (zero external dependencies like Pandas to keep it robust), ready to power the final UI.

## KEY DESIGN DECISIONS AND WHY
- **Continuous embeddings instead of discrete query buckets:** Needed for the contextual bandit to learn non-linear decision boundaries as query diversity grows.
- **Mock mode with deterministic hashing:** The same query always produces the same mock embedding/response, so bandit convergence can be tested without waiting on real inference every time.
- **Simulated pricing mapped to real hosted-model tiers:** Maps local models (e.g., `llama3.2:1b` as a cheap-tier proxy, `mistral` as an expensive-tier proxy). This keeps the project free to run while preserving a realistic cost/quality/latency tradeoff.
- **Cumulative tracking and Doubly Robust OPE:** Chosen specifically to demonstrate advanced MLE understanding that goes beyond simple static train/test splits.

## CURRENT STATUS
- Phases 1, 2, and 3 are 100% complete.
- Phase 4 is near completion (Stages 8a, 8b, 8c, and 9a are done).
- All 28 unit tests across the project are passing seamlessly (`pytest tests/`).
- Ready to build the final visualization layer (Streamlit dashboard UI).

## CONVENTIONS TO KEEP CONSISTENT
- Folder/file naming as established above.
- Clean, well-commented, idiomatic Python, with comments explaining "why" not just "what" - this needs to be defensible in interviews.
- Build one phase at a time, pause after each phase for review before continuing.
