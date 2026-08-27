# HANDOFF: LLM Inference Cost & Latency Optimization Router

## PROJECT DESCRIPTION
- **What this project is:** An LLM Inference Cost & Latency Optimization Router. It routes incoming queries to one of several LLMs using a continuous contextual bandit that learns online which model to use for which query, balancing cost, latency, and response quality, instead of a traditionally trained classifier.
- **Why it exists:** Built as a portfolio project for ML/AI placement interviews. It is designed to demonstrate systems thinking and decision-making under uncertainty (not just standard model training), and to mirror a real problem every company running LLM products faces (routing between cheap and expensive models).
- **Scope:** This is the full, universal-best version of the project, not a scoped-down one. Correctness and depth are prioritized over speed.
- **Zero-cost constraint:** Uses Ollama-served local open-source models (`llama3.2:1b`, `llama3.2:3b`, `mistral`, `phi3`) instead of paid hosted APIs. It uses simulated per-call cost based on public pricing of comparable hosted models, plus a mock mode for fast, free iteration.

## WHAT'S BUILT SO FAR (Phase 1 - Foundation)
- **Project scaffold:** `router/`, `judge/`, `dashboard/`, `experiments/`, `observability/`, `tests/`
- **Unified client wrapper (`router/client.py`):** Calls local Ollama models, tracks simulated cost and real latency, and includes a mock mode toggle.
- **Query embeddings (`router/embeddings.py`):** Continuous context vectors via `sentence-transformers`, with multi-turn conversation support.
- **Tests (`tests/test_client.py`, `tests/test_embeddings.py`):** All passing (5/5).

## KEY DESIGN DECISIONS AND WHY
- **Continuous embeddings instead of discrete query buckets:** Needed for the contextual bandit to learn non-linear decision boundaries as query diversity grows.
- **Mock mode with deterministic hashing:** The same query always produces the same mock embedding/response, so bandit convergence can be tested without waiting on real inference every time.
- **Simulated pricing mapped to real hosted-model tiers:** Maps local models (e.g., `llama3.2:1b` as a cheap-tier proxy, `mistral` as an expensive-tier proxy). This keeps the project free to run while preserving a realistic cost/quality/latency tradeoff.

## CURRENT STATUS
- Phase 1 complete and verified: all tests passing, Ollama models pulled and working locally.
- Committed to git, pushed to GitHub.

## CONVENTIONS TO KEEP CONSISTENT
- Folder/file naming as established above.
- Clean, well-commented, idiomatic Python, with comments explaining "why" not just "what" - this needs to be defensible in interviews.
- Build one phase at a time, pause after each phase for review before continuing.
