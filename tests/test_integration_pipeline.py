"""
STAGE 10b: End-to-end integration tests.

Exercises the full pipeline as it actually runs in production: a request
hits the FastAPI endpoint, flows through the real (unmocked) embedder,
LinUCB bandit, and LLM judge, gets escalated when the router's own real
thresholds say so, gets persisted by the real StructuredLogger, and is then
correctly consumed by the Stage 8b/9a observability and dashboard layers.

Unlike tests/test_api.py (checks the API contract with a router fixture
that mocks out embedding generation) and tests/test_router_core.py
(unit-tests escalation logic against a heavily mocked bandit/judge), these
tests wire every real component together and only take control where a test
needs to *observe or force* a specific pipeline branch (e.g. priming the
bandit's confidence for one arm to deterministically hit the
low-judge-score escalation path).
"""

import json
from collections import Counter

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.rate_limiter as rate_limiter_module
from api.dependencies import get_router
from api.main import app
from api.rate_limiter import RateLimiter
from dashboard.data_layer import DashboardDataLayer
from judge.judge import LLMJudge
from observability.alerts import AlertManager
from observability.logger import StructuredLogger
from observability.metrics import SLATracker
from router.bandit import LinUCBRouter
from router.client import UnifiedLLMClient
from router.embeddings import ContextEmbedder
from router.router_core import OptimizationRouter

MODELS = list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys())
API_KEY = "integration-test-key"


def _build_real_router(tmp_path):
    """
    Wires up the actual Phase 1-4 stack - real embedder, real bandit, real
    judge - with mock-mode LLM calls (per the project's zero-cost design)
    and a private log file so tests never touch production telemetry.
    """
    client = UnifiedLLMClient(mock_mode=True)
    embedder = ContextEmbedder(mock_mode=True)
    bandit = LinUCBRouter(models=MODELS, embedding_dim=embedder.embedding_dim)
    judge = LLMJudge(client)
    log_path = tmp_path / "integration_logs.jsonl"
    structured_logger = StructuredLogger(log_filepath=str(log_path))

    router = OptimizationRouter(
        client=client,
        embedder=embedder,
        bandit=bandit,
        judge=judge,
        fallback_model="mistral",
        structured_logger=structured_logger,
    )
    return router, str(log_path)


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    """Wires the real pipeline into the FastAPI app; yields (TestClient, router, log_path)."""
    router, log_path = _build_real_router(tmp_path)

    monkeypatch.setenv("ROUTER_API_KEYS", API_KEY)
    # Generous limit - these tests are about the pipeline, not rate limiting
    # (that's covered by tests/test_api.py).
    monkeypatch.setattr(rate_limiter_module, "rate_limiter", RateLimiter(max_requests=1000, window_seconds=60))

    app.dependency_overrides[get_router] = lambda: router
    test_client = TestClient(app)
    yield test_client, router, log_path
    app.dependency_overrides.clear()


def _post(client, query, conversation_history=None):
    payload = {"query": query}
    if conversation_history is not None:
        payload["conversation_history"] = conversation_history
    return client.post("/v1/route", json=payload, headers={"X-API-Key": API_KEY})


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# 1. Single query, full pipeline: API response and persisted log must agree
# ---------------------------------------------------------------------------

def test_single_query_flows_through_bandit_judge_and_logger(api_client):
    client, router, log_path = api_client

    resp = _post(client, "Write a Python function to reverse a string.")
    assert resp.status_code == 200
    body = resp.json()

    logs = _read_jsonl(log_path)
    assert len(logs) == 1
    entry = logs[0]

    # The API response and the structured telemetry describe the same event.
    assert entry["model_used"] == body["model_used"]
    assert entry["judge_score"] == pytest.approx(body["judge_score"])
    assert entry["actual_cost"] == pytest.approx(body["cost"])
    assert entry["actual_latency_ms"] == pytest.approx(body["latency_ms"])
    assert entry["escalated"] == body["escalated"]
    assert entry["escalation_reason"] == body["escalation_reason"]
    assert entry["query"] == "Write a Python function to reverse a string."


# ---------------------------------------------------------------------------
# 2. Cold-start: forced warm-start trial, not a low-confidence escalation
# ---------------------------------------------------------------------------

def test_cold_start_first_query_is_a_forced_warm_start_not_a_confidence_escalation(api_client):
    """
    Bug fix (post-Stage 10a): a brand-new bandit used to deadlock - every
    arm ties at expected_reward == 0.0, which always fails the confidence
    gate, which always re-routes to the fallback model, which means a
    non-fallback arm could NEVER earn its first real trial. LinUCBRouter now
    forces every arm to be tried once (warm-start) before the confidence
    gate applies at all, so the very first query should go to the *first*
    model in the router's arm list - not skip straight to the fallback via
    "low_confidence".

    llama3.2:1b (first in PRICING_PER_1M_TOKENS order) is deterministically
    capped by the mock judge below JUDGE_SCORE_THRESHOLD (base 0.4 + at most
    +0.198 jitter, always < 0.6), so this specific query still ends up
    escalating - but via the real "low_judge_score" path, not because the
    bandit was never allowed to try the arm in the first place.
    """
    client, router, log_path = api_client
    first_arm = router.bandit.models[0]
    assert first_arm == "llama3.2:1b"

    resp = _post(client, "hello")
    body = resp.json()

    assert body["escalated"] is True
    assert body["escalation_reason"] == "low_judge_score"
    assert body["model_used"] == router.fallback_model

    # The deadlock fix's whole point: the de-prioritized first arm still
    # earned a real update from its own trial, not just the fallback.
    assert router.bandit.pull_counts[first_arm] == 1
    assert router.bandit.pull_counts[router.fallback_model] == 1

    logs = _read_jsonl(log_path)
    assert logs[0]["escalated"] is True
    assert logs[0]["escalation_reason"] == "low_judge_score"


def test_warm_start_covers_every_arm_within_the_first_few_queries(api_client):
    """
    Structural verification of the deadlock fix: every arm should have
    received at least one real trial well before traffic volume would make
    that plausible by chance alone under the old (broken) behavior, where a
    non-fallback arm could never be tried at all.
    """
    client, router, log_path = api_client

    for i in range(len(router.bandit.models)):
        _post(client, f"warm start probe query {i}")

    assert all(count >= 1 for count in router.bandit.pull_counts.values())


# ---------------------------------------------------------------------------
# 3. Low-judge-score escalation via a genuine (non-warm-start) UCB pick
# ---------------------------------------------------------------------------

def test_confident_but_low_quality_arm_escalates_on_judge_score(api_client):
    """
    Exhausts warm-start for every arm with throwaway updates, then primes
    llama3.2:1b so it's confidently selected via genuine UCB comparison
    (not a forced warm-start trial) for one specific query's real
    (mock-mode) embedding. Confirms the router's real judge-score fallback
    still kicks in: the mock judge deterministically caps llama3.2:1b's
    score below the 0.6 JUDGE_SCORE_THRESHOLD, so the router should retry
    with mistral and keep the better result.
    """
    client, router, log_path = api_client
    query = "integration test query"

    dummy_context = router.embedder.get_embedding([{"role": "user", "content": "warm-start filler"}])
    for model in router.bandit.models:
        router.bandit.update(model, dummy_context, reward=0.01)

    # Compute the exact context vector route_and_execute will derive for
    # this query (mock embeddings are deterministic per-process for
    # identical input text), then feed the bandit a few confident,
    # high-reward observations for llama3.2:1b at that vector - enough to
    # win the UCB comparison against the other arms' single throwaway update.
    context_vector = router.embedder.get_embedding([{"role": "user", "content": query}])
    for _ in range(3):
        router.bandit.update("llama3.2:1b", context_vector, reward=1.0)

    selected_model, expected_reward, _, is_forced = router.bandit.select_model(context_vector)
    assert selected_model == "llama3.2:1b"
    assert is_forced is False
    assert expected_reward >= router.CONFIDENCE_THRESHOLD

    resp = _post(client, query)
    body = resp.json()

    assert body["escalated"] is True
    assert body["escalation_reason"] == "low_judge_score"
    assert body["model_used"] == "mistral"

    logs = _read_jsonl(log_path)
    assert logs[-1]["escalation_reason"] == "low_judge_score"


# ---------------------------------------------------------------------------
# 4. Multi-query batch: bandit state actually changes, log volume matches
# ---------------------------------------------------------------------------

def test_batch_of_queries_updates_bandit_state_and_logs_every_call(api_client):
    client, router, log_path = api_client
    queries = [
        "What is the integral of x^2?",
        "Write a SQL query to join two tables.",
        "Tell me a joke.",
        "Explain what a closure is in JavaScript.",
        "How's the weather today?",
    ]

    initial_A = {m: router.bandit.A[m].copy() for m in MODELS}

    for q in queries:
        resp = _post(client, q)
        assert resp.status_code == 200

    logs = _read_jsonl(log_path)
    assert len(logs) == len(queries)

    # At least one arm's bandit state must have moved from its initial
    # identity matrix - proof route_and_execute actually called
    # bandit.update() for real, not just returned a static response.
    changed = any(not np.allclose(router.bandit.A[m], initial_A[m]) for m in MODELS)
    assert changed


# ---------------------------------------------------------------------------
# 5. Downstream tracking: Stage 8b SLA tracker + Stage 8c alerting consume
#    the logs produced by this real pipeline correctly
# ---------------------------------------------------------------------------

def test_logged_traffic_feeds_sla_tracker_and_alerting(api_client):
    client, router, log_path = api_client
    for q in ["query one", "query two", "query three", "query four"]:
        _post(client, q)

    tracker = SLATracker()
    tracker.process_log_file(log_path)
    sla_metrics = tracker.get_metrics()

    logged_models = {log["model_used"] for log in _read_jsonl(log_path)}
    assert set(sla_metrics.keys()) == logged_models
    for metrics in sla_metrics.values():
        assert metrics["p50_latency_ms"] >= 0.0
        assert 0.0 <= metrics["escalation_rate"] <= 1.0

    # The cold-start bandit guarantees the very first query escalates (see
    # test_cold_start_query_escalates_on_low_confidence), so a strict
    # escalation-rate threshold must trip at least one alert - confirming
    # SLATracker's output feeds directly into AlertManager with no
    # transformation needed, end to end.
    alert_manager = AlertManager(max_escalation_rate=0.1)
    alerts = alert_manager.check_metrics(sla_metrics)
    assert len(alerts) > 0
    assert any(a["metric"] == "escalation_rate" for a in alerts)


# ---------------------------------------------------------------------------
# 6. Downstream tracking: Stage 9a dashboard data layer reflects exactly
#    what flowed through the API
# ---------------------------------------------------------------------------

def test_logged_traffic_feeds_dashboard_data_layer_correctly(api_client):
    client, router, log_path = api_client
    queries = ["a", "b", "c"]
    costs, scores = [], []

    for q in queries:
        body = _post(client, q).json()
        costs.append(body["cost"])
        scores.append(body["judge_score"])

    data_layer = DashboardDataLayer(log_path)
    summary = data_layer.get_summary_metrics()

    assert summary["total_queries"] == len(queries)
    assert summary["total_cost"] == pytest.approx(sum(costs))
    assert summary["avg_judge_score"] == pytest.approx(sum(scores) / len(scores))

    per_model = data_layer.get_per_model_stats()
    assert sum(stats["query_count"] for stats in per_model.values()) == len(queries)


# ---------------------------------------------------------------------------
# 7. Bug fix: a degraded dominant model no longer traps the system
# ---------------------------------------------------------------------------

def test_shocked_dominant_model_traffic_migrates_away(api_client):
    """
    Bug fix: the low-confidence escalation gate used to always route to a
    hardcoded fallback_model unconditionally - so if the currently-dominant
    model (which happens to be the configured fallback) itself degraded,
    the system had no way to notice and kept escalating right back into the
    very model that just failed.

    Repeats one query many times so the bandit builds real, convergent
    evidence for that exact context, shocks whichever model that evidence
    converged on, and confirms traffic actually migrates elsewhere
    afterward.

    The judge's mock scoring is replaced with a deterministic version (real
    `set_shock` state is still consulted, so the shock mechanism itself
    stays real) purely to remove its `hash(prompt + response)` jitter -
    that jitter is reseeded by Python's per-process hash randomization, so
    leaving it in made this specific test's *convergence speed* flaky
    across separate test runs even though the underlying fix is correct
    every time; every other test in this file has no such dependency on
    run-to-run hash stability.
    """
    client, router, log_path = api_client
    query = "what is the capital of france"

    base_scores = {"llama3.2:1b": 0.3, "llama3.2:3b": 0.75, "phi3": 0.75, "mistral": 0.9}

    def deterministic_evaluate(q, response_text, model):
        score = base_scores.get(model, 0.5)
        penalty = router.judge._shock_state.get(model, 0.0)
        return max(0.0, score - penalty), {"reasoning": "deterministic test double"}

    router.judge.evaluate = deterministic_evaluate

    served_before = Counter()
    for _ in range(40):
        body = _post(client, query).json()
        served_before[body["model_used"]] += 1

    dominant_model, dominant_count = served_before.most_common(1)[0]
    # Sanity check this test actually set up genuine convergence, not noise.
    assert dominant_count >= 30

    router.judge.set_shock(dominant_model, 0.6)

    served_after = Counter()
    for _ in range(40):
        body = _post(client, query).json()
        served_after[body["model_used"]] += 1

    assert served_after[dominant_model] < dominant_count / 2
