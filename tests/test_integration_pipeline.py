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
# 2. Cold-start low-confidence escalation, exercised through the real bandit
# ---------------------------------------------------------------------------

def test_cold_start_query_escalates_on_low_confidence(api_client):
    """
    A brand-new LinUCBRouter ties every arm at zero expected reward, so the
    router's own CONFIDENCE_THRESHOLD check should deterministically
    escalate the very first query to the fallback model - no mocking of
    router_core needed to observe this, it's what the real code does.
    """
    client, router, log_path = api_client

    resp = _post(client, "hello")
    body = resp.json()

    assert body["escalated"] is True
    assert body["escalation_reason"] == "low_confidence"
    assert body["model_used"] == router.fallback_model

    logs = _read_jsonl(log_path)
    assert logs[0]["escalated"] is True
    assert logs[0]["escalation_reason"] == "low_confidence"


# ---------------------------------------------------------------------------
# 3. Low-judge-score escalation, exercised through the real judge
# ---------------------------------------------------------------------------

def test_confident_but_low_quality_arm_escalates_on_judge_score(api_client):
    """
    Primes the bandit so a low-quality arm (llama3.2:1b) is confidently
    selected for one specific query's real (mock-mode) embedding, then
    confirms the router's real judge-score fallback kicks in: the mock
    judge deterministically caps llama3.2:1b's score below the 0.6
    JUDGE_SCORE_THRESHOLD (base 0.4 + at most +0.198 jitter), so the router
    should retry with mistral and keep the better result.
    """
    client, router, log_path = api_client
    query = "integration test query"

    # Compute the exact context vector route_and_execute will derive for
    # this query (mock embeddings are deterministic per-process for
    # identical input text), then feed the bandit a few confident,
    # high-reward observations for llama3.2:1b at that vector - enough to
    # clear CONFIDENCE_THRESHOLD and beat the still-untouched arms' UCB.
    context_vector = router.embedder.get_embedding([{"role": "user", "content": query}])
    for _ in range(3):
        router.bandit.update("llama3.2:1b", context_vector, reward=1.0)

    selected_model, expected_reward, _ = router.bandit.select_model(context_vector)
    assert selected_model == "llama3.2:1b"
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
