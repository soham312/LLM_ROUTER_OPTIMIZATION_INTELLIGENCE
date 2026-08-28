from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.rate_limiter as rate_limiter_module
from api import auth as auth_module
from api.dependencies import get_router
from api.main import app
from api.rate_limiter import RateLimiter
from judge.judge import LLMJudge
from observability.logger import StructuredLogger
from router.bandit import LinUCBRouter
from router.client import UnifiedLLMClient
from router.embeddings import ContextEmbedder
from router.router_core import OptimizationRouter


# ---------------------------------------------------------------------------
# RateLimiter - pure unit tests, no FastAPI/HTTP involved
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_requests_under_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("client-a")  # should not raise


def test_rate_limiter_blocks_requests_over_the_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("client-a")
    limiter.check("client-a")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("client-a")

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


def test_rate_limiter_tracks_keys_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check("client-a")
    limiter.check("client-b")  # different bucket - should not raise


def test_rate_limiter_sliding_window_expires_old_requests(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(rate_limiter_module.time, "monotonic", lambda: fake_now[0])

    limiter = RateLimiter(max_requests=1, window_seconds=10)
    limiter.check("client-a")

    with pytest.raises(HTTPException):
        limiter.check("client-a")

    fake_now[0] += 11  # advance past the window
    limiter.check("client-a")  # should succeed again now that the old entry aged out


# ---------------------------------------------------------------------------
# Auth - pure unit tests
# ---------------------------------------------------------------------------

def test_verify_api_key_accepts_a_configured_key(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEYS", "key-one,key-two")
    assert auth_module.verify_api_key(api_key="key-one") == "key-one"


def test_verify_api_key_rejects_an_unknown_key(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEYS", "key-one")
    with pytest.raises(HTTPException) as exc_info:
        auth_module.verify_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_verify_api_key_rejects_a_missing_key(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEYS", "key-one")
    with pytest.raises(HTTPException) as exc_info:
        auth_module.verify_api_key(api_key=None)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# End-to-end API tests via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def wired_router(tmp_path):
    """
    A small, fully mock-mode OptimizationRouter for exercising the real
    routing path through the API - deterministic embedding dimension and a
    tmp_path log file so tests never touch the production log or download a
    real embedding model.
    """
    client = UnifiedLLMClient(mock_mode=True)
    embedder = ContextEmbedder(mock_mode=True)
    embedder.get_embedding = MagicMock(return_value=np.ones(10))
    bandit = LinUCBRouter(models=list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys()), embedding_dim=10)
    judge = LLMJudge(client)
    structured_logger = StructuredLogger(log_filepath=str(tmp_path / "api_test_logs.jsonl"))

    return OptimizationRouter(
        client=client,
        embedder=embedder,
        bandit=bandit,
        judge=judge,
        fallback_model="mistral",
        structured_logger=structured_logger,
    )


@pytest.fixture
def client(wired_router, monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEYS", "test-key")
    # A generous per-test limiter so unrelated tests don't compete for quota.
    monkeypatch.setattr(rate_limiter_module, "rate_limiter", RateLimiter(max_requests=100, window_seconds=60))

    app.dependency_overrides[get_router] = lambda: wired_router
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_health_check_does_not_require_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_route_without_api_key_is_rejected(client):
    resp = client.post("/v1/route", json={"query": "hello"})
    assert resp.status_code == 401


def test_route_with_wrong_api_key_is_rejected(client):
    resp = client.post("/v1/route", json={"query": "hello"}, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


def test_route_with_valid_api_key_returns_routing_result(client):
    resp = client.post("/v1/route", json={"query": "hello"}, headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["model_used"] in UnifiedLLMClient.PRICING_PER_1M_TOKENS
    assert 0.0 <= body["judge_score"] <= 1.0
    assert body["cost"] >= 0.0
    assert body["latency_ms"] >= 0.0
    assert isinstance(body["escalated"], bool)


def test_route_accepts_conversation_history(client):
    resp = client.post(
        "/v1/route",
        json={
            "query": "and in Python?",
            "conversation_history": [{"role": "user", "content": "How do I reverse a string?"}],
        },
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200


def test_route_rejects_empty_query(client):
    resp = client.post("/v1/route", json={"query": ""}, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 422


def test_rate_limit_enforced_end_to_end(client, monkeypatch):
    monkeypatch.setattr(rate_limiter_module, "rate_limiter", RateLimiter(max_requests=2, window_seconds=60))

    first = client.post("/v1/route", json={"query": "q1"}, headers={"X-API-Key": "test-key"})
    second = client.post("/v1/route", json={"query": "q2"}, headers={"X-API-Key": "test-key"})
    third = client.post("/v1/route", json={"query": "q3"}, headers={"X-API-Key": "test-key"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_rate_limit_does_not_penalize_unauthenticated_requests(client, monkeypatch):
    # An invalid key should fail with 401 without ever touching the rate
    # limiter bucket for the real key.
    monkeypatch.setattr(rate_limiter_module, "rate_limiter", RateLimiter(max_requests=1, window_seconds=60))

    client.post("/v1/route", json={"query": "q1"}, headers={"X-API-Key": "wrong-key"})
    client.post("/v1/route", json={"query": "q2"}, headers={"X-API-Key": "wrong-key"})

    # The real key's bucket should still be untouched.
    resp = client.post("/v1/route", json={"query": "q3"}, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
