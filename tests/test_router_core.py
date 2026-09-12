import pytest
import numpy as np
from unittest.mock import MagicMock
from router.client import (
    LLMResponse,
    SimulatedConnectionRefusedError,
    SimulatedServerError,
    SimulatedTimeoutError,
    UnifiedLLMClient,
)
from router.embeddings import ContextEmbedder
from router.bandit import LinUCBRouter
from judge.judge import LLMJudge
from router.router_core import OptimizationRouter, RoutingFailoverError

@pytest.fixture
def mocked_components():
    client = UnifiedLLMClient(mock_mode=True)
    # Mock client generate
    client.generate = MagicMock(return_value=LLMResponse(
        id="123", model="model_a", response_text="Test response", 
        prompt_tokens=10, completion_tokens=10, total_tokens=20, 
        latency_ms=100.0, simulated_cost=0.0, is_mock=True
    ))
    
    embedder = ContextEmbedder(mock_mode=True)
    embedder.get_embedding = MagicMock(return_value=np.ones(10))
    
    bandit = LinUCBRouter(["model_a", "mistral"], embedding_dim=10)
    judge = LLMJudge(client)
    
    return client, embedder, bandit, judge

def test_escalation_low_confidence(mocked_components):
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    router.CONFIDENCE_THRESHOLD = 0.5
    
    # Mock bandit to return very low confidence (not a forced-exploration pick)
    bandit.select_model = MagicMock(return_value=("model_a", 0.1, 0.1, False))
    
    # Judge will just return a decent score
    judge.evaluate = MagicMock(return_value=(0.8, {}))
    
    result = router.route_and_execute("Test query")
    
    assert result["escalated"] is True
    assert result["escalation_reason"] == "low_confidence"
    assert result["model_used"] == "mistral"
    # Ensure client was called with fallback model
    client.generate.assert_called_with("mistral", "Test query")

def test_escalation_low_judge_score(mocked_components):
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    router.CONFIDENCE_THRESHOLD = 0.1
    router.JUDGE_SCORE_THRESHOLD = 0.6
    
    # Mock bandit to return high confidence for model_a
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))
    
    # Mock judge to return a low score for model_a, but high for mistral
    def mock_eval(query, response, model):
        if model == "model_a":
            return 0.3, {}
        return 0.9, {}
    judge.evaluate = MagicMock(side_effect=mock_eval)
    
    result = router.route_and_execute("Test query")

    assert result["escalated"] is True
    assert result["escalation_reason"] == "low_judge_score"
    assert result["model_used"] == "mistral"
    assert result["judge_score"] == 0.9


def test_low_judge_score_escalation_still_updates_the_original_arm(mocked_components):
    """
    Bug fix: previously, escalating to the fallback on a low judge score
    meant only the fallback model's arm ever got a bandit.update() call -
    the originally selected arm's own (poor) outcome was silently dropped,
    so it could never learn from its own trial. Both arms actually ran and
    produced real judge scores, so both must be updated.
    """
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    router.CONFIDENCE_THRESHOLD = 0.1
    router.JUDGE_SCORE_THRESHOLD = 0.6

    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))

    def mock_eval(query, response, model):
        return (0.3, {}) if model == "model_a" else (0.9, {})
    judge.evaluate = MagicMock(side_effect=mock_eval)

    assert bandit.pull_counts["model_a"] == 0
    assert bandit.pull_counts["mistral"] == 0

    router.route_and_execute("Test query")

    # Both the de-prioritized original arm and the fallback that actually
    # ran should have received a real update.
    assert bandit.pull_counts["model_a"] == 1
    assert bandit.pull_counts["mistral"] == 1


def test_low_confidence_escalation_avoids_a_degraded_fallback():
    """
    Bug fix: escalation used to always target the hardcoded fallback_model
    unconditionally, even if that specific model's own track record had
    degraded (e.g. via a shock) - it had no way to notice. Seeds the bandit
    so "mistral" (the configured fallback) has a real, poor track record at
    this context and "model_b" has a real, strong one, then confirms
    escalation targets "model_b" instead of blindly using the fallback.
    """
    client = UnifiedLLMClient(mock_mode=True)
    client.generate = MagicMock(return_value=LLMResponse(
        id="123", model="model_a", response_text="Test response",
        prompt_tokens=10, completion_tokens=10, total_tokens=20,
        latency_ms=100.0, simulated_cost=0.0, is_mock=True
    ))

    embedder = ContextEmbedder(mock_mode=True)
    context = np.ones(10)
    embedder.get_embedding = MagicMock(return_value=context)

    bandit = LinUCBRouter(["model_a", "model_b", "mistral"], embedding_dim=10)
    judge = LLMJudge(client)

    # Real, opposite track records at this exact context - not warm-start
    # ties, so there's nothing left for the "prefer fallback" tie-break to
    # apply to.
    bandit.update("mistral", context, reward=0.0)
    bandit.update("model_b", context, reward=1.0)

    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    router.CONFIDENCE_THRESHOLD = 0.5

    bandit.select_model = MagicMock(return_value=("model_a", 0.1, 0.1, False))
    judge.evaluate = MagicMock(return_value=(0.8, {}))

    result = router.route_and_execute("Test query")

    assert result["escalated"] is True
    assert result["escalation_reason"] == "low_confidence"
    assert result["model_used"] == "model_b"
    client.generate.assert_called_with("model_b", "Test query")


# --- Backend failover (as opposed to quality escalation above) ----------

def test_backend_failure_fails_over_to_best_known_model(mocked_components):
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")

    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))
    judge.evaluate = MagicMock(return_value=(0.8, {}))

    failover_response = LLMResponse(
        id="456", model="mistral", response_text="Failover response",
        prompt_tokens=10, completion_tokens=10, total_tokens=20,
        latency_ms=100.0, simulated_cost=0.0, is_mock=True
    )
    client.generate = MagicMock(side_effect=[
        SimulatedConnectionRefusedError("connection refused to model_a"),
        failover_response,
    ])

    result = router.route_and_execute("Test query")

    assert result["model_used"] == "mistral"
    assert result["bandit_selected_model"] == "model_a"
    assert result["attempted_backend"] == "model_a"
    assert result["backend_failed_over"] is True
    assert result["backend_failure_reason"].startswith("connection_refused:")
    assert result["backend_failure_code"] == "connection_refused"
    assert client.generate.call_count == 2
    client.generate.assert_any_call("model_a", "Test query")
    client.generate.assert_any_call("mistral", "Test query")

def test_backend_failure_raises_when_failover_target_also_fails(mocked_components):
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))

    client.generate = MagicMock(side_effect=[
        SimulatedConnectionRefusedError("model_a is down"),
        SimulatedServerError("mistral", 503),
    ])

    with pytest.raises(RoutingFailoverError) as exc_info:
        router.route_and_execute("Test query")

    err = exc_info.value
    # Both failure reasons must be preserved, distinctly - not just the
    # last exception seen.
    assert err.primary_model == "model_a"
    assert err.primary_code == "connection_refused"
    assert err.failover_model == "mistral"
    assert err.failover_code == "http_503"
    assert "connection_refused" in str(err)
    assert "http_503" in str(err)

def test_backend_failure_raises_with_no_alternate_backend_available():
    client = UnifiedLLMClient(mock_mode=True)
    client.generate = MagicMock(side_effect=SimulatedTimeoutError("model_a timed out"))

    embedder = ContextEmbedder(mock_mode=True)
    embedder.get_embedding = MagicMock(return_value=np.ones(10))

    bandit = LinUCBRouter(["model_a"], embedding_dim=10)  # only one arm in the whole pool
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))

    judge = LLMJudge(client)
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="model_a")

    with pytest.raises(RoutingFailoverError) as exc_info:
        router.route_and_execute("Test query")

    err = exc_info.value
    assert err.primary_model == "model_a"
    assert err.primary_code == "timeout"
    assert err.failover_model is None
    assert err.failover_code == "no_alternate_backend_available"
    assert "no healthy alternate backend" in str(err)

def test_no_backend_failure_reports_defaults(mocked_components):
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))
    judge.evaluate = MagicMock(return_value=(0.8, {}))

    result = router.route_and_execute("Test query")

    assert result["backend_failed_over"] is False
    assert result["backend_failure_reason"] is None
    assert result["backend_failure_code"] is None
    assert result["bandit_selected_model"] == "model_a"
    assert result["attempted_backend"] == "model_a"

def test_quality_retry_hitting_a_faulted_backend_does_not_crash_the_request(mocked_components):
    """
    Regression test for a coexistence bug the backend-failover work
    surfaced: step 6's judge-score retry picks bandit.best_known_model()
    excluding only whichever model is *currently* selected - which can
    be the very backend step 4's own failover just moved away from, or
    simply a backend that happens to be down for unrelated reasons. That
    retry call used to be unguarded, so if the retry target also raised,
    the whole request crashed with a raw, unclassified exception instead
    of falling back to the response already in hand. It must now be
    caught, logged, and the existing (already-successful) response kept.
    """
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))

    primary_response = LLMResponse(
        id="789", model="model_a", response_text="Primary response",
        prompt_tokens=10, completion_tokens=10, total_tokens=20,
        latency_ms=100.0, simulated_cost=0.0, is_mock=True
    )
    client.generate = MagicMock(side_effect=[
        primary_response,
        SimulatedServerError("mistral", 503),
    ])

    def mock_eval(query, response, model):
        return (0.3, {})  # below JUDGE_SCORE_THRESHOLD -> triggers the quality retry
    judge.evaluate = MagicMock(side_effect=mock_eval)

    result = router.route_and_execute("Test query")

    assert result["model_used"] == "model_a"
    assert result["response"] is primary_response
    assert result["escalated"] is False
    assert result["escalation_reason"] is None
    assert client.generate.call_count == 2
    assert bandit.pull_counts["model_a"] == 1
    assert bandit.pull_counts["mistral"] == 0

def test_failed_primary_backend_is_never_rewarded_only_the_one_that_responded_is(mocked_components):
    """
    A backend that raised never produced a response, so it has no judge
    score to compute a reward from - unlike the quality-escalation paths,
    which always update both arms because both actually ran. This means
    the bandit currently has no signal at all telling it to route away
    from a hard-down backend; it will keep getting selected at whatever
    rate the bandit already assigned it until something else changes its
    belief. That's a real, separate gap from failover working correctly.
    """
    client, embedder, bandit, judge = mocked_components
    router = OptimizationRouter(client, embedder, bandit, judge, fallback_model="mistral")
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1, False))
    judge.evaluate = MagicMock(return_value=(0.8, {}))

    failover_response = LLMResponse(
        id="456", model="mistral", response_text="Failover response",
        prompt_tokens=10, completion_tokens=10, total_tokens=20,
        latency_ms=100.0, simulated_cost=0.0, is_mock=True
    )
    client.generate = MagicMock(side_effect=[
        SimulatedConnectionRefusedError("connection refused"),
        failover_response,
    ])

    assert bandit.pull_counts["model_a"] == 0
    assert bandit.pull_counts["mistral"] == 0

    router.route_and_execute("Test query")

    assert bandit.pull_counts["model_a"] == 0
    assert bandit.pull_counts["mistral"] == 1
