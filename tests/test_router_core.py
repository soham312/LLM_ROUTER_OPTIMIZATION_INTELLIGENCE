import pytest
import numpy as np
from unittest.mock import MagicMock
from router.client import UnifiedLLMClient, LLMResponse
from router.embeddings import ContextEmbedder
from router.bandit import LinUCBRouter
from judge.judge import LLMJudge
from router.router_core import OptimizationRouter

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
