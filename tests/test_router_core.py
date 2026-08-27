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
    
    # Mock bandit to return very low confidence
    bandit.select_model = MagicMock(return_value=("model_a", 0.1, 0.1))
    
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
    bandit.select_model = MagicMock(return_value=("model_a", 0.9, 0.1))
    
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
