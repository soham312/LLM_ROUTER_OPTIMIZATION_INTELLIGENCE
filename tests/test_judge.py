import pytest
from unittest.mock import MagicMock
from judge.judge import LLMJudge
from router.client import UnifiedLLMClient, LLMResponse

def test_judge_mock_scoring():
    client = UnifiedLLMClient(mock_mode=True)
    judge = LLMJudge(client=client)
    
    score, metadata = judge.evaluate("Hello", "Hi there", "mistral")
    assert 0.0 <= score <= 1.0
    assert "Mock evaluation" in metadata["reasoning"]
    
def test_judge_length_bias_penalty():
    client = UnifiedLLMClient(mock_mode=False)
    # Force mock_mode to False even if ollama is missing, so we test the real logic path
    client.mock_mode = False
    
    # Mock the underlying client to return a score of 9 (0.9 normalized)
    client.generate = MagicMock(return_value=LLMResponse(
        id="123", model="mistral", response_text="SCORE: 9", 
        prompt_tokens=10, completion_tokens=10, total_tokens=20, 
        latency_ms=100.0, simulated_cost=0.1, is_mock=False
    ))
    judge = LLMJudge(client=client)
    
    # Prompt is 2 words. Response is > 10 words and > 100 words? No, let's make response 101 words.
    short_prompt = "Short prompt"
    long_response = "word " * 105
    
    score, metadata = judge.evaluate(short_prompt, long_response, "mistral")
    
    assert metadata["raw_score"] == 0.9
    assert metadata["length_penalty"] == 0.1
    assert score == 0.8
    
def test_judge_shock():
    client = UnifiedLLMClient(mock_mode=True)
    judge = LLMJudge(client=client)
    
    base_score, _ = judge.evaluate("Test", "Response", "mistral")
    
    judge.set_shock("mistral", 0.5)
    shock_score, metadata = judge.evaluate("Test", "Response", "mistral")
    
    assert metadata["shock_penalty"] == 0.5
    assert shock_score == max(0.0, base_score - 0.5)
    
    judge.clear_shock("mistral")
    cleared_score, _ = judge.evaluate("Test", "Response", "mistral")
    assert cleared_score == base_score
