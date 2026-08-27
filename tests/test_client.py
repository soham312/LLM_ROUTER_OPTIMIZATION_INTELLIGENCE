import pytest
from router.client import UnifiedLLMClient

def test_mock_client_generation():
    client = UnifiedLLMClient(mock_mode=True)
    response = client.generate("llama3.2:1b", "Hello world")
    
    assert response.is_mock is True
    assert response.model == "llama3.2:1b"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert response.simulated_cost > 0
    assert response.latency_ms > 0

def test_cost_calculation():
    client = UnifiedLLMClient(mock_mode=True)
    
    # 1 million tokens for llama3.2:1b should be $0.10
    cost = client._calculate_cost("llama3.2:1b", 1_000_000)
    assert cost == 0.10
    
    # 1 million tokens for mistral should be $1.00
    cost = client._calculate_cost("mistral", 1_000_000)
    assert cost == 1.00
